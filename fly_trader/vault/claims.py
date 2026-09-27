"""Claim payouts (docs/vault/SPEC.md §2): pull signed claims from the relay, verify them here, pay all SOL owed.

received -> verified -> (waiting_liquidity) -> sending -> paid | rejected | failed

* Both texts are re-rendered from the fields; a relay-supplied text is never trusted.
* One claim in flight per holder (partial unique index); the amount is fixed when the claim moves to ``sending``.
* The payment row (signature, blockhash, last valid height) is written BEFORE broadcast; after a crash the claim is
  re-checked on chain and re-signed only once its blockhash can no longer land, so a holder is never paid twice.
* Claims are paid from the Squads treasury under its weekly spending limit L2, which only the payout key may use; the
  signer (fly_trader/signer) builds and signs the payment and keeps its own once-per-claim ledger. A claim the
  treasury or this week's L2 cannot cover waits (``waiting_liquidity``) and alerts, never fails.
"""
from __future__ import annotations

import logging
import time

from psycopg.types.json import Jsonb

from .. import config
from ..chain.cluster_guard import assert_vault_signing_allowed
from ..signer.client import SignerUnavailable
from ..signer.policy import PolicyError
from ..db.apilog import record_event
from ..db.connection import transaction
from . import alerts, claim_message as cm, evm, sigs, state, walletlock

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 4
STALE_AFTER_S = 3600          # a claim first seen more than an hour after its expiry is refused (signed long ago)
TERMINAL = ("paid", "rejected", "failed")


def _set(conn, cid: int, status: str, **cols) -> None:
    sets = ", ".join([f"{k} = %({k})s" for k in cols] + ["status = %(status)s", "updated_at = now()", "reported_at = NULL"])
    conn.execute(f"UPDATE vault_claims SET {sets} WHERE id = %(id)s", {**cols, "status": status, "id": cid})


def ingest(conn, c: dict) -> int | None:
    """Store a relay claim once; returns our id (None if already known)."""
    r = conn.execute(
        "INSERT INTO vault_claims (relay_id, evm, sol, nonce, domain, uri, chain_id, sol_chain, issued_at, expires_at, evm_sig, sol_sig, status) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'received') ON CONFLICT DO NOTHING RETURNING id",
        (int(c["id"]), evm.to_checksum(c["evm"]) if evm.is_address(c.get("evm", "")) else str(c.get("evm")), c.get("sol"), c.get("nonce"),
         c.get("domain"), c.get("uri"), c.get("chain_id"), c.get("sol_chain"), c.get("issued_at"), c.get("expires_at"),
         c.get("evm_sig"), c.get("sol_sig"))).fetchone()
    return int(r["id"]) if r else None


def verify(row: dict, rh_rpc=None, now: float | None = None) -> tuple[bool, str]:
    """(ok, reason). Everything except the owed amount, which is read under the claim's own transaction."""
    now = time.time() if now is None else now
    try:
        f = cm.ClaimFields(domain=row["domain"], uri=row["uri"], chain_id=int(row["chain_id"]), sol_chain=row["sol_chain"],
                           evm=row["evm"], sol=row["sol"], nonce=row["nonce"], issued_at=row["issued_at"], expires_at=row["expires_at"])
        f.check()
    except (ValueError, TypeError, KeyError) as e:
        return False, f"malformed claim: {e}"
    if f.domain != config.VAULT_SITE_DOMAIN or f.uri != config.VAULT_SITE_URI:
        return False, "claim was signed for another site"
    if int(f.chain_id) != config.RH_CHAIN_ID or f.sol_chain != ("mainnet" if config.VAULT_CLUSTER == "mainnet-beta" else "devnet"):
        return False, "claim was signed for another network"
    seen = row.get("created_at")
    seen_ts = seen.timestamp() if hasattr(seen, "timestamp") else now
    if seen_ts > cm.parse_rfc3339(f.expires_at) + STALE_AFTER_S or cm.parse_rfc3339(f.issued_at) > now + 300:
        return False, "claim expired"
    try:
        kind = sigs.verify_evm(cm.evm_text(f), row["evm_sig"] or "", f.evm, rh_rpc)
        sigs.verify_sol(cm.sol_text(f), row["sol_sig"] or "", f.sol)
    except sigs.SignatureInvalid as e:
        return False, str(e)
    row["sig_kind"] = kind
    return True, kind


def _owed(conn, evm_addr: str) -> int:
    r = conn.execute("SELECT owed FROM vault_accounts WHERE evm = %s", (evm_addr,)).fetchone()
    return int(r["owed"]) if r else 0


def signer_claim_id(row: dict) -> str:
    """The claim's key in the signer's own ledger: our id plus the holder's signed nonce, so a rebuilt database that
    reuses an id can never make the signer pay a different claim under it."""
    return f"{int(row['id'])}:{row['nonce']}"


def _record_paid(conn, cid: int, row: dict, sig: str, slot: int, fee: int) -> None:
    _set(conn, cid, "paid", slot=slot, fee_lamports=fee)
    conn.execute("INSERT INTO vault_flows (signature, slot, block_time, direction, kind, counterparty, lamports, fee_lamports, classified_by, note) "
                 "VALUES (%s,%s,now(),'out','claim',%s,%s,%s,'sender',%s) ON CONFLICT DO NOTHING",
                 (sig, slot, row["sol"], int(row["lamports"]), fee, f"claim {cid} for {row['evm']}"))


def pay(cid: int, rpc, signer, wait=None) -> str:
    """Send (or resume) the payment of a claim in ``sending``. Returns the resulting status."""
    from ..execution.broker_live import await_confirmation
    assert_vault_signing_allowed()
    wait = wait or await_confirmation
    with walletlock.exclusive(timeout_s=180):
        with transaction() as conn:
            row = dict(conn.execute("SELECT * FROM vault_claims WHERE id = %s", (cid,)).fetchone())
        if row["status"] != "sending":
            return row["status"]
        if row.get("tx_signature"):                              # resume after a crash: did the last attempt land?
            st = (rpc.get_signature_statuses([row["tx_signature"]], search_history=True) or [None])[0]
            if st and st.get("err") is None and st.get("confirmationStatus") in ("confirmed", "finalized"):
                # Landed. Record it from the history lookup: await_confirmation polls without history, so once the
                # status has aged out of the recent cache it would report "expired" and we would pay again.
                with transaction() as conn:
                    _record_paid(conn, cid, row, row["tx_signature"], int(st.get("slot") or 0), 5000)
                return "paid"
            if st and st.get("err") is None:
                status, info = wait(rpc, row["tx_signature"], row.get("last_valid_block_height"))
                if status == "confirmed":
                    with transaction() as conn:
                        _record_paid(conn, cid, row, row["tx_signature"], int(info.get("slot") or 0), 5000)
                    return "paid"
            if st is None and rpc.get_block_height() <= int(row.get("last_valid_block_height") or 0):
                status, info = wait(rpc, row["tx_signature"], row.get("last_valid_block_height"))
                if status == "confirmed":
                    with transaction() as conn:
                        _record_paid(conn, cid, row, row["tx_signature"], int(info.get("slot") or 0), 5000)
                    return "paid"
        if int(row.get("attempts") or 0) >= MAX_ATTEMPTS:
            with transaction() as conn:
                _set(conn, cid, "failed", reason="payment did not land after several attempts")
            alerts.send(f"claim {cid} failed after {MAX_ATTEMPTS} attempts ({row['lamports']} lamports to {row['sol']})")
            return "failed"
        try:
            built = signer.call("pay_claim", claim_id=signer_claim_id(row), dest=row["sol"], lamports=int(row["lamports"]))
        except PolicyError as e:
            if e.code in ("cap", "liquidity"):
                with transaction() as conn:
                    _set(conn, cid, "waiting_liquidity", reason=str(e))
                what = ("this week's payout limit (L2) is used up: raise it from the owner wallet in the Squads app"
                        if e.code == "cap" else "the treasury holds too little")
                alerts.send(f"claim {cid} waiting ({int(row['lamports']) / config.LAMPORTS_PER_SOL:.4f} SOL): {what}", key="claim_liquidity")
                return "waiting_liquidity"
            if e.code == "in_flight":
                return "sending"
            if e.code == "paid" and row.get("tx_signature"):      # the signer saw it land; our resume check will record it
                return "sending"
            with transaction() as conn:
                _set(conn, cid, "failed", reason=f"the signer refused: {e}")
            alerts.send(f"claim {cid}: the signer refused the payment: {e}")
            return "failed"
        except SignerUnavailable as e:
            log.warning("claim %d: signer unavailable: %s", cid, e)
            return "sending"
        sig, lvbh = built["signature"], int(built["last_valid_block_height"])
        with transaction() as conn:                              # persisted before it can land
            conn.execute("UPDATE vault_claims SET tx_signature = %s, blockhash = %s, last_valid_block_height = %s, attempts = attempts + 1, "
                         "updated_at = now() WHERE id = %s", (sig, built.get("blockhash"), lvbh, cid))
        try:
            rpc.send_transaction(built["tx"])
        except Exception as e:
            log.warning("claim %d send failed: %s", cid, type(e).__name__)
        status, info = wait(rpc, sig, lvbh)
        with transaction() as conn:
            if status == "confirmed":
                _record_paid(conn, cid, row, sig, int((info or {}).get("slot") or 0), 5000)
                return "paid"
            if status == "failed_on_chain":
                _set(conn, cid, "failed", reason=f"payment failed on chain: {(info or {}).get('err')}")
                alerts.send(f"claim {cid} payment failed on chain")
                return "failed"
        return "sending"                                         # expired unlanded: the next round re-signs


def process(relay, rpc, signer, rh_rpc=None, now: float | None = None) -> dict:
    """One round: ingest new relay claims, verify, pay, report. Never raises into the worker loop."""
    out = {"ingested": 0, "paid": 0, "rejected": 0}
    after = int(state.get("relay_claims_after") or 0)
    for c in relay.pending_claims(after=after):
        try:
            with transaction() as conn:
                if ingest(conn, c) is not None:
                    out["ingested"] += 1
        except Exception as e:                                   # a row we cannot store is skipped, not retried forever
            log.warning("relay claim %s could not be stored: %s", c.get("id"), e)
            alerts.send(f"relay claim {c.get('id')} was malformed and skipped: {e}", key="claim_malformed")
        try:
            after = max(after, int(c["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    state.put("relay_claims_after", after)
    if state.halted():
        return {**out, "halted": True}
    with transaction() as conn:
        todo = [dict(r) for r in conn.execute("SELECT * FROM vault_claims WHERE status IN ('received', 'verified', 'waiting_liquidity', 'sending') "
                                               "ORDER BY id").fetchall()]
    for row in todo:
        cid = int(row["id"])
        try:
            if row["status"] == "received":
                ok, why = verify(row, rh_rpc, now)
                with transaction() as conn:
                    if not ok:
                        _set(conn, cid, "rejected", reason=why); out["rejected"] += 1
                        continue
                    busy = conn.execute("SELECT 1 FROM vault_claims WHERE evm = %s AND status IN ('verified','waiting_liquidity','sending') "
                                        "AND id <> %s", (row["evm"], cid)).fetchone()
                    if busy:
                        _set(conn, cid, "rejected", reason="another claim for this address is in flight"); out["rejected"] += 1
                        continue
                    _set(conn, cid, "verified", sig_kind=row.get("sig_kind"))
                row["status"] = "verified"
            if row["status"] in ("verified", "waiting_liquidity"):
                with walletlock.exclusive(timeout_s=180), transaction() as conn:
                    owed = _owed(conn, row["evm"]) + (int(row["lamports"] or 0) if row["status"] == "waiting_liquidity" else 0)
                    if owed < config.CLAIM_MIN_LAMPORTS:
                        _set(conn, cid, "rejected", reason=f"nothing to claim (owed {owed} lamports)"); out["rejected"] += 1
                        continue
                    _set(conn, cid, "sending", lamports=owed)
            if pay(cid, rpc, signer) == "paid":
                out["paid"] += 1
        except Exception as e:
            log.exception("claim %d", cid)
            record_event("error", "vault", f"claim {cid} error", {"error": type(e).__name__})
    report(relay)
    return out


def report(relay) -> int:
    with transaction() as conn:
        rows = conn.execute("SELECT id, relay_id, status, reason, lamports, tx_signature FROM vault_claims "
                            "WHERE reported_at IS NULL AND relay_id IS NOT NULL ORDER BY id LIMIT 200").fetchall()
    if not rows:
        return 0
    relay.report([{"id": int(r["relay_id"]), "status": r["status"], "reason": r["reason"], "lamports": r["lamports"],
                   "tx": None} for r in rows])     # the payout tx names the fly's wallet: never published
    with transaction() as conn:
        conn.execute("UPDATE vault_claims SET reported_at = now() WHERE id = ANY(%s)", ([int(r["id"]) for r in rows],))
    return len(rows)
