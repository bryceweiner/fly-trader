"""A claim's ETH leg: the holder's owed ETH (the vault's ETH pot, vault/settle_eth.py) paid in native ETH on Robinhood
Chain to the EVM address that signed the claim. The claim text (v1) is unchanged: the EVM address is already in it.

none -> owed -> sending -> paid | failed        (independent of the SOL leg's status)

* The amount is fixed in the same database transaction that fixes the SOL amount (claims.process), from
  ``vault_eth_accounts`` — so a later claim can never be paid the same wei again.
* The payment is a plain value transfer from the RH bot wallet (rh/wallet.RhWallet, the same key and guard as live RH
  trading). Its ``rh_txs`` row and the claim's link to it are committed together before broadcast; after a crash the
  row is settled on chain first (receipt, superseded at its nonce, or cancelled) and re-sent only if it can no longer
  land, so a holder is never paid twice.
* Signing refused (RH_LIVE_ENABLED off, key or address missing) leaves the leg owed and alerts; it is never lost.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from . import alerts

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 4
REBROADCAST_S = 300.0
WEI = 10 ** 18


def owed(conn, evm_addr: str) -> int:
    r = conn.execute("SELECT owed FROM vault_eth_accounts WHERE evm = %s", (evm_addr,)).fetchone()
    return int(r["owed"]) if r else 0


def fix_amount(conn, cid: int, evm_addr: str) -> bool:
    """Fix the ETH leg of claim ``cid`` (once). True when there is an ETH leg to pay."""
    if not config.RH_ENABLED:
        return False
    row = conn.execute("SELECT eth_status FROM vault_claims WHERE id = %s", (cid,)).fetchone()
    if row and row["eth_status"] != "none":
        return row["eth_status"] in ("owed", "sending", "paid")
    eth = owed(conn, evm_addr)
    if eth < config.CLAIM_MIN_WEI:
        return False
    conn.execute("UPDATE vault_claims SET eth_wei = %s, eth_status = 'owed', updated_at = now() WHERE id = %s AND eth_status = 'none'", (eth, cid))
    return True


def _wallet():
    from ..rh.rpc import RhRpc
    from ..rh.wallet import RhWallet
    return RhWallet(RhRpc())


def _paid(conn, row: dict, tx_id: int) -> None:
    conn.execute("UPDATE vault_claims SET eth_status = 'paid', eth_reason = NULL, updated_at = now(), reported_at = NULL WHERE id = %s", (row["id"],))
    conn.execute("UPDATE rh_txs SET applied_at = now() WHERE id = %s", (tx_id,))
    conn.execute("INSERT INTO rh_wallet_flows (direction, kind, wei, note) VALUES ('out', 'payout', %s, %s)", (int(row["eth_wei"]), f"claim {row['id']} for {row['evm']}"))


def _retry(conn, row: dict, why: str) -> None:
    conn.execute("UPDATE vault_claims SET eth_status = 'owed', eth_tx_id = NULL, eth_reason = %s, updated_at = now() WHERE id = %s", (why[:200], row["id"]))


def pay_one(wallet, row: dict) -> str:
    """Advance one claim's ETH leg. Returns its status."""
    from ..rh.wallet import SendFailed
    cid = int(row["id"])
    with wallet.lock:
        if row.get("eth_tx_id"):                                    # settle the last attempt before anything else
            with transaction() as conn:
                tx = dict(conn.execute("SELECT * FROM rh_txs WHERE id = %s", (row["eth_tx_id"],)).fetchone())
            rc = wallet.rpc.receipt(tx["hash"])
            if rc is None and tx["status"] in ("signed", "sent"):
                age = (datetime.now(timezone.utc) - tx["created_at"]).total_seconds()
                if wallet.rpc.tx_count(wallet.address, "latest") <= int(tx["nonce"]):
                    if age < REBROADCAST_S:
                        rc = wallet.wait(tx["hash"], timeout_s=30.0)
                        if rc is None:
                            return "sending"
                    else:                                           # cannot land any more once its nonce is taken: take it
                        c = wallet.cancel(int(tx["nonce"]), int(tx["max_fee_wei"] or 0))
                        rc_c = wallet.wait(c["hash"], timeout_s=120.0)
                        if rc_c is not None:                        # its gas is a wallet flow like any other
                            with transaction() as conn:
                                wallet.record_receipt(conn, int(c["id"]), rc_c)
                                conn.execute("UPDATE rh_txs SET applied_at = now() WHERE id = %s", (c["id"],))
                        rc = wallet.rpc.receipt(tx["hash"])
            with transaction() as conn:
                if rc is not None and rc.get("blockNumber"):
                    res = wallet.record_receipt(conn, int(tx["id"]), rc)
                    if res["ok"]:
                        _paid(conn, row, int(tx["id"])); return "paid"
                    conn.execute("UPDATE rh_txs SET applied_at = now() WHERE id = %s", (tx["id"],))
                    _retry(conn, row, "payout reverted")
                else:
                    conn.execute("UPDATE rh_txs SET status = 'replaced', applied_at = now(), updated_at = now() WHERE id = %s AND status IN ('signed', 'sent')", (tx["id"],))
                    _retry(conn, row, "payout did not land")
            row = {**row, "eth_tx_id": None}
        if int(row.get("eth_attempts") or 0) >= MAX_ATTEMPTS:
            with transaction() as conn:
                conn.execute("UPDATE vault_claims SET eth_status = 'failed', eth_reason = 'the ETH payout did not land after several attempts', updated_at = now(), "
                             "reported_at = NULL WHERE id = %s", (cid,))
            alerts.send(f"claim {cid}: ETH payout failed after {MAX_ATTEMPTS} attempts ({int(row['eth_wei']) / WEI:.6f} ETH to {row['evm']})")
            return "failed"
        def link(conn, tid):
            conn.execute("UPDATE vault_claims SET eth_tx_id = %s, eth_status = 'sending', eth_attempts = eth_attempts + 1, updated_at = now() WHERE id = %s", (tid, cid))
        try:
            s = wallet.send(to=row["evm"], data=b"", value=int(row["eth_wei"]), kind="payout", on_persist=link)
        except SendFailed as e:
            with transaction() as conn:
                _retry(conn, row, f"send failed: {e}")
            return "owed"
        rc = wallet.wait(s["hash"], timeout_s=120.0)
        if rc is None:
            return "sending"
        with transaction() as conn:
            res = wallet.record_receipt(conn, s["id"], rc)
            if res["ok"]:
                _paid(conn, row, s["id"]); return "paid"
            conn.execute("UPDATE rh_txs SET applied_at = now() WHERE id = %s", (s["id"],))
            _retry(conn, row, "payout reverted")
        return "owed"


def process(wallet=None) -> dict:
    """Pay every owed or unfinished ETH leg. Never raises into the claims round."""
    out = {"paid": 0, "waiting": 0}
    with transaction() as conn:
        todo = [dict(r) for r in conn.execute("SELECT * FROM vault_claims WHERE eth_status IN ('owed', 'sending') ORDER BY id").fetchall()]
    if not todo:
        return out
    if wallet is None:
        from ..rh.guard import SigningRefused
        try:
            wallet = _wallet()
        except SigningRefused as e:
            out["waiting"] = len(todo)
            alerts.send(f"{len(todo)} ETH claim payout(s) wait: Robinhood Chain signing is off ({e})", key="eth_claims_signing", cooldown_s=3600)
            return out
    for row in todo:
        try:
            st = pay_one(wallet, row)
            out["paid"] += st == "paid"; out["waiting"] += st in ("owed", "sending")
        except Exception as e:                                      # noqa: BLE001 — one leg's failure never blocks the rest
            log.exception("claim %s ETH leg", row["id"])
            alerts.send(f"claim {row['id']}: ETH leg error {type(e).__name__}", key=f"eth_leg_{row['id']}", cooldown_s=1800)
    return out
