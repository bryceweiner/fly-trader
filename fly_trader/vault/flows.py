"""Every lamport that enters or leaves the book other than by trading (plan "Core accounting"). The book is two
accounts: the trading wallet and the Squads treasury (vault/payout.py); each has its own scan cursor.

The scanner walks each account's finalized transaction history and classifies each transaction by who signed it:

* **ours** (the trading or the payout key signed): swaps, token-account closes, claim payouts, treasury top-ups and
  sweeps. Those write their own ``vault_flows`` rows (or orders/fills rows) when they are sent; the rest move lamports
  inside R (fees, rent) and need no row. A transaction our keys signed that no table knows about is
  ``unknown_outbound``: a possible key leak, so entries and claims halt until the operator looks
  (``fly-trader vault reclassify``).
* **theirs**: SOL (or wSOL) arriving is a ``deposit`` when the sender is in ``FUNDING_ADDRESSES``, else ``profit`` (a
  gift counts for the lockers). SOL leaving the treasury with a signature of one of its multisig members is the owner
  taking principal out (``withdrawal``, alerted). Lamports leaving any other way are an ``anomaly`` and the vault halts.

Signatures are 88-char base58 strings, which the log scrubber redacts as secrets; logs here carry 12-char prefixes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)

SCAN_NAME = "solana_wallet"
SCAN_TREASURY = "solana_treasury"
PAGE = 1000


@dataclass
class Flow:
    signature: str
    slot: int
    block_time: int | None
    direction: str
    kind: str
    counterparty: str | None
    lamports: int
    fee_lamports: int = 0
    note: str | None = None


def _keys(tx: dict) -> list[dict]:
    keys = (((tx or {}).get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    return [k if isinstance(k, dict) else {"pubkey": str(k), "signer": False} for k in keys]


def _wsol_delta(meta: dict, wallet: str) -> int:
    """Change in wrapped SOL held by token accounts the wallet owns (a wSOL gift or deposit)."""
    def total(rows):
        return sum(int((r.get("uiTokenAmount") or {}).get("amount") or 0) for r in rows or []
                   if r.get("owner") == wallet and r.get("mint") == config.WSOL_MINT)
    return total(meta.get("postTokenBalances")) - total(meta.get("preTokenBalances"))


def classify(tx: dict, wallet: str, known_ours: bool, funding: set[str], our_keys: set[str] | None = None,
             owners: set[str] | None = None) -> Flow | None:
    """The flow one finalized transaction makes to ``wallet`` (the trading wallet or the treasury), or None when it is
    trading/internal or moves nothing. ``our_keys``: the server's signing keys (default: ``wallet`` itself);
    ``owners``: the treasury multisig's members, whose outbound transfers are principal withdrawals."""
    keys = _keys(tx)
    meta = tx.get("meta") or {}
    sig = ((tx.get("transaction") or {}).get("signatures") or [None])[0]
    slot, bt = int(tx.get("slot") or 0), tx.get("blockTime")
    idx = next((i for i, k in enumerate(keys) if k.get("pubkey") == wallet), None)
    if idx is None or sig is None:
        return None
    pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
    delta = (int(post[idx]) - int(pre[idx])) if idx < len(pre) and idx < len(post) else 0
    signers = {k.get("pubkey") for k in keys if k.get("signer")}
    ours = bool(signers & set(our_keys or {wallet}))
    if ours:
        if known_ours:
            return None
        return Flow(sig, slot, bt, "out" if delta <= 0 else "in", "unknown_outbound", keys[0].get("pubkey"), abs(delta),
                    int(meta.get("fee") or 0), "signed by our key but unknown to orders/claims/withdrawals/ATA closes")
    if meta.get("err") is not None:
        return None                                     # a failed transaction someone else paid for moves nothing of ours
    amount = delta + _wsol_delta(meta, wallet)
    if amount < 0:
        if signers & set(owners or ()):                  # the owner moved principal out of the treasury (Squads app)
            dest, gain = None, 0
            for i, k in enumerate(keys):
                if i != idx and i < len(pre) and i < len(post) and int(post[i]) - int(pre[i]) > gain:
                    dest, gain = k.get("pubkey"), int(post[i]) - int(pre[i])
            return Flow(sig, slot, bt, "out", "withdrawal", dest, -amount, 0, "owner withdrawal from the treasury")
        return Flow(sig, slot, bt, "out", "anomaly", keys[0].get("pubkey"), -amount, 0, "lamports left without our signature")
    if amount == 0:
        return None
    # the sender: the signer whose lamports fell the most, else the fee payer
    sender, drop = keys[0].get("pubkey"), 0
    for i, k in enumerate(keys):
        if k.get("signer") and i < len(pre) and i < len(post) and int(pre[i]) - int(post[i]) > drop:
            sender, drop = k.get("pubkey"), int(pre[i]) - int(post[i])
    kind = "deposit" if sender in funding else "profit"
    return Flow(sig, slot, bt, "in", kind, sender, amount)


def known_ours(conn, signatures: list[str]) -> set[str]:
    """Signatures our own code sent: swaps, fills, claims, withdrawals, token-account closes."""
    if not signatures:
        return set()
    rows = conn.execute(
        "SELECT signature AS s FROM orders WHERE signature = ANY(%(s)s) "
        "UNION SELECT signature FROM fills WHERE signature = ANY(%(s)s) "
        "UNION SELECT tx_signature FROM vault_claims WHERE tx_signature = ANY(%(s)s) "
        "UNION SELECT signature FROM vault_flows WHERE signature = ANY(%(s)s) AND kind IN ('withdrawal', 'claim', 'sweep', 'topup') "
        "UNION SELECT detail->>'signature' FROM wallet_events WHERE detail->>'signature' = ANY(%(s)s) "
        "UNION SELECT (value->>'signature') FROM vault_kv WHERE key LIKE 'ours:%%' AND value->>'signature' = ANY(%(s)s)",
        {"s": list(signatures)}).fetchall()
    return {r["s"] for r in rows if r["s"]}


def insert(conn, f: Flow, classified_by: str = "auto") -> bool:
    row = conn.execute(
        "INSERT INTO vault_flows (signature, slot, block_time, direction, kind, counterparty, lamports, fee_lamports, classified_by, note) "
        "VALUES (%s,%s,to_timestamp(%s),%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (signature, direction, kind) DO NOTHING RETURNING id",
        (f.signature, f.slot, f.block_time, f.direction, f.kind, f.counterparty, f.lamports, f.fee_lamports, classified_by, f.note)).fetchone()
    return row is not None


def cursor(name: str = SCAN_NAME) -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT cursor, slot, detail FROM vault_scan WHERE name = %s", (name,)).fetchone()
    return dict(r) if r else {"cursor": None, "slot": None, "detail": None}


def scanned_through() -> int:
    """Every flow with slot <= this is in vault_flows (both accounts: the lower of the two cursors)."""
    slots = [int(cursor(SCAN_NAME).get("slot") or 0)]
    if config.VAULT_MULTISIG:
        slots.append(int(cursor(SCAN_TREASURY).get("slot") or 0))
    return min(slots)


def scan(rpc, wallet: str, funding: set[str] | None = None, max_pages: int = 50, name: str = SCAN_NAME,
         our_keys: set[str] | None = None, owners: set[str] | None = None) -> dict:
    """Classify finalized transactions newer than the cursor. Idempotent; stops at the first unreadable transaction."""
    funding = set(funding if funding is not None else config.FUNDING_ADDRESSES)
    SCAN = name
    head = int(rpc.call("getSlot", [{"commitment": "finalized"}]))
    cur = cursor(SCAN)
    until = cur.get("cursor")
    newest: list[dict] = []
    before = None
    for _ in range(max_pages):
        page = rpc.get_signatures_for_address(wallet, before=before, until=until, limit=PAGE, commitment="finalized")
        if not page:
            break
        newest.extend(page)
        before = page[-1]["signature"]
        if len(page) < PAGE:
            break
    else:
        log.warning("vault flow scan: more than %d pages behind; continuing next round", max_pages)
        head = None                                        # not caught up: do not advance the scanned-through slot
    counts: dict[str, int] = {}
    halted, withdrawals = [], []
    for entry in reversed(newest):                         # oldest first so the cursor only ever moves forward
        sig = entry["signature"]
        tx = rpc.get_transaction(sig)
        if tx is None:
            log.info("vault flow scan: %s… not readable yet; resuming next round", sig[:12])
            head = None
            break
        with transaction() as conn:
            f = classify(tx, wallet, sig in known_ours(conn, [sig]), funding, our_keys, owners)
            if f is not None and insert(conn, f):
                counts[f.kind] = counts.get(f.kind, 0) + 1
                if f.kind in ("unknown_outbound", "anomaly"):
                    halted.append(f)
                elif f.kind == "withdrawal" and name == SCAN_TREASURY:
                    withdrawals.append(f)
            conn.execute(
                "INSERT INTO vault_scan (name, cursor, slot, detail, updated_at) VALUES (%s,%s,%s,%s,now()) "
                "ON CONFLICT (name) DO UPDATE SET cursor = EXCLUDED.cursor, detail = EXCLUDED.detail, updated_at = now()",
                (SCAN, sig, cur.get("slot"), Jsonb({"last_slot": int(entry.get("slot") or 0)})))
    if head is not None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO vault_scan (name, slot, updated_at) VALUES (%s,%s,now()) "
                "ON CONFLICT (name) DO UPDATE SET slot = GREATEST(COALESCE(vault_scan.slot, 0), EXCLUDED.slot), updated_at = now()",
                (SCAN, head))
    chk = None
    if halted:
        from ..chain.rpc import check_rpc
        chk = check_rpc()
    for f in halted:
        if chk is not None and not _confirmed_elsewhere(chk, f, wallet, funding, our_keys, owners):
            from . import alerts
            alerts.send(f"the RPC providers disagree about transaction {f.signature[:12]}… ({f.kind} on the primary): not halting, "
                        "look at it on an explorer", key="rpc_disagree", cooldown_s=1800)
            continue
        _halt(f)
    for f in withdrawals:
        from . import alerts
        alerts.send(f"owner withdrawal from the treasury: {f.lamports / config.LAMPORTS_PER_SOL:.4f} SOL to {f.counterparty}"
                    + ("" if f.counterparty in funding else " (NOT a funding address: if this was not you, the owner wallet is compromised)"))
    if counts:
        log.info("vault flow scan: %s", counts)
    return {"new": counts, "through_slot": head, "seen": len(newest)}


def scan_all(rpc, trading: str, our_keys: set[str]) -> dict:
    """Both accounts of the book: the trading wallet and (when configured) the treasury."""
    from .custody import owners, treasury_address
    out = {"wallet": scan(rpc, trading, our_keys=our_keys)}
    t = treasury_address()
    if t:
        out["treasury"] = scan(rpc, t, name=SCAN_TREASURY, our_keys=our_keys, owners=owners())
    return out


def _confirmed_elsewhere(chk, f: Flow, wallet: str, funding: set[str], our_keys, owners) -> bool:
    """The second provider classifies the same transaction the same way (else the primary may be lying)."""
    try:
        tx = chk.get_transaction(f.signature)
    except Exception:
        return True                                      # the check provider is down: the primary's word stands
    if tx is None:
        return False
    with transaction() as conn:
        g = classify(tx, wallet, f.signature in known_ours(conn, [f.signature]), funding, our_keys, owners)
    return g is not None and g.kind == f.kind and g.lamports == f.lamports


def _halt(f: Flow) -> None:
    from . import state
    reason = f"{f.kind} transaction {f.signature[:12]}… ({f.lamports} lamports)"
    state.halt(reason, {"signature": f.signature, "kind": f.kind, "lamports": f.lamports})
    record_event("error", "vault", "vault halted: " + reason, {"signature_prefix": f.signature[:12], "kind": f.kind})


def reclassify(signature: str, kind: str, note: str | None = None) -> int:
    """Operator override for one transaction's flow: deposit | profit | internal (drops the row). Returns rows changed.
    A flow already counted by a settlement stays counted there; the change moves only future settlements (a deposit
    re-labelled after its week was paid out becomes a carried loss that later profit refills first)."""
    if kind not in ("deposit", "profit", "internal"):
        raise ValueError("kind must be deposit, profit or internal")
    with transaction() as conn:
        rows = conn.execute("SELECT id, kind, direction FROM vault_flows WHERE signature = %s", (signature,)).fetchall()
        if not rows:
            raise LookupError("no flow with that signature")
        if kind == "internal":
            conn.execute("DELETE FROM vault_flows WHERE signature = %s AND kind IN ('unknown_outbound', 'anomaly', 'profit', 'deposit')", (signature,))
            conn.execute("INSERT INTO vault_kv (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         ("ours:" + signature[:32], Jsonb({"signature": signature, "note": note})))
            return len(rows)
        n = 0
        for r in rows:
            if r["direction"] == "in" and r["kind"] in ("deposit", "profit"):
                n += conn.execute("UPDATE vault_flows SET kind = %s, classified_by = 'operator', note = COALESCE(%s, note) WHERE id = %s",
                                  (kind, note, r["id"])).rowcount
        return n
