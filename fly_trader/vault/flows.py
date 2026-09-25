"""Every lamport that enters or leaves the trading wallet other than by trading (plan "Core accounting").

The scanner walks the wallet's finalized transaction history and classifies each transaction by who signed it:

* **ours** (our key signed): swaps, token-account closes, claim payouts, principal withdrawals. Claims and withdrawals
  write their own ``vault_flows`` rows when they are sent; the rest move lamports inside R (fees, rent) and need no row.
  A transaction our key signed that no table knows about is ``unknown_outbound``: a possible key leak, so entries and
  claims halt until the operator looks (``fly-trader vault reclassify``).
* **theirs**: SOL (or wSOL) arriving is a ``deposit`` when the sender is in ``FUNDING_ADDRESSES``, else ``profit`` (a
  gift counts for the lockers). Lamports leaving without our signature cannot happen on a system account; if they do,
  the row is an ``anomaly`` and the vault halts.

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


def classify(tx: dict, wallet: str, known_ours: bool, funding: set[str]) -> Flow | None:
    """The flow one finalized transaction makes, or None when it is trading/internal or moves nothing."""
    keys = _keys(tx)
    meta = tx.get("meta") or {}
    sig = ((tx.get("transaction") or {}).get("signatures") or [None])[0]
    slot, bt = int(tx.get("slot") or 0), tx.get("blockTime")
    idx = next((i for i, k in enumerate(keys) if k.get("pubkey") == wallet), None)
    if idx is None or sig is None:
        return None
    pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
    delta = (int(post[idx]) - int(pre[idx])) if idx < len(pre) and idx < len(post) else 0
    ours = bool(keys[idx].get("signer"))
    if ours:
        if known_ours:
            return None
        return Flow(sig, slot, bt, "out" if delta <= 0 else "in", "unknown_outbound", keys[0].get("pubkey"), abs(delta),
                    int(meta.get("fee") or 0), "signed by our key but unknown to orders/claims/withdrawals/ATA closes")
    if meta.get("err") is not None:
        return None                                     # a failed transaction someone else paid for moves nothing of ours
    amount = delta + _wsol_delta(meta, wallet)
    if amount < 0:
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
        "UNION SELECT signature FROM vault_flows WHERE signature = ANY(%(s)s) AND kind IN ('withdrawal', 'claim', 'sweep') "
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


def cursor() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT cursor, slot, detail FROM vault_scan WHERE name = %s", (SCAN_NAME,)).fetchone()
    return dict(r) if r else {"cursor": None, "slot": None, "detail": None}


def scanned_through() -> int:
    """Every flow with slot <= this is in vault_flows."""
    return int(cursor().get("slot") or 0)


def scan(rpc, wallet: str, funding: set[str] | None = None, max_pages: int = 50) -> dict:
    """Classify finalized transactions newer than the cursor. Idempotent; stops at the first unreadable transaction."""
    funding = set(funding if funding is not None else config.FUNDING_ADDRESSES)
    head = int(rpc.call("getSlot", [{"commitment": "finalized"}]))
    cur = cursor()
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
    halted = []
    for entry in reversed(newest):                         # oldest first so the cursor only ever moves forward
        sig = entry["signature"]
        tx = rpc.get_transaction(sig)
        if tx is None:
            log.info("vault flow scan: %s… not readable yet; resuming next round", sig[:12])
            head = None
            break
        with transaction() as conn:
            f = classify(tx, wallet, sig in known_ours(conn, [sig]), funding)
            if f is not None and insert(conn, f):
                counts[f.kind] = counts.get(f.kind, 0) + 1
                if f.kind in ("unknown_outbound", "anomaly"):
                    halted.append(f)
            conn.execute(
                "INSERT INTO vault_scan (name, cursor, slot, detail, updated_at) VALUES (%s,%s,%s,%s,now()) "
                "ON CONFLICT (name) DO UPDATE SET cursor = EXCLUDED.cursor, detail = EXCLUDED.detail, updated_at = now()",
                (SCAN_NAME, sig, cur.get("slot"), Jsonb({"last_slot": int(entry.get("slot") or 0)})))
    if head is not None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO vault_scan (name, slot, updated_at) VALUES (%s,%s,now()) "
                "ON CONFLICT (name) DO UPDATE SET slot = GREATEST(COALESCE(vault_scan.slot, 0), EXCLUDED.slot), updated_at = now()",
                (SCAN_NAME, head))
    for f in halted:
        _halt(f)
    if counts:
        log.info("vault flow scan: %s", counts)
    return {"new": counts, "through_slot": head, "seen": len(newest)}


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
