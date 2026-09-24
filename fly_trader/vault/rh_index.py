"""FlyVault + timelock event indexer over finalized Robinhood Chain blocks (docs/vault/SPEC.md §1).

Only ``finalized`` blocks are indexed (~18 min behind the head on this Arbitrum Orbit chain), so a settlement never
counts a lock that could still be reorged away. The getLogs range adapts: it starts wide and halves on any error.
"""
from __future__ import annotations

import logging

from psycopg.types.json import Jsonb

from .. import config
from ..db.connection import transaction
from . import evm, state

log = logging.getLogger(__name__)

SCAN_NAME = "rh_vault"
MAX_RANGE = 500_000
MIN_RANGE = 500

T_LOCKED = evm.topic("Locked(address,uint256,uint256)")
T_REQUESTED = evm.topic("WithdrawRequested(address,uint256,uint256,uint64,uint256)")
T_CANCELLED = evm.topic("RequestCancelled(address,uint256,uint256,uint256)")
T_WITHDRAWN = evm.topic("Withdrawn(address,uint256,uint256)")
T_UPGRADED = evm.topic("Upgraded(address)")
T_PAUSED = evm.topic("Paused(address)")
T_UNPAUSED = evm.topic("Unpaused(address)")
VAULT_TOPICS = [T_LOCKED, T_REQUESTED, T_CANCELLED, T_WITHDRAWN, T_UPGRADED, T_PAUSED, T_UNPAUSED]
T_SCHEDULED = evm.topic("CallScheduled(bytes32,uint256,address,uint256,bytes,bytes32,uint256)")
T_EXECUTED = evm.topic("CallExecuted(bytes32,uint256,address,uint256,bytes)")
T_TL_CANCELLED = evm.topic("Cancelled(bytes32)")


def decode(lg: dict) -> dict | None:
    """One FlyVault log as a vault_events row (without chain/time fields), or None for foreign topics."""
    t = lg["topics"]
    w = evm.data_words(lg.get("data") or "0x")
    t0 = t[0].lower()
    if t0 == T_LOCKED:
        return {"event": "Locked", "account": evm.word_to_address(t[1]), "amount": evm.word_to_int(w[0]), "locked_after": evm.word_to_int(w[1])}
    if t0 == T_REQUESTED:
        return {"event": "WithdrawRequested", "account": evm.word_to_address(t[1]), "request_id": evm.word_to_int(t[2][2:]),
                "amount": evm.word_to_int(w[0]), "ready_at": evm.word_to_int(w[1]), "locked_after": evm.word_to_int(w[2])}
    if t0 == T_CANCELLED:
        return {"event": "RequestCancelled", "account": evm.word_to_address(t[1]), "request_id": evm.word_to_int(t[2][2:]),
                "amount": evm.word_to_int(w[0]), "locked_after": evm.word_to_int(w[1])}
    if t0 == T_WITHDRAWN:
        return {"event": "Withdrawn", "account": evm.word_to_address(t[1]), "request_id": evm.word_to_int(t[2][2:]), "amount": evm.word_to_int(w[0])}
    if t0 == T_UPGRADED:
        return {"event": "Upgraded", "account": evm.word_to_address(t[1])}
    if t0 in (T_PAUSED, T_UNPAUSED):
        return {"event": "Paused" if t0 == T_PAUSED else "Unpaused", "account": evm.word_to_address(w[0]) if w else None}
    return None


def _cursor() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT cursor, slot, detail FROM vault_scan WHERE name = %s", (SCAN_NAME,)).fetchone()
    return dict(r) if r else {}


def index_state() -> dict:
    """What settlement needs: finalized-through block and time, the implementation and its code hash."""
    c = _cursor()
    d = c.get("detail") or {}
    return {"through_block": int(c.get("slot") or 0), "through_time": int(d.get("through_time") or 0),
            "impl": d.get("impl"), "impl_codehash": d.get("impl_codehash"), "paused": bool(d.get("paused")),
            "upgrade_scheduled": d.get("upgrade_scheduled"), "range": int(d.get("range") or MAX_RANGE)}


def _block_times(rpc: evm.EvmRpc, logs: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for lg in logs:
        n = int(lg["blockNumber"], 16)
        if n in out:
            continue
        if lg.get("blockTimestamp"):
            out[n] = int(lg["blockTimestamp"], 16)
        else:
            out[n] = int(rpc.block(n)["timestamp"], 16)
    return out


def _timelock_ops(rpc: evm.EvmRpc, frm: int, to: int, ops: dict, times: dict) -> dict:
    """Pending timelock operations aimed at the vault proxy: id -> {eta}; executed/cancelled ones drop out."""
    if not config.VAULT_TIMELOCK:
        return ops
    logs = rpc.get_logs(config.VAULT_TIMELOCK, frm, to, [[T_SCHEDULED, T_EXECUTED, T_TL_CANCELLED]])
    times.update(_block_times(rpc, logs))
    vault = (config.VAULT_ADDRESS or "").lower()
    for lg in logs:
        t0, op = lg["topics"][0].lower(), lg["topics"][1]
        if t0 == T_SCHEDULED:
            w = evm.data_words(lg["data"])
            target = ("0x" + w[0][-40:]).lower()
            delay = evm.word_to_int(w[5]) if len(w) > 5 else 0
            if target == vault:
                ops[op] = {"eta": times[int(lg["blockNumber"], 16)] + delay, "id": op}
        else:
            ops.pop(op, None)
    return ops


def run_once(rpc: evm.EvmRpc | None = None, max_steps: int = 20) -> dict:
    """Index finalized blocks after the cursor. Returns index_state()."""
    if not config.VAULT_ADDRESS:
        return index_state()
    rpc = rpc or evm.EvmRpc(config.RH_RPC_URL, config.RH_CHAIN_ID)
    fin = rpc.block("finalized")
    head, head_time = int(fin["number"], 16), int(fin["timestamp"], 16)
    st = index_state()
    frm = max(st["through_block"] + 1, config.VAULT_START_BLOCK)
    rng = st["range"]
    ops = dict((st.get("upgrade_scheduled") or {}).get("ops") or {})
    paused = st["paused"]
    through, through_time = st["through_block"], st["through_time"]
    for _ in range(max_steps):
        if frm > head:
            through, through_time = head, head_time
            break
        to = min(head, frm + rng - 1)
        try:
            logs = rpc.get_logs(config.VAULT_ADDRESS, frm, to, [VAULT_TOPICS])
            times = _block_times(rpc, logs)
            ops = _timelock_ops(rpc, frm, to, ops, times)
        except Exception as e:
            if rng <= MIN_RANGE:
                raise
            rng = max(MIN_RANGE, rng // 2)
            log.info("rh index: getLogs %d..%d failed (%s); range -> %d", frm, to, type(e).__name__, rng)
            continue
        with transaction() as conn:
            for lg in logs:
                row = decode(lg)
                if row is None:
                    continue
                if row["event"] in ("Paused", "Unpaused"):
                    paused = row["event"] == "Paused"
                n = int(lg["blockNumber"], 16)
                conn.execute(
                    "INSERT INTO vault_events (chain_id, block_number, log_index, tx_hash, block_time, event, account, request_id, amount, "
                    "locked_after, ready_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (config.RH_CHAIN_ID, n, int(lg["logIndex"], 16), lg["transactionHash"], times[n], row["event"],
                     row.get("account"), row.get("request_id"), row.get("amount"), row.get("locked_after"), row.get("ready_at")))
        through = to
        through_time = head_time if to == head else int(rpc.block(to)["timestamp"], 16)
        frm = to + 1
        rng = min(MAX_RANGE, rng * 2)
    impl = rpc.implementation(config.VAULT_ADDRESS, hex(through) if through else "finalized")
    detail = {"through_time": through_time, "impl": impl, "impl_codehash": rpc.code_hash(impl, "finalized"), "paused": paused,
              "range": rng, "upgrade_scheduled": {"ops": ops, "next": min(ops.values(), key=lambda o: o["eta"]) if ops else None}}
    with transaction() as conn:
        conn.execute("INSERT INTO vault_scan (name, slot, detail, updated_at) VALUES (%s,%s,%s,now()) "
                     "ON CONFLICT (name) DO UPDATE SET slot = EXCLUDED.slot, detail = EXCLUDED.detail, updated_at = now()",
                     (SCAN_NAME, through, Jsonb(detail)))
    return index_state()


def totals(conn) -> dict:
    """Current earning and pending totals and the number of earners, from the indexed events."""
    rows = conn.execute("SELECT DISTINCT ON (account) account, locked_after FROM vault_events WHERE chain_id = %s AND locked_after IS NOT NULL "
                        "ORDER BY account, block_number DESC, log_index DESC", (config.RH_CHAIN_ID,)).fetchall()
    locked = sum(int(r["locked_after"]) for r in rows)
    req = conn.execute("SELECT COALESCE(sum(amount) FILTER (WHERE event = 'WithdrawRequested'), 0) - "
                       "COALESCE(sum(amount) FILTER (WHERE event IN ('RequestCancelled', 'Withdrawn')), 0) AS p "
                       "FROM vault_events WHERE chain_id = %s", (config.RH_CHAIN_ID,)).fetchone()
    return {"total_locked": locked, "total_pending": int(req["p"] or 0), "earners": sum(1 for r in rows if int(r["locked_after"]) > 0)}
