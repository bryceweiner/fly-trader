"""The weekly settlement's ETH pot: Robinhood Chain memecoin profit shared with the same lockers, by the same weights,
with the same formulas as the SOL pot (vault/settle.py) — in wei, paid in native ETH on Robinhood Chain.

    R   = N + C + L - D + Wd + P        N native ETH of the RH bot wallet, C cost of open live_rh positions, L basis of
                                        open base lots, D deposits, Wd withdrawals, P ETH claims paid
    pot = max(0, R - A)                 A = all ETH ever allocated (losses and dust carry forward)
    out = min(pot, N - (A - P) - gas)   never allocate ETH that is not liquid

It runs after each SOL settlement is allocated (Robinhood Chain has then finalized past the period end, so the weights
are final): one ETH row per SOL settlement, taken while no RH intent is in flight (the wallet then matches the books).
A paper vault (``VAULT_BOOK`` not 'live') shares the paper RH fly book's closed-trade profit instead.
"""
from __future__ import annotations

import hashlib
import json
import logging

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..markets import RH
from . import settle, state

log = logging.getLogger(__name__)
WEI = 10 ** 18


def book() -> str:
    return RH.live_book if not settle.paper() else RH.fly_book


def _w(x) -> int:
    return int(round(float(x or 0.0) * WEI))


def allocated_total(conn) -> int:
    return int(conn.execute("SELECT COALESCE(sum(allocated), 0) AS a FROM vault_eth_settlements WHERE status = 'allocated'").fetchone()["a"])


def paid_total(conn) -> int:
    return int(conn.execute("SELECT COALESCE(sum(eth_wei), 0) AS p FROM vault_claims WHERE eth_status = 'paid'").fetchone()["p"])


def reserved_in_trading(conn) -> int:
    """ETH owed to lockers (allocated, not yet paid): the live RH book never trades it (rh/live.reserved_wei)."""
    if settle.paper():
        return 0
    return max(0, allocated_total(conn) - paid_total(conn))


def in_flight(conn) -> bool:
    a = conn.execute("SELECT count(*) AS n FROM rh_intents WHERE state NOT IN ('done', 'failed', 'aborted')").fetchone()["n"]
    b = conn.execute("SELECT count(*) AS n FROM rh_txs WHERE applied_at IS NULL AND status IN ('signed', 'sent') AND kind <> 'payout'").fetchone()["n"]
    return bool(a or b)


def snapshot(conn, native_wei: int | None) -> dict:
    """R's terms now. ``native_wei``: the RH wallet's balance (None for a paper vault)."""
    bk = book()
    cost = _w(conn.execute("SELECT COALESCE(sum(cost_sol), 0) AS c FROM positions WHERE book = %s AND status = 'open'", (bk,)).fetchone()["c"])
    paid = paid_total(conn)
    if settle.paper():
        start = int(state.get("vault_started_at", conn=conn) or 0)
        r = conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s FROM positions WHERE book = %s AND status = 'closed' AND closed_at >= to_timestamp(%s)",
                         (bk, start)).fetchone()["s"]
        m = conn.execute("SELECT sol_free FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (bk,)).fetchone()
        return {"native": _w(m["sol_free"]) if m else 0, "open_cost": cost, "lots": 0, "deposits": 0, "withdrawals": 0, "payouts": paid, "realized": _w(r)}
    lots = _w(conn.execute("SELECT COALESCE(sum(basis_eth), 0) AS b FROM rh_base_lots WHERE status = 'open'").fetchone()["b"])
    f = conn.execute("SELECT COALESCE(sum(wei) FILTER (WHERE kind = 'deposit'), 0) AS d, COALESCE(sum(wei) FILTER (WHERE kind = 'withdrawal'), 0) AS wd "
                     "FROM rh_wallet_flows").fetchone()
    d, wd = int(f["d"]), int(f["wd"])
    return {"native": int(native_wei or 0), "open_cost": cost, "lots": lots, "deposits": d, "withdrawals": wd, "payouts": paid,
            "realized": settle.realized(int(native_wei or 0), lots, cost, d, wd, paid)}


def settle_one(sid: int, native_wei: int | None) -> dict:
    """The ETH pot of SOL settlement ``sid`` (already allocated)."""
    with transaction() as conn:
        s = conn.execute("SELECT * FROM vault_settlements WHERE id = %s", (sid,)).fetchone()
        if s is None or s["status"] != "allocated":
            return {"skipped": "the SOL settlement is not allocated"}
        if conn.execute("SELECT 1 FROM vault_eth_settlements WHERE settlement_id = %s", (sid,)).fetchone():
            return {"skipped": "done"}
        if not settle.paper() and in_flight(conn):
            return {"waiting": "an RH intent is in flight"}
        t0, t1 = int(s["period_start"].timestamp()), int(s["period_end"].timestamp())
        snap = snapshot(conn, native_wei)
        start, evs = settle.event_inputs(conn, config.RH_CHAIN_ID, t0, t1)
        w = settle.weights(start, evs, t0, t1)
        before = allocated_total(conn)
        p = settle.plan(snap["realized"], before, snap["payouts"], snap["native"], 0 if settle.paper() else int(config.VAULT_RH_GAS_RESERVE_WEI))
        alloc = settle.allocate(p["amount"], w)
        total = sum(alloc.values())
        inputs = {"settlement": sid, "period": [t0, t1], "book": book(), "snapshot": snap, "allocated_before": before, "plan": p,
                  "weights": {a: str(v) for a, v in sorted(w.items())}, "allocations": {a: str(v) for a, v in sorted(alloc.items())}}
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        eid = int(conn.execute(
            "INSERT INTO vault_eth_settlements (settlement_id, period_start, period_end, status, book, native, open_cost, lots, deposits, withdrawals, payouts, "
            "realized, allocated_before, pot, liquid_cap, allocated, carried, earners, inputs_sha256) "
            "VALUES (%s,%s,%s,'allocated',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (sid, s["period_start"], s["period_end"], book(), snap["native"], snap["open_cost"], snap["lots"], snap["deposits"], snap["withdrawals"],
             snap["payouts"], snap["realized"], before, p["pot"], p["liquid_cap"], total, p["pot"] - total, len(alloc), digest)).fetchone()["id"])
        for a, wei in alloc.items():
            conn.execute("INSERT INTO vault_eth_allocations (eth_settlement_id, evm, weight, wei) VALUES (%s,%s,%s,%s)", (eid, a, w[a], wei))
        state.put(f"settlement_eth_inputs:{sid}", inputs, conn=conn)
    record_event("info", "vault", "ETH settlement allocated", {"settlement": sid, "pot_wei": str(p["pot"]), "allocated_wei": str(total), "earners": len(alloc)})
    return {"allocated": total, "pot": p["pot"], "earners": len(alloc), "carried": p["pot"] - total, "sha256": digest}


def run_once(balance=None) -> dict:
    """Settle the ETH pot of every allocated SOL settlement that has none, in period order. ``balance()``: the RH
    wallet's native wei (live vault); the step waits while it is unavailable."""
    if not config.RH_ENABLED or state.halted():
        return {}
    out = {}
    with transaction() as conn:
        todo = [int(r["id"]) for r in conn.execute("SELECT s.id FROM vault_settlements s LEFT JOIN vault_eth_settlements e ON e.settlement_id = s.id "
                                                   "WHERE s.status = 'allocated' AND e.id IS NULL ORDER BY s.period_end").fetchall()]
    for sid in todo:
        native = None
        if not settle.paper():
            if balance is None:
                out[sid] = {"waiting": "no RH wallet"}; break
            native = int(balance())
        res = settle_one(sid, native); out[sid] = res
        if "allocated" not in res and res.get("skipped") != "done":
            break
        if "allocated" in res:
            from . import alerts
            alerts.send(f"settlement #{sid} ETH: {res['allocated'] / WEI:.6f} ETH to {res['earners']} holders (pot {res['pot'] / WEI:.6f}, carried {res['carried'] / WEI:.6f})")
    return out
