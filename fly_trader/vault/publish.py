"""The public stats (docs/vault/SPEC.md §4): one snapshot every minute plus history deltas and holder accounts, pushed
to the relay. Built only from the database and cached prices, so a push never waits on the chain; any failure is
logged and retried next minute and never reaches trading."""
from __future__ import annotations

import json
import logging
import time

from .. import config
from ..agent import handover, rails
from ..db.connection import transaction
from . import prices, rh_index, settle, state

log = logging.getLogger(__name__)
LAMPORTS = config.LAMPORTS_PER_SOL
NAV_EVERY_S = 300                       # history points: one NAV mark per 5 min is plenty for a chart
# Nothing public may help anyone front-run the fly: wallet figures and NAV points are published only once they are at
# least this old, open positions are never published (a trade appears when it closes), and neither is the performance
# index (it shows how close the kill switch is) nor the next settlement time; the snapshot's time is the hour. The fly's
# wallet address is never published either, nor anything that leads to it (transaction signatures, deposit senders).
PUBLIC_DELAY_S = 3600


def _ui(conn, key: str):
    r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (key,)).fetchone()
    if not r or r["value"] is None:
        return None
    return r["value"] if isinstance(r["value"], (dict, list)) else json.loads(r["value"])


def _t(d) -> int | None:
    return int(d.timestamp()) if d is not None else None


def settlement_item(r) -> dict:
    return {"id": int(r["id"]), "period_start": _t(r["period_start"]), "period_end": _t(r["period_end"]),
            "realized": int(r["realized"] or 0), "pot": int(r["pot"] or 0), "allocated": int(r["allocated"] or 0),
            "carried": int(r["carried"] or 0), "earners": int(r["earners"] or 0), "total_weight": str(r["total_weight"] or 0),
            "status": r["status"]}


def stats(conn, wallet: str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    m = conn.execute("SELECT * FROM vault_nav WHERE ts <= to_timestamp(%s) ORDER BY ts DESC LIMIT 1", (now - PUBLIC_DELAY_S,)).fetchone()
    w = {"native": int(m["native"]) if m else 0, "nav": int(m["wealth"]) if m else 0}
    cost, _mints = settle.open_positions(conn)
    f = settle.flow_totals(conn, 2**62)
    if settle.paper():                                     # a paper book: R is its closed-trade profit since the vault started
        start = int(state.get("vault_started_at", conn=conn) or 0)
        rs = conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s FROM positions WHERE book = %s AND status = 'closed' "
                          "AND closed_at >= to_timestamp(%s)", (config.VAULT_BOOK, start)).fetchone()["s"]
        r_now = int(round(float(rs) * LAMPORTS))
        f["deposits"] = int(round(config.CAPITAL_SOL * LAMPORTS))
    else:
        r_now = settle.realized(live_native(conn), 0, cost, f["deposits"], f["withdrawals"], f["payouts"])
    a = settle.allocated_total(conn)
    booked = conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s FROM positions WHERE book = %s AND status = 'closed'", (config.VAULT_BOOK,)).fetchone()["s"]
    c = rails.load_circuit(conn)
    ho = handover.state(conn)
    fs = _ui(conn, "fly_status") or {}
    live = bool(ho) and bool((fs.get("live") or {}).get("wallet_sol") is not None)
    fly_state = "halted" if (state.halted() or c.kill_switch) else "live" if live else "paper" if fs else "starting"
    ix = rh_index.index_state()
    tot = rh_index.totals(conn)
    last = conn.execute("SELECT * FROM vault_settlements WHERE status = 'allocated' ORDER BY period_end DESC LIMIT 1").fetchone()
    up = (ix.get("upgrade_scheduled") or {}).get("next")
    return {
        "v": 1, "ts": int(now) // 3600 * 3600, "book": config.VAULT_BOOK, "cluster": "mainnet" if config.VAULT_CLUSTER == "mainnet-beta" else config.VAULT_CLUSTER,
        "fly": {"state": fly_state, "handover": bool(ho), "kill_switch": bool(c.kill_switch),
                "entries_paused": bool(c.entries_paused),
                "model": {"fly": fs.get("bootstrap"), "selector": (_ui(conn, "pinned_selector_snapshot") or {}).get("id"),
                          "release": (_ui(conn, "release") or {}).get("seq")}},
        "wallet": w,
        "ledger": {"deposits": f["deposits"], "withdrawals": f["withdrawals"], "claims_paid": f["payouts"], "realized": r_now,
                   "booked_realized": int(round(float(booked) * LAMPORTS)), "allocated": a, "reserved": max(0, a - f["payouts"]),
                   "pot": max(0, r_now - a)},
        "prices": prices.snapshot(),
        "vault": {"address": config.VAULT_ADDRESS, "chain_id": config.RH_CHAIN_ID, "total_locked": str(tot["total_locked"]),
                  "total_pending": str(tot["total_pending"]), "earners": tot["earners"], "finalized_block": ix["through_block"],
                  "paused": ix["paused"], "impl": ix["impl"], "upgrade_scheduled": {"eta": up["eta"], "id": up["id"]} if up else None},
        "settlement": {"last": settlement_item(last) if last else None},
    }


def live_native(conn) -> int:
    """The newest mark's SOL, for R only (R moves when a trade closes, which the trades history shows anyway)."""
    m = conn.execute("SELECT native FROM vault_nav ORDER BY ts DESC LIMIT 1").fetchone()
    return int(m["native"]) if m else 0


def _symbol(conn, mint: str) -> str:
    r = conn.execute("SELECT symbol FROM tokens WHERE mint = %s", (mint,)).fetchone()
    return (r["symbol"] if r and r["symbol"] else mint[:4] + "…")


def history(conn, cur: dict) -> tuple[dict, dict]:
    """New history items since the cursors in ``cur``; returns (history, new cursors)."""
    out: dict = {}
    nav_since = float(cur.get("nav", 0))
    rows = conn.execute("SELECT ts, wealth FROM vault_nav WHERE extract(epoch FROM ts) >= %s AND ts <= now() - make_interval(secs => %s) "
                        "AND consistent ORDER BY ts", (nav_since + NAV_EVERY_S, PUBLIC_DELAY_S)).fetchall()
    sol = prices.sol_usd()[0]
    pts, last = [], nav_since
    for r in rows:
        t = r["ts"].timestamp()
        if t - last >= NAV_EVERY_S:
            pts.append({"ts": int(t), "nav": int(r["wealth"]), "sol_usd": sol}); last = t
    if pts:
        out["nav"] = pts
    trades = conn.execute("SELECT id, mint, opened_at, closed_at, cost_sol, realized_sol, forced_exit_kind FROM positions "
                          "WHERE book = %s AND status = 'closed' AND id > %s ORDER BY id LIMIT 500", (config.VAULT_BOOK, int(cur.get("trades", 0)))).fetchall()
    if trades:
        out["trades"] = [{"id": int(t["id"]), "mint": t["mint"], "symbol": _symbol(conn, t["mint"]), "opened_at": _t(t["opened_at"]),
                          "closed_at": _t(t["closed_at"]), "cost": int(round(float(t["cost_sol"]) * LAMPORTS)),
                          "proceeds": int(round((float(t["cost_sol"]) + float(t["realized_sol"] or 0)) * LAMPORTS)),
                          "realized": int(round(float(t["realized_sol"] or 0) * LAMPORTS)), "exit_kind": t["forced_exit_kind"] or "hold"}
                         for t in trades]
    fl = conn.execute("SELECT id, block_time, signature, direction, kind, counterparty, lamports FROM vault_flows "
                      "WHERE id > %s AND kind IN ('deposit', 'profit', 'withdrawal', 'claim') ORDER BY id LIMIT 500", (int(cur.get("flows", 0)),)).fetchall()
    if fl:
        # no signature or counterparty: either leads straight to the fly's wallet, which is not published
        out["flows"] = [{"id": int(x["id"]), "ts": _t(x["block_time"]), "direction": x["direction"],
                         "kind": x["kind"], "lamports": int(x["lamports"])} for x in fl]
    st = conn.execute("SELECT * FROM vault_settlements WHERE status = 'allocated' AND id > %s ORDER BY id", (int(cur.get("settlements", 0)),)).fetchall()
    if st:
        out["settlements"] = [settlement_item(s) for s in st]
    new = dict(cur)
    if pts:
        new["nav"] = pts[-1]["ts"]
    for k, rows_ in (("trades", trades), ("flows", fl), ("settlements", st)):
        if rows_:
            new[k] = int(rows_[-1]["id"])
    return out, new


def accounts(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM vault_accounts ORDER BY evm").fetchall()
    out = []
    for r in rows:
        allocs = conn.execute("SELECT s.period_end, a.lamports, a.weight, s.total_weight FROM vault_allocations a JOIN vault_settlements s "
                              "ON s.id = a.settlement_id WHERE a.evm = %s ORDER BY s.period_end DESC LIMIT 52", (r["evm"],)).fetchall()
        cl = conn.execute("SELECT id, status, lamports, sol, tx_signature, created_at, updated_at FROM vault_claims WHERE evm = %s "
                          "ORDER BY id DESC LIMIT 20", (r["evm"],)).fetchall()
        out.append({"evm": r["evm"], "allocated": int(r["allocated"]), "claimed": int(r["claimed"]), "in_flight": int(r["in_flight"]),
                    "owed": int(r["owed"]),
                    "allocations": [{"period_end": _t(a["period_end"]), "lamports": int(a["lamports"]), "weight": str(a["weight"]),
                                     "share": float(a["weight"]) / float(a["total_weight"]) if a["total_weight"] else 0.0} for a in allocs],
                    "claims": [{"id": int(c["id"]), "status": c["status"], "lamports": int(c["lamports"] or 0), "sol": c["sol"],
                                "tx": "", "created_at": _t(c["created_at"]),
                                "updated_at": _t(c["updated_at"])} for c in cl]})
    return out


def push(relay, wallet: str) -> dict:
    with transaction() as conn:
        s = stats(conn, wallet)
        cur = state.get("publish_cursors", {}, conn=conn)
        h, new = history(conn, cur)
        acc = accounts(conn)
    relay.push(stats=s, history=h or None, accounts=acc or None)
    state.put("publish_cursors", new)
    return {"history": {k: len(v) for k, v in h.items()}, "accounts": len(acc)}
