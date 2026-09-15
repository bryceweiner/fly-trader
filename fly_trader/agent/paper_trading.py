"""One trading minute of a paper book — the rules both race books share (the selector's and the fly's):

exits: positions held ``horizon_s`` or longer are sold at this minute's price (or the last traded price: the training
label's exit); entries: scores at or above the book's line open a position sized from the book's certainty bands
(``agent/sizing.py``), one position per token, unless blocked (global kill switch, paused entries, the book's own
drawdown halt, or the caller's ``block``); marks and wealth per minute (positions net of exit cost, exposure at cost,
drawdown ≥ 0); the book's drawdown halt is checked on its own peak (``rails.check_book_drawdown``). Every score, entry,
exit and blocked pick is a ``decisions`` row.
"""
from __future__ import annotations

import json

from . import rails, sizing
from ..execution import ledger
from ..market.exit_cost import exit_cost_fraction


def trade_minute(ctx, *, book: str, run_id: str, beat_no: int, broker, kind: str, mints: list, infos: list, scores, threshold: float,
                 table: list | None, horizon_s: float, block: str | None = None) -> dict:
    conn, m1 = ctx.conn, ctx.m1
    beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active, notes) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                        (run_id, m1, beat_no, len(mints), json.dumps({"minute": m1.isoformat(), "mints_traded": len(ctx.agg), "book": book}))).fetchone()["id"]
    n_exit = 0
    for p in ledger.open_positions(conn, book):
        held = (m1 - p["opened_at"]).total_seconds()
        if held < horizon_s:
            continue
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,false,%s,%s) RETURNING id",
                           (beat, run_id, m1, p["mint"], p["pool"], f"{kind}_exit", float(p["cost_sol"]), f"held {held/60:.0f} min", json.dumps({"book": book}))).fetchone()["id"]
        px = ctx.prices.get(p["mint"]) or float(p.get("last_mark_price") or p["entry_price"])
        broker.sell(conn, position=p, decision_id=did, price=ctx.prices.get(p["mint"]), res_quote_sol=ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"]),
                    mcap_sol=ctx.mcap(p["mint"], px), program_label=p.get("program_label"), forced_kind=None, pool_fee=ctx.fees.get(p["mint"]), ts=m1)
        n_exit += 1
    opens_now = ledger.open_positions(conn, book); open_mints = {p["mint"] for p in opens_now}
    cash = ledger.paper_cash(conn, book); n_enter = 0; picks = []; entries = []
    bankroll = cash + sum(float(p["cost_sol"]) for p in opens_now)         # wealth at cost: the sizing base (agent/sizing.py)
    circuit = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = 1").fetchone()
    blocked = (block or ("kill switch" if circuit and circuit["kill_switch"] else None) or ("paused" if circuit and circuit["entries_paused"] else None)
               or ("book halted" if rails.book_halted(conn, book) else None))
    for m, i, sc in zip(mints, infos, scores):
        if sc < threshold:
            continue
        picks.append((m, float(sc)))
        size, why = sizing.size_position(float(sc), threshold, table or [], bankroll, cash, i["resq"])
        if m in open_mints or blocked or size <= 0:
            rail = blocked or ("held" if m in open_mints else "sizing")
            conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                         (beat, run_id, m1, m, i["pool"], float(sc), size, rail, why if rail == "sizing" else f"{kind} pick not taken",
                          json.dumps({"book": book, "score": float(sc), "threshold": threshold, "sizing": why})))
            continue
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, reason, book_targets, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,false,%s,%s,%s) RETURNING id",
                           (beat, run_id, m1, m, i["pool"], f"{kind}_enter", float(sc), size, f"score {sc:.3f} ≥ {threshold:.3f}; {why}", [book],
                            json.dumps({"book": book, "score": float(sc), "threshold": threshold, "size_sol": size, "sizing": why, "resq": i["resq"], "age_h": i["age_h"], "price": i["price"]}))).fetchone()["id"]
        fill = broker.buy(conn, decision_id=did, mint=m, pool=i["pool"], size_sol=size, price=i["price"], res_quote_sol=i["resq"],
                          mcap_sol=i["mcap"], decimals=i["decimals"], program_label=i["program_label"], pool_fee=i["fee_rate"], ts=m1)
        if fill.ok:
            n_enter += 1; open_mints.add(m); cash -= size
            entries.append({"mint": m, "decision_id": did, "score": float(sc), "size_sol": size, "info": i})
        else:
            conn.execute("UPDATE decisions SET rail = %s, reason = %s WHERE id = %s", ("paper_fill", fill.reason, did))
    ledger.mark_positions(conn, book, ctx.prices, {}, m1)
    opens = ledger.open_positions(conn, book); gross_v = 0.0; exit_cost = 0.0; exposure = 0.0
    for p in opens:
        px = float(p.get("last_mark_price") or p["entry_price"])
        gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * px
        rq = ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"])
        gross_v += gross; exposure += float(p["cost_sol"])
        exit_cost += gross * exit_cost_fraction(gross, rq, ctx.mcap(p["mint"], px), p.get("program_label"), ctx.fees.get(p["mint"]))
    cash = ledger.paper_cash(conn, book); positions_net = gross_v - exit_cost; wealth = cash + positions_net
    peak_row = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s", (book,)).fetchone(); peak = max(float(peak_row["pk"] or 0.0), wealth)
    conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                 (beat, book, m1, cash, positions_net, exit_cost, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure, len(opens)))
    rails.check_book_drawdown(conn, book, wealth, peak)        # the book's own kill switch: blocks its next minutes' entries
    return {"beat_id": beat, "entries": entries, "entered": n_enter, "exited": n_exit, "open": len(opens), "cash": cash, "wealth": wealth, "picks": len(picks), "blocked": blocked,
            "top_scores": sorted(picks, key=lambda x: -x[1])[:5], "open_mints": sorted({p["mint"] for p in opens})}
