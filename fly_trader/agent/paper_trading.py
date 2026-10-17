"""One trading minute of a paper book — the rules both race books share (the selector's and the fly's):

exits: positions held their own hold (the strategy's, stored on the position) or ``horizon_s`` are sold at this minute's
price (or the last traded price: the training label's exit); entries: scores at or above the book's line open a position sized from the book's certainty bands
(``agent/sizing.py``), one position per token, unless blocked (global kill switch, paused entries, the book's own
drawdown halt, a minute traded more than ``ENTRY_MAX_LAG_S`` after its end, or the caller's ``block``); marks and wealth
per minute (positions net of exit cost, exposure at cost, drawdown ≥ 0); the book's drawdown halt is checked on its own
peak (``rails.check_book_drawdown``). Every score, entry, exit and blocked pick is a ``decisions`` row.

Live money first (CLAUDE.md): ``plan_minute`` decides the minute without booking a trade -- the exits due and what they
would bring, the entries on the cash and tokens that leaves, each paper fill quoted (``PaperBroker.quote_*``, the numbers
the booking writes) -- so a live mirror can send the wallet's orders from its ``entries`` before ``book_minute`` books
the paper trades (agent/fly_session.FlyBook books them only after every chain's live orders are out).
``trade_minute`` is both in one call, the plan's entries handed to ``on_entries`` in between.

Costs (market/exit_cost.py): a sale or a mark pays the mint's last known pool fee when its minute reported none
(``Minute.fee``), and an RH coin quoted in another asset than ETH pays its route through the base (``Minute.cost_label``).
Each entry hands the live mirrors its fill's all-in price (``ref_price``: the decision's price × (1 + its modelled cost)),
the most a live buy of it may pay (``MAX_ENTRY_CHASE`` above it).
"""
from __future__ import annotations

import json

import numpy as np

from . import rails, sizing
from .. import config
from ..execution import ledger
from ..market.exit_cost import exit_cost_fraction


def plan_minute(ctx, *, book: str, run_id: str, beat_no: int, broker, kind: str, mints: list, infos: list, scores, threshold,
                table: list | None, horizon_s: float, block: str | None = None, holds=None, strategies=None, allow=None, reasons=None,
                tables: list | None = None, flat: bool = False) -> dict:
    """The minute's beat, exits and entries, decided and quoted but not booked: every entry and blocked pick is already a
    ``decisions`` row (the live mirror buys under the entry's id); ``book_minute`` books the rest.

    Optional per-row arguments (the selector's strategy stack, train/strategies.decide): ``threshold`` may be per row,
    ``holds`` (s), ``strategies``, ``tables`` (sizing) and ``allow`` / ``reasons`` (a blocked row is recorded with its reason).
    Left out, the call behaves exactly as the single-strategy engine. ``flat``: every buy the same share of the bankroll
    (agent/sizing.size_position), the fly's rule (``config.FLY_SIZING``)."""
    conn, m1 = ctx.conn, ctx.m1
    beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active, notes) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                        (run_id, m1, beat_no, len(mints), json.dumps({"minute": m1.isoformat(), "mints_traded": len(ctx.agg), "book": book}))).fetchone()["id"]
    exits = []                                                   # (position, held, its sale's arguments, what it would bring)
    opens_before = ledger.open_positions(conn, book)
    for p in opens_before:
        held = (m1 - p["opened_at"]).total_seconds()
        if held < float(p.get("hold_s") or horizon_s):
            continue
        px = ctx.prices.get(p["mint"]) or float(p.get("last_mark_price") or p["entry_price"])
        kw = dict(price=ctx.prices.get(p["mint"]), res_quote_sol=ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"]), mcap_sol=ctx.mcap(p["mint"], px),
                  program_label=ctx.cost_label(p["mint"], p.get("program_label")), pool_fee=ctx.fee(p["mint"]))
        exits.append((p, held, kw, broker.quote_sell(position=p, **kw)))
    leaving = {int(p["id"]) for p, *_ in exits}
    opens_now = [p for p in opens_before if int(p["id"]) not in leaving]; open_mints = {p["mint"] for p in opens_now}
    cash = ledger.paper_cash(conn, book) + sum(q for *_, q in exits); n_enter = 0; picks = []; entries = []; buys = []
    bankroll = cash + sum(float(p["cost_sol"]) for p in opens_now)         # wealth at cost: the sizing base (agent/sizing.py)
    from ..markets import for_book
    market = for_book(book)                                     # the book's market: its circuit, money settings and sizes
    circuit = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = %s", (market.circuit_id,)).fetchone()
    # a minute traded long after its end (a lagging stream): its price is old, a buy now would pay a newer one (paper and live alike)
    blocked = (block or ("stale minute" if ctx.lag_s > config.ENTRY_MAX_LAG_S else None) or ("kill switch" if circuit and circuit["kill_switch"] else None)
               or ("paused" if circuit and circuit["entries_paused"] else None) or ("book halted" if rails.book_halted(conn, book) else None))
    thr_v = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(mints),)) if len(mints) else np.array([])
    for k, (m, i, sc) in enumerate(zip(mints, infos, scores)):
        thr_k = float(thr_v[k])
        if not np.isfinite(sc) or sc < thr_k:
            continue
        picks.append((m, float(sc)))
        tab_k = tables[k] if tables is not None else table
        size, why = sizing.size_position(float(sc), thr_k, tab_k or [], bankroll, cash, i["resq"], flat=flat, market=market)
        denied = allow is not None and not bool(allow[k])
        if m in open_mints or blocked or denied or size <= 0:
            rail = blocked or ("held" if m in open_mints else (str(reasons[k]) if denied and reasons is not None else "filtered") if denied else "sizing")
            conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                         (beat, run_id, m1, m, i["pool"], float(sc), size, rail, why if rail == "sizing" else f"{kind} pick not taken",
                          json.dumps({"book": book, "score": float(sc), "threshold": thr_k, "sizing": why,
                                      "strategy": None if strategies is None else strategies[k]})))
            continue
        hold_k = float(holds[k]) if holds is not None else None; strat_k = None if strategies is None else strategies[k]
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, reason, book_targets, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,false,%s,%s,%s) RETURNING id",
                           (beat, run_id, m1, m, i["pool"], f"{kind}_enter", float(sc), size, f"{strat_k + ': ' if strat_k else ''}score {sc:.3f} ≥ {thr_k:.3f}; {why}", [book],
                            json.dumps({"book": book, "score": float(sc), "threshold": thr_k, "size_sol": size, "sizing": why, "resq": i["resq"], "age_h": i["age_h"], "price": i["price"],
                                        "strategy": strat_k, "hold_s": hold_k}))).fetchone()["id"]
        args = dict(size_sol=size, price=i["price"], res_quote_sol=i["resq"], mcap_sol=i["mcap"], decimals=i["decimals"],
                    program_label=ctx.cost_label(m, i["program_label"]), pool_fee=ctx.fee(m))
        q = broker.quote_buy(**args)
        if q.ok:
            n_enter += 1; open_mints.add(m); cash -= size; buys.append((did, m, i, args, hold_k, strat_k))
            entries.append({"mint": m, "decision_id": did, "score": float(sc), "size_sol": size, "info": i, "threshold": thr_k, "table": tab_k, "hold_s": hold_k, "strategy": strat_k,
                            "ref_price": size / (q.token_delta / 10 ** int(i["decimals"]))})
        else:
            conn.execute("UPDATE decisions SET rail = %s, reason = %s WHERE id = %s", ("paper_fill", q.reason, did))
    return {"book": book, "run_id": run_id, "kind": kind, "broker": broker, "beat_id": beat, "exits": exits, "buys": buys, "entries": entries,
            "entered": n_enter, "picks": picks, "blocked": blocked, "market": market}


def book_minute(ctx, plan: dict) -> dict:
    """Books ``plan`` (``plan_minute``): its exits, then its entries, at the prices it quoted; then the book's marks, its
    wealth and its drawdown halt."""
    conn, m1 = ctx.conn, ctx.m1
    book, beat, broker, market = plan["book"], plan["beat_id"], plan["broker"], plan["market"]
    n_exit = 0
    for p, held, kw, _ in plan["exits"]:
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,false,%s,%s) RETURNING id",
                           (beat, plan["run_id"], m1, p["mint"], p["pool"], f"{plan['kind']}_exit", float(p["cost_sol"]), f"held {held/60:.0f} min", json.dumps({"book": book}))).fetchone()["id"]
        broker.sell(conn, position=p, decision_id=did, forced_kind=None, ts=m1, **kw)
        n_exit += 1
    for did, m, i, args, hold_k, strat_k in plan["buys"]:
        fill = broker.buy(conn, decision_id=did, mint=m, pool=i["pool"], ts=m1, hold_s=hold_k, strategy=strat_k, **args)
        if not fill.ok:                                          # its own quote planned it: only a changed broker could refuse it now
            conn.execute("UPDATE decisions SET rail = %s, reason = %s WHERE id = %s", ("paper_fill", fill.reason, did))
    ledger.mark_positions(conn, book, ctx.prices, {}, m1)
    opens = ledger.open_positions(conn, book); gross_v = 0.0; exit_cost = 0.0; exposure = 0.0
    for p in opens:
        px = float(p.get("last_mark_price") or p["entry_price"])
        gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * px
        rq = ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"])
        gross_v += gross; exposure += float(p["cost_sol"])
        exit_cost += gross * exit_cost_fraction(gross, rq, ctx.mcap(p["mint"], px), ctx.cost_label(p["mint"], p.get("program_label")), ctx.fee(p["mint"]))
    cash = ledger.paper_cash(conn, book); positions_net = gross_v - exit_cost; wealth = cash + positions_net
    peak_row = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s", (book,)).fetchone(); peak = max(float(peak_row["pk"] or 0.0), wealth)
    conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                 (beat, book, m1, cash, positions_net, exit_cost, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure, len(opens)))
    rails.check_book_drawdown(conn, book, wealth, peak, drawdown=market.kill_drawdown(), unit=market.unit)   # the book's own kill switch
    picks = plan["picks"]
    return {"beat_id": beat, "entries": plan["entries"], "entered": plan["entered"], "exited": n_exit, "open": len(opens), "cash": cash, "wealth": wealth,
            "picks": len(picks), "blocked": plan["blocked"], "top_scores": sorted(picks, key=lambda x: -x[1])[:5], "open_mints": sorted({p["mint"] for p in opens})}


def trade_minute(ctx, *, on_entries=None, **kw) -> dict:
    """``plan_minute`` (its arguments), the plan's entries handed to ``on_entries(beat_id, entries)`` before a paper trade is
    booked (a live mirror's orders go out there; what it returns is ``live``), then ``book_minute``."""
    plan = plan_minute(ctx, **kw)
    live = on_entries(plan["beat_id"], plan["entries"]) if on_entries is not None else None
    return {**book_minute(ctx, plan), "live": live}
