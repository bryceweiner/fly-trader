"""The Kalshi paper books' shared arithmetic (agent/paper_trading.py for prediction markets): cash and bankroll from the
capital cap, position sizing from the fly's certainty bands and a walk of the order book (vendored ``fill_budget``), the
taker arm's entries, settlement of positions when their market resolves, marks and wealth per minute (``wealth_marks``,
in USD), and the book's own drawdown halt. Every entry, block and settlement is a ``decisions`` row (``mint`` = ticker,
``pool`` = side, ``size_sol`` = dollars).

Books: ``paper_kalshi_taker`` (IOC at the worst level of the book walk) and ``paper_kalshi_maker`` (kalshi/maker.py);
their live mirrors ``live_kalshi_taker`` / ``live_kalshi_maker`` (kalshi/live.py) reuse ``settle`` and ``mark``.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

from .. import config
from ..agent import rails, sizing
from .vendor.book_sizing import BookLevel, OrderBook, fill_budget
from .vendor.kalshi_client import effective_price_cents, kalshi_fee_rate_cents

log = logging.getLogger(__name__)
BOOKS = {"taker": "paper_kalshi_taker", "maker": "paper_kalshi_maker"}
LIVE_BOOKS = {"taker": "live_kalshi_taker", "maker": "live_kalshi_maker"}
MAX_POSITION_FRACTION = 0.10           # of the cap in one position
PAYOUT_CENTS = 100.0


def cap_cents() -> float:
    return float(config.KALSHI_CAPITAL_USD) * 100.0


def open_positions(conn, book: str) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM kalshi_positions WHERE book = %s AND status = 'open' ORDER BY id", (book,)).fetchall()]


def realized_cents(conn, book: str) -> float:
    r = conn.execute("SELECT COALESCE(sum(realized_cents), 0) AS s FROM kalshi_positions WHERE book = %s AND status <> 'open'", (book,)).fetchone()
    return float(r["s"] or 0.0)


def open_cost_cents(conn, book: str) -> float:
    r = conn.execute("SELECT COALESCE(sum(cost_cents + fee_cents), 0) AS s FROM kalshi_positions WHERE book = %s AND status = 'open'", (book,)).fetchone()
    return float(r["s"] or 0.0)


def resting_cents(conn, book: str) -> float:
    """Collateral tied up by resting orders (price × count, the maker's cost if filled)."""
    r = conn.execute("SELECT COALESCE(sum(price_cents * count), 0) AS s FROM kalshi_orders WHERE book = %s AND status = 'resting'", (book,)).fetchone()
    return float(r["s"] or 0.0)


def cash_cents(conn, book: str) -> float:
    """A paper book's cash: the cap, plus what settled, minus what is out in positions and resting orders."""
    return cap_cents() + realized_cents(conn, book) - open_cost_cents(conn, book) - resting_cents(conn, book)


def side_quote(quotes: dict, ticker: str, side: str) -> tuple[float, float]:
    """(bid, ask) of ``side`` in cents from a YES quote (yes_bid, yes_ask); NaN where missing."""
    q = quotes.get(ticker)
    if not q:
        return math.nan, math.nan
    yb, ya = q[0], q[1]
    yb = yb if yb is not None and math.isfinite(yb) else math.nan; ya = ya if ya is not None and math.isfinite(ya) else math.nan
    return (yb, ya) if side == "yes" else (100.0 - ya, 100.0 - yb)


def top_of_book(quotes: dict, ticker: str, side: str) -> OrderBook:
    """The order book from the top of book alone (bid sizes are contracts): what the taker arm walks when no depth is held."""
    q = quotes.get(ticker) or (None, None, 0.0, 0.0)
    yb, ya = q[0], q[1]; bs = float(q[2] or 0.0) if len(q) > 2 else 0.0; as_ = float(q[3] or 0.0) if len(q) > 3 else 0.0
    yes_bids = (BookLevel(int(round(yb)), int(max(bs, 1)) if bs else 10_000),) if yb is not None and 1 <= yb <= 99 else ()
    no_bids = (BookLevel(int(round(100 - ya)), int(max(as_, 1)) if as_ else 10_000),) if ya is not None and 1 <= ya <= 99 else ()
    return OrderBook(yes_bids=yes_bids, no_bids=no_bids, ticker=ticker)


def budget_cents(edge: float, line: float, table: list | None, bankroll: float, cash: float, deployed: float) -> tuple[float, str]:
    """Dollars (in cents) to put behind one pick: the certainty band's Kelly fraction of the bankroll (agent/sizing.py),
    capped at ``MAX_POSITION_FRACTION`` of the cap, the slate fraction of the cap across every open position, and the cash
    above the floor."""
    band = sizing.band_for(table or [], float(edge) - float(line))
    k = float((band or {}).get("kelly") or 0.0)
    if k <= 0:
        return 0.0, "no certainty band with an edge" if not band else f"band at margin ≥ {band['lo']:.3f} has no edge (kelly 0)"
    want = k * bankroll
    cap = cap_cents(); room_slate = cap * config.KALSHI_MAX_SLATE_FRACTION - deployed
    room_cash = cash - config.KALSHI_CASH_FLOOR_USD * 100.0
    size = min(want, cap * MAX_POSITION_FRACTION, room_slate, room_cash)
    if size <= 0:
        return 0.0, "slate full" if room_slate <= 0 else "cash at the floor"
    return float(size), f"kelly {k:.2f} × bankroll ${bankroll / 100:.2f}" + (" (capped)" if size < want else "")


def maker_fee_cents(price_cents: float, ticker: str, charged: bool) -> float:
    return config.KALSHI_MAKER_FEE_FACTOR / config.KALSHI_FEE_FACTOR * kalshi_fee_rate_cents(price_cents, ticker) if charged else 0.0


def decision(conn, *, beat_id: int, run_id: str, ts: datetime, ticker: str, side: str, kind: str, edge: float | None, usd: float, reason: str, detail: dict,
             rail: str | None = None, book: str | None = None) -> int:
    return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail, book_targets) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,false,%s,%s,%s,%s) RETURNING id",
                            (beat_id, run_id, ts, ticker, side, kind, edge, usd, rail, reason, json.dumps({**detail, "book": book}, default=str), [book] if book else None)).fetchone()["id"])


def open_position(conn, *, book: str, ticker: str, side: str, contracts: float, cost_cents: float, fee_cents: float, avg_price: float, decision_id: int | None,
                  order_id: str | None, strategy: str | None, arm: str, score: float | None, line: float | None, ts: datetime) -> int:
    return int(conn.execute("INSERT INTO kalshi_positions (book, ticker, side, contracts, cost_cents, fee_cents, avg_price_cents, opened_at, entry_decision_id, order_id, strategy, arm, score, line) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                            (book, ticker, side, contracts, cost_cents, fee_cents, avg_price, ts, decision_id, order_id, strategy, arm, score, line)).fetchone()["id"])


def blocked_reason(conn, book: str) -> str | None:
    c = rails.load_circuit(conn, rails.KALSHI_CIRCUIT)
    if c.kill_switch:
        return "kill switch"
    if c.entries_paused:
        return "paused"
    if rails.book_halted(conn, book):
        return "book halted"
    return None


def taker_entries(conn, *, book: str, run_id: str, beat_id: int, ts: datetime, picks: list[dict], quotes: dict, orderbooks: dict | None = None, kind: str = "kalshi_taker") -> dict:
    """The taker arm's entries for one minute. ``picks``: [{ticker, side, p, edge, line, table, strategy, hold_s}] at or above the
    line; each is sized (``budget_cents``) and walked through its order book (``orderbooks[ticker]`` when held, else the top
    of book) at ``p``; a fill is opened at its fee-inclusive VWAP with the walk's worst level as the limit (what the live
    mirror submits as an IOC). One position per (market, side); the other side of a held market is skipped."""
    opens = open_positions(conn, book); held = {(p["ticker"], p["side"]) for p in opens}; held_markets = {p["ticker"] for p in opens}
    cash = cash_cents(conn, book); deployed = open_cost_cents(conn, book); bankroll = cash + deployed
    blocked = blocked_reason(conn, book); entries = []; n = 0
    for pk in picks:
        tk, side = pk["ticker"], pk["side"]; edge = float(pk["edge"]); usd0 = 0.0
        if (tk, side) in held or tk in held_markets or blocked:
            rail = blocked or ("held" if (tk, side) in held else "other side held")
            decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=usd0, reason=f"{kind} pick not taken", rail=rail,
                     detail={"edge": edge, "line": pk["line"], "strategy": pk.get("strategy"), "p": pk.get("p")}, book=book)
            continue
        size, why = budget_cents(edge, pk["line"], pk.get("table"), bankroll, cash, deployed)
        if size <= 0:
            decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=0.0, reason=why, rail="sizing",
                     detail={"edge": edge, "line": pk["line"], "strategy": pk.get("strategy"), "p": pk.get("p")}, book=book)
            continue
        ob = (orderbooks or {}).get(tk) or top_of_book(quotes, tk, side)
        fill = fill_budget(ob, side, float(pk["p"]), size, min_payout_cents=config.KALSHI_MIN_PAYOUT_CENTS)
        if not fill.is_fillable:
            decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=size / 100.0, reason=f"book walk: {fill.stopped_reason}",
                     rail="paper_fill", detail={"edge": edge, "line": pk["line"], "strategy": pk.get("strategy"), "p": pk.get("p"), "budget_cents": size, "best_ask": fill.best_ask_cents}, book=book)
            continue
        cost = fill.cost_cents - fill.fee_cents; fee = fill.fee_cents
        did = decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind=f"{kind}_enter", edge=edge, usd=fill.cost_cents / 100.0,
                       reason=f"{pk.get('strategy') or 'ev'}: edge {edge:.3f} ≥ {pk['line']:.3f}; {why}; {fill.quantity} × {fill.vwap_cents:.1f}c (limit {fill.limit_price_cents}c)",
                       detail={"edge": edge, "line": pk["line"], "strategy": pk.get("strategy"), "p": pk.get("p"), "contracts": fill.quantity, "vwap_cents": fill.vwap_cents,
                               "limit_price_cents": fill.limit_price_cents, "fee_cents": fee, "levels": fill.levels_taken, "budget_cents": size, "hold_s": pk.get("hold_s")}, book=book)
        pid = open_position(conn, book=book, ticker=tk, side=side, contracts=float(fill.quantity), cost_cents=float(cost), fee_cents=float(fee), avg_price=cost / fill.quantity,
                            decision_id=did, order_id=None, strategy=pk.get("strategy"), arm="taker", score=edge, line=pk["line"], ts=ts)
        held.add((tk, side)); held_markets.add(tk); cash -= fill.cost_cents; deployed += fill.cost_cents; n += 1
        entries.append({"position_id": pid, "decision_id": did, "ticker": tk, "side": side, "contracts": fill.quantity, "limit_price_cents": fill.limit_price_cents,
                        "vwap_cents": fill.vwap_cents, "cost_cents": fill.cost_cents, "p": pk.get("p"), "edge": edge, "line": pk["line"], "strategy": pk.get("strategy")})
    return {"entered": n, "entries": entries, "blocked": blocked, "cash_cents": cash}


def settle(conn, *, book: str, results: dict, ts: datetime, run_id: str | None = None, beat_id: int | None = None, payouts: dict | None = None) -> list[dict]:
    """Close the book's open positions in markets that resolved (``results``: ticker → 'yes'|'no'); a contract on the
    winning side pays 100c. ``payouts`` (live): ticker → (payout_cents, fee_cents) from the exchange's settlement row."""
    out = []
    for p in open_positions(conn, book):
        res = results.get(p["ticker"])
        if res not in ("yes", "no"):
            continue
        won = res == p["side"]
        if payouts and p["ticker"] in payouts:
            payout, fee_s = payouts[p["ticker"]]
        else:
            payout, fee_s = (PAYOUT_CENTS * float(p["contracts"]) if won else 0.0), 0.0
        realized = payout - float(p["cost_cents"]) - float(p["fee_cents"]) - fee_s
        did = None
        if run_id is not None and beat_id is not None:
            did = decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=p["ticker"], side=p["side"], kind="kalshi_settle", edge=p.get("score"), usd=payout / 100.0,
                           reason=f"settled {res}: {'won' if won else 'lost'} {realized / 100:+.2f} USD", detail={"payout_cents": payout, "realized_cents": realized, "position": int(p["id"])}, book=book)
        conn.execute("UPDATE kalshi_positions SET status = 'settled', closed_at = %s, result = %s, payout_cents = %s, realized_cents = %s, exit_decision_id = %s WHERE id = %s",
                     (ts, res, payout, realized, did, int(p["id"])))
        out.append({**p, "result": res, "payout_cents": payout, "realized_cents": realized})
    return out


def exit_value_cents(p: dict, quotes: dict) -> float:
    """What the position would fetch now: contracts × the side's bid, net of the taker fee (the training label's exit)."""
    bid, _ask = side_quote(quotes, p["ticker"], p["side"])
    if not math.isfinite(bid):
        bid = float(p.get("last_mark_cents") or p.get("avg_price_cents") or 0.0)
    bid = min(max(bid, 0.0), 100.0)
    fee = kalshi_fee_rate_cents(min(max(bid, 1.0), 99.0), p["ticker"]) if 0 < bid < 100 else 0.0
    return float(p["contracts"]) * max(bid - fee, 0.0)


def mark(conn, *, book: str, quotes: dict, ts: datetime, beat_id: int, cash: float | None = None) -> dict:
    """Marks, the wealth row (USD) and the book's own drawdown halt."""
    opens = open_positions(conn, book); value = 0.0; exposure = 0.0
    for p in opens:
        bid, _ = side_quote(quotes, p["ticker"], p["side"])
        if math.isfinite(bid):
            conn.execute("UPDATE kalshi_positions SET last_mark_cents = %s, last_mark_ts = %s WHERE id = %s", (bid, ts, int(p["id"])))
        value += exit_value_cents(p, quotes); exposure += float(p["cost_cents"]) + float(p["fee_cents"])
    cash = cash_cents(conn, book) if cash is None else float(cash)
    wealth = (cash + value + resting_cents(conn, book)) / 100.0
    pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s", (book,)).fetchone(); peak = max(float(pk["pk"] or 0.0), wealth)
    conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                 "ON CONFLICT (beat_id, book) DO NOTHING",
                 (beat_id, book, ts, cash / 100.0, value / 100.0, 0.0, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure / 100.0, len(opens)))
    halted = rails.check_book_drawdown(conn, book, wealth, peak, drawdown=config.KALSHI_KILL_SWITCH_DRAWDOWN, unit="USD")
    return {"wealth": wealth, "peak": peak, "cash": cash / 100.0, "positions_value": value / 100.0, "open": len(opens), "halted": halted}


def eff_ask(ticker: str, ask: float) -> float:
    return effective_price_cents(min(max(ask, 1.0), 99.0), ticker)
