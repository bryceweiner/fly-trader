"""The maker arm (paper book ``paper_kalshi_maker``; mirrored live by kalshi/live.py): for every pick whose maker edge
clears the fly's line, a ``post_only`` bid one tick above the side's best bid and below its ask — the order the training
label assumes (kalshi/decisions.maker_rest / maker_filled). Each minute the plan is re-made:

- a new pick posts (``kalshi_orders`` status 'resting', expiring at min(close − ``KALSHI_MAKER_QUIET_MIN``, now +
  ``KALSHI_MAKER_TTL_H``)); nothing posts inside the quiet margin or beyond ``KALSHI_MAKER_MAX_RESTING`` open orders;
- a resting order stays at its price until it fills or expires, as the label's order does: re-pricing it when the book
  moves would be another order than the one trained on (and live, our own bid would be the best bid it chased);
- a resting order whose (market, side) the fly no longer picks is cancelled;
- paper adjudication against the minute's extremes (``kalshi_minutes``): a bid at ``r`` fills when the side's lowest ask
  reached ``r``, or a taker sold the side below ``r`` — or at ``r`` when the bid was first at its price (it improved on
  the best bid when posted); a fill opens a position with the maker fee where the series charges one; an order past its
  expiry expires.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

from .. import config
from . import paper as P
from .decisions import maker_filled, maker_rest

log = logging.getLogger(__name__)
BOOK = P.BOOKS["maker"]


def rest_price(side_bid: float, side_ask: float) -> int:
    return int(maker_rest(side_bid, side_ask))


def resting(conn, book: str = BOOK) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM kalshi_orders WHERE book = %s AND status = 'resting' ORDER BY id", (book,)).fetchall()]


def _set_status(conn, oid: int, status: str, reason: str | None = None) -> None:
    conn.execute("UPDATE kalshi_orders SET status = %s, error = COALESCE(%s, error), updated_at = now() WHERE id = %s", (status, reason, oid))


def plan(conn, *, book: str, run_id: str, beat_id: int, ts: datetime, now: float, picks: list[dict], quotes: dict, metas: dict, maker_fee: dict | None = None) -> dict:
    """One minute of the maker arm. ``picks``: [{ticker, side, p, edge, line, table, strategy}] with edge ≥ line (the
    maker strategies'); ``quotes``: ticker → (yes_bid, yes_ask, ...); ``metas``: ticker → MarketMeta (close_ts, maker_fee)."""
    have = resting(conn, book); by_key = {(o["ticker"], o["side"]): o for o in have}
    opens = P.open_positions(conn, book); held = {(p["ticker"], p["side"]) for p in opens}; held_markets = {p["ticker"] for p in opens}
    cash = P.cash_cents(conn, book); deployed = P.open_cost_cents(conn, book) + P.resting_cents(conn, book); bankroll = cash + deployed
    blocked = P.blocked_reason(conn, book)
    wanted: dict[tuple, dict] = {}
    for pk in picks:
        key = (pk["ticker"], pk["side"]); meta = metas.get(pk["ticker"])
        if meta is None or meta.close_ts is None or key in held or pk["ticker"] in held_markets:
            continue
        if meta.close_ts - now <= config.KALSHI_MAKER_QUIET_MIN * 60.0:
            continue
        bid, ask = P.side_quote(quotes, pk["ticker"], pk["side"])
        if not (math.isfinite(bid) and math.isfinite(ask)) or not (1 <= bid < ask <= 99):
            continue
        wanted[key] = {**pk, "rest": rest_price(bid, ask), "bid": bid, "ask": ask, "close_ts": meta.close_ts, "maker_fee": bool(meta.maker_fee)}
    posted, canceled, replaced = [], 0, 0
    # cancellations: no longer picked (a picked order keeps its price: see the module docstring)
    for key, o in by_key.items():
        if wanted.get(key) is None:
            _set_status(conn, int(o["id"]), "canceled", "no longer picked"); canceled += 1
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=o["ticker"], side=o["side"], kind="kalshi_maker_cancel", edge=None, usd=float(o["price_cents"]) * float(o["count"]) / 100.0,
                       reason="no longer picked", detail={"order": int(o["id"]), "price_cents": o["price_cents"]}, book=book)
    n_rest = sum(1 for o in by_key.values() if o is not None)
    for key, w in wanted.items():
        if by_key.get(key) is not None:
            continue                                                    # still resting at the right price
        tk, side = key; edge = float(w["edge"])
        if blocked:
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=0.0, reason="maker pick not posted", rail=blocked,
                       detail={"edge": edge, "line": w["line"], "strategy": w.get("strategy")}, book=book)
            continue
        if n_rest >= config.KALSHI_MAKER_MAX_RESTING:
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=0.0, reason="maker pick not posted", rail="max resting",
                       detail={"edge": edge, "line": w["line"], "resting": n_rest}, book=book)
            continue
        size, why = P.budget_cents(edge, w["line"], w.get("table"), bankroll, cash, deployed)
        rest = w["rest"]; fee1 = P.maker_fee_cents(rest, tk, w["maker_fee"]); eff = rest + fee1
        count = math.floor(size / eff) if size > 0 else 0
        if count < 1 or count * (100.0 - eff) < config.KALSHI_MIN_PAYOUT_CENTS:
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="blocked", edge=edge, usd=size / 100.0,
                       reason=why if size <= 0 else f"{count} contracts at {rest}c pay too little", rail="sizing", detail={"edge": edge, "line": w["line"], "budget_cents": size}, book=book)
            continue
        exp_ts = min(w["close_ts"] - config.KALSHI_MAKER_QUIET_MIN * 60.0, now + config.KALSHI_MAKER_TTL_H * 3600.0)
        did = P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=tk, side=side, kind="kalshi_maker_post", edge=edge, usd=count * rest / 100.0,
                         reason=f"{w.get('strategy') or 'ev'}: maker edge {edge:.3f} ≥ {w['line']:.3f}; {why}; {count} × {rest}c resting (bid {w['bid']:.0f}c, ask {w['ask']:.0f}c)",
                         detail={"edge": edge, "line": w["line"], "strategy": w.get("strategy"), "p": w.get("p"), "count": count, "rest_cents": rest, "fee_cents_each": fee1,
                                 "expiration_ts": exp_ts}, book=book)
        oid = int(conn.execute("INSERT INTO kalshi_orders (book, decision_id, ticker, side, action, price_cents, count, tif, post_only, expiration_ts, status, fill_count, remaining, "
                               "fair_cents, margin_cents, strategy, request) VALUES (%s,%s,%s,%s,'buy',%s,%s,'good_till_canceled',true,%s,'resting',0,%s,%s,%s,%s,%s) RETURNING id",
                               (book, did, tk, side, rest, float(count), datetime.fromtimestamp(exp_ts, timezone.utc), float(count), float(w.get("p") or 0.0) * 100.0,
                                float(w["ask"]) - rest, w.get("strategy"), json.dumps({"p": w.get("p"), "edge": edge, "line": w["line"], "bid": w["bid"], "ask": w["ask"]},
                                                                                  default=str))).fetchone()["id"])
        n_rest += 1; deployed += count * rest; cash -= count * rest
        posted.append({"order_row": oid, "decision_id": did, "ticker": tk, "side": side, "price_cents": rest, "count": count, "expiration_ts": exp_ts, "p": w.get("p"), "edge": edge,
                       "line": w["line"], "strategy": w.get("strategy"), "maker_fee": w["maker_fee"]})
    return {"posted": posted, "canceled": canceled, "replaced": replaced, "resting": n_rest, "blocked": blocked}


def side_extremes(ex: tuple, side: str) -> tuple[float, float]:
    """(lowest ask, lowest taker sale) of one side from a minute's ``(yes_ask_low, yes_bid_high, yes_sold_low,
    yes_bought_high)``: the NO ask is 100 − the YES bid, and a taker sold NO when it bought YES."""
    f = lambda v: float(v) if v is not None and math.isfinite(float(v)) else math.nan
    ask_low, bid_high, sold_low, bought_high = (tuple(ex) + (None, None))[:4]
    if side == "yes":
        return f(ask_low), f(sold_low)
    return 100.0 - f(bid_high), 100.0 - f(bought_high)


def adjudicate(conn, *, book: str, run_id: str, beat_id: int, ts: datetime, now: float, extremes: dict, metas: dict) -> dict:
    """Paper fills and expiries for the resting orders. ``extremes``: ticker → (yes_ask_low, yes_bid_high, yes_sold_low,
    yes_bought_high) of the minute (absent when the market had no message this minute: nothing traded)."""
    filled, expired = [], 0
    for o in resting(conn, book):
        exp = o["expiration_ts"].timestamp() if o.get("expiration_ts") else math.inf
        if now >= exp:
            _set_status(conn, int(o["id"]), "expired"); expired += 1
            continue
        ex = extremes.get(o["ticker"])
        if not ex:
            continue
        r = float(o["price_cents"]); ask_low, sold_low = side_extremes(ex, o["side"])
        bid_then = float((o.get("request") or {}).get("bid") or r)          # an order without its posting bid is taken as joining the queue
        if not bool(maker_filled(r, bid_then, ask_low, sold_low)):
            continue
        meta = metas.get(o["ticker"]); charged = bool(meta.maker_fee) if meta is not None else False
        count = float(o["count"]); fee = P.maker_fee_cents(r, o["ticker"], charged) * count; cost = r * count
        conn.execute("UPDATE kalshi_orders SET status = 'filled', fill_count = %s, remaining = 0, avg_fill_cents = %s, fee_cents = %s, updated_at = now() WHERE id = %s", (count, r, fee, int(o["id"])))
        did = P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=o["ticker"], side=o["side"], kind="kalshi_maker_fill", edge=None, usd=cost / 100.0,
                         reason=f"resting {int(count)} × {int(r)}c filled (side ask low {ask_low:.0f}c, taker sale low {sold_low:.0f}c)", detail={"order": int(o["id"]), "fee_cents": fee}, book=book)
        pid = P.open_position(conn, book=book, ticker=o["ticker"], side=o["side"], contracts=count, cost_cents=cost, fee_cents=fee, avg_price=r, decision_id=o.get("decision_id"),
                              order_id=o.get("order_id"), strategy=o.get("strategy"), arm="maker", score=None, line=None, ts=ts)
        filled.append({"order_row": int(o["id"]), "position_id": pid, "decision_id": did, "ticker": o["ticker"], "side": o["side"], "count": count, "price_cents": r})
    return {"filled": filled, "expired": expired}
