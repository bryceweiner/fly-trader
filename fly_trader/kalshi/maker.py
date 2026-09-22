"""The maker arm (paper book ``paper_kalshi_maker``; mirrored live by kalshi/live.py): for every pick whose maker edge
clears the fly's line, a ``post_only`` bid resting one tick inside the side's ask — the price the training label assumed
(kalshi/decisions.py: rest = ask − 1, filled iff a later minute's ask reached it). Each minute the plan is re-made:

- a new pick posts (``kalshi_orders`` status 'resting', expiring at min(close − ``KALSHI_MAKER_QUIET_MIN``, now +
  ``KALSHI_MAKER_TTL_H``)); nothing posts inside the quiet margin or beyond ``KALSHI_MAKER_MAX_RESTING`` open orders;
- a resting order whose rest price no longer sits one tick inside the ask is cancelled and re-posted (cancel/replace);
- a resting order whose (market, side) the fly no longer picks is cancelled;
- paper adjudication against the minute's extremes (``kalshi_minutes.yes_ask_low`` / ``yes_bid_high``): a YES bid at
  ``r`` fills when the minute's lowest YES ask ≤ r; a NO bid at ``r`` when 100 − the highest YES bid ≤ r; a fill opens a
  position with the maker fee where the series charges one; an order past its expiry expires.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

from .. import config
from . import paper as P
from .decisions import MAKER_DISCOUNT

log = logging.getLogger(__name__)
BOOK = P.BOOKS["maker"]


def rest_price(side_ask: float) -> int:
    return int(min(max(round(side_ask) - MAKER_DISCOUNT, 1), 99))


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
        _bid, ask = P.side_quote(quotes, pk["ticker"], pk["side"])
        if not math.isfinite(ask) or not (2 <= ask <= 99):
            continue
        wanted[key] = {**pk, "rest": rest_price(ask), "ask": ask, "close_ts": meta.close_ts, "maker_fee": bool(meta.maker_fee)}
    posted, canceled, replaced = [], 0, 0
    # cancellations: no longer picked, or the rest price moved
    for key, o in by_key.items():
        w = wanted.get(key)
        if w is None:
            _set_status(conn, int(o["id"]), "canceled", "no longer picked"); canceled += 1
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=o["ticker"], side=o["side"], kind="kalshi_maker_cancel", edge=None, usd=float(o["price_cents"]) * float(o["count"]) / 100.0,
                       reason="no longer picked", detail={"order": int(o["id"]), "price_cents": o["price_cents"]}, book=book)
        elif int(o["price_cents"]) != w["rest"]:
            _set_status(conn, int(o["id"]), "replaced", f"ask moved to {w['ask']:.0f}c"); replaced += 1; by_key[key] = None
            P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=o["ticker"], side=o["side"], kind="kalshi_maker_cancel", edge=w["edge"], usd=float(o["price_cents"]) * float(o["count"]) / 100.0,
                       reason=f"replace: rest {o['price_cents']}c → {w['rest']}c", detail={"order": int(o["id"])}, book=book)
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
                         reason=f"{w.get('strategy') or 'ev'}: maker edge {edge:.3f} ≥ {w['line']:.3f}; {why}; {count} × {rest}c resting (ask {w['ask']:.0f}c)",
                         detail={"edge": edge, "line": w["line"], "strategy": w.get("strategy"), "p": w.get("p"), "count": count, "rest_cents": rest, "fee_cents_each": fee1,
                                 "expiration_ts": exp_ts}, book=book)
        oid = int(conn.execute("INSERT INTO kalshi_orders (book, decision_id, ticker, side, action, price_cents, count, tif, post_only, expiration_ts, status, fill_count, remaining, "
                               "fair_cents, margin_cents, strategy, request) VALUES (%s,%s,%s,%s,'buy',%s,%s,'good_till_canceled',true,%s,'resting',0,%s,%s,%s,%s,%s) RETURNING id",
                               (book, did, tk, side, rest, float(count), datetime.fromtimestamp(exp_ts, timezone.utc), float(count), float(w.get("p") or 0.0) * 100.0,
                                float(w["ask"]) - rest, w.get("strategy"), json.dumps({"p": w.get("p"), "edge": edge, "line": w["line"]}, default=str))).fetchone()["id"])
        n_rest += 1; deployed += count * rest; cash -= count * rest
        posted.append({"order_row": oid, "decision_id": did, "ticker": tk, "side": side, "price_cents": rest, "count": count, "expiration_ts": exp_ts, "p": w.get("p"), "edge": edge,
                       "line": w["line"], "strategy": w.get("strategy"), "maker_fee": w["maker_fee"]})
    return {"posted": posted, "canceled": canceled, "replaced": replaced, "resting": n_rest, "blocked": blocked}


def adjudicate(conn, *, book: str, run_id: str, beat_id: int, ts: datetime, now: float, extremes: dict, metas: dict) -> dict:
    """Paper fills and expiries for the resting orders. ``extremes``: ticker → (yes_ask_low, yes_bid_high) of the minute
    (None when the market had no message this minute: nothing traded through)."""
    filled, expired = [], 0
    for o in resting(conn, book):
        exp = o["expiration_ts"].timestamp() if o.get("expiration_ts") else math.inf
        if now >= exp:
            _set_status(conn, int(o["id"]), "expired"); expired += 1
            continue
        ex = extremes.get(o["ticker"])
        if not ex:
            continue
        ask_low, bid_high = ex
        r = float(o["price_cents"])
        if o["side"] == "yes":
            hit = ask_low is not None and math.isfinite(ask_low) and ask_low <= r
        else:
            hit = bid_high is not None and math.isfinite(bid_high) and (100.0 - bid_high) <= r
        if not hit:
            continue
        meta = metas.get(o["ticker"]); charged = bool(meta.maker_fee) if meta is not None else False
        count = float(o["count"]); fee = P.maker_fee_cents(r, o["ticker"], charged) * count; cost = r * count
        conn.execute("UPDATE kalshi_orders SET status = 'filled', fill_count = %s, remaining = 0, avg_fill_cents = %s, fee_cents = %s, updated_at = now() WHERE id = %s", (count, r, fee, int(o["id"])))
        did = P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ts, ticker=o["ticker"], side=o["side"], kind="kalshi_maker_fill", edge=None, usd=cost / 100.0,
                         reason=f"resting {int(count)} × {int(r)}c filled (ask low {ask_low}, bid high {bid_high})", detail={"order": int(o["id"]), "fee_cents": fee}, book=book)
        pid = P.open_position(conn, book=book, ticker=o["ticker"], side=o["side"], contracts=count, cost_cents=cost, fee_cents=fee, avg_price=r, decision_id=o.get("decision_id"),
                              order_id=o.get("order_id"), strategy=o.get("strategy"), arm="maker", score=None, line=None, ts=ts)
        filled.append({"order_row": int(o["id"]), "position_id": pid, "decision_id": did, "ticker": o["ticker"], "side": o["side"], "count": count, "price_cents": r})
    return {"filled": filled, "expired": expired}
