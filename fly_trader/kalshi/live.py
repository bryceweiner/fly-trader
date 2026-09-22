"""Live execution for the Kalshi fly (books ``live_kalshi_taker`` / ``live_kalshi_maker``): the paper arms' decisions of
each minute mirrored on the dedicated subaccount, only when ``KALSHI_LIVE_ENABLED=1`` and every prerequisite holds
(config.kalshi_live_prerequisites_missing). Per minute:

- taker: every paper taker entry becomes an IOC order at the walk's worst level for as many contracts as the paper fill,
  sized down to the subaccount's cash (the cap is never exceeded); the exchange's fill count, cost and fees open the live
  position (``client_order_id`` 'fly-<decision id>', ``subaccount`` on every order);
- maker: every paper post becomes a GTC ``post_only`` order with the paper expiry; every paper cancel or replace cancels
  the live order carrying the same decision; fills arrive on the stream's ``fill`` channel (``kalshi_fills``, keyed by
  order id) and open live positions; resting orders are reconciled with ``GET /portfolio/orders`` every ``RECONCILE_S``;
- settlements: ``GET /portfolio/settlements`` closes live positions with the exchange's payout and fee; a market whose
  result the stream wrote settles at 100c per winning contract when the exchange row is late;
- wealth: subaccount cash + positions at the bid + resting collateral → ``wealth_marks`` per live book (USD); the kill
  switch (circuit 2, ``KALSHI_KILL_SWITCH_DRAWDOWN`` on the combined live wealth) blocks entries and, with
  ``KILL_SWITCH_LIQUIDATE``, cancels every resting order and sells every position IOC at the bid;
- gap: over the last ``GAP_TRADES`` settled live positions per arm, a mean return trailing the paper mirror's same
  decisions by more than ``GAP_MAX`` pauses entries (circuit 2) and raises an alert. Learning is unaffected.
Order failures feed the circuit breaker (``rails.record_failure``, circuit 2); a tripped circuit blocks entries.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..agent import rails
from ..db.apilog import record_event
from . import paper as P
from .client import KalshiApiError
from .vendor.kalshi_client import kalshi_fee_rate_cents

log = logging.getLogger(__name__)
BOOKS = P.LIVE_BOOKS
MIRROR = P.BOOKS
GAP_TRADES, GAP_MAX = 50, 0.02
RECONCILE_S, SETTLE_LOOKBACK_S = 300.0, 3 * 86400.0


def _c(v) -> float:
    try:
        return float(v) * 100.0 if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def _n(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def order_facts(o: dict) -> dict:
    """What an exchange order object says about its fills: count, cost and fee in cents (the V2 ``*_dollars`` / ``*_fp`` fields)."""
    filled = _n(o.get("fill_count_fp") if "fill_count_fp" in o else o.get("fill_count"))
    remaining = _n(o.get("remaining_count_fp") if "remaining_count_fp" in o else o.get("remaining_count"))
    cost = _c(o.get("taker_fill_cost_dollars")) if o.get("taker_fill_cost_dollars") is not None else _n(o.get("taker_fill_cost"), math.nan)
    if not math.isfinite(cost):
        cost = _c(o.get("maker_fill_cost_dollars")) if o.get("maker_fill_cost_dollars") is not None else _n(o.get("maker_fill_cost"), math.nan)
    fee = _c(o.get("taker_fees_dollars")) if o.get("taker_fees_dollars") is not None else _n(o.get("taker_fees"), math.nan)
    if not math.isfinite(fee):
        fee = _c(o.get("maker_fees_dollars")) if o.get("maker_fees_dollars") is not None else _n(o.get("maker_fees"), 0.0)
    price = _c(o.get("yes_price_dollars")) if o.get("yes_price_dollars") is not None else _n(o.get("yes_price"), math.nan)
    if o.get("side") == "no":
        np_ = _c(o.get("no_price_dollars")) if o.get("no_price_dollars") is not None else _n(o.get("no_price"), math.nan)
        price = np_ if math.isfinite(np_) else (100.0 - price if math.isfinite(price) else math.nan)
    return {"order_id": o.get("order_id"), "status": o.get("status"), "filled": filled, "remaining": remaining, "cost_cents": cost, "fee_cents": fee if math.isfinite(fee) else 0.0,
            "price_cents": price}


class KalshiLiveMirror:
    def __init__(self, rest=None):
        if rest is None:
            from .client import rest as _rest
            rest = _rest()
        self.rest = rest; self.sub = config.KALSHI_SUBACCOUNT
        self.last_reconcile = 0.0; self.last_settle_ts = int(time.time() - SETTLE_LOOKBACK_S); self.cash_cents: float | None = None

    # ---- exchange state ----
    def balance_cents(self) -> float:
        b = self.rest.balance(self.sub)
        v = b.get("balance_dollars")
        return _c(v) if v is not None else _n(b.get("balance"), 0.0)

    def _record_order(self, conn, *, book: str, decision_id: int | None, ticker: str, side: str, price: int, count: float, tif: str, post_only: bool, exp_ts: float | None,
                      req: dict, resp: dict | None, error: str | None, strategy: str | None, action: str = "buy") -> int:
        f = order_facts(resp or {})
        return int(conn.execute(
            "INSERT INTO kalshi_orders (book, order_id, client_order_id, decision_id, ticker, side, action, price_cents, count, tif, post_only, expiration_ts, status, fill_count, remaining, "
            "avg_fill_cents, fee_cents, subaccount, strategy, request, response, error) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (book, f["order_id"], req.get("client_order_id"), decision_id, ticker, side, action, price, count, tif, post_only,
             datetime.fromtimestamp(exp_ts, timezone.utc) if exp_ts else None, "error" if error else (f["status"] or "submitted"), f["filled"], f["remaining"],
             (f["cost_cents"] / f["filled"]) if f["filled"] and math.isfinite(f["cost_cents"]) else None, f["fee_cents"], self.sub, strategy,
             json.dumps(req, default=str), json.dumps(resp, default=str) if resp is not None else None, error)).fetchone()["id"])

    def _submit(self, conn, *, book: str, decision_id: int | None, ticker: str, side: str, price: int, count: float, tif: str, post_only: bool, exp_ts: float | None,
                strategy: str | None, action: str = "buy") -> tuple[dict | None, int]:
        coid = f"fly-{decision_id}" if decision_id is not None else None
        req = {"ticker": ticker, "side": side, "price_cents": price, "count": count, "tif": tif, "post_only": post_only, "expiration_ts": int(exp_ts) if exp_ts else None,
               "client_order_id": coid, "subaccount": self.sub, "action": action}
        try:
            resp = self.rest.create_order(ticker, side, price, count, tif=tif, post_only=post_only, expiration_ts=int(exp_ts) if exp_ts else None, client_order_id=coid,
                                          subaccount=self.sub, action=action)
            rails.record_success(conn, rails.KALSHI_CIRCUIT)
            return resp, self._record_order(conn, book=book, decision_id=decision_id, ticker=ticker, side=side, price=price, count=count, tif=tif, post_only=post_only, exp_ts=exp_ts,
                                            req=req, resp=resp, error=None, strategy=strategy, action=action)
        except (KalshiApiError, RuntimeError, OSError) as e:
            rails.record_failure(conn, f"kalshi order: {e}"[:300], rails.KALSHI_CIRCUIT)
            log.warning("kalshi order failed (%s %s %s x%s @%sc): %s", action, side, ticker, count, price, e)
            return None, self._record_order(conn, book=book, decision_id=decision_id, ticker=ticker, side=side, price=price, count=count, tif=tif, post_only=post_only, exp_ts=exp_ts,
                                            req=req, resp=None, error=str(e)[:300], strategy=strategy, action=action)

    # ---- the minute ----
    def minute(self, ctx, *, run_id: str, beat_id: int, taker_entries: list, maker_plan: dict) -> dict:
        conn, m1, now = ctx.conn, ctx.m1, ctx.m1_epoch
        try:
            self.cash_cents = self.balance_cents()
        except Exception as e:
            log.warning("kalshi balance: %s", e)
        out = {"stage": "trading live", "cash": (self.cash_cents or 0.0) / 100.0}
        out["settled"] = self.settlements(conn, ctx, run_id, beat_id)
        out["maker_fills"] = self.maker_fills(conn, ctx, run_id, beat_id)
        c = rails.load_circuit(conn, rails.KALSHI_CIRCUIT)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if c.tripped else "paused" if c.entries_paused else None
        out["blocked"] = blocked
        out["taker"] = self.taker(conn, ctx, run_id, beat_id, taker_entries, blocked) if config.KALSHI_TAKER_LIVE else {"skipped": "KALSHI_TAKER_LIVE=0"}
        out["maker"] = self.maker(conn, ctx, run_id, beat_id, maker_plan, blocked) if config.KALSHI_MAKER_LIVE else {"skipped": "KALSHI_MAKER_LIVE=0"}
        if now - self.last_reconcile >= RECONCILE_S:
            self.last_reconcile = now; out["reconciled"] = self.reconcile(conn)
        wealth = 0.0
        for arm, book in BOOKS.items():
            cash_part = (self.cash_cents or 0.0) if arm == "taker" else 0.0                      # the subaccount's cash counts once, on the taker book
            m = P.mark(conn, book=book, quotes=ctx.quotes, ts=m1, beat_id=beat_id, cash=cash_part); out[f"wealth_{arm}"] = m["wealth"]; wealth += m["wealth"]
        pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = 'live_kalshi'").fetchone(); peak = max(float(pk["pk"] or 0.0), wealth)
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,'live_kalshi',%s,%s,%s,0,%s,%s,%s,%s,%s) "
                     "ON CONFLICT (beat_id, book) DO NOTHING", (beat_id, m1, (self.cash_cents or 0.0) / 100.0, wealth - (self.cash_cents or 0.0) / 100.0, wealth, peak,
                                                               (1.0 - wealth / peak) if peak > 0 else 0.0, wealth - (self.cash_cents or 0.0) / 100.0,
                                                               len(P.open_positions(conn, BOOKS["taker"])) + len(P.open_positions(conn, BOOKS["maker"]))))
        out["wealth"] = wealth
        killed = rails.check_drawdown(conn, wealth, peak, rails.KALSHI_CIRCUIT, drawdown=config.KALSHI_KILL_SWITCH_DRAWDOWN, unit="USD")
        out["liquidated"] = self.liquidate(conn, ctx, run_id, beat_id) if killed and config.KILL_SWITCH_LIQUIDATE else 0
        out["gap"] = {arm: self.gap(conn, arm) for arm in BOOKS}
        return out

    def taker(self, conn, ctx, run_id, beat_id, entries: list, blocked: str | None) -> dict:
        book = BOOKS["taker"]; opens = P.open_positions(conn, book); held = {p["ticker"] for p in opens}
        cash = self.cash_cents if self.cash_cents is not None else 0.0
        floor = config.KALSHI_CASH_FLOOR_USD * 100.0; n = 0; skipped = []
        for e in entries:
            tk, side = e["ticker"], e["side"]
            if tk in held:
                skipped.append((tk, "held")); continue
            if blocked:
                skipped.append((tk, blocked)); continue
            price = int(e["limit_price_cents"] or round(e["vwap_cents"]))
            eff = price + kalshi_fee_rate_cents(price, tk)
            count = min(float(e["contracts"]), math.floor(max(cash - floor, 0.0) / eff))
            if count < 1:
                skipped.append((tk, "cash")); continue
            resp, row = self._submit(conn, book=book, decision_id=e["decision_id"], ticker=tk, side=side, price=price, count=count, tif="immediate_or_cancel", post_only=False,
                                     exp_ts=None, strategy=e.get("strategy"))
            if resp is None:
                continue
            f = order_facts(resp)
            if f["filled"] > 0:
                cost = f["cost_cents"] if math.isfinite(f["cost_cents"]) and f["cost_cents"] > 0 else f["filled"] * price
                P.open_position(conn, book=book, ticker=tk, side=side, contracts=f["filled"], cost_cents=cost, fee_cents=f["fee_cents"], avg_price=cost / f["filled"],
                                decision_id=e["decision_id"], order_id=f["order_id"], strategy=e.get("strategy"), arm="taker", score=e.get("edge"), line=e.get("line"), ts=ctx.m1)
                conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), %s) WHERE id = %s", (book, e["decision_id"]))
                cash -= cost + f["fee_cents"]; held.add(tk); n += 1
            else:
                skipped.append((tk, f"no fill ({f['status']})"))
        self.cash_cents = cash
        return {"entered": n, "skipped": skipped[:10]}

    def maker(self, conn, ctx, run_id, beat_id, plan: dict, blocked: str | None) -> dict:
        book = BOOKS["maker"]; n_post = n_cancel = 0
        # cancels and replaces first: the paper order rows that left 'resting' this minute carry the decision ids of the live orders to cancel
        gone = conn.execute("SELECT o.decision_id, o.ticker FROM kalshi_orders o WHERE o.book = %s AND o.status IN ('canceled','replaced') AND o.updated_at >= %s",
                            (MIRROR["maker"], ctx.m0)).fetchall()
        for g in gone:
            for lo in conn.execute("SELECT id, order_id FROM kalshi_orders WHERE book = %s AND decision_id = %s AND status IN ('resting','submitted') AND order_id IS NOT NULL",
                                   (book, g["decision_id"])).fetchall():
                try:
                    self.rest.cancel_order(lo["order_id"], self.sub); rails.record_success(conn, rails.KALSHI_CIRCUIT)
                    conn.execute("UPDATE kalshi_orders SET status = 'canceled', updated_at = now() WHERE id = %s", (lo["id"],)); n_cancel += 1
                except (KalshiApiError, RuntimeError, OSError) as e:
                    rails.record_failure(conn, f"kalshi cancel: {e}"[:300], rails.KALSHI_CIRCUIT); log.warning("cancel %s failed: %s", lo["order_id"], e)
        skipped = []
        for o in plan.get("posted") or []:
            if blocked:
                skipped.append((o["ticker"], blocked)); continue
            resp, _row = self._submit(conn, book=book, decision_id=o["decision_id"], ticker=o["ticker"], side=o["side"], price=int(o["price_cents"]), count=float(o["count"]),
                                      tif="good_till_canceled", post_only=True, exp_ts=o.get("expiration_ts"), strategy=o.get("strategy"))
            if resp is not None:
                n_post += 1
                conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), %s) WHERE id = %s", (book, o["decision_id"]))
        return {"posted": n_post, "canceled": n_cancel, "skipped": skipped[:10]}

    def maker_fills(self, conn, ctx, run_id, beat_id) -> int:
        """Fills of our live maker orders (the stream writes ``kalshi_fills``; the fill row joins its order's book) open positions."""
        book = BOOKS["maker"]; n = 0
        rows = conn.execute("""SELECT f.trade_id, f.order_id, f.ticker, f.side, f.price_cents, f.count, f.fee_cents, f.ts, o.decision_id, o.strategy, o.id AS oid
                               FROM kalshi_fills f JOIN kalshi_orders o ON o.order_id = f.order_id WHERE o.book = %s AND f.action = 'buy'
                               AND NOT EXISTS (SELECT 1 FROM kalshi_positions p WHERE p.order_id = f.order_id AND p.book = %s)""", (book, book)).fetchall()
        by_order: dict[str, list] = {}
        for r in rows:
            by_order.setdefault(r["order_id"], []).append(r)
        for oid, fs in by_order.items():
            count = sum(float(f["count"]) for f in fs); px = float(fs[0]["price_cents"]) if fs[0]["side"] == "yes" else 100.0 - float(fs[0]["price_cents"])
            if fs[0]["side"] == "yes":
                px = float(fs[0]["price_cents"])
            cost = sum(float(f["count"]) * (float(f["price_cents"]) if f["side"] == "yes" else 100.0 - float(f["price_cents"])) for f in fs)
            fee = sum(float(f["fee_cents"] or 0.0) for f in fs)
            P.open_position(conn, book=book, ticker=fs[0]["ticker"], side=fs[0]["side"], contracts=count, cost_cents=cost, fee_cents=fee, avg_price=cost / count,
                            decision_id=fs[0]["decision_id"], order_id=oid, strategy=fs[0]["strategy"], arm="maker", score=None, line=None, ts=ctx.m1)
            conn.execute("UPDATE kalshi_orders SET fill_count = %s, status = CASE WHEN remaining IS NOT NULL AND remaining - %s <= 0 THEN 'filled' ELSE status END, updated_at = now() WHERE id = %s",
                         (count, count, fs[0]["oid"]))
            n += 1
        return n

    def reconcile(self, conn) -> dict:
        """Resting live maker orders against the exchange: gone or filled there → closed here."""
        book = BOOKS["maker"]; ours = conn.execute("SELECT id, order_id FROM kalshi_orders WHERE book = %s AND status IN ('resting','submitted') AND order_id IS NOT NULL", (book,)).fetchall()
        if not ours:
            return {"resting": 0}
        try:
            live = {o.get("order_id"): o for o in self.rest.orders(status="resting", subaccount=self.sub)}
        except Exception as e:
            log.warning("kalshi orders: %s", e); return {"error": str(e)[:200]}
        closed = 0
        for r in ours:
            o = live.get(r["order_id"])
            if o is None:
                try:
                    o = self.rest.order(r["order_id"])
                except Exception:
                    o = {}
                f = order_facts(o.get("order") or o)
                conn.execute("UPDATE kalshi_orders SET status = %s, fill_count = %s, remaining = %s, updated_at = now() WHERE id = %s",
                             (f["status"] or "gone", f["filled"], f["remaining"], r["id"])); closed += 1
        return {"resting": len(ours) - closed, "closed": closed}

    def settlements(self, conn, ctx, run_id, beat_id) -> int:
        """Exchange settlement rows (payout and fee per market) close live positions; a market the stream marked resolved
        without a row yet settles at face value."""
        payouts: dict[str, tuple] = {}; results: dict[str, str] = {}
        try:
            for s in self.rest.settlements(self.sub, min_ts=self.last_settle_ts):
                tk = s.get("ticker"); res = s.get("market_result")
                if tk and res in ("yes", "no"):
                    rev = _c(s.get("revenue_dollars")) if s.get("revenue_dollars") is not None else _n(s.get("revenue"), math.nan)
                    fee = _c(s.get("fee_dollars")) if s.get("fee_dollars") is not None else _n(s.get("fee_cost") or s.get("fee"), 0.0)
                    if math.isfinite(rev):
                        payouts[tk] = (rev, fee)
                    results[tk] = res
            self.last_settle_ts = int(ctx.m1_epoch) - 3600
        except Exception as e:
            log.warning("kalshi settlements: %s", e)
        n = 0
        for book in BOOKS.values():
            opens = P.open_positions(conn, book)
            res = {**ctx.settled([p["ticker"] for p in opens]), **results}
            n += len(P.settle(conn, book=book, results=res, ts=ctx.m1, run_id=run_id, beat_id=beat_id, payouts=payouts))
        return n

    def liquidate(self, conn, ctx, run_id, beat_id) -> int:
        n = 0
        for lo in conn.execute("SELECT id, order_id FROM kalshi_orders WHERE book = %s AND status IN ('resting','submitted') AND order_id IS NOT NULL", (BOOKS["maker"],)).fetchall():
            try:
                self.rest.cancel_order(lo["order_id"], self.sub); conn.execute("UPDATE kalshi_orders SET status = 'canceled', error = 'kill switch', updated_at = now() WHERE id = %s", (lo["id"],))
            except Exception as e:
                log.warning("kill switch cancel %s: %s", lo["order_id"], e)
        for book in BOOKS.values():
            for p in P.open_positions(conn, book):
                bid, _ = P.side_quote(ctx.quotes, p["ticker"], p["side"])
                if not math.isfinite(bid) or not (1 <= bid <= 99):
                    continue
                did = P.decision(conn, beat_id=beat_id, run_id=run_id, ts=ctx.m1, ticker=p["ticker"], side=p["side"], kind="kalshi_exit", edge=None, usd=float(p["cost_cents"]) / 100.0,
                                 reason="kill switch: liquidating", detail={"position": int(p["id"])}, book=book)
                resp, _ = self._submit(conn, book=book, decision_id=did, ticker=p["ticker"], side=p["side"], price=int(bid), count=float(p["contracts"]), tif="immediate_or_cancel",
                                       post_only=False, exp_ts=None, strategy=p.get("strategy"), action="sell")
                if resp is not None:
                    f = order_facts(resp)
                    if f["filled"] > 0:
                        proceeds = f["cost_cents"] if math.isfinite(f["cost_cents"]) and f["cost_cents"] > 0 else f["filled"] * bid
                        realized = proceeds - float(p["cost_cents"]) * (f["filled"] / float(p["contracts"])) - float(p["fee_cents"]) - f["fee_cents"]
                        if f["filled"] >= float(p["contracts"]) - 1e-9:
                            conn.execute("UPDATE kalshi_positions SET status = 'closed', closed_at = %s, payout_cents = %s, realized_cents = %s, exit_decision_id = %s WHERE id = %s",
                                         (ctx.m1, proceeds, realized, did, int(p["id"])))
                        else:
                            conn.execute("UPDATE kalshi_positions SET contracts = contracts - %s, cost_cents = cost_cents * (1 - %s / contracts) WHERE id = %s", (f["filled"], f["filled"], int(p["id"])))
                        n += 1
        if n:
            record_event("error", "kalshi_live", f"kill switch: liquidated {n} live position(s)", {"positions": n})
        return n

    def gap(self, conn, arm: str) -> dict:
        rows = conn.execute("""SELECT l.realized_cents / NULLIF(l.cost_cents + l.fee_cents, 0) AS rl, p.realized_cents / NULLIF(p.cost_cents + p.fee_cents, 0) AS rp
                               FROM kalshi_positions l JOIN kalshi_positions p ON p.entry_decision_id = l.entry_decision_id AND p.book = %s AND p.status <> 'open'
                               WHERE l.book = %s AND l.status <> 'open' AND l.entry_decision_id IS NOT NULL ORDER BY l.closed_at DESC LIMIT %s""", (MIRROR[arm], BOOKS[arm], GAP_TRADES)).fetchall()
        rl = np.array([r["rl"] for r in rows if r["rl"] is not None and r["rp"] is not None], dtype=np.float64)
        rp = np.array([r["rp"] for r in rows if r["rl"] is not None and r["rp"] is not None], dtype=np.float64)
        out = {"trades": int(len(rl)), "live_mean": float(rl.mean()) if len(rl) else None, "mirror_mean": float(rp.mean()) if len(rp) else None}
        if len(rl) >= GAP_TRADES and rl.mean() - rp.mean() < -GAP_MAX:
            c = rails.load_circuit(conn, rails.KALSHI_CIRCUIT)
            if not c.entries_paused:
                conn.execute("UPDATE circuit_state SET entries_paused = true, updated_at = now() WHERE id = %s", (rails.KALSHI_CIRCUIT,))
                conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('gap_pause', %s)", (json.dumps({**out, "arm": arm, "circuit": rails.KALSHI_CIRCUIT}),))
                record_event("error", "kalshi_live", f"{arm}: entries paused: live trails its paper mirror by {(rp.mean() - rl.mean()) * 100:.1f} points per trade over {len(rl)} trades", out)
            out["paused"] = True
        return out
