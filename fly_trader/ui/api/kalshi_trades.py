"""Kalshi trades: settled positions with their results, resting and recent maker orders, exchange fills and the decision
log, per book (one topic per view; ``book`` is one of state.KALSHI_BOOKS)."""
from __future__ import annotations

from ...db.queries import q, q1
from ..hub import TopicError
from .kalshi_overview import book_param, kalshi_topic

# the decision log's pills: label → decision kinds (the Streamlit page's mapping and default selection)
KINDS = {"Taker bought": ("kalshi_taker_enter",), "Maker posted": ("kalshi_maker_post",), "Maker filled": ("kalshi_maker_fill",), "Maker cancelled": ("kalshi_maker_cancel",),
         "Settled": ("kalshi_settle",), "Exits": ("kalshi_exit",), "Not taken": ("blocked",)}
DEFAULT_KINDS = ["Taker bought", "Maker posted", "Maker filled", "Settled"]


def _f(x) -> float | None:
    return None if x is None else float(x)


@kalshi_topic("kalshi_trades.settled", every=15)
def settled(params: dict) -> dict:
    book = book_param(params)
    c = q1("SELECT count(*) AS n, sum(realized_cents) AS pnl, avg(realized_cents / NULLIF(cost_cents + fee_cents, 0)) AS avg_ret, count(*) FILTER (WHERE realized_cents > 0) AS wins, "
           "sum(fee_cents) AS fees FROM kalshi_positions WHERE book = %s AND status <> 'open'", (book,)) or {}
    rows = q("""SELECT p.ticker, p.side, p.contracts, p.avg_price_cents, p.fee_cents, p.result, p.payout_cents, p.realized_cents, p.cost_cents, p.strategy, p.opened_at, p.closed_at, p.status, m.title
                FROM kalshi_positions p LEFT JOIN kalshi_markets m USING (ticker) WHERE p.book = %s AND p.status <> 'open' ORDER BY p.closed_at DESC LIMIT 300""", (book,))
    out = [{"ticker": r["ticker"], "market": (r["title"] or r["ticker"])[:70], "side": r["side"], "strategy": r["strategy"] or "—", "contracts": float(r["contracts"]),
            "paid": float(r["avg_price_cents"] or 0), "fee": float(r["fee_cents"] or 0) / 100, "result": r["result"] or r["status"], "payout": float(r["payout_cents"] or 0) / 100,
            "pnl": float(r["realized_cents"] or 0) / 100,
            "ret": float(r["realized_cents"] or 0) / (float(r["cost_cents"] or 0) + float(r["fee_cents"] or 0)) * 100 if r["cost_cents"] else None,
            "opened": r["opened_at"], "settled": r["closed_at"]} for r in rows]
    # the book's cumulative realized P&L (every settled position, thinned to about 180 points) and its split by strategy
    cum = q("""WITH c AS (SELECT closed_at, sum(realized_cents) OVER (ORDER BY closed_at, id) AS cum, row_number() OVER (ORDER BY closed_at, id) AS i, count(*) OVER () AS n
                          FROM kalshi_positions WHERE book = %s AND status <> 'open' AND closed_at IS NOT NULL)
               SELECT closed_at, cum FROM c WHERE i %% GREATEST(1, n / 180) = 0 OR i = n ORDER BY closed_at""", (book,))
    strat = q("""SELECT COALESCE(strategy, '—') AS strategy, count(*) AS n, sum(realized_cents) AS pnl, count(*) FILTER (WHERE realized_cents > 0) AS wins
                 FROM kalshi_positions WHERE book = %s AND status <> 'open' GROUP BY 1 ORDER BY 2 DESC""", (book,))
    n = int(c.get("n") or 0)
    return {"n": n, "pnl": float(c.get("pnl") or 0) / 100.0, "avg_ret": _f(c.get("avg_ret")), "wins": int(c.get("wins") or 0), "fees": float(c.get("fees") or 0) / 100.0, "rows": out,
            "cum": [[r["closed_at"], float(r["cum"] or 0) / 100.0] for r in cum],
            "by_strategy": [{"strategy": r["strategy"], "n": int(r["n"]), "pnl": float(r["pnl"] or 0) / 100.0, "wins": int(r["wins"] or 0)} for r in strat]}


@kalshi_topic("kalshi_trades.orders", every=15)
def orders(params: dict) -> dict:
    book = book_param(params)
    rows = q("""SELECT o.ts, o.status, o.ticker, o.side, o.price_cents, o.count, o.fill_count, o.tif, o.post_only, o.expiration_ts, o.strategy, o.order_id, o.error, m.title
                FROM kalshi_orders o LEFT JOIN kalshi_markets m USING (ticker) WHERE o.book = %s ORDER BY o.id DESC LIMIT 300""", (book,))
    out = [{"time": r["ts"], "status": r["status"], "ticker": r["ticker"], "market": (r["title"] or r["ticker"])[:60], "side": r["side"], "price": int(r["price_cents"]),
            "count": float(r["count"]), "filled": float(r["fill_count"] or 0), "type": ("post-only " if r["post_only"] else "") + r["tif"].replace("_", " "),
            "expires": r["expiration_ts"], "strategy": r["strategy"] or "—", "order_id": r["order_id"] or "", "error": r["error"] or ""} for r in rows]
    return {"resting": sum(1 for r in rows if r["status"] == "resting"), "rows": out}


@kalshi_topic("kalshi_trades.fills", every=15)
def fills(params: dict) -> dict:
    """A live book's fills include the subaccount's fills not yet attributed to a book (``book IS NULL``)."""
    book = book_param(params)
    rows = q("SELECT ts, ticker, side, action, price_cents, count, fee_cents, is_taker, order_id, trade_id FROM kalshi_fills WHERE book = %s OR (%s LIKE 'live%%' AND book IS NULL) ORDER BY ts DESC LIMIT 300",
             (book, book))
    return {"rows": [{"time": r["ts"], "ticker": r["ticker"], "side": r["side"], "action": r["action"], "price": float(r["price_cents"]), "count": float(r["count"]),
                      "fee": float(r["fee_cents"] or 0) / 100, "taker": bool(r["is_taker"]), "order": r["order_id"], "trade": r["trade_id"]} for r in rows]}


@kalshi_topic("kalshi_trades.decisions", every=15)
def decisions(params: dict) -> dict:
    book = book_param(params)
    pick = params.get("kinds", DEFAULT_KINDS)
    if not isinstance(pick, list) or any(p not in KINDS for p in pick):
        raise TopicError(f"kinds must be a list of {', '.join(KINDS)}")
    want = [k for p in pick for k in KINDS[p]]
    rows = q("SELECT ts, kind, mint, pool, m_hat, size_sol, rail, reason FROM decisions WHERE kind = ANY(%s) AND detail->>'book' = %s ORDER BY id DESC LIMIT 300", (want, book)) if want else []
    action = {v[0]: k for k, v in KINDS.items()}
    return {"kinds": list(KINDS), "default": DEFAULT_KINDS,
            "rows": [{"time": r["ts"], "action": action.get(r["kind"], r["kind"]), "market": r["mint"], "side": r["pool"], "edge": _f(r["m_hat"]),
                      "dollars": float(r["size_sol"] or 0), "why": (f"{r['rail']}: " if r["rail"] else "") + (r["reason"] or "")} for r in rows]}
