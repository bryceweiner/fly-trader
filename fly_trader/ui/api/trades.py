"""Trades: closed trades with their results, the decision log (every buy, sell and pick not taken) and fills, for either
memecoin chain."""
from __future__ import annotations

from ... import markets
from ...db.queries import q, q1
from ...ops.wallets import explorer
from ..hub import TopicError, topic
from ..state import system_state, labels, market_param

# The decision log's "Show" pills → the decision kinds each covers (the selector's, the fly's and the older generic kinds).
KINDS = {"Bought": ("selector_enter", "fly_enter", "enter"), "Sold": ("selector_exit", "fly_exit", "exit"), "Not taken": ("blocked",)}
ACTION = {k: p for p, ks in KINDS.items() for k in ks}
LIMIT = 300


def _book(params: dict, mk: markets.MarketSpec) -> str:
    """The book asked for when it belongs to this chain, else the chain's selector book — the Streamlit page reset its
    remembered book the same way when the chain changed under it."""
    b = params.get("book")
    if not isinstance(b, str) or not b or len(b) > 64 or markets.for_book(b) is not mk:
        return mk.selector_book
    return b


@topic("trades.books", every=30)
def books(params: dict) -> dict:
    """The Book picker: the chain's own books first (even before their first trade), then any other book seen in
    positions or fills that belongs to this chain (replay_* test books count as Solana's)."""
    mk = market_param(params)
    seen = [r["book"] for r in q("SELECT DISTINCT book FROM positions UNION SELECT DISTINCT book FROM fills ORDER BY 1")]
    out = list(mk.books) + [b for b in seen if b not in mk.books and markets.for_book(b) is mk]
    live = bool(system_state(mk)["live_money"])                  # while live money trades, the page opens on the live book
    return {"chain": mk.chain, "unit": mk.unit, "books": out, "default": mk.live_book if live else mk.selector_book, "live": live,
            "help": f"{mk.selector_book} is the selector's paper book, {mk.fly_book} the fly's, {mk.live_book} the bot wallet; amounts in {mk.unit}."}


def _thin(points: list, n: int = 180) -> list:
    """At most ``n`` points of a series, evenly spaced, always keeping the last (sparklines and charts need no more)."""
    if len(points) <= n:
        return points
    step = len(points) / (n - 1)
    return [points[int(i * step)] for i in range(n - 1)] + [points[-1]]


@topic("trades.summary", every=15)
def summary(params: dict) -> dict:
    """The book's results at a glance: the closed-trade metrics, the open positions, realized P&L cumulated trade by
    trade and per day, and how the trades exited and which strategy took them."""
    mk = market_param(params); book = _book(params, mk)
    c = q1("SELECT count(*) AS n, sum(realized_sol) AS pnl, avg(realized_sol / NULLIF(cost_sol, 0)) AS avg_ret, "
           "count(*) FILTER (WHERE realized_sol > 0) AS wins FROM positions WHERE book = %s AND status = 'closed'", (book,)) or {}
    o = q1("SELECT count(*) AS n, COALESCE(sum(cost_sol), 0) AS cost FROM positions WHERE book = %s AND status = 'open'", (book,)) or {}
    n, wins = int(c.get("n") or 0), int(c.get("wins") or 0)
    cum, run = [], 0.0
    for r in q("SELECT closed_at, realized_sol FROM positions WHERE book = %s AND status = 'closed' AND closed_at IS NOT NULL ORDER BY closed_at, id", (book,)):
        run += float(r["realized_sol"] or 0); cum.append([r["closed_at"], run])
    daily = list(reversed(q("SELECT (closed_at AT TIME ZONE 'UTC')::date AS day, count(*) AS n, sum(realized_sol) AS pnl, count(*) FILTER (WHERE realized_sol > 0) AS wins "
                            "FROM positions WHERE book = %s AND status = 'closed' AND closed_at IS NOT NULL GROUP BY 1 ORDER BY 1 DESC LIMIT 90", (book,))))
    exits = q("SELECT COALESCE(forced_exit_kind, 'time') AS kind, count(*) AS n, sum(realized_sol) AS pnl FROM positions WHERE book = %s AND status = 'closed' "
              "GROUP BY 1 ORDER BY 2 DESC, 1", (book,))
    strategies = q("SELECT COALESCE(strategy, '') AS strategy, count(*) AS n, sum(realized_sol) AS pnl, count(*) FILTER (WHERE realized_sol > 0) AS wins "
                   "FROM positions WHERE book = %s AND status = 'closed' GROUP BY 1 ORDER BY 2 DESC, 1", (book,))
    return {"chain": mk.chain, "book": book, "unit": mk.unit,
            "metrics": {"n": n, "pnl": c.get("pnl"), "avg_ret": c.get("avg_ret"), "wins": wins, "win_rate": wins / n if n else None,
                        "open": int(o.get("n") or 0), "open_cost": float(o.get("cost") or 0)},
            "cum": _thin(cum), "daily": [{"day": d["day"].isoformat(), "n": d["n"], "pnl": float(d["pnl"] or 0), "wins": d["wins"]} for d in daily],
            "exits": [{"kind": e["kind"], "n": e["n"], "pnl": float(e["pnl"] or 0)} for e in exits],
            "strategies": [{"strategy": x["strategy"] or None, "n": x["n"], "pnl": float(x["pnl"] or 0), "wins": x["wins"]} for x in strategies]}


@topic("trades.closed", every=10)
def closed(params: dict) -> dict:
    mk = market_param(params); book = _book(params, mk)
    rows = q("SELECT mint, opened_at, closed_at, cost_sol, realized_sol, forced_exit_kind, strategy FROM positions WHERE book = %s AND status = 'closed' "
             "ORDER BY closed_at DESC LIMIT %s", (book, LIMIT))
    names = labels([r["mint"] for r in rows])
    out = []
    for r in rows:
        cost, pnl = float(r["cost_sol"] or 0), float(r["realized_sol"] or 0)
        out.append({"mint": r["mint"], "token": names.get(r["mint"], r["mint"][:6]), "opened": r["opened_at"], "closed": r["closed_at"],
                    "held_min": (r["closed_at"] - r["opened_at"]).total_seconds() / 60 if r["closed_at"] else None, "cost": cost, "pnl": pnl,
                    "ret": pnl / float(r["cost_sol"]) * 100 if r["cost_sol"] else None, "exit": r["forced_exit_kind"] or "time", "strategy": r["strategy"]})
    return {"chain": mk.chain, "book": book, "unit": mk.unit, "rows": out}


@topic("trades.decisions", every=10)
def decisions(params: dict) -> dict:
    mk = market_param(params); book = _book(params, mk)
    show = params.get("show", list(KINDS))
    if not isinstance(show, list) or not all(isinstance(s, str) for s in show):
        raise TopicError("show must be a list of " + ", ".join(KINDS))
    want = [k for p in show if p in KINDS for k in KINDS[p]]
    rows = q("SELECT ts, kind, mint, m_hat, size_sol, rail, reason FROM decisions WHERE kind = ANY(%s) AND (detail->>'book' = %s OR %s = ANY(book_targets)) "
             "ORDER BY id DESC LIMIT %s", (want, book, book, LIMIT)) if want else []
    names = labels([r["mint"] for r in rows])
    out = [{"ts": r["ts"], "action": ACTION.get(r["kind"], r["kind"]), "kind": r["kind"], "mint": r["mint"], "token": names.get(r["mint"], (r["mint"] or "")[:6]),
            "score": float(r["m_hat"]) if r["m_hat"] is not None else None, "size": float(r["size_sol"] or 0),
            "why": (f"{r['rail']}: " if r["rail"] else "") + (r["reason"] or "")} for r in rows]
    return {"chain": mk.chain, "book": book, "unit": mk.unit, "show": [p for p in KINDS if p in show], "rows": out}


@topic("trades.fills", every=10)
def fills(params: dict) -> dict:
    mk = market_param(params); book = _book(params, mk)
    rows = q("SELECT ts, side, mint, sol_delta_lamports, price_sol, fee_lamports, verified_by, signature FROM fills WHERE book = %s ORDER BY id DESC LIMIT %s",
             (book, LIMIT))
    names = labels([r["mint"] for r in rows])
    out = [{"ts": r["ts"], "side": r["side"], "mint": r["mint"], "token": names.get(r["mint"], (r["mint"] or "")[:6]), "amount": (r["sol_delta_lamports"] or 0) / 1e9,
            "price": float(r["price_sol"]) if r["price_sol"] is not None else None, "fee": (r["fee_lamports"] or 0) / 1e9,
            "source": "simulated" if r["verified_by"] == "model" else r["verified_by"], "signature": r["signature"],
            "url": explorer(mk.chain, "tx", r["signature"]) if r["signature"] else None} for r in rows]
    return {"chain": mk.chain, "book": book, "unit": mk.unit, "rows": out}
