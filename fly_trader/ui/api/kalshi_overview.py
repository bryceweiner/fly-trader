"""Kalshi overview: what the visual fly is doing in plain words, both arms' books, this minute's scoring, and the open
positions by category (the brain with both flies lit is api/brain.py). Every Kalshi page's topic answers
``{"dormant": true}`` while Kalshi is not set up (no KALSHI_API_KEY_ID): the menu hides these pages then, but a page
opened by its URL must explain itself rather than fail on tables a dormant install never filled."""
from __future__ import annotations

import functools
from collections import Counter
from datetime import datetime, timezone

from ... import config
from ...db.queries import q, q1
from ..hub import ActionError, TopicError, topic
from ..state import KALSHI_BOOKS, kalshi_enabled, kalshi_state

DORMANT = {"dormant": True}
NOT_SET_UP = "Kalshi is not set up on this install: set KALSHI_API_KEY_ID (and KALSHI_PRIVATE_KEY_PATH) in .env and restart the console."


def kalshi_topic(name: str, every: float):
    """``@topic`` for a Kalshi page: the dormant marker instead of a read while Kalshi is off."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(params: dict):
            return dict(DORMANT) if not kalshi_enabled() else fn(params)
        topic(name, every)(wrapper)
        return fn
    return deco


def require_kalshi() -> None:
    if not kalshi_enabled():
        raise ActionError(NOT_SET_UP)


def book_param(params: dict) -> str:
    b = params.get("book") or "paper_kalshi_taker"
    if b not in KALSHI_BOOKS:
        raise TopicError(f"unknown Kalshi book {b!r}")
    return b


def narrative(k: dict) -> str:
    fly = k["fly"]; lines = []
    if fly.get("stage") and fly.get("stage") != "not trading":
        arms = fly.get("arms") or {}; ln = fly.get("lines") or {}
        parts = ", ".join(f"**{s}** ({arms.get(s, '?')}, edge ≥ {float(ln.get(s, 0)):.3f})" for s in ln)
        money = ("Both arms also trade Kalshi subaccount " + f"{config.KALSHI_SUBACCOUNT} with real dollars (cap ${config.KALSHI_CAPITAL_USD:g})." if k["live_money"]
                 else "No real money is at risk" + (": live mirroring is off (KALSHI_LIVE_ENABLED=0)." if not config.KALSHI_LIVE_ENABLED else f" yet: {', '.join(k['prerequisites_missing'])}."))
        lines.append(f"**{k['trading_why']}** The visual fly (bootstrap **#{fly.get('bootstrap')}**) scores both sides of every quoted Kalshi market inside "
                     f"{config.KALSHI_MAX_DAYS_TO_CLOSE} days of its close, every minute: it predicts the probability the side pays and trades the **edge** — that probability "
                     f"minus the fee-inclusive price. Strategies: {parts}. The **taker arm** buys at the ask (an IOC at the book walk's worst level); the **maker arm** rests a "
                     f"bid one tick inside the ask and cancels or replaces it as the quote moves. Every position runs to settlement, which is also what teaches the fly. {money}")
    else:
        lines.append(f"**{k['trading_why']}**" + ("" if k["runner"] else " Start the Kalshi trading engine on the Processes page."))
    if k["training"]:
        tr = k["train"]; pr = k["train_progress"]
        lines.append(f"Kalshi training is running: {tr.get('stage', '')}" + (f" · {pr.get('stage', '')} ({pr.get('step')}/{pr.get('total')})" if pr.get("total") else "") + ".")
    return "\n\n".join(lines)


def _f(x) -> float | None:
    return None if x is None else float(x)


@kalshi_topic("kalshi_overview.summary", every=5)
def summary(params: dict) -> dict:
    """The narrative, each paper arm's book (wealth against the shared cap, settled P&L and winners, exposure, drawdown),
    the live mirror when live is on or wanted, and this minute's scoring."""
    k = kalshi_state(); arms = []
    for arm, book in (("taker", "paper_kalshi_taker"), ("maker", "paper_kalshi_maker")):
        w = k["books"].get(book) or {}
        c = q1("SELECT count(*) AS n, sum(realized_cents) AS pnl, count(*) FILTER (WHERE realized_cents > 0) AS wins FROM kalshi_positions WHERE book = %s AND status <> 'open'", (book,)) or {}
        spark = [float(r["wealth"]) for r in q("SELECT wealth FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 180", (book,)) if r["wealth"] is not None][::-1]
        arms.append({"arm": arm, "book": book, "label": KALSHI_BOOKS[book], "marked": bool(w), "wealth": _f(w.get("wealth")), "exposure": float(w.get("exposure") or 0),
                     "n_open": int(w.get("n_open") or 0), "drawdown": _f(w.get("drawdown")), "pnl": float(c.get("pnl") or 0) / 100.0, "settled": int(c.get("n") or 0),
                     "wins": int(c.get("wins") or 0), "spark": spark})
    fly = k["fly"]; live = None
    if k["live_money"] or config.KALSHI_LIVE_ENABLED:
        lv = fly.get("live") or {}; g = lv.get("gap") or {}
        live = {"stage": lv.get("stage"), "wealth": _f(lv.get("wealth")), "cash": _f(lv.get("cash")), "wealth_taker": _f(lv.get("wealth_taker")),
                "wealth_maker": _f(lv.get("wealth_maker")), "error": lv.get("error"),
                "gap": {a: {"live_mean": _f((g.get(a) or {}).get("live_mean")), "mirror_mean": _f((g.get(a) or {}).get("mirror_mean"))} for a in ("taker", "maker")}}
    b = fly.get("books") or {}; tb = b.get("taker") or {}; mb = b.get("maker") or {}
    minute = {"minute": fly.get("minute"), "markets_active": int(fly.get("markets_active") or 0), "eligible": int(fly.get("eligible") or 0), "picks": int(fly.get("picks") or 0),
              "taker_entered": int(tb.get("entered") or 0), "maker_posted": int(mb.get("posted") or 0), "maker_resting": int(mb.get("resting") or 0),
              "settled": int(tb.get("settled") or 0) + int(mb.get("settled") or 0), "stage": fly.get("stage"), "at": k["fly_at"]}
    return {"narrative": narrative(k), "capital": float(config.KALSHI_CAPITAL_USD), "capital_g": f"{config.KALSHI_CAPITAL_USD:g}", "arms": arms, "live": live,
            "minute": minute, "runner": k["runner"], "trading": k["trading"], "trading_why": k["trading_why"]}


@kalshi_topic("kalshi_overview.positions", every=15)
def positions(params: dict) -> dict:
    """Wealth by book over 30 days (averaged into at most ~1,200 points per book: a month of minute marks is 43,200) and
    the open positions with their market, closing time and category."""
    marks = q("""WITH w AS (SELECT ts, book, wealth FROM wealth_marks WHERE book = ANY(%s) AND ts > now() - interval '30 days'),
                      b AS (SELECT GREATEST(60, EXTRACT(epoch FROM max(ts) - min(ts)) / 1200) AS s FROM w)
                 SELECT to_timestamp(floor(EXTRACT(epoch FROM w.ts) / b.s) * b.s) AS ts, w.book, avg(w.wealth) AS wealth
                 FROM w, b GROUP BY 1, 2 ORDER BY 1""", (list(KALSHI_BOOKS),))
    by_book: dict[str, list] = {}
    for r in marks:
        by_book.setdefault(r["book"], []).append([r["ts"], _f(r["wealth"])])
    rows = q("""SELECT p.book, p.ticker, p.side, p.contracts, p.cost_cents, p.fee_cents, p.avg_price_cents, p.last_mark_cents, p.strategy, p.opened_at, m.close_time, m.title,
                       COALESCE(s.category, e.category) AS category
                FROM kalshi_positions p LEFT JOIN kalshi_markets m USING (ticker) LEFT JOIN kalshi_events e USING (event_ticker) LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker
                WHERE p.book = ANY(%s) AND p.status = 'open' ORDER BY p.opened_at DESC""", (list(KALSHI_BOOKS),))
    now = datetime.now(timezone.utc)
    out = [{"book": KALSHI_BOOKS.get(r["book"], r["book"]), "ticker": r["ticker"], "market": (r["title"] or r["ticker"])[:60], "side": r["side"], "strategy": r["strategy"] or "—",
            "contracts": float(r["contracts"]), "paid": float(r["avg_price_cents"] or 0), "now": _f(r["last_mark_cents"]),
            "closes_in_h": (r["close_time"] - now).total_seconds() / 3600 if r["close_time"] else None, "category": r["category"] or "—"} for r in rows]
    return {"series": [{"name": KALSHI_BOOKS[b], "data": by_book[b]} for b in KALSHI_BOOKS if b in by_book], "rows": out,
            "categories": Counter(r["category"] for r in out).most_common()}
