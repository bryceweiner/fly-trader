"""Overview: what the system is doing in plain words, the book the money is in (live, else the selector's paper book), this minute's scoring, the race between the selector
and the plastic fly, and the open positions. (The brain at the top of the page is ui/api/brain.py.)"""
from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timedelta, timezone

from ... import config, markets
from ...agent.selector_session import pinned_snapshot
from ...db.queries import q, q1
from ..cache import ttl_cache
from ..hub import topic
from ..state import ago, backtest_line, labels, market_info, market_param, pct, system_state

MAX_POINTS = 1500             # a 240 px line chart cannot show more; a month of minute marks is ~45,000 per book


def narrative(s: dict) -> str:
    m = s["model"]; lines = []; mk = s["market"]; u = mk.unit; rh = mk.chain != "sol"
    if m:
        run = m.get("run") or {}; meta = m["meta"]; mm = meta.get("model") or {}
        money = "No real money is at risk." if not s["live_money"] else f"Trades spend real {u} from the bot wallet."
        what = "graduated Pons pool on Robinhood Chain (prices in ETH, a non-ETH quote converted at the minute's base price)" if rh else "PumpSwap token"
        resq = f"{mk.min_resq():.3g}" if rh else mm.get("min_resq_sol", "—"); vol = f"{mk.min_vol_15m():.3g}" if rh else mm.get("min_vol_15m_sol", "—")
        who = ("The selector picks what the bot wallet trades, live," if s["live_money"] and not s["handover"]
               else "The selector races the fly on paper" if s["live_money"] else "The selector is paper trading")
        lines.append(f"**{s['trading_why']}** {who} with model **#{m['id']}** (trained through {meta.get('trained_through', '—')}). "
                     f"Every minute it predicts the net {meta.get('horizon_min', '—')}-minute return of each {what} that traded, at least {mm.get('min_age_h', '—')} hours "
                     f"past graduation, with a pool of at least {resq} {u} and {vol} {u} traded in the last 15 minutes, and buys those predicted to make at least "
                     f"**{pct(float(run.get('threshold') or meta.get('threshold') or 0), 1)}** — sized by how certain the "
                     f"model is (up to {config.MAX_POSITION_FRACTION:.0%} of the bankroll, never the {mk.gas_reserve():g} {u} gas reserve) — "
                     f"and sells it **{run.get('horizon_min') or meta.get('horizon_min') or 30} minutes** later. {money}")
        lines.append(backtest_line(meta))
        if pinned_snapshot() == m["id"]:
            lines[-1] += " **It trades because you pinned it** (Model page); unpin it to let a current model take over."
    fly = s["fly"]
    if fly.get("stage") and fly.get("stage") != "not trading":
        if s["live_money"]:
            seat = ("holds the seat and trades the bot wallet live" if s["handover"] else
                    "trades the bot wallet live: it buys what the selector buys and sizes it the same, unless its learning vetoes the row")
            lines.append(f"**The plastic fly** {seat}; its paper book is the mirror. It learns every minute from each scored token's realized return, "
                         "two hours later" + ("." if s["handover"] else ", and still races the selector for the seat."))
        else:
            lines.append("**The plastic fly** " + ("holds the seat" if s["handover"] else "races the selector on its own paper book")
                         + f": it buys at a predicted {pct(fly.get('line'), 1)} and learns every minute from each scored token's realized return, two hours later.")
    else:
        lines.append(f"**{s['trading_why']}** Start the trading engine on the Processes page to trade.")
    if s["training"]:
        tr = s["train_status"]; step, total = tr.get("step"), tr.get("total")
        lines.append(f"Training is running: {tr.get('stage', '')}" + (f" ({step}/{total})" if step is not None and total else "") + ".")
    return "\n\n".join(lines)


def _fly_running(fly: dict) -> bool:
    return bool(fly.get("stage")) and fly.get("stage") != "not trading"


def marks(books: list[str], days: int | None = None, points: int = MAX_POINTS) -> list[dict]:
    """Wealth marks of the books (``days``: only the most recent), thinned to about ``points`` per book: the newest mark
    of each time bucket, so the line keeps its shape and ends at the latest mark."""
    win = " AND ts > now() - make_interval(days => %s)" if days else ""; args = (books, days) if days else (books,)
    span = q1(f"SELECT count(*) AS n, min(ts) AS lo, max(ts) AS hi FROM wealth_marks WHERE book = ANY(%s){win}", args) or {}
    if not span.get("n"):
        return []
    if span["n"] <= points * max(1, len(books)):
        return q(f"SELECT ts, book, wealth FROM wealth_marks WHERE book = ANY(%s){win} ORDER BY ts", args)
    width = max(60, math.ceil((span["hi"] - span["lo"]).total_seconds() / points))
    return q(f"""SELECT max(ts) AS ts, book, (array_agg(wealth ORDER BY ts DESC))[1] AS wealth FROM wealth_marks WHERE book = ANY(%s){win}
                 GROUP BY book, date_bin(make_interval(secs => %s), ts, timestamptz '2000-01-01') ORDER BY 1""", (*args, width))


def _series(rows: list[dict], books: list[str]) -> list[dict]:
    by: dict[str, list] = {}
    for r in rows:
        if r["wealth"] is not None:
            by.setdefault(r["book"], []).append([r["ts"], float(r["wealth"])])
    return [{"name": b, "data": by[b]} for b in books if b in by]


@topic("overview.summary", every=5)
def summary(params: dict) -> dict:
    """The narrative and the money's metrics — the live book while live money trades, else the selector's paper book:
    wealth, realized P&L, closed (wins), open, drawdown, with sparklines (wealth and drawdown over the book's whole
    history, realized P&L cumulative over the last 180 trades)."""
    mk = market_param(params); s = system_state(mk); live = bool(s["live_money"]); BOOK = mk.live_book if live else mk.selector_book; u = mk.unit
    first = q1("SELECT wealth FROM wealth_marks WHERE book = %s ORDER BY ts LIMIT 1", (BOOK,)) if live else None
    cap = float(first["wealth"]) if first and first.get("wealth") is not None else mk.capital()
    w = q1("SELECT wealth, exposure, n_open, drawdown FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (BOOK,)) or {}
    c = q1("SELECT count(*) AS n, sum(realized_sol) AS pnl, count(*) FILTER (WHERE realized_sol > 0) AS wins FROM positions WHERE book = %s AND status = 'closed'", (BOOK,)) or {}
    spark = [float(r["wealth"]) for r in marks([BOOK], points=180) if r["wealth"] is not None]      # the book since it started, thinned
    peak, dd_spark = 0.0, []
    for x in spark:                         # drawdown below the running peak, on the same points
        peak = max(peak, x); dd_spark.append(1 - x / peak if peak > 0 else 0.0)
    trades = [float(r["realized_sol"] or 0) for r in q("SELECT realized_sol FROM positions WHERE book = %s AND status = 'closed' ORDER BY closed_at DESC NULLS LAST, id DESC LIMIT 180", (BOOK,))][::-1]
    base = float(c.get("pnl") or 0) - sum(trades); pnl_spark = []
    for r in trades:                        # cumulative realized P&L, ending at the total
        base += r; pnl_spark.append(base)
    return {"market": market_info(mk), "narrative": narrative(s), "marked": bool(w), "wealth": w.get("wealth"), "exposure": w.get("exposure"),
            "n_open": int(w.get("n_open") or 0), "drawdown": w.get("drawdown"), "capital": cap, "closed": int(c.get("n") or 0), "pnl": c.get("pnl"),
            "wins": int(c.get("wins") or 0), "spark": spark, "pnl_spark": pnl_spark, "dd_spark": dd_spark, "live": live, "book": BOOK,
            "label": "Live wealth" if live else "Paper wealth",
            "wealth_help": (f"The bot wallet: {u} plus open positions net of exit costs; started at {cap:g} {u}." if live
                            else f"Started at {cap:g} {u}; open positions valued net of exit costs.")}


@topic("overview.minute", every=3)
def minute(params: dict) -> dict:
    """This minute's scoring from the selector's status blob: what traded, what was eligible, the best scores and the picks."""
    mk = market_param(params); s = system_state(mk); sel = s["selector"]; u = mk.unit
    top = [t for t in (sel.get("top_scores") or []) if t and t[0]]; names = labels([t[0] for t in top])
    return {"unit": u, "runner": s["runner"], "trading": s["trading"], "trading_why": s["trading_why"], "scored_at": s["selector_at"],
            "minute": sel.get("minute"), "minute_hm": str(sel["minute"])[11:16] if sel.get("minute") else None,
            "mints_traded": int(sel.get("mints_traded") or 0), "eligible": int(sel.get("eligible") or 0), "score_max": float(sel.get("score_max") or 0),
            "threshold": float(sel.get("threshold") or 0), "picks": int(sel.get("picks") or 0), "entered": int(sel.get("entered") or 0),
            "exited": int(sel.get("exited") or 0), "latency_s": sel.get("latency_s"),
            "top": [{"mint": t[0], "token": names.get(t[0], t[0][:6]), "score": float(t[1])} for t in top],
            "eligible_help": f"Pool ≥ {mk.min_resq():.3g} {u} and ≥ {mk.min_vol_15m():.3g} {u} traded in the last 15 minutes."}


@topic("overview.race", every=30)
def race(params: dict) -> dict:
    """The race: the selector's and the fly's 14-day realized P&L, the handover rule or the handover, and wealth by book."""
    mk = market_param(params); s = system_state(mk); ho = s["handover"]; fly = s["fly"]; u = mk.unit; books = list(mk.books)
    rows = marks(books, days=30)
    if not rows and not _fly_running(fly):
        return {"hidden": True, "unit": u}
    since = datetime.now(timezone.utc) - timedelta(days=14)
    pnl = {r["book"]: r for r in q("SELECT book, COALESCE(sum(realized_sol), 0) AS pnl, count(*) AS n FROM positions WHERE book = ANY(%s) "
                                   "AND status = 'closed' AND closed_at >= %s GROUP BY book", (books, since))}
    dates = [(since + timedelta(days=i)).date() for i in range(15)]; daily = {b: [0.0] * 15 for b in books}      # each day's realized P&L (bar sparks)
    for r in q("SELECT book, (closed_at AT TIME ZONE 'UTC')::date AS d, sum(realized_sol) AS pnl FROM positions WHERE book = ANY(%s) AND status = 'closed' "
               "AND closed_at >= %s GROUP BY 1, 2", (books, since)):
        if r["d"] in dates:
            daily[r["book"]][dates.index(r["d"])] = float(r["pnl"] or 0)
    rs = fly.get("race_started_at")                          # the fly's own race clock (a paper reset does not restart the race)
    start = datetime.fromtimestamp(float(rs), timezone.utc) if rs else (q1("SELECT min(ts) AS t FROM wealth_marks WHERE book = %s", (mk.fly_book,)) or {}).get("t")
    success = caption = None
    if ho:
        success = (f"The fly took the selector's seat {ago(ho.get('at'))}: {float(ho.get('fly_pnl_sol') or 0):+.4g} {u} vs {float(ho.get('selector_pnl_sol') or 0):+.4g} {u} "
                   f"over {ho.get('days')} days. " + ("It trades the bot wallet; its paper book is the mirror." if s["live_money"]
                                                      else f"Live trading waits for {'RH_LIVE_ENABLED' if mk.chain != 'sol' else 'LIVE_ENABLED'}=1, the live prerequisites and a funded wallet."))
    else:
        days = (datetime.now(timezone.utc) - start).total_seconds() / 86400 if start else 0.0
        caption = ("The fly takes the seat when its realized P&L over the last 14 days is at least the selector's, with at least 30 trades"
                   + (f" · racing for {days:.1f} days." if start else " · the race starts when the fly first trades."))
    titles = ((mk.selector_book, "Selector · 14 days"), (mk.fly_book, "Fly · 14 days")) + (((mk.live_book, "Live · 14 days"),) if ho or s["live_money"] else ())
    return {"hidden": False, "unit": u, "handover": bool(ho), "success": success, "caption": caption,
            "metrics": [{"book": b, "title": t, "pnl": (pnl.get(b) or {}).get("pnl"), "n": int((pnl.get(b) or {}).get("n") or 0), "daily": daily[b]} for b, t in titles],
            "days": dates,
            "series": _series(rows, books)}


@topic("overview.book", every=15)
def book(params: dict) -> dict:
    """The money's book — live while live money trades, else the selector's paper book: its whole wealth history and
    its open positions (held, exits in, cost, gross return)."""
    mk = market_param(params); live = bool(system_state(mk)["live_money"]); BOOK = mk.live_book if live else mk.selector_book; u = mk.unit
    hist = _series(marks([BOOK]), [BOOK])
    rows = q("SELECT mint, opened_at, cost_sol, entry_price, last_mark_price, hold_s, strategy FROM positions WHERE book = %s AND status = 'open' ORDER BY opened_at", (BOOK,))
    out = []
    if rows:
        hold = int((system_state(mk).get("model") or {}).get("run", {}).get("horizon_min") or 30)
        now = datetime.now(timezone.utc); names = labels([r["mint"] for r in rows])
        for r in rows:
            held = (now - r["opened_at"]).total_seconds() / 60
            out.append({"mint": r["mint"], "token": names.get(r["mint"], r["mint"][:6]), "strategy": r["strategy"] or "—", "held_min": held,
                        "exits_in_min": max(0.0, (float(r["hold_s"]) / 60 if r["hold_s"] else hold) - held), "cost": float(r["cost_sol"]),
                        "ret": (float(r["last_mark_price"]) / float(r["entry_price"]) - 1) * 100 if r["entry_price"] and r["last_mark_price"] else None})
    return {"unit": u, "history": hist[0]["data"] if hist else [], "positions": out, "live": live, "book": BOOK,
            "title": "Live wealth" if live else "Paper wealth", "sub": "the bot wallet since it started trading" if live else "the selector book since it started"}


# ---------------------------------------------------------------- trades by hour of day
def _hour_of(ts) -> int | None:
    """The UTC hour of a timestamp: a datetime, or the text Postgres wrote into an archive ('2026-09-15 03:55:00+00')."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, str):
        t = ts.strip().replace(" ", "T", 1)
        if re.search(r"[+-]\d\d$", t):
            t += ":00"
        try:
            ts = datetime.fromisoformat(t)
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).hour


def _num(v) -> float | None:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _archive_signature() -> tuple:
    """Every archived positions file (a reset archives what it deletes): path, mtime, size — the cache key."""
    out = []
    for p in sorted(config.PG_ARCHIVE_DIR.glob("*/positions.csv")):
        try:
            st = p.stat(); out.append((str(p), st.st_mtime, st.st_size))
        except OSError:
            continue
    return tuple(out)


@ttl_cache(None, maxsize=2)
def _archived(signature: tuple) -> list[tuple[int, str, int, float | None]]:
    """(id, book, UTC hour opened, realized) of every archived position: archives never change, a new reset adds a file."""
    rows = []
    for path, _mtime, _size in signature:
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    h = _hour_of(r.get("opened_at"))
                    if h is None or not r.get("id") or not r.get("book"):
                        continue
                    rows.append((int(r["id"]), r["book"], h, _num(r.get("realized_sol"))))
        except (OSError, ValueError, csv.Error):
            continue
    return rows


def _empty_hours() -> list[dict]:
    return [{"hour": h, "trades": 0, "closed": 0, "wins": 0, "pnl": 0.0} for h in range(24)]


def hour_buckets(rows, live_books: set[str]) -> dict[str, dict[str, list[dict]]]:
    """``rows``: (book, UTC hour opened, realized or None). Per chain, 24 hourly buckets for every book ("all") and for the
    live books alone ("live"): trades entered, how many closed, winners, realized P&L in the chain's unit."""
    out: dict[str, dict[str, list[dict]]] = {}
    for book, hour, realized in rows:
        if hour is None or "kalshi" in book:
            continue
        chain = markets.for_book(book).chain
        per = out.setdefault(chain, {"all": _empty_hours(), "live": _empty_hours()})
        for scope in ("all", "live") if book in live_books else ("all",):
            b = per[scope][hour]
            b["trades"] += 1
            if realized is not None:
                b["closed"] += 1; b["pnl"] += realized; b["wins"] += int(realized > 0)
    return out


@topic("overview.hours", every=60)
def hours(params: dict) -> dict:
    """Every memecoin trade ever made — the positions table and every reset's archive (data/pg_archive/*/positions.csv) —
    by the UTC hour it was entered, per chain (Solana, then Robinhood Chain when enabled), all books and live alone."""
    seen: set[int] = set(); rows = []
    for r in q("SELECT id, book, opened_at, realized_sol FROM positions WHERE opened_at IS NOT NULL"):
        seen.add(int(r["id"])); rows.append((r["book"], _hour_of(r["opened_at"]), _num(r["realized_sol"])))
    sig = _archive_signature()
    for pid, book, h, realized in _archived(sig):
        if pid not in seen:                     # a row both archived and still in the table (a failed delete) counts once
            seen.add(pid); rows.append((book, h, realized))
    enabled = markets.enabled()
    per = hour_buckets(rows, {m.live_book for m in enabled})
    return {"tz": "UTC", "archives": len(sig),
            "chains": [{"chain": m.chain, "name": m.name, "unit": m.unit, **per.get(m.chain, {"all": _empty_hours(), "live": _empty_hours()})} for m in enabled]}

