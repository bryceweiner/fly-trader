"""Global search (the top bar): a token by symbol or address, a Kalshi market by ticker or title, a transaction by its
signature or hash. Every result names where it leads: the inspector for a token or market, a page for the rest.

The big tables have no text indexes (corpus_meta 127k rows, rh_tokens 700k, kalshi_markets millions), so each part is
bounded: symbols and addresses scan the small token tables; Kalshi matches an exact ticker, the markets the books touched,
and the markets the visual fly can trade (closing within the evidence window), whose titles are read once every five
minutes (a second's scan) and searched in memory."""
from __future__ import annotations

from datetime import datetime, timezone

import psycopg

from ... import config, markets
from ...db.queries import q, read_connection
from ..cache import ttl_cache
from ..hub import topic
from ..state import kalshi_enabled, labels

MIN_Q, MAX_Q, LIMIT = 2, 120, 20
CAP = {"token": 10, "market": 8, "tx": 8}      # per kind, so one kind never crowds the others out
KALSHI_TIMEOUT_MS = 8000


def _like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _route(to: str) -> dict:
    return {"kind": "route", "to": to}


def _token(chain: str, mint: str) -> dict:
    return {"kind": "token", "chain": chain, "mint": mint}


def _tokens(s: str, rh: bool) -> list[dict]:
    out = []; pat = f"%{_like(s)}%"; pre = f"{_like(s)}%"
    rows = q("""SELECT mint, sym, name FROM (
                  SELECT mint, NULLIF(symbol, '') AS sym, name, graduated_at FROM corpus_meta WHERE symbol ILIKE %s OR mint LIKE %s
                  UNION SELECT mint, NULLIF(symbol, '') AS sym, name, graduated_at FROM tokens WHERE symbol ILIKE %s OR mint LIKE %s) x
                ORDER BY (lower(sym) = lower(%s)) DESC NULLS LAST, (mint IN (SELECT mint FROM positions)) DESC, graduated_at DESC NULLS LAST LIMIT %s""",
             (pat, pre, pat, pre, s, CAP["token"] * 2))
    seen = set()
    for r in rows:
        if r["mint"] in seen:
            continue
        seen.add(r["mint"])
        out.append({"kind": "token", "label": r["sym"] or r["mint"][:6] + "…", "sub": f"Solana · {r['mint'][:4]}…{r['mint'][-4:]}" + (f" · {r['name']}" if r["name"] else ""),
                    "target": _token("sol", r["mint"])})
    if len(s) >= 32 and s not in seen and not s.lower().startswith("0x"):     # a full mint the metadata never saw (still scored, traded or held)
        out.insert(0, {"kind": "token", "label": labels([s]).get(s, s[:6] + "…"), "sub": f"Solana · {s[:4]}…{s[-4:]}", "target": _token("sol", s)})
    if rh and s.lower().startswith("0x"):
        lo = s.lower()
        for r in q("SELECT token, NULLIF(symbol, '') AS sym, name, status FROM rh_tokens WHERE token LIKE %s ORDER BY graduated_at DESC NULLS LAST LIMIT %s",
                   (f"{_like(lo)}%", CAP["token"])):
            out.append({"kind": "token", "label": r["sym"] or f"{r['token'][:6]}…", "sub": f"Robinhood Chain · {r['token'][:6]}…{r['token'][-4:]} · {r['status']}",
                        "target": _token("rh", r["token"])})
    return out[:CAP["token"]]


@ttl_cache(300, maxsize=1)
def _open_markets() -> list[tuple]:
    """(ticker, title, close) of every active market the visual fly may trade: closing within KALSHI_MAX_DAYS_TO_CLOSE."""
    with read_connection() as conn, conn.transaction():
        conn.execute(f"SET LOCAL statement_timeout = {KALSHI_TIMEOUT_MS}")
        rows = conn.execute("""SELECT ticker, title, close_time FROM kalshi_markets WHERE status = 'active'
                               AND close_time BETWEEN now() AND now() + make_interval(secs => %s) ORDER BY close_time""",
                            (config.KALSHI_MAX_DAYS_TO_CLOSE * 86400.0,)).fetchall()
    return [(r["ticker"], r["title"] or "", r["close_time"]) for r in rows]


def _kalshi(s: str) -> list[dict]:
    up, lo = s.upper(), s.lower(); out = []; seen = set()
    rows = q("""SELECT ticker, title, status FROM kalshi_markets WHERE ticker = %s
                UNION SELECT m.ticker, m.title, m.status FROM kalshi_markets m
                  WHERE m.ticker IN (SELECT ticker FROM kalshi_positions UNION SELECT ticker FROM kalshi_orders) AND (m.ticker LIKE %s OR m.title ILIKE %s)
                LIMIT %s""", (up, f"%{_like(up)}%", f"%{_like(s)}%", CAP["market"]))
    for r in rows:
        seen.add(r["ticker"])
        out.append({"kind": "market", "label": r["ticker"], "sub": f"{r['title'] or ''} · {r['status'] or ''}".strip(" ·"), "target": {"kind": "market", "ticker": r["ticker"]}})
    try:
        opened = _open_markets()
    except psycopg.errors.QueryCanceled:
        return out + [{"kind": "note", "label": "Kalshi titles unavailable", "sub": "Reading the open markets took too long; an exact ticker still works.", "target": None}]
    now = datetime.now(timezone.utc)
    for tk, title, close in opened:
        if len(out) >= CAP["market"]:
            break
        if tk in seen or (up not in tk and lo not in title.lower()):
            continue
        h = (close - now).total_seconds() / 3600 if close else None
        out.append({"kind": "market", "label": tk, "sub": f"{title} · closes in {h:.0f} h" if h is not None else title, "target": {"kind": "market", "ticker": tk}})
    return out


def _txs(s: str) -> list[dict]:
    pre = f"{_like(s)}%"; out = []
    fills = q("SELECT id, ts, book, side, mint, signature FROM fills WHERE signature LIKE %s ORDER BY id DESC LIMIT %s", (pre, CAP["tx"]))
    names = labels([r["mint"] for r in fills])
    for r in fills:
        out.append({"kind": "tx", "label": f"{r['signature'][:10]}…", "sub": f"fill · {r['book']} · {r['side']} {names.get(r['mint'], r['mint'][:6])}",
                    "target": _token(markets.for_book(r["book"]).chain, r["mint"])})
    for r in q("SELECT id, ts, kind, detail->>'signature' AS sig FROM wallet_events WHERE detail->>'signature' LIKE %s ORDER BY id DESC LIMIT %s", (pre, CAP["tx"])):
        out.append({"kind": "tx", "label": f"{r['sig'][:10]}…", "sub": f"Solana wallet · {r['kind']}", "target": _route("/safety")})
    lo = s.lower()
    if lo.startswith("0x"):
        for r in q("""SELECT t.hash, t.kind, t.status, p.mint, p.book FROM rh_txs t LEFT JOIN positions p ON p.id = t.position_id
                      WHERE t.hash LIKE %s ORDER BY t.id DESC LIMIT %s""", (f"{_like(lo)}%", CAP["tx"])):
            out.append({"kind": "tx", "label": f"{r['hash'][:10]}…", "sub": f"Robinhood Chain tx · {r['kind']} · {r['status']}" + (f" · {r['book']}" if r["book"] else ""),
                        "target": _token("rh", r["mint"]) if r["mint"] else _route("/rh")})
    for r in q("SELECT signature, direction, kind FROM vault_flows WHERE signature LIKE %s ORDER BY id DESC LIMIT %s", (pre, CAP["tx"])):
        out.append({"kind": "tx", "label": f"{r['signature'][:10]}…", "sub": f"vault flow · {r['direction']} · {r['kind']}", "target": _route("/vault")})
    return out[:CAP["tx"]]


@topic("search.query", every=60)
def query(params: dict) -> dict:
    """Up to ``LIMIT`` results for ``q``: tokens, then Kalshi markets (when Kalshi is set up), then transactions."""
    s = str(params.get("q") or "").strip()
    if len(s) < MIN_Q:
        return {"q": s, "results": [], "note": f"Type at least {MIN_Q} characters."}
    if len(s) > MAX_Q:
        return {"q": s[:MAX_Q], "results": [], "note": "That is longer than any token, ticker or signature."}
    results = _tokens(s, bool(config.RH_ENABLED))
    if kalshi_enabled():
        results += _kalshi(s)
    if len(s) >= 6:                                    # signatures are 64–88 characters: a prefix of six narrows them enough
        results += _txs(s)
    return {"q": s, "results": results[:LIMIT], "note": None if results else "Nothing matches."}
