"""The live fly ticker: the newest things the flies did, across chains — buys, sells and picks not taken on the fly books,
the bot wallets' live trades and fills, and the visual fly's Kalshi entries, posts, fills and settlements. The Overview
card lists them; every page toasts the new live-money ones (console/src/fly/Ticker.tsx)."""
from __future__ import annotations

from datetime import datetime

from ... import markets
from ...db.queries import q
from ..hub import topic
from ..state import KALSHI_BOOKS, kalshi_enabled, labels

N = 60
FLY_KINDS = ["fly_enter", "fly_exit", "fly_rh_enter", "fly_rh_exit", "blocked"]
KALSHI_KINDS = ["kalshi_taker_enter", "kalshi_maker_post", "kalshi_maker_fill", "kalshi_settle", "kalshi_exit"]
KALSHI_ACTION = {"kalshi_taker_enter": "bought", "kalshi_maker_post": "posted", "kalshi_maker_fill": "filled", "kalshi_maker_cancel": "cancelled",
                 "kalshi_settle": "settled", "kalshi_exit": "sold"}


RAIL_TEXT = {"held": "already holding it", "filtered": "a filter said no", "paper_fill": "the simulated fill failed"}


def action_of(kind: str) -> str:
    """A decision kind as a verb: bought / sold / not taken, or the Kalshi arm's own (posted, filled, settled…)."""
    if kind in KALSHI_ACTION:
        return KALSHI_ACTION[kind]
    return "bought" if kind.endswith("_enter") or kind == "enter" else "sold" if kind.endswith("_exit") or kind == "exit" else "not taken" if kind == "blocked" else kind


def actor_of(kind: str, book: str | None) -> str:
    """Who decided: the live mirror (a live book), the fly, the selector, or the visual fly."""
    b = book or ""
    if b.startswith("live"):
        return "live"
    if kind.startswith("kalshi") or "kalshi" in b:
        return "visual fly"
    if kind.startswith("selector") or b.endswith("selector"):
        return "selector"
    return "fly"


def _ms(ts: datetime) -> float:
    return ts.timestamp()


def _num(v, nd: int) -> str:
    return "—" if v is None else f"{float(v):,.{nd}f}"


def memecoin_items(fly_books: list[str], live_books: list[str]) -> list[dict]:
    books = fly_books + live_books; out = []
    dec = q("""SELECT id, ts, kind, mint, m_hat, size_sol, rail, reason, COALESCE(detail->>'book', book_targets[1]) AS book, detail->>'strategy' AS strategy
               FROM decisions WHERE kind = ANY(%s) AND (detail->>'book' = ANY(%s) OR book_targets && %s::text[]) ORDER BY id DESC LIMIT %s""",
            (FLY_KINDS, books, books, N))
    fills = q("SELECT id, ts, book, side, mint, sol_delta_lamports, price_sol, fee_lamports, verified_by, signature FROM fills WHERE book = ANY(%s) ORDER BY id DESC LIMIT %s",
              (books, N))
    names = labels([r["mint"] for r in dec] + [r["mint"] for r in fills])
    for r in dec:
        m = markets.for_book(r["book"] or ""); live = (r["book"] or "").startswith("live"); act = action_of(r["kind"])
        size = float(r["size_sol"]) if r["size_sol"] is not None else None
        if act == "bought":
            detail = f"{_num(size, 4)} {m.unit}" + (f" — {r['reason']}" if r["reason"] else "")
        elif act == "sold":
            detail = f"cost {_num(size, 4)} {m.unit}" + (f" — {r['reason']}" if r["reason"] else "")
        else:
            why = RAIL_TEXT.get(r["rail"] or "", r["rail"] or "")
            detail = why + (f": {r['reason']}" if r["reason"] and "pick not taken" not in r["reason"] else "")
            if r["m_hat"] is not None:
                detail += f" (score {float(r['m_hat']):.3f})"
        who = "Live" if live else "Fly"
        out.append({"id": f"d:{r['id']}", "ts": r["ts"], "chain": m.chain, "book": r["book"], "kind": r["kind"], "action": act,
                    "side": "buy" if act == "bought" else "sell" if act == "sold" else None, "mint": r["mint"], "ticker": None,
                    "token": names.get(r["mint"], (r["mint"] or "")[:6]), "size": size, "unit": m.unit, "price": None, "live": live,
                    "strategy": r["strategy"], "detail": detail.strip(),
                    "text": f"{who} {act} {names.get(r['mint'], r['mint'])} on {m.name}: {detail.strip()}"})
    for r in fills:
        m = markets.for_book(r["book"]); live = r["book"].startswith("live"); amt = abs((r["sol_delta_lamports"] or 0) / 1e9)
        src = "simulated" if r["verified_by"] == "model" else r["verified_by"] or ""
        detail = f"{'bought' if r['side'] == 'buy' else 'sold'} for {_num(amt, 4)} {m.unit}" + (f" at {float(r['price_sol']):.3e}" if r["price_sol"] else "") + (f" · {src}" if src else "")
        out.append({"id": f"f:{r['id']}", "ts": r["ts"], "chain": m.chain, "book": r["book"], "kind": "fill", "action": "fill", "side": r["side"], "mint": r["mint"],
                    "ticker": None, "token": names.get(r["mint"], (r["mint"] or "")[:6]), "size": amt, "unit": m.unit, "price": r["price_sol"], "live": live,
                    "strategy": None, "detail": detail, "signature": r["signature"],
                    "text": f"{'LIVE fill' if live else 'Paper fill'} on {m.name}: {names.get(r['mint'], r['mint'])} {detail}"})
    return out


def rh_live_items(live_book: str) -> list[dict]:
    """The Robinhood Chain live book's swaps (rh_txs through their intents): its trades leave no ``fills`` rows."""
    rows = q("""SELECT t.id, COALESCE(t.applied_at, t.created_at) AS ts, t.status, t.hash, i.kind, i.token, i.book FROM rh_txs t JOIN rh_intents i ON i.id = t.intent_id
                WHERE t.kind = 'swap' AND i.book = %s ORDER BY t.id DESC LIMIT %s""", (live_book, N))
    names = labels([r["token"] for r in rows]); out = []
    for r in rows:
        side = "buy" if r["kind"] == "buy" else "sell"
        detail = f"{r['kind']} swap {r['status'].replace('_', ' ')}"
        out.append({"id": f"r:{r['id']}", "ts": r["ts"], "chain": "rh", "book": r["book"], "kind": "rh_swap", "action": "fill", "side": side, "mint": r["token"],
                    "ticker": None, "token": names.get(r["token"], (r["token"] or "")[:6]), "size": None, "unit": "ETH", "price": None, "live": True,
                    "strategy": None, "detail": detail, "signature": r["hash"], "text": f"LIVE on Robinhood Chain: {names.get(r['token'], r['token'])} {detail}"})
    return out


def kalshi_items() -> list[dict]:
    dec = q("""SELECT id, ts, kind, mint AS ticker, pool AS side, m_hat, size_sol, rail, reason, detail->>'book' AS book FROM decisions
               WHERE kind = ANY(%s) AND detail->>'book' = ANY(%s) ORDER BY id DESC LIMIT %s""", (KALSHI_KINDS, list(KALSHI_BOOKS), N))
    fills = q("SELECT trade_id, ts, book, ticker, side, action, price_cents, count, fee_cents FROM kalshi_fills ORDER BY ts DESC LIMIT %s", (N,))
    tickers = list({r["ticker"] for r in dec} | {r["ticker"] for r in fills})
    titles = {r["ticker"]: r["title"] for r in q("SELECT ticker, title FROM kalshi_markets WHERE ticker = ANY(%s)", (tickers,))} if tickers else {}
    out = []
    for r in dec:
        live = (r["book"] or "").startswith("live"); act = action_of(r["kind"]); usd = float(r["size_sol"]) if r["size_sol"] is not None else None
        detail = f"{str(r['side'] or '').upper()} ${_num(usd, 2)}" + (f" · edge {float(r['m_hat']):+.3f}" if r["m_hat"] is not None else "") + (f" — {r['reason']}" if r["reason"] else "")
        out.append({"id": f"d:{r['id']}", "ts": r["ts"], "chain": "kalshi", "book": r["book"], "kind": r["kind"], "action": act, "side": r["side"], "mint": None,
                    "ticker": r["ticker"], "token": titles.get(r["ticker"]) or r["ticker"], "size": usd, "unit": "USD", "price": None, "live": live,
                    "strategy": None, "detail": detail, "text": f"{'Live' if live else 'Visual fly'} {act} {r['ticker']}: {detail}"})
    for r in fills:                              # exchange fills: only the live subaccount trades on the exchange
        detail = f"{r['action'] or 'buy'} {_num(r['count'], 0)} {str(r['side']).upper()} at {_num(r['price_cents'], 0)}¢"
        out.append({"id": f"k:{r['trade_id']}", "ts": r["ts"], "chain": "kalshi", "book": r["book"] or "live_kalshi", "kind": "kalshi_fill", "action": "fill",
                    "side": r["side"], "mint": None, "ticker": r["ticker"], "token": titles.get(r["ticker"]) or r["ticker"],
                    "size": float(r["price_cents"]) * float(r["count"]) / 100 if r["price_cents"] is not None and r["count"] is not None else None, "unit": "USD",
                    "price": r["price_cents"], "live": True, "strategy": None, "detail": detail, "text": f"LIVE Kalshi fill {r['ticker']}: {detail}"})
    return out


@topic("ticker.recent", every=2)
def recent(params: dict) -> dict:
    """The newest ``N`` fly events, newest first; ids are stable ("d:<decision>", "f:<fill>", "r:<rh tx>", "k:<kalshi trade>")."""
    ms = markets.enabled()
    items = memecoin_items([m.fly_book for m in ms], [m.live_book for m in ms])
    if any(m.chain == "rh" for m in ms):
        items += rh_live_items(markets.RH.live_book)
    if kalshi_enabled():
        items += kalshi_items()
    items.sort(key=lambda r: (_ms(r["ts"]), r["id"]), reverse=True)
    return {"items": items[:N], "kalshi": kalshi_enabled(), "chains": [m.chain for m in ms] + (["kalshi"] if kalshi_enabled() else [])}
