"""The fly decision inspector: the drawer every token, Kalshi market and search result opens (``?inspect=`` in the URL).

For a memecoin token: every minute the fly scored it (``fly_scored``: its score, its teacher's score, its buy line, the
teacher's own decision and the realized label once the row's hold ran out), what the books did with it (decisions,
positions, fills on every book of its chain), and the fly-wide KC→MBON pathways its mushroom body has learned. For a
Kalshi market: the market, the visual fly's scores (``kalshi_fly_scored``) and its decisions, orders, positions and fills.
"""
from __future__ import annotations

import re

from ... import config, markets
from ...db.queries import q, q1
from ..hub import TopicError, topic
from ..state import KALSHI_BOOKS, ago, kalshi_enabled, labels, setting
from . import brain
from .ticker import action_of, actor_of

MINT_RE = re.compile(r"^[A-Za-z0-9]{20,64}$")         # base58 mints (32–44 chars) and 0x addresses (42)
TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
SERIES_ROWS = 720                                      # newest scored rows drawn (a strategy's trigger fires at most once a minute)
TOP_PATHWAYS = 15
STATE_HELP = {"pending": "its hold has not run out yet", "resolved": "its realized return is known", "unknown": "a bar was missing or the exit price jumped 50×, so it taught nothing",
              "dropped": "a new bootstrap replaced the one that scored it before it resolved"}


def _token_params(params: dict) -> tuple[str, str]:
    chain = params.get("chain") or "sol"
    if chain not in markets.MARKETS:
        raise TopicError(f"chain must be one of {list(markets.MARKETS)}")
    mint = str(params.get("mint") or "").strip()
    if not MINT_RE.match(mint):
        raise TopicError("mint must be a token address")
    return chain, (mint.lower() if chain == "rh" else mint)


def _ticker_param(params: dict) -> str:
    t = str(params.get("ticker") or "").strip()
    if not TICKER_RE.match(t):
        raise TopicError("ticker must be a Kalshi market ticker")
    return t


def _f(v) -> float | None:
    return None if v is None else float(v)


def _compact(rows: list[dict], keys: tuple) -> list[dict]:
    """The chart's rows: only what it draws, scores to six decimals (stored as float4, their tails are noise)."""
    return [{k: round(r[k], 6) if isinstance(r[k], float) else r[k] for k in keys} for r in rows]


def explorer(chain: str, kind: str, value: str) -> str:
    from ...ops.wallets import explorer as ex
    return ex(chain, kind, value)


# ---------------------------------------------------------------- the memecoin fly
def _pathways(mode: str) -> dict:
    """The fly-wide top learned KC→MBON pathways (Overview's 3D view draws the same), with each neuron's cell type."""
    fly, _ = setting(markets.SOL.fly_status_key)          # system_state()["fly"] without the rest of the system state: one brain for both chains
    out = {"mode": mode, "items": [], "note": None, "changed": 0, "reference": None, "bootstrap": fly.get("bootstrap"), "running": brain._running(fly),
           "drift": fly.get("drift_by_strategy") or {}}
    if not out["running"]:
        out["note"] = "The fly is not running, so no learned pathway is read."
        return out
    try:
        meta = brain.geometry()
        p = brain.pathways(mode, fly, meta)
    except Exception as e:                                 # no annotations or connectome on this install, a state file mid-write
        out["note"] = f"The learned pathways cannot be read here: {type(e).__name__}: {e}"
        return out
    names, idx = (meta.get("cell_type") or {}).get("names") or [], (meta.get("cell_type") or {}).get("index") or []
    ct = lambda i: names[idx[i]] if 0 <= i < len(idx) and idx[i] < len(names) else None   # noqa: E731
    out.update(note=p.get("note"), changed=p.get("changed") or 0, reference=p.get("reference"), n_pairs=brain.N_PAIRS,
               items=[{**it, "kc_type": ct(it["kc"]), "mbon_type": ct(it["mbon"])} for it in (p.get("items") or [])[:TOP_PATHWAYS]])
    return out


def _explain_token(latest: list[dict], chain: str) -> str:
    """Score now vs line, for the newest scored minute: what each number is and what the fly would do with it."""
    if not latest:
        return ("The fly has not scored this token in the last 35 days on this chain. It scores a token only in minutes where it is "
                "eligible (liquidity, volume, age) and some strategy's trigger fires.")
    teacher = config.FLY_SCORE == "teacher"; gated = config.FLY_TEACHER_GATE; veto = config.FLY_LEARNED_VETO
    t0 = latest[0]["ts"]
    out = [f"**Score now vs line** — last scored {ago(t0)} ({t0.strftime('%d %b %H:%M')} UTC) on {markets.for_chain(chain).name}."]
    for r in latest:
        s, f, line = _f(r["score"]), _f(r["frozen_score"]), _f(r["line"])
        if s is None or line is None:
            continue
        corr = s - f if f is not None else None
        head = f"**{r['strategy']}** ({r['hold_min'] or '?'} min hold): the fly scored it **{s:.4f}**"
        if teacher and f is not None:
            head += f" = its teacher's (the selector's) score {f:.4f} {corr:+.4f} learned by its mushroom body"
        elif f is not None:
            head += f" (its frozen bootstrap would score {f:.4f})"
        side = "above" if s >= line else "below"
        head += f"; its own line is **{line:.4f}**, so it is {side} the line by {s - line:+.4f}."
        ta = r.get("teacher_allow")
        head += (" Its teacher would have **bought** this row (its trigger, line and every filter passed)." if ta
                 else " Its teacher would **not** have bought this row (below its line or a filter said no)." if ta is False
                 else " The teacher's decision was not recorded for this row.")
        if gated:
            if ta and corr is not None and corr < -veto:
                head += f" Under the teacher gate the fly would **veto** it: its learning moved the score {corr:+.4f}, more than {veto:g} below the teacher's."
            elif ta:
                head += (f" Under the teacher gate the fly **buys** what its teacher buys, sized as the selector sizes it, unless its learning moved the "
                         f"score more than {veto:g} below the teacher's (here {corr:+.4f}), and the book has room.") if corr is not None else ""
            else:
                head += " Under the teacher gate the fly **does not buy** a row its teacher rejects, whatever its own score."
        else:
            head += f" Ungated, the fly buys at or above its own line: it would {'buy' if s >= line else 'not buy'} (if the book has room)."
        st = r.get("state") or "pending"
        if r.get("label") is not None:
            head += f" Realized: **{float(r['label']) * 100:+.2f}%** over the hold."
        else:
            head += f" Outcome {st}: {STATE_HELP.get(st, st)}."
        out.append(head)
    out.append(("Under the gate the fly's own line does not decide a buy (the selector's decision does); the line is recalibrated every day so the fly "
                "would take as many trades as its teacher (rule match), and its rollback checks judge its learning at it." if gated else
                "The line is recalibrated every day from the last week's resolved scores.")
               + " A realized label is the row's net return: entry at the next traded minute's open, exit at the last close within the hold, the exit cost at both ends — "
                 "exactly how the training data labels a row.")
    return "\n\n".join(out)


def _minute_bar(chain: str, mint: str, at) -> dict | None:
    """The token's minute at ``at`` (the newest scored minute; None: its newest traded minute): its close and its volume
    (buys + sells, in its chain's unit), with the close five minutes earlier and the mean volume of the five minutes before
    it (a minute without trades counts as none) -- the inspector's up/down arrows."""
    tbl, vol = ("rh_minutes", "COALESCE(buy_eth, 0) + COALESCE(sell_eth, 0)") if chain == "rh" else ("pump_minutes", "COALESCE(buy_sol, 0) + COALESCE(sell_sol, 0)")
    cur = q1(f"SELECT ts, close, {vol} AS volume FROM {tbl} WHERE mint = %s{' AND ts <= %s' if at else ''} ORDER BY ts DESC LIMIT 1", (mint, at) if at else (mint,))
    if not cur:
        return None
    then = q1(f"SELECT close FROM {tbl} WHERE mint = %s AND ts <= %s - interval '5 minutes' AND close IS NOT NULL ORDER BY ts DESC LIMIT 1", (mint, cur["ts"]))
    before = q1(f"SELECT sum({vol}) AS v FROM {tbl} WHERE mint = %s AND ts >= %s - interval '5 minutes' AND ts < %s", (mint, cur["ts"], cur["ts"]))
    return {"ts": cur["ts"], "price": _f(cur["close"]), "volume": _f(cur["volume"]), "price_5m": _f(then["close"]) if then else None,
            "volume_5m": (_f(before["v"]) or 0.0) / 5.0}


@topic("inspector.token", every=10)
def token(params: dict) -> dict:
    """One memecoin token through the fly's eyes and the books' ledgers."""
    chain, mint = _token_params(params); m = markets.for_chain(chain); books = list(m.books)
    mode = params.get("mode") or "bootstrap"
    if mode not in brain.MODES:
        raise TopicError(f"mode must be one of {list(brain.MODES)}")
    meta = q1("""SELECT COALESCE(NULLIF(cm.symbol, ''), NULLIF(t.symbol, '')) AS symbol, COALESCE(NULLIF(cm.name, ''), NULLIF(t.name, '')) AS name,
                        COALESCE(cm.graduated_at, t.graduated_at) AS graduated_at, cm.creator
                 FROM (SELECT %s::text AS mint) x LEFT JOIN corpus_meta cm ON cm.mint = x.mint LEFT JOIN tokens t ON t.mint = x.mint""", (mint,)) or {}
    if chain == "rh" and not meta.get("graduated_at"):
        rt = q1("SELECT NULLIF(symbol, '') AS symbol, NULLIF(name, '') AS name, graduated_at, creator FROM rh_tokens WHERE token = %s", (mint,))
        meta = {k: v for k, v in (rt or {}).items()}
    series = q("""SELECT ts, strategy, hold_min, score, frozen_score, line, frozen_line, label, state, teacher_allow, bootstrap_id
                  FROM fly_scored WHERE mint = %s AND chain = %s ORDER BY ts DESC LIMIT %s""", (mint, chain, SERIES_ROWS))
    series.reverse()
    stats = q("""SELECT strategy, count(*) AS n, count(*) FILTER (WHERE state = 'pending') AS pending, count(label) AS resolved,
                        count(*) FILTER (WHERE score >= line) AS cleared, count(*) FILTER (WHERE frozen_score >= frozen_line) AS frozen_cleared,
                        count(*) FILTER (WHERE teacher_allow) AS teacher_allowed, count(teacher_allow) AS teacher_known,
                        avg(label) FILTER (WHERE score >= line) AS label_cleared, count(label) FILTER (WHERE score >= line) AS n_label_cleared,
                        avg(label) FILTER (WHERE score < line) AS label_not, count(label) FILTER (WHERE score < line) AS n_label_not,
                        avg(label) FILTER (WHERE teacher_allow) AS label_allowed, count(label) FILTER (WHERE teacher_allow) AS n_label_allowed,
                        avg(label) AS label_all, avg(score - frozen_score) AS mean_correction, min(ts) AS first, max(ts) AS last
                 FROM fly_scored WHERE mint = %s AND chain = %s GROUP BY ROLLUP (strategy) ORDER BY strategy NULLS FIRST""", (mint, chain))
    total = next((s for s in stats if s["strategy"] is None), None) if stats else None
    last_ts = series[-1]["ts"] if series else None
    latest = [r for r in reversed(series) if r["ts"] == last_ts]
    series = _compact(series, ("ts", "strategy", "score", "frozen_score", "line", "label", "state", "teacher_allow"))
    decisions = q("""SELECT id, ts, kind, m_hat, size_sol, rail, reason, COALESCE(detail->>'book', book_targets[1]) AS book, detail->>'strategy' AS strategy,
                            (detail->>'threshold')::float AS threshold, (detail->>'correctness')::float AS correctness
                     FROM decisions WHERE mint = %s AND (detail->>'book' = ANY(%s) OR book_targets && %s::text[]) ORDER BY id DESC LIMIT 200""", (mint, books, books))
    for d in decisions:
        d["action"], d["actor"] = action_of(d["kind"]), actor_of(d["kind"], d["book"])
    positions = q("""SELECT id, book, status, opened_at, closed_at, strategy, hold_s, cost_sol, entry_price, last_mark_price, last_mark_ts, exit_price, realized_sol,
                            fees_sol, forced_exit_kind FROM positions WHERE mint = %s AND book = ANY(%s) ORDER BY opened_at DESC LIMIT 100""", (mint, books))
    for p in positions:
        cost = _f(p["cost_sol"]) or 0.0
        p["return"] = (_f(p["realized_sol"]) / cost if cost and p["realized_sol"] is not None else None) if p["status"] != "open" else \
            (float(p["last_mark_price"]) / float(p["entry_price"]) - 1 if p["last_mark_price"] and p["entry_price"] else None)
        p["held_min"] = ((p["closed_at"] or p["last_mark_ts"] or p["opened_at"]) - p["opened_at"]).total_seconds() / 60
    fills = q("""SELECT id, ts, book, side, sol_delta_lamports, price_sol, fee_lamports, verified_by, signature FROM fills
                 WHERE mint = %s AND book = ANY(%s) ORDER BY id DESC LIMIT 200""", (mint, books))
    for f in fills:
        f["amount"] = (f["sol_delta_lamports"] or 0) / 1e9; f["fee"] = (f["fee_lamports"] or 0) / 1e9
        f["source"] = "simulated" if f["verified_by"] == "model" else f["verified_by"]
        f["tx_url"] = explorer(chain, "tx", f["signature"]) if f["signature"] and f["book"].startswith("live") else None
    closed = [p for p in positions if p["status"] != "open"]
    return {"chain": chain, "chain_name": m.name, "unit": m.unit, "mint": mint, "label": labels([mint]).get(mint, mint[:6] + "…"),
            "symbol": meta.get("symbol"), "name": meta.get("name"), "graduated_at": meta.get("graduated_at"), "creator": meta.get("creator"),
            "explorer": explorer(chain, "address", mint), "books": {"selector": m.selector_book, "fly": m.fly_book, "live": m.live_book},
            "minute": _minute_bar(chain, mint, last_ts), "fly_score": config.FLY_SCORE, "teacher_gate": bool(config.FLY_TEACHER_GATE), "learned_veto": config.FLY_LEARNED_VETO,
            "series": series, "stats": stats, "total": total, "latest": latest, "explain": _explain_token(latest, chain),
            "decisions": decisions, "positions": positions, "fills": fills,
            "pnl": {"closed": len(closed), "realized": sum(_f(p["realized_sol"]) or 0.0 for p in closed), "open": sum(1 for p in positions if p["status"] == "open"),
                    "wins": sum(1 for p in closed if (_f(p["realized_sol"]) or 0) > 0)},
            "pathways": _pathways(mode)}


# ---------------------------------------------------------------- the visual (Kalshi) fly
def _explain_market(latest: list[dict]) -> str:
    if not latest:
        return ("The visual fly has not scored this market in the last 35 days. It scores a (market, side) only in minutes where the market is "
                "quoted, eligible (closes within the evidence window) and some strategy's trigger fires.")
    t0 = latest[0]["ts"]
    out = [f"**Score now vs line** — last scored {ago(t0)} ({t0.strftime('%d %b %H:%M')} UTC)."]
    for r in latest:
        s, f, line = _f(r["score"]), _f(r["frozen_score"]), _f(r["line"])
        if s is None or line is None:
            continue
        txt = (f"**{r['strategy']}**, {str(r['side']).upper()}: edge **{s:+.4f}** (its probability that this side wins minus the arm's fee-inclusive price)"
               + (f"; the frozen bootstrap's edge {f:+.4f}, so learning moved it {s - f:+.4f}" if f is not None else "")
               + f"; its edge line is **{line:+.4f}**, so it is {'above' if s >= line else 'below'} the line by {s - line:+.4f} and the fly would "
               + ("take it (if the book has room)." if s >= line else "pass."))
        txt += (f" Realized: **{float(r['label']) * 100:+.2f}%** per dollar." if r.get("label") is not None else f" Outcome {r.get('state') or 'pending'}: "
                + {"pending": "the market has not settled yet", "unknown": "never resolved (a maker bid that did not fill, or no settlement a week after close)"}.get(r.get("state") or "pending", r.get("state") or ""))
        out.append(txt)
    out.append("The taker label is the settlement return at the effective ask; the maker label the return of a bid one tick inside the ask, "
               "only when a later minute's ask reached it. Lines are recalibrated every day from the last week's settled scores.")
    return "\n\n".join(out)


@topic("inspector.market", every=10)
def market(params: dict) -> dict:
    """One Kalshi market through the visual fly's eyes and the Kalshi books' ledgers."""
    ticker = _ticker_param(params); books = list(KALSHI_BOOKS)
    mk = q1("""SELECT m.ticker, m.title, m.yes_sub_title, m.no_sub_title, m.status, m.open_time, m.close_time, m.result, m.settlement_ts, m.event_ticker,
                      e.title AS event_title, COALESCE(s.category, e.category) AS category, s.title AS series_title, e.series_ticker
               FROM kalshi_markets m LEFT JOIN kalshi_events e ON e.event_ticker = m.event_ticker LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker
               WHERE m.ticker = %s""", (ticker,))
    series = q("""SELECT ts, side, strategy, score, frozen_score, line, frozen_line, label, state, due_ts, resolved_at
                  FROM kalshi_fly_scored WHERE ticker = %s ORDER BY ts DESC LIMIT %s""", (ticker, SERIES_ROWS))
    series.reverse()
    stats = q("""SELECT strategy, count(*) AS n, count(*) FILTER (WHERE state = 'pending') AS pending, count(label) AS resolved,
                        count(*) FILTER (WHERE score >= line) AS cleared, avg(label) FILTER (WHERE score >= line) AS label_cleared,
                        count(label) FILTER (WHERE score >= line) AS n_label_cleared, avg(label) FILTER (WHERE score < line) AS label_not,
                        count(label) FILTER (WHERE score < line) AS n_label_not, avg(score - frozen_score) AS mean_correction, min(ts) AS first, max(ts) AS last
                 FROM kalshi_fly_scored WHERE ticker = %s GROUP BY ROLLUP (strategy) ORDER BY strategy NULLS FIRST""", (ticker,))
    last_ts = series[-1]["ts"] if series else None
    latest = [r for r in reversed(series) if r["ts"] == last_ts]
    series = _compact(series, ("ts", "side", "strategy", "score", "frozen_score", "line", "label", "state"))
    decisions = q("""SELECT id, ts, kind, pool AS side, m_hat, size_sol, rail, reason, detail->>'book' AS book FROM decisions
                     WHERE mint = %s AND detail->>'book' = ANY(%s) ORDER BY id DESC LIMIT 200""", (ticker, books))
    for d in decisions:
        d["action"], d["actor"] = action_of(d["kind"]), actor_of(d["kind"], d["book"])
    orders = q("""SELECT id, ts, book, status, side, action, price_cents, count, fill_count, tif, post_only, expiration_ts, strategy, order_id, error
                  FROM kalshi_orders WHERE ticker = %s ORDER BY id DESC LIMIT 100""", (ticker,))
    positions = q("""SELECT id, book, status, side, strategy, arm, contracts, cost_cents, fee_cents, avg_price_cents, opened_at, closed_at, result, payout_cents,
                            realized_cents, score, line, last_mark_cents FROM kalshi_positions WHERE ticker = %s ORDER BY opened_at DESC LIMIT 100""", (ticker,))
    for p in positions:
        base = (_f(p["cost_cents"]) or 0) + (_f(p["fee_cents"]) or 0)
        p["return"] = _f(p["realized_cents"]) / base if p["realized_cents"] is not None and base else None
    fills = q("""SELECT trade_id, ts, book, side, action, price_cents, count, fee_cents, is_taker, order_id FROM kalshi_fills
                 WHERE ticker = %s ORDER BY ts DESC LIMIT 100""", (ticker,))
    closed = [p for p in positions if p["status"] != "open"]
    return {"ticker": ticker, "enabled": kalshi_enabled(), "market": mk, "label": (mk or {}).get("title") or ticker, "books": KALSHI_BOOKS,
            "series": series, "stats": stats, "total": next((s for s in stats if s["strategy"] is None), None), "latest": latest, "explain": _explain_market(latest),
            "decisions": decisions, "orders": orders, "positions": positions, "fills": fills,
            "pnl": {"closed": len(closed), "realized_usd": sum(_f(p["realized_cents"]) or 0.0 for p in closed) / 100, "open": len(positions) - len(closed),
                    "wins": sum(1 for p in closed if (_f(p["realized_cents"]) or 0) > 0)}}
