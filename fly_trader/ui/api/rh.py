"""Robinhood Chain: the Pons indexer, the quote assets' marks, the per-chain verdict and seat, and the ETH wallet's
intents, transactions, legs and base lots. Rails (circuit 3) and limits live on the Safety page.

Every topic answers ``{"enabled": False, "message": ...}`` while RH_ENABLED is off (the page shows that message)."""
from __future__ import annotations

from datetime import datetime, timezone

from ... import config, markets
from ...db.queries import q, q1
from ..hub import TopicError, topic
from ..state import labels, setting, system_state

RH = markets.RH
OFF = "Robinhood Chain is off (RH_ENABLED=0)."
VIEWS = ("Intents", "Transactions", "Legs")
MARK_STALE_S = 600            # a quote asset's ETH mark older than this blocks entries quoted in it


def _off() -> dict | None:
    return None if config.RH_ENABLED else {"enabled": False, "message": OFF}


def explorer(kind: str, value: str = "") -> str:
    """The RH block explorer's page of an address or a transaction (ops/wallets.py; testnet or mainnet by RH_TESTNET)."""
    from ...ops.wallets import explorer as ex
    return ex("rh", kind, value)


@topic("rh.indexer", every=10)
def indexer(params: dict) -> dict:
    """The indexer's own status row (fast); its table counts and hourly activity are rh.index_activity (slow counts)."""
    if off := _off():
        return off
    s, at = setting("rh_stream_status")
    if not s:
        return {"enabled": True, "status": None}
    mt = s.get("minutes_through")
    return {"enabled": True, "updated_at": at,
            "status": {"mode": str(s.get("mode") or "—"), "behind_blocks": int(s.get("behind_blocks") or 0), "head": s.get("head"), "index_through": s.get("index_through"),
                       "minutes_through": datetime.fromtimestamp(int(mt), timezone.utc).strftime("%b %d %H:%M") if mt else None, "error": s.get("error") or None}}


@topic("rh.index_activity", every=10)
def index_activity(params: dict) -> dict:
    """Launches, graduations, swaps and minutes indexed (whole-table counts: seconds on the live database, so apart from the
    status), and the last 24 hours per hour."""
    if off := _off():
        return off
    counts = {r["t"]: int(r["n"]) for r in q("SELECT 'launches' AS t, count(*) AS n FROM rh_tokens UNION ALL SELECT 'graduated', count(*) FROM rh_tokens WHERE status = 'graduated' "
                                             "UNION ALL SELECT 'swaps', count(*) FROM rh_swaps UNION ALL SELECT 'minutes', count(*) FROM rh_minutes")}
    hours = q("""SELECT h.h AS hour, COALESCE(s.n, 0) AS swaps, COALESCE(m.n, 0) AS minutes, COALESCE(l.n, 0) AS launches
                 FROM generate_series(date_trunc('hour', now()) - interval '23 hours', date_trunc('hour', now()), interval '1 hour') AS h(h)
                 LEFT JOIN (SELECT date_trunc('hour', ts) AS h, count(*) AS n FROM rh_swaps WHERE ts > date_trunc('hour', now()) - interval '23 hours' GROUP BY 1) s ON s.h = h.h
                 LEFT JOIN (SELECT date_trunc('hour', ts) AS h, count(*) AS n FROM rh_minutes WHERE ts > date_trunc('hour', now()) - interval '23 hours' GROUP BY 1) m ON m.h = h.h
                 LEFT JOIN (SELECT date_trunc('hour', launch_ts) AS h, count(*) AS n FROM rh_tokens WHERE launch_ts > date_trunc('hour', now()) - interval '23 hours' GROUP BY 1) l
                   ON l.h = h.h ORDER BY 1""")
    return {"enabled": True, "counts": counts, "hours": hours}


@topic("rh.assets", every=30)
def assets(params: dict) -> dict:
    if off := _off():
        return off
    rows = q("""SELECT a.symbol, a.class, a.decimals, a.asset, p.ts, p.price_eth, p.stale FROM rh_assets a
                LEFT JOIN LATERAL (SELECT ts, price_eth, stale FROM rh_base_prices b WHERE b.asset = a.asset ORDER BY ts DESC LIMIT 1) p ON true
                WHERE a.approved ORDER BY a.class, a.symbol""")
    now = datetime.now(timezone.utc)
    out = [{"asset": r["symbol"] or r["asset"][:10], "address": r["asset"], "class": r["class"], "eth_per_unit": 1.0 if r["class"] == "eth" else r["price_eth"],
            "native": r["class"] == "eth", "marked_at": r["ts"],
            "entries": "open" if r["class"] == "eth" or (r["ts"] and not r["stale"] and (now - r["ts"]).total_seconds() < MARK_STALE_S) else "blocked: mark stale"}
           for r in rows]
    return {"enabled": True, "rows": out}


@topic("rh.verdict", every=30)
def verdict(params: dict) -> dict:
    """The combined replay's verdict per chain and whether the RH fly holds the seat (and trades real ETH)."""
    if off := _off():
        return off
    v, v_at = setting("fly_replay"); ho, _ = setting(RH.handover_key); s = system_state(RH)
    chains = []
    for c, r in ((v or {}).get("chains") or {}).items():
        ev = r.get("evaluation") or {}
        chains.append({"chain": c, "name": markets.MARKETS[c].name if c in markets.MARKETS else str(c), "passed": bool(r.get("passed")),
                       "evaluated": bool(ev), "n": int(ev.get("n") or 0), "mean": ev.get("mean"), "reason": r.get("reason")})
    return {"enabled": True, "chains": chains, "replay_at": v_at, "handover": {"at": ho.get("at")} if ho else None, "live_money": bool(s["live_money"])}


@topic("rh.wallet", every=15)
def wallet(params: dict) -> dict:
    if off := _off():
        return off
    wm = q1("SELECT ts, native_wei, consistent FROM rh_wallet_marks ORDER BY ts DESC LIMIT 1") or {}
    w = q1("SELECT wealth, exposure, n_open, drawdown FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (RH.live_book,)) or {}
    spark = q("SELECT ts, wealth FROM (SELECT ts, wealth FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 180) w ORDER BY ts", (RH.live_book,))
    lots = q("""SELECT l.asset, a.symbol, l.qty_raw::text AS qty_raw, l.basis_eth, l.source, l.position_id, l.opened_at FROM rh_base_lots l
                LEFT JOIN rh_assets a ON a.asset = l.asset WHERE l.status = 'open' ORDER BY l.id""")
    addr = config.RH_BOT_ADDRESS or None
    return {"enabled": True, "book": RH.live_book, "address": addr, "address_url": explorer("address", addr) if addr else None,
            "reconciled": bool(wm), "marked_at": wm.get("ts"), "native_eth": int(wm["native_wei"]) / 1e18 if wm.get("native_wei") is not None else None,
            "wealth": w.get("wealth"), "n_open": int(w.get("n_open") or 0), "consistent": bool(wm.get("consistent", True)), "lots": lots,
            "spark": [float(r["wealth"]) for r in spark if r["wealth"] is not None]}


@topic("rh.activity", every=15)
def activity(params: dict) -> dict:
    """The live RH book's last 100 intents, transactions or legs (``view``)."""
    if off := _off():
        return off
    view = params.get("view") or "Intents"
    if view not in VIEWS:
        raise TopicError(f"view must be one of {', '.join(VIEWS)}")
    if view == "Intents":
        rows = q("SELECT id, created_at, kind, token, route, state, attempts, error FROM rh_intents ORDER BY id DESC LIMIT 100")
        names = labels([r["token"] for r in rows if r["token"]])
        for r in rows:
            r["mint"] = r["token"]; r["token"] = names.get(r["token"], (r["token"] or "")[:10])
    elif view == "Transactions":
        rows = q("SELECT id, created_at, kind, nonce, status, gas_used, fee_wei::float / 1e18 AS fee_eth, router, hash FROM rh_txs ORDER BY id DESC LIMIT 100")
    else:
        rows = q("SELECT id, ts, leg_no, asset_in, asset_out, amount_in_raw::text AS amount_in, amount_out_raw::text AS amount_out, eth_in, eth_out, verified_by "
                 "FROM rh_legs ORDER BY id DESC LIMIT 100")
    return {"enabled": True, "view": view, "rows": rows, "tx_url": explorer("tx")}
