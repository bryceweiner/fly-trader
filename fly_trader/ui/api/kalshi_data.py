"""Kalshi data: the live feed, the history corpus (dataset seed, candles, feature builds) and the catalogue's coverage."""
from __future__ import annotations

import glob
from datetime import datetime, timezone

from ... import config
from ...db.queries import q, q1
from .. import runtime
from ..cache import ttl_cache
from ..state import STALE_FEED_S, running, setting
from .kalshi_overview import kalshi_topic


@ttl_cache(60)
def feature_parts() -> dict:
    from ...kalshi.mature import part_current
    files = sorted(glob.glob(str(config.KALSHI_DIR / "features" / "*" / "part-*.parquet")))
    days = sorted({f.split("/")[-2] for f in files})
    return {"n": len(files), "days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
            "current": sum(1 for f in files if part_current(f))}


@ttl_cache(30)
def quoted_per_minute() -> list:
    """Markets with a minute row, per minute over the last three hours (the feed's pulse for the chart)."""
    return [[r["ts"], int(r["n"])] for r in q("SELECT ts, count(*) AS n FROM kalshi_minutes WHERE ts > now() - interval '3 hours' GROUP BY ts ORDER BY ts")]


@ttl_cache(60)
def feature_days() -> list:
    """Feature rows built per day (kalshi_days), the last 90 days built."""
    return [{"day": r["day"], "markets": int(r["markets"] or 0), "rows": int(r["rows"] or 0)} for r in q("SELECT day, markets, rows FROM kalshi_days ORDER BY day DESC LIMIT 90")][::-1]


@ttl_cache(60)
def corpus_counts() -> dict:
    c = {r["status"]: int(r["n"]) for r in q("SELECT status, count(*) AS n FROM kalshi_corpus GROUP BY status")}
    m = q1("SELECT count(*) AS n, count(*) FILTER (WHERE result IN ('yes','no')) AS settled, count(*) FILTER (WHERE result IS NULL AND close_time > now()) AS open, min(close_time) AS first FROM kalshi_markets") or {}
    seed = q1("SELECT value FROM ui_settings WHERE key = 'kalshi_dataset_seed'")
    cats = q("""SELECT COALESCE(s.category, e.category, 'unknown') AS category, count(*) AS markets FROM kalshi_markets m LEFT JOIN kalshi_events e USING (event_ticker)
                LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker WHERE m.result IN ('yes','no') GROUP BY 1 ORDER BY 2 DESC LIMIT 15""")
    return {"corpus": c, "markets": m, "seed": seed["value"] if seed else None, "categories": cats}


def _badge(alive: bool, ok: bool = True) -> dict:
    return {"text": "running" if alive and ok else "stale" if alive else "stopped", "color": "green" if alive and ok else "orange" if alive else "gray"}


@kalshi_topic("kalshi_data.pipelines", every=10)
def pipelines(params: dict) -> dict:
    ws = runtime.sup().status(); now = datetime.now(timezone.utc)
    ks, ks_at = setting("kalshi_stream_status"); hs, hs_at = setting("kalshi_history_status")
    ft = ks.get("flushed_through"); age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None
    # running(): a CLI feed or history process counts, as in the status widgets
    feed = {"badge": _badge(running(ws, "kalshi_stream"), age is not None and age < STALE_FEED_S), "events_per_s": float(ks.get("events_per_s") or 0), "age": age,
            "tickers_quoted": int(ks.get("tickers_quoted") or 0), "settled": int(ks.get("settled") or 0), "reconnects": int(ks.get("reconnects") or 0),
            "private": bool(ks.get("private")), "dropped_late": int(ks.get("dropped_late") or 0), "updated_at": ks_at, "per_minute": quoted_per_minute()}
    cc = corpus_counts(); c = cc["corpus"]; total = sum(c.values()); filled = c.get("done", 0) + c.get("built", 0)
    history = {"badge": _badge(running(ws, "kalshi_history")), "settled_markets": int((cc["markets"] or {}).get("settled") or 0), "filled": filled, "total": total,
               "built": c.get("built", 0), "errors": c.get("error", 0), "progress": min(1.0, filled / total) if total else None, "start": config.KALSHI_HISTORY_START,
               "stage": hs.get("stage") or None, "eta_h": float(hs["eta_h"]) if hs.get("eta_h") else None, "seed_done": bool(cc["seed"]), "updated_at": hs_at}
    corpus = [{"status": k, "n": v} for k, v in sorted(c.items(), key=lambda kv: -kv[1])]
    return {"feed": feed, "history": history, "corpus": corpus, "features": {**feature_parts(), "per_day": feature_days()}, "categories": cc["categories"]}
