"""Data pipelines: is each source current, and how far along are the historical builds — the Solana market feed, the
pumpapi.io history archive, the training feature parts, token metadata and the Jupiter token stats. One topic, as the
Streamlit page was one fragment: every panel refreshes together every 10 s — except the Jupiter coverage count (about 11 s on
the live database), which has its own minute-long topic so it never holds the rest back. The feature parts are cached for a minute."""
from __future__ import annotations

import glob
import threading
from collections import deque
from datetime import datetime, timezone

from ... import config
from ...db.queries import q, q1
from .. import runtime
from ..cache import ttl_cache
from ..hub import topic
from ..state import STALE_FEED_S, setting

FEED_MINUTES = 180            # the market feed chart: trades and tokens per minute over the last three hours
ARCHIVE_DAYS = 90             # the history archive chart: graduations per assembled day
_eps: deque = deque(maxlen=120)   # (status time, events/s) samples the feed reported, newest last: the Events / s sparkline
_eps_lock = threading.Lock()


def state_badge(alive: bool, ok: bool = True, optional: bool = False) -> dict:
    """running (green) / stale (orange: the worker is up but its output is old) / stopped (red; gray for an optional worker)."""
    return {"text": "running" if alive and ok else "stale" if alive else "stopped",
            "color": "green" if alive and ok else "orange" if alive else "gray" if optional else "red"}


def _eps_history(at, eps: float) -> list[float]:
    """The feed overwrites its status row every few seconds; keep the samples this console has seen (one per status update)."""
    with _eps_lock:
        if at is not None and (not _eps or _eps[-1][0] != at):
            _eps.append((at, eps))
        return [v for _, v in _eps]


@ttl_cache(60)
def feature_parts() -> dict:
    """The built feature days (one parquet part per day) and how many carry the live feature engine's current version:
    reading every part's schema is slow, so a minute's staleness is fine."""
    from ...market.features import FEATURE_VERSION
    from ...train.mature import part_version
    files = sorted(glob.glob(str(config.CORPUS_DIR / "features_mature" / "*" / "part.parquet")))
    days = [f.split("/")[-2] for f in files]
    return {"n": len(files), "first": days[0] if days else None, "last": days[-1] if days else None,
            "current": sum(1 for f in files if part_version(f) == FEATURE_VERSION)}


@ttl_cache(60)
def token_stats_coverage() -> dict:
    """Tokens the selector could trade in the last hour (pool above its gate) and how many have Jupiter stats from the
    last hour, plus when the stats history starts."""
    from ...train.decisions import MIN_RESQ_SOL
    c = q1("""SELECT count(*) AS universe, count(*) FILTER (WHERE EXISTS (SELECT 1 FROM token_stats s WHERE s.mint = u.mint AND s.ts > now() - interval '1 hour')) AS covered
              FROM (SELECT DISTINCT mint FROM pump_minutes WHERE ts > now() - interval '1 hour' AND resq_sol >= %s) u""", (MIN_RESQ_SOL,)) or {}
    first = q1("SELECT min(ts) AS first FROM token_stats") or {}
    return {"universe": int(c.get("universe") or 0), "covered": int(c.get("covered") or 0), "first": first.get("first")}


@topic("data.pipelines", every=10)
def pipelines(params: dict) -> dict:
    ws = runtime.sup().status(); now = datetime.now(timezone.utc)
    alive = lambda n: bool(ws.get(n, {}).get("alive"))      # noqa: E731
    ps, ps_at = setting("pumpstream_status"); rs, rs_at = setting("replay_status")
    ft = ps.get("flushed_through")
    age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None      # since the newest complete minute closed
    done, total = int(rs.get("hours_done") or 0), int(rs.get("hours_total") or 0)
    days = q1("SELECT count(*) AS n, max(day) AS last FROM replay_days") or {}
    m = q1("SELECT count(*) AS n, count(creator) AS creators, count(own_dd60) AS outcomes FROM corpus_meta") or {}
    cr = q1("SELECT count(*) AS n FROM pump_events WHERE action = 'create'") or {}
    eps = float(ps.get("events_per_s") or 0)
    per_min = q("SELECT ts, sum(n_buys + n_sells) AS trades, count(*) AS tokens FROM pump_minutes WHERE ts > now() - make_interval(mins => %s) GROUP BY ts ORDER BY ts",
                (FEED_MINUTES,))
    per_day = q("SELECT day, graduations, tokens_written FROM (SELECT * FROM replay_days ORDER BY day DESC LIMIT %s) d ORDER BY day", (ARCHIVE_DAYS,))
    return {
        "feed": {"badge": state_badge(alive("pumpstream"), age is not None and age < STALE_FEED_S), "events_per_s": eps, "eps_spark": _eps_history(ps_at, eps),
                 "age": age, "migrates": int(ps.get("migrates") or 0), "reconnects": int(ps.get("reconnects") or 0),
                 "dropped_band": int(ps.get("dropped_band") or 0), "dropped_late": int(ps.get("dropped_late") or 0), "updated_at": ps_at,
                 "per_minute": [[r["ts"], int(r["trades"] or 0), int(r["tokens"] or 0)] for r in per_min]},
        "archive": {"badge": state_badge(alive("replay")), "hours_done": done, "hours_total": total, "progress": min(1.0, done / total) if total else None,
                    "days": int(days.get("n") or 0), "newest_day": days.get("last"), "errors": int(rs.get("errors") or 0),
                    "replay_start": config.REPLAY_START, "last_hour": str(rs.get("last_hour") or "—")[:13],
                    "eta_h": float(rs["eta_h"]) if rs.get("eta_h") else None, "updated_at": rs_at,
                    "per_day": [[r["day"], int(r["graduations"] or 0), int(r["tokens_written"] or 0)] for r in per_day]},
        "features": feature_parts(),
        "meta": {"graduations": int(m.get("n") or 0), "creators": int(m.get("creators") or 0), "launches": int(cr.get("n") or 0)},
        "stats": {"badge": state_badge(alive("discover"), optional=True)},
    }


@topic("data.token_stats", every=60)
def token_stats(params: dict) -> dict:
    """The Jupiter stats coverage on its own (a slow count, cached for a minute)."""
    return token_stats_coverage()
