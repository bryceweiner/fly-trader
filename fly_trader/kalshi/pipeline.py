"""The Kalshi training pipeline — the ``kalshi_train`` worker (train/pipeline.py for prediction markets). Every
``INTERVAL_DAYS`` days, or at once when the operator asks:

1. The Kalshi selector retrains on the whole corpus (kalshi/selector.py) — walk-forward by settlement day at Kalshi
   costs; deployable only if its evaluation half made money and beat random picks.
2. The Kalshi fly is bootstrapped (kalshi/fly.py, the selector teaches it once) when none exists for
   ``KALSHI_FLY_VERSION`` or on request; then it learns from settlements through its mushroom body. There is no race and
   no handover (decision of 2026-09-22): once a deployable Kalshi fly trades, weekly runs stop and re-bootstraps are on
   request (or after three rollbacks, from kalshi/fly_session.py).

A run starts only when the corpus build is complete (kalshi/mature.build_complete); otherwise it waits an hour.
State for the console: ``ui_settings['kalshi_training_pipeline']``.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from ..train.pipeline import INTERVAL_DAYS, RETRY_S, NotReady

log = logging.getLogger(__name__)
KEY = "kalshi_training_pipeline"


def state() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (KEY,)).fetchone()
    v = r["value"] if r else {}
    return v if isinstance(v, dict) else json.loads(v or "{}")


def _save(**kv) -> dict:
    v = {**state(), **kv}
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (KEY, json.dumps(v, default=str)))
    return v


def request_run(fly: bool = False, reason: str | None = None) -> None:
    kv = {"run_requested": True, "retry_after": None}
    if fly:
        kv.update(fly_requested=True, fly_reason=reason or "requested")
    _save(**kv)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fly_trades() -> bool:
    """A deployable Kalshi fly exists and the replay passed: the weekly cadence stops, runs are on request only."""
    from . import fly as KF, fly_replay
    with transaction() as conn:
        have = KF.latest_deployable(conn)
    return have is not None and (fly_replay.verdict() or {}).get("passed") is True


def due(st: dict, now: datetime | None = None, fly_trades: bool = False) -> bool:
    now = now or _now()
    if st.get("retry_after") and now < datetime.fromisoformat(st["retry_after"]):
        return False
    if fly_trades:
        return bool(st.get("run_requested"))
    last = st.get("last_run_at")
    return bool(st.get("run_requested")) or last is None or now >= datetime.fromisoformat(last) + timedelta(days=INTERVAL_DAYS)


def run(stop_event: threading.Event | None = None) -> dict:
    from . import fly as KF, mature, selector
    ok, why = mature.build_complete()
    if not ok:
        raise NotReady(why)
    st = state()
    _save(stage="training the Kalshi selector", started_at=_now().isoformat(), run_requested=False, retry_after=None, last_error=None, waiting=None)
    wf = selector.main(stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return _save(stage="stopped")
    _save(selector_snapshot=wf.get("snapshot_id"), selector_deployable=wf.get("deployable"), selector_reason=wf.get("deploy_reason"))
    fly_snapshot = st.get("fly_snapshot")
    with transaction() as conn:
        have = KF.latest_current(conn)
    if st.get("fly_requested") or have is None:
        _save(stage="bootstrapping the Kalshi fly (the selector teaches it once)")
        boot = KF.main(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            return _save(stage="stopped")
        fly_snapshot = boot.get("snapshot_id")
    now = _now()
    out = _save(stage="done", last_run_at=now.isoformat(), next_run_at=(now + timedelta(days=INTERVAL_DAYS)).isoformat(), fly_snapshot=fly_snapshot,
                fly_requested=False, fly_reason=None)
    record_event("info", "kalshi_pipeline", "kalshi training pipeline finished", {k: out.get(k) for k in ("selector_snapshot", "selector_deployable", "fly_snapshot")})
    return out


def main(stop_event: threading.Event | None = None) -> None:
    setup("kalshi_train")
    record_event("info", "kalshi_pipeline", "kalshi training pipeline worker started", {"interval_days": INTERVAL_DAYS})
    while not (stop_event is not None and stop_event.is_set()):
        try:
            if due(state(), fly_trades=_fly_trades()):
                run(stop_event)
        except NotReady as e:
            log.info("kalshi training waits for data: %s", e)
            _save(stage="waiting for data", waiting=str(e), retry_after=(_now() + timedelta(seconds=RETRY_S)).isoformat())
        except Exception as e:
            log.exception("kalshi training pipeline failed")
            record_event("error", "kalshi_pipeline", f"kalshi training pipeline failed: {type(e).__name__}: {e}")
            _save(stage="failed", last_error=f"{type(e).__name__}: {e}"[:300], retry_after=(_now() + timedelta(seconds=RETRY_S)).isoformat())
        (stop_event or threading.Event()).wait(30.0)
