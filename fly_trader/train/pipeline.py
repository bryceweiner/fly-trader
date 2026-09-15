"""The training pipeline — the ``train`` worker. Every ``INTERVAL_DAYS`` days, or at once when the operator asks:

1. The selector retrains on the whole corpus (``train/selector.main``): every day after the warm-up is backtested by a
   model that never saw it, then the model is fit on every day and saved. It is put to work only if its backtest made
   money after costs and beat random picks (``deployable``); the trading engine switches to it without a restart.
2. The fly learns to imitate the selector (``train/fly_selector.main``: the selector's scores are its training targets)
   and is scored on the last 21 days against the selector and random picks.

A run starts only when the feature build is complete (every day aggregated and built with the current definitions);
otherwise it waits and checks again every hour. State for the console: ``ui_settings['training_pipeline']``.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup

log = logging.getLogger(__name__)
INTERVAL_DAYS = 7
RETRY_S = 3600
KEY = "training_pipeline"


class NotReady(RuntimeError):
    pass


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


def request_run() -> None:
    """Run as soon as the worker sees it (the console's Retrain now)."""
    _save(run_requested=True, retry_after=None)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def due(st: dict, now: datetime | None = None) -> bool:
    now = now or _now()
    if st.get("retry_after") and now < datetime.fromisoformat(st["retry_after"]):
        return False
    last = st.get("last_run_at")
    return bool(st.get("run_requested")) or last is None or now >= datetime.fromisoformat(last) + timedelta(days=INTERVAL_DAYS)


def run(stop_event: threading.Event | None = None) -> dict:
    from . import fly_selector, mature, selector
    ok, why = mature.build_complete()
    if not ok:
        raise NotReady(why)
    _save(stage="training the selector", started_at=_now().isoformat(), run_requested=False, retry_after=None, last_error=None, waiting=None)
    wf = selector.main(stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return _save(stage="stopped")
    _save(stage="training the fly to imitate the selector", selector_snapshot=wf.get("snapshot_id"), selector_deployable=wf.get("deployable"))
    verdict = fly_selector.main(line=wf.get("line"), stop_event=stop_event)       # the fly trades the selector's line, unaltered
    if stop_event is not None and stop_event.is_set():
        return _save(stage="stopped")
    now = _now()
    out = _save(stage="done", last_run_at=now.isoformat(), next_run_at=(now + timedelta(days=INTERVAL_DAYS)).isoformat(), fly_snapshot=verdict.get("snapshot_id"))
    record_event("info", "pipeline", "training pipeline finished", {k: out.get(k) for k in ("selector_snapshot", "selector_deployable", "fly_snapshot")})
    return out


def main(stop_event: threading.Event | None = None) -> None:
    setup("train")
    record_event("info", "pipeline", "training pipeline worker started", {"interval_days": INTERVAL_DAYS})
    while not (stop_event is not None and stop_event.is_set()):
        if due(state()):
            try:
                run(stop_event)
            except NotReady as e:
                log.info("training waits for data: %s", e)
                _save(stage="waiting for data", waiting=str(e), retry_after=(_now() + timedelta(seconds=RETRY_S)).isoformat())
            except Exception as e:
                log.exception("training pipeline failed")
                record_event("error", "pipeline", f"training pipeline failed: {type(e).__name__}: {e}")
                _save(stage="failed", last_error=f"{type(e).__name__}: {e}"[:300], retry_after=(_now() + timedelta(seconds=RETRY_S)).isoformat())
        if stop_event is not None:
            stop_event.wait(30.0)
        else:
            threading.Event().wait(30.0)
