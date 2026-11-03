"""Real-time training status for the console: one JSON row in ui_settings['training_status'], updated
by the trainer every few seconds (stage, step/total, ETA, latest metrics), plus a cooperative stop event."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from ..db.connection import transaction

_state: dict = {}
_lock = threading.Lock()
_last_write = 0.0
STOP: threading.Event | None = None


def _finite(x):
    """JSON (and Postgres jsonb) have no inf/nan: map them to null recursively."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def set_stop_event(ev: threading.Event | None) -> None:
    global STOP
    STOP = ev


def should_stop() -> bool:
    return STOP is not None and STOP.is_set()


def update(stage: str, step: int | None = None, total: int | None = None, force: bool = False, **extra) -> None:
    global _last_write
    with _lock:
        _state.update({"stage": stage, "step": step, "total": total, "updated_at": datetime.now(timezone.utc).isoformat()})
        if "started_at" not in _state:
            _state["started_at"] = _state["updated_at"]
        if "stage_started_at" not in _state or _state.get("_stage") != stage:
            _state["stage_started_at"] = time.time(); _state["_stage"] = stage
        if step and total:
            el = time.time() - _state["stage_started_at"]
            _state["eta_s"] = el / max(step, 1) * (total - step)
        _state.update(extra)
        now = time.time()
        if not force and now - _last_write < 3.0:
            return
        _last_write = now
        payload = _finite({k: v for k, v in _state.items() if not k.startswith("_")})
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('training_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps(payload, default=str),))
    except Exception:
        pass


def clear() -> None:
    with _lock:
        _state.clear()
