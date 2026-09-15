"""The `runner` worker: the selector session (``agent/selector_session.py``) trading the paper book every minute on
the market feed. One runner per database (a Postgres advisory lock). ``LIVE_ENABLED=1`` passes the signing guard first."""
from __future__ import annotations

import logging

from .. import config
from ..db.apilog import record_event
from ..db.connection import connect
from ..logging_setup import setup

log = logging.getLogger(__name__)


RUNNER_LOCK_KEY = 0x666C795F72756E  # "fly_run": one runner per database, held for the runner's lifetime


def acquire_runner_lock():
    """Session advisory lock on a dedicated autocommit connection; closing the connection releases it.
    Raises RuntimeError when another runner (process or console thread) already holds it."""
    conn = connect(autocommit=True)
    try:
        ok = conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (RUNNER_LOCK_KEY,)).fetchone()["ok"]
    except Exception:
        conn.close()
        raise
    if not ok:
        conn.close()
        raise RuntimeError("another runner holds the runner lock; refusing to start a second one")
    return conn


def main(stop_event=None) -> None:
    setup("runner")
    try:
        lock_conn = acquire_runner_lock()
    except RuntimeError as e:
        log.error("%s", e)
        record_event("error", "runner", str(e))
        raise
    try:
        _main(stop_event)
    finally:
        lock_conn.close()


def _main(stop_event=None) -> None:
    live = False
    if config.LIVE_ENABLED:
        from ..chain.cluster_guard import assert_signing_allowed
        assert_signing_allowed()
        live = True
    if config.RESET_ON_START:                      # a fresh runner never carries over a previous run's books or stats
        from ..ops.reset import reset_training_state
        reset_training_state(reason="runner start")
    from . import selector_session
    record_event("info", "runner", "runner started", {"live": live})
    selector_session.main(stop_event=stop_event, live=live)
