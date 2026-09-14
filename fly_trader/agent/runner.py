"""The `runner` worker: one brain over the live pump.fun slots plus (optionally) corpus-replay slots.
Wall-clock beats; paper books always; live book when LIVE_ENABLED=1 and the signing guard passes.
SIGTERM → finish (journal flush, stop snapshot, run row). The replay clock is persisted in
ui_settings so a restart resumes the corpus where it left off."""
from __future__ import annotations

import json
import logging
import signal
import time

from .. import config
from ..brain.lif import Connectome
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from .beat import Session, SessionOptions

log = logging.getLogger(__name__)


def _resume_clock(name: str) -> float | None:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (f"replay_clock:{name}",)).fetchone()
    if not r:
        return None
    v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])
    return float(v["clock"]) if v.get("corpus") == config.REPLAY_CORPUS else None


def main(stop_event=None) -> None:
    import threading
    setup("runner")
    live = False
    if config.LIVE_ENABLED:
        from ..chain.cluster_guard import assert_signing_allowed
        assert_signing_allowed()
        live = True
    if config.BRAIN_MODE == "selector":            # the strategy: minute-by-minute selector on the live tape (paper book, live once wired)
        from . import selector_session
        record_event("info", "runner", "runner started", {"mode": "selector", "live": live})
        selector_session.main(stop_event=stop_event, live=live)
        return
    if config.RESET_ON_START:
        from ..ops.reset import reset_training_state
        reset_training_state(reason="runner start")
    c = Connectome.load()
    if config.BRAIN_MODE == "policy":
        from .beat_policy import PolicySession
        s = PolicySession(c, SessionOptions(run_kind="live", ticks=0, learn=False, persist_slots=True))
    else:
        s = Session(c, SessionOptions(run_kind="live", ticks=config.BEAT_TICKS, learn=True, persist_slots=True), corpus=None)
    s.add_live_group(config.SLOTS, books=("paper_free", "paper_mirror"), live=live)
    replay_note = None   # corpus replay removed from the runner (operator decision 2026-09-13)
    s.start()
    stop = {"flag": False}
    if stop_event is None and threading.current_thread() is threading.main_thread():
        def _stop(*_):
            stop["flag"] = True
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    if stop_event is not None:
        _orig = stop
        class _Flag(dict):
            def __getitem__(self, k):
                return stop_event.is_set() or dict.__getitem__(self, k)
        stop = _Flag(flag=False)
    record_event("info", "runner", "runner started", {"run_id": s.run_id, "live": live, "slots": config.SLOTS, "mode": config.BRAIN_MODE,
                                                       "ticks": config.BEAT_TICKS, "device": str(c.device), "replay": replay_note})
    log.info("runner started run_id=%s live=%s device=%s batch=%d %s", s.run_id, live, c.device, s.B, replay_note)
    n = 0
    try:
        while not stop["flag"]:
            t0 = time.time()
            out = s.run_beat(t0)
            n += 1
            if n % 20 == 0:
                log.info("beat %d active=%d decisions=%d forced=%d gpu_ms=%d total_ms=%d mean_mbon=%.3f kc=%.3f blocks=%s replay=%s",
                         n, out["active"], out["decisions"], out["forced"], out["gpu_ms"], out["total_ms"], out["mean_mbon"], out["kc"],
                         out["blocks"], (out["groups"].get("replay") or {}).get("clock"))
            remaining = config.BEAT_S - (time.time() - t0)
            while remaining > 0 and not stop["flag"]:
                time.sleep(min(0.2, remaining))
                remaining = config.BEAT_S - (time.time() - t0)
    except Exception as e:
        log.exception("runner crashed")
        record_event("error", "runner", f"runner crashed: {type(e).__name__}: {e}")
        s.finish("crashed", {"beats": n})
        raise
    s.finish("stopped", {"beats": n})
    record_event("info", "runner", "runner stopped", {"beats": n})
