"""Wall-clock context readiness (VOC dexlp/env/context_window.py semantics).

The runner may only enter positions once it has observed CONTEXT_READY_COVERAGE of a full window of
tape activity. A universe-wide silence longer than CONTEXT_GAP_S (feed outage) resets coverage from
the end of the gap. Works identically on simulated time in pretraining.
"""
from __future__ import annotations

from .. import config


class ContextWindow:
    def __init__(self, window_s: float | None = None, gap_s: float | None = None, ready_cov: float | None = None):
        self.window_s = window_s or config.CONTEXT_WINDOW_S
        self.gap_s = gap_s or config.CONTEXT_GAP_S
        self.ready_cov = ready_cov or config.CONTEXT_READY_COVERAGE
        self.start_ts: float | None = None
        self.last_activity_ts: float | None = None
        self.gaps = 0

    def observe(self, ts: float, had_activity: bool) -> None:
        if not had_activity:
            return
        if self.start_ts is None:
            self.start_ts = ts  # coverage starts at the first activity, not at the first (possibly silent) beat
        elif self.last_activity_ts is not None and ts - self.last_activity_ts > self.gap_s:
            self.gaps += 1
            self.start_ts = ts  # coverage restarts after an outage
        self.last_activity_ts = ts

    def coverage(self, now: float) -> float:
        if self.start_ts is None:
            return 0.0
        return max(0.0, min(1.0, (now - self.start_ts) / self.window_s))

    def ready(self, now: float) -> bool:
        return self.coverage(now) >= self.ready_cov

    def feed_age_s(self, now: float) -> float | None:
        return None if self.last_activity_ts is None else max(0.0, now - self.last_activity_ts)
