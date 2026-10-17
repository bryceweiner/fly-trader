"""Live money first and fast over paper, always (CLAUDE.md): when the live books' work runs.

- ``LiveClock``: the live books' sales due at a minute boundary go out the moment it passes, from a thread of their own
  (``runner-live-clock``: each live book's ``live_exits``, agent/fly_session.FlyBook), never behind the stream writing
  the minute or the engine's work on another minute or chain.
- The engine runs every chain's newest minute for its live books first (the fly scores, its paper book plans, the
  wallet's entries go out, committed at once), and only then any book's paper trades, learning and bookkeeping
  (agent/minute_engine.main).
- ``may_mark``: a wealth mark reads the wallet under the worker's lock, so it is taken between minutes once no order is
  queued or in flight, and never within ``MARK_QUIET_S`` of a boundary, when the clock's sales go out.
- ``mid_minute``: slow wallet bookkeeping (dead-bag sweeps, the RH reconcile) runs mid-minute, after the minute's
  entries and done before the next boundary's sales.
"""
from __future__ import annotations

import logging
import math
import threading
import time

from ..db.apilog import record_event

log = logging.getLogger(__name__)
MARK_QUIET_S = 3.0            # no wallet read for a mark this close to a minute boundary: the clock's sales must not wait for it
MID_MINUTE = (20.0, 40.0)     # seconds into a minute when slow wallet bookkeeping may start


def may_mark(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now % 60.0 < 60.0 - MARK_QUIET_S


def mid_minute(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return MID_MINUTE[0] <= now % 60.0 < MID_MINUTE[1]


class LiveClock:
    """Calls every book's ``live_exits(m1_epoch)`` (books without one are skipped) at each minute boundary."""

    def __init__(self, books: list, stop_event: threading.Event | None = None):
        self.books, self.outer, self.stop_flag = books, stop_event, threading.Event()
        self.thread = threading.Thread(target=self._loop, name="runner-live-clock", daemon=True)   # routed to the runner's log

    def start(self) -> "LiveClock":
        self.thread.start()
        return self

    def stopped(self) -> bool:
        return self.stop_flag.is_set() or (self.outer is not None and self.outer.is_set())

    def _loop(self) -> None:
        while not self.stopped():
            nxt = math.floor(time.time() / 60.0) * 60.0 + 60.0
            while not self.stopped() and time.time() < nxt:
                self.stop_flag.wait(nxt - time.time())
            if self.stopped():
                break
            self.tick(math.floor(time.time() / 60.0) * 60.0)        # the boundary just passed (a late wake takes the newest)

    def tick(self, m1_epoch: float) -> None:
        for b in list(self.books):
            fn = getattr(b, "live_exits", None)
            if fn is None:
                continue
            try:
                out = fn(m1_epoch)
                n = (out or {}).get("exited", 0) + (out or {}).get("liquidated", 0)
                if n:
                    log.info("%s: %d live sale(s) sent at the minute boundary", b.name, n)
            except Exception as e:
                log.exception("%s: live exits at the minute boundary failed", b.name)
                record_event("error", b.name, f"live exits at the minute boundary failed: {type(e).__name__}: {str(e)[:160]}")

    def close(self, timeout: float = 10.0) -> None:
        self.stop_flag.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout)
