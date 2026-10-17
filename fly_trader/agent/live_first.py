"""Live money first and fast over paper, always (CLAUDE.md): when the live books' work runs.

- ``LiveClock``: the live chains' sales due at a minute boundary go out the moment it passes, from a thread of their own
  (``runner-live-clock``), started with the runner -- before its books load and the engine warms up, so a restart never
  holds a sale back -- and at once for the minute it starts in (a restart may have straddled a boundary).
- One live mirror per chain per process (``mirror``): made by whichever comes first, the clock at a boundary or the
  fly's first live minute, and stopped by the runner when it stops (``stop_mirrors``), never by a book.
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
from datetime import datetime, timezone

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)
MARK_QUIET_S = 3.0            # no wallet read for a mark this close to a minute boundary: the clock's sales must not wait for it
MID_MINUTE = (20.0, 40.0)     # seconds into a minute when slow wallet bookkeeping may start
_MIRRORS: dict[str, object] = {}
_MIRRORS_LOCK = threading.Lock()


def may_mark(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now % 60.0 < 60.0 - MARK_QUIET_S


def mid_minute(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return MID_MINUTE[0] <= now % 60.0 < MID_MINUTE[1]


def signing_refused(chain: str) -> str | None:
    """Why the chain's wallet may not sign now (None: it may)."""
    if chain != "sol":
        missing = config.rh_live_prerequisites_missing()
        return ", ".join(missing) if missing else None
    from ..chain.cluster_guard import signing_allowed
    return None if signing_allowed() else "signing is not allowed (LIVE_ENABLED=1 and the live prerequisites)"


def trades_live(conn, chain: str, live: bool) -> bool:
    """The chain's wallet trades: the runner may sign (``live``), and the chain's fly holds the seat or
    LIVE_WITHOUT_HANDOVER says live money trades whoever holds it (the signing gates still apply)."""
    from . import handover
    return bool(live) and (config.LIVE_WITHOUT_HANDOVER or handover.state(conn, chain) is not None)


def mirror(chain: str, horizon_s: float | None = None):
    """The chain's live mirror (agent/fly_live.LiveMirror, rh/live.RhLiveMirror), one per process. ``horizon_s``: the fly's
    longest hold, the hold of a position that carries none; the fly sets it when it runs a live minute (until then such
    a position is not sold)."""
    with _MIRRORS_LOCK:
        m = _MIRRORS.get(chain)
        if m is None:
            if chain == "sol":
                from .fly_live import LiveMirror
                m = LiveMirror(horizon_s)
                record_event("info", "fly_live", "the fly trades the bot wallet", {"chain": chain})
            else:
                from ..rh.live import RhLiveMirror
                m = RhLiveMirror(horizon_s)
                record_event("info", "rh_live", "the RH fly trades the ETH wallet", {"chain": chain})
            _MIRRORS[chain] = m
        elif horizon_s is not None:
            m.H = float(horizon_s)
        return m


def mirror_of(chain: str):
    """The chain's live mirror once made, else None."""
    return _MIRRORS.get(chain)


def chain_exits(chain: str, m1_epoch: float, live: bool, run_id: str | None = None) -> dict | None:
    """The chain's live sales due at the minute boundary ``m1_epoch``, while its wallet trades, from the chain's mirror
    (made now when the fly has not run a live minute yet: its book may still be loading)."""
    if not live or signing_refused(chain):
        return None
    with transaction() as conn:
        if not trades_live(conn, chain, live):
            return None
        return mirror(chain).exits(conn, m1=datetime.fromtimestamp(m1_epoch, timezone.utc), run_id=run_id)


def idle_mirrors() -> None:
    """Between minutes: each mirror's bookkeeping that waits for an idle wallet (the wealth mark, sweeps, the reconcile)."""
    for chain, m in list(_MIRRORS.items()):
        try:
            m.idle()
        except Exception:
            log.exception("%s live mirror: idle bookkeeping failed", chain)


def stop_mirrors() -> None:
    """Stop every mirror (each waits for its order in flight) and forget them: a runner started again in this process
    makes new ones."""
    with _MIRRORS_LOCK:
        ms = list(_MIRRORS.items()); _MIRRORS.clear()
    for chain, m in ms:
        try:
            m.stop()
        except Exception:
            log.exception("%s live mirror failed to stop", chain)


class LiveClock:
    """Sends every live chain's sales due at each minute boundary (``chain_exits``). ``books``: the runner's list; a
    chain's live book lends its run id to the sales once it has loaded."""

    def __init__(self, books: list, chains: list[str], live: bool, stop_event: threading.Event | None = None):
        self.books, self.chains, self.live = books, list(chains), bool(live)
        self.outer, self.stop_flag = stop_event, threading.Event()
        self.thread = threading.Thread(target=self._loop, name="runner-live-clock", daemon=True)   # routed to the runner's log

    def start(self) -> "LiveClock":
        self.thread.start()
        return self

    def stopped(self) -> bool:
        return self.stop_flag.is_set() or (self.outer is not None and self.outer.is_set())

    def _loop(self) -> None:
        self.tick(math.floor(time.time() / 60.0) * 60.0)               # the boundary a restart may have straddled
        while not self.stopped():
            nxt = math.floor(time.time() / 60.0) * 60.0 + 60.0
            while not self.stopped() and time.time() < nxt:
                self.stop_flag.wait(nxt - time.time())
            if self.stopped():
                break
            self.tick(math.floor(time.time() / 60.0) * 60.0)        # the boundary just passed (a late wake takes the newest)

    def tick(self, m1_epoch: float) -> None:
        for chain in self.chains:
            run_id = next((b.run_id for b in list(self.books) if getattr(b, "LIVE", False) and getattr(b, "chain", "sol") == chain), None)
            try:
                out = chain_exits(chain, m1_epoch, self.live, run_id)
                n = (out or {}).get("exited", 0) + (out or {}).get("liquidated", 0)
                if n:
                    log.info("%s: %d live sale(s) sent at the minute boundary", chain, n)
            except Exception as e:
                log.exception("%s: live exits at the minute boundary failed", chain)
                record_event("error", "fly_live" if chain == "sol" else "rh_live", f"live exits at the minute boundary failed: {type(e).__name__}: {str(e)[:160]}",
                             {"chain": chain})

    def close(self, timeout: float = 10.0) -> None:
        self.stop_flag.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout)
