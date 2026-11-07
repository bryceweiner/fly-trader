"""Crash exit, Solana (Bryce, 2026-11-07: "If we see it drop 25% in 5 minutes or less we should exit immediately. If it
passes the bar later and we rebuy, then fine").

On 2026-11-07 the wallet bought 519AMD at 04:08; from 04:31 to 04:38 its holders sold ~75 SOL and the price fell 90 %;
the position was held to its 240 minutes and sold at -82 % (-0.195 SOL). Replayed over every Solana trade since 10-17
(data/scratch/crash_exit_20261107): selling the moment a held coin is 25 % under its high of the last 5 minutes made
live +0.30 SOL since 10-28 instead of -0.24, and 519AMD -25 % instead of -82 %; the same rule read once a minute, on
the minute's close, made nothing (-0.22). So it watches every swap, not the minutes. On Robinhood Chain it lost in every
variant: not used there.

- ``observe`` (ingest/pumpstream.Aggregator.ingest, each PumpSwap leg it keeps): a coin a Solana book holds keeps its
  prices of the last ``WINDOW_S``; a price ``DROP`` or more under their high (the entry price counted while inside the
  window, nothing before the entry) is a crash: noted (``crashed``), and the wallet's position, if any, is sold at once
  from a thread of its own (agent/fly_live.LiveMirror.crash_exit, the forced slippage), tried again on later prices
  while it is still open. Cheap on the stream's thread: dictionary reads; the database and the order go to ``_POOL``.
- Paper books (agent/paper_trading.plan_minute) sell a position whose coin crashed after its entry, in the minute's
  bookkeeping, as forced 'crash' exits, at the first price the stream saw ``FILL_DELAY_S`` or more after the crash
  (about when the wallet's sale lands; the crash's own price until one comes): a paper fill at the crash's very price
  would beat the wallet's on every crash and read as an execution gap (agent/fly_live.LiveMirror.gap).
- A crashed coin may be bought again whenever it passes the bar again.
- The held coins are re-read every ``REFRESH_S`` (a new position is watched within seconds of its entry).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)
DROP = 0.25                 # a fall this deep under the window's high is a crash
WINDOW_S = 300.0            # the window's length: 5 minutes
REFRESH_S = 5.0             # the held coins re-read this often
RETRY_S = 15.0              # a crashed coin's live sale is sent again this long after the last try while it stays open
KEEP_S = 86_400.0           # a crash is remembered this long (a paper position of the coin opened before it still sells)
FILL_DELAY_S = 5.0          # paper's crash sale: the first price this long after the crash (the wallet's sale: order, signer, landing)
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crash-exit")


class Watch:
    def __init__(self):
        self.lock = threading.Lock()
        self.held: dict[str, tuple[float, float]] = {}          # mint -> (first open position's entry, epoch s; its entry price)
        self.ticks: dict[str, deque] = {}                       # mint -> (epoch s, price) of the last WINDOW_S
        self.crashes: dict[str, list] = {}                      # mint -> [epoch s, price, the window's high, paper's fill or None]
        self.sent: dict[str, float] = {}                        # mint -> when its live sale was last sent
        self.refreshed = 0.0; self.refreshing = False

    # -------------------------------------------------------------- the stream's thread
    def observe(self, mint: str, ts: float, price: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now - self.refreshed >= REFRESH_S and not self.refreshing:
            self.refreshing = True
            _POOL.submit(self._refresh)
        h = self.held.get(mint)
        if h is None or price <= 0 or ts < h[0]:
            return
        dq = self.ticks.get(mint)
        if dq is None:
            dq = self.ticks[mint] = deque()
        dq.append((ts, price))
        while dq and dq[0][0] < ts - WINDOW_S:
            dq.popleft()
        high = max(p for _, p in dq)
        if h[0] >= ts - WINDOW_S:
            high = max(high, h[1])
        c = self.crashes.get(mint)
        known = c is not None and c[0] >= h[0]                  # this holding's crash, its live sale maybe not done yet
        if known and c[3] is None and ts >= c[0] + FILL_DELAY_S:
            c[3] = price
        if not known and price > (1.0 - DROP) * high:
            return
        with self.lock:
            first = not known
            if first:
                c = self.crashes[mint] = [ts, price, high, None]; self.sent.pop(mint, None)
            due = now - self.sent.get(mint, 0.0) >= RETRY_S
            if due:
                self.sent[mint] = now
        if first:
            log.warning("crash: %s at %.6g, %.0f%% under its 5-minute high %.6g", mint, price, (1 - price / high) * 100, high)
        if due:
            _POOL.submit(self._sell, mint, c[1], c[2], first)

    # -------------------------------------------------------------- its own threads
    def _refresh(self) -> None:
        try:
            with transaction() as conn:
                rows = conn.execute("""SELECT mint, min(extract(epoch FROM opened_at)) AS t0,
                                              (array_agg(entry_price ORDER BY opened_at))[1] AS px
                                       FROM positions WHERE status = 'open' AND COALESCE(chain, 'sol') = 'sol' GROUP BY mint""").fetchall()
            held = {r["mint"]: (float(r["t0"]), float(r["px"] or 0.0)) for r in rows}
            with self.lock:
                self.held = held
                for m in [m for m in self.ticks if m not in held]:
                    self.ticks.pop(m, None)
                cut = time.time() - KEEP_S
                for m in [m for m, c in self.crashes.items() if c[0] < cut]:
                    self.crashes.pop(m, None); self.sent.pop(m, None)
        except Exception:
            log.exception("crash exit: reading the held coins failed")
        finally:
            self.refreshed = time.time(); self.refreshing = False

    def _sell(self, mint: str, price: float, high: float, first: bool) -> None:
        why = f"crash: {(1 - price / high) * 100:.0f}% under its 5-minute high"
        try:
            from . import live_first
            n = live_first.crash_sale("sol", mint, why)
            if first:
                record_event("warning", "crash_exit", f"{mint} fell {(1 - price / high) * 100:.0f}% within 5 minutes: "
                             + (f"the wallet's position sold ({n})" if n else "no live position to sell; paper sells it this minute"),
                             {"mint": mint, "price": price, "high": high, "live_sales": n})
        except Exception:
            log.exception("crash exit: selling %s failed", mint)

    # -------------------------------------------------------------- the engine's thread
    def crashed(self, mint: str, since) -> tuple[float, float] | None:
        """(epoch s, paper's price) of ``mint``'s crash at or after ``since`` (a datetime: a position's entry), else None."""
        c = self.crashes.get(mint)
        if c is None or c[0] < since.timestamp():
            return None
        return c[0], c[3] if c[3] is not None else c[1]


WATCH = Watch()
