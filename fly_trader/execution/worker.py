"""Execution queue for the live book: the caller never blocks on Jupiter.

The caller enqueues ExecRequest objects; ``threads`` background threads (config.EXEC_THREADS) execute them through
LiveBroker, several mints' swaps in flight at once and each mint's one at a time, in order (each fill is read from its
own transaction, execution/broker_live.py); results are drained by the caller, which
opens/closes positions accordingly. A mint with a request queued or in flight takes no other, except a buy queued
behind a lone sale of it (its paper book sold and bought the token again in one minute): the sale runs first, and the
buy is refused as held if the sale left the position open. ``busy`` is held (shared) from each swap to its booking, so
whoever reads the wallet and the books under it alone sees them agree. A sale that leaves the live book without its mint marks the mint's
token account ``emptied``: the mirror closes it mid-minute with nothing queued (agent/fly_live.close_accounts), never in
front of an order, and books the rent it returns. ``stop`` lets the requests in flight finish (bounded) and drops the
queued ones: exits are retried next minute, entries by the mirror (agent/fly_live.py).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..agent import rails
from . import ledger

log = logging.getLogger(__name__)
STOP_JOIN_S = 120.0           # the longest a stop waits for the request in flight (a swap polls until its blockhash expires)


@dataclass
class ExecRequest:
    decision_id: int | None
    mint: str
    pool: str | None
    side: str                     # buy | sell
    amount_in: int                # lamports (buy) or raw token units (sell)
    slippage_bps: int
    max_slippage_bps: int
    decimals: int
    forced_kind: str | None = None
    position_id: int | None = None
    size_sol: float = 0.0
    hold_s: float | None = None   # buy: the entry's own hold and strategy, booked on its position
    strategy: str | None = None
    ref_price: float | None = None   # buy: its decision's paper fill price (SOL per token): a quote above it by more than MAX_ENTRY_CHASE is refused
    retry: int = 0                # a buy sent again after a failure (agent/fly_live.py): its failure was counted on the circuit the first time
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExecResult:
    request: ExecRequest
    ok: bool
    fill_id: int | None = None
    signature: str | None = None
    token_delta: int = 0
    lamports_delta: int = 0
    price_sol: float | None = None
    error: str | None = None
    position_id: int | None = None
    realized_sol: float | None = None
    retryable: bool = False       # a failed buy that moved nothing (no fill row, no unknown outcome): it may be sent again
    closed: bool = False          # a sale that closed its position (or swept a written-off one)


class SharedLock:
    """``busy``: every swap in flight holds it shared, any number at once (``shared``); a wealth mark or a token-account
    close holds it alone (``acquire``/``release``, as a Lock), so none reads the wallet and the books while a swap is
    between its fill and its booking. A swap waits only behind one holding it alone, never behind one waiting (live
    money first)."""

    def __init__(self):
        self._cv = threading.Condition(); self._shared = 0; self._alone = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        with self._cv:
            free = lambda: not self._shared and not self._alone
            if not (free() if not blocking else self._cv.wait_for(free, None if timeout is None or timeout < 0 else timeout)):
                return False
            self._alone = True
            return True

    def release(self) -> None:
        with self._cv:
            self._alone = False; self._cv.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    @contextmanager
    def shared(self):
        with self._cv:
            self._cv.wait_for(lambda: not self._alone)
            self._shared += 1
        try:
            yield
        finally:
            with self._cv:
                self._shared -= 1; self._cv.notify_all()


class ExecutionWorker:
    def __init__(self, broker, threads: int | None = None):
        self.broker = broker
        self.results: queue.Queue[ExecResult] = queue.Queue()
        self.inflight: dict[str, list[str]] = {}       # mint -> sides queued or in flight, in order
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.waiting: dict[str, deque] = {}            # mint -> its requests not started yet, in order
        self.ready: deque[str] = deque()               # mints whose next request may start (none of theirs in flight)
        self.busy = SharedLock()                       # held (shared) from each swap to its booking
        self.stop_flag = False
        self.on_done = None                            # called with each result once booked (the mirror's live stats: agent/fly_live.py)
        self.emptied: set[str] = set()                 # mints whose sale left their token account empty: closed by the mirror, mid-minute
        n = max(1, int(threads if threads is not None else config.EXEC_THREADS))
        self.threads = [threading.Thread(target=self._loop, name="runner-exec" + (f"-{i}" if i else ""), daemon=True) for i in range(n)]   # routed to the runner's log
        self.thread = self.threads[0]
        for t in self.threads:
            t.start()

    def submit(self, req: ExecRequest) -> bool:
        """Queue ``req``; False when its mint already has a request queued or in flight (a buy may queue behind a lone
        sale of it) or the worker is stopping. A mint with nothing queued or in flight starts at once on a free thread."""
        with self.lock:
            sides = self.inflight.get(req.mint)
            if self.stop_flag or (sides and not (req.side == "buy" and sides == ["sell"])):
                return False
            self.inflight.setdefault(req.mint, []).append(req.side)
            self.waiting.setdefault(req.mint, deque()).append(req)
            if not sides:
                self.ready.append(req.mint); self.cv.notify()
        return True

    def pending(self) -> set[str]:
        with self.lock:
            return set(self.inflight)

    def selling(self) -> set[str]:
        """Mints whose only request queued or in flight is a sale: a buy of one may queue behind it."""
        with self.lock:
            return {m for m, sides in self.inflight.items() if sides == ["sell"]}

    def drain(self) -> list[ExecResult]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def _release(self, req: ExecRequest) -> None:
        """``req`` is done (or dropped): its mint's next request, if one is queued, may start."""
        with self.lock:
            sides = self.inflight.get(req.mint)
            if sides and req.side in sides:
                sides.remove(req.side)
            if not sides:
                self.inflight.pop(req.mint, None)
            if self.waiting.get(req.mint):
                if not self.stop_flag:
                    self.ready.append(req.mint); self.cv.notify()
            else:
                self.waiting.pop(req.mint, None)

    def stop(self, timeout: float = STOP_JOIN_S) -> None:
        """Let the requests in flight finish — a process killed mid-swap can leave a landed swap unbooked — drop the
        queued ones, and wait for the threads (at most ``timeout`` in all)."""
        with self.lock:
            self.stop_flag = True
            dropped = [r for q in self.waiting.values() for r in q]
            self.waiting.clear(); self.ready.clear(); self.cv.notify_all()
        for r in dropped:
            self._release(r)
        deadline = time.monotonic() + timeout
        others = [t for t in self.threads if t is not threading.current_thread()]
        for t in others:
            t.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in others):
            log.error("execution worker: a request in flight did not finish within %.0f s of the stop", timeout)

    def _loop(self) -> None:
        while True:
            with self.lock:
                while not self.stop_flag and not self.ready:
                    self.cv.wait(1.0)
                if self.stop_flag:                      # stopping: a queued request is dropped, never started
                    return
                mint = self.ready.popleft(); req = self.waiting[mint].popleft()
            with self.busy.shared():
                res = self._execute(req)
                if res.ok and res.closed:
                    self._emptied(req)
            self._release(req)
            self.results.put(res)
            if self.on_done is not None:
                try:
                    self.on_done(res)
                except Exception:
                    log.exception("execution worker: the after-order hook failed (the order stands)")

    def _execute(self, req: ExecRequest) -> ExecResult:
        from ..vault import walletlock                 # the vault fly: swap + ledger commit are one step for its snapshots
        try:
            with walletlock.maybe_exclusive(config.VAULT_ENABLED), transaction() as conn:
                if req.side == "buy" and conn.execute("SELECT 1 FROM positions WHERE book = 'live' AND mint = %s AND status = 'open'", (req.mint,)).fetchone():
                    return ExecResult(req, False, error="held: the live book still holds this token (the sale before this buy did not close it)", retryable=True)
                fr = self.broker.swap(conn, decision_id=req.decision_id, book="live", side=req.side, mint=req.mint,
                                      amount_in=req.amount_in, slippage_bps=req.slippage_bps,
                                      max_slippage_bps=req.max_slippage_bps, step_bps=config.SLIPPAGE_STEP_BPS,
                                      ref_price=req.ref_price if req.side == "buy" else None, decimals=req.decimals)
                if not fr.ok:
                    if self._counts(req) and fr.status != "price_ran":     # the market ran away from the decision: no execution failed
                        rails.record_failure(conn, fr.error or fr.status or "swap failed")
                    return ExecResult(req, False, error=fr.error or fr.status, retryable=req.side == "buy" and fr.fill_id is None)
                rails.record_success(conn)
                ts = datetime.now(timezone.utc)
                if req.side == "buy":
                    cost = -fr.lamports_delta / config.LAMPORTS_PER_SOL
                    price = fr.price_sol or (cost / max(fr.token_delta / (10 ** req.decimals), 1e-12))
                    pid = ledger.open_position(conn, book="live", mint=req.mint, pool=req.pool, qty_raw=fr.token_delta,
                                               cost_sol=cost, entry_price=price, decision_id=req.decision_id,
                                               fees_sol=0.0, ts=ts, hold_s=req.hold_s, strategy=req.strategy,
                                               deposit_sol=max(fr.deposit_lamports, 0) / config.LAMPORTS_PER_SOL)
                    rails.record_notional(conn, cost, "entry")
                    return ExecResult(req, True, fr.fill_id, fr.signature, fr.token_delta, fr.lamports_delta, price, position_id=pid)
                else:
                    if fr.deposit_lamports < 0:                # the swap closed the token account itself: its deposit is back
                        ledger.refund_deposits(conn, "live", req.mint)
                    proceeds = fr.lamports_delta / config.LAMPORTS_PER_SOL
                    price = fr.price_sol if fr.price_sol is not None else (proceeds / max(abs(fr.token_delta) / (10 ** req.decimals), 1e-12))
                    if req.forced_kind == "dead_bag_sweep":    # a written-off position's tokens: what they fetch is that position's
                        pid = ledger.last_dead_bag(conn, "live", req.mint)
                        if pid is not None:
                            ledger.credit(conn, pid, proceeds)
                        return ExecResult(req, True, fr.fill_id, fr.signature, fr.token_delta, fr.lamports_delta, price,
                                          position_id=pid, realized_sol=proceeds if pid is not None else None, closed=pid is not None)
                    pos = conn.execute("SELECT * FROM positions WHERE id=%s FOR UPDATE", (req.position_id,)).fetchone()
                    remaining = int(pos["qty"]) + fr.token_delta if pos else 0  # token_delta negative
                    closed = False
                    if pos is None or pos["status"] != "open":   # closed elsewhere: the fill row stands, nothing is booked twice
                        log.warning("sell of %s filled but position %s is not open; not booked", req.mint, req.position_id)
                        realized = None
                    elif remaining > max(1, int(pos["qty"]) // 200):  # partial fill: keep the remainder open
                        realized = ledger.realize_partial(conn, position_id=req.position_id, qty_raw=-fr.token_delta,
                                                          cost_part=float(pos["cost_sol"]) * (1 - remaining / max(int(pos["qty"]), 1)),
                                                          proceeds_sol=proceeds, fees_sol=0.0, price=price, ts=ts)
                    else:
                        realized = ledger.close_position(conn, position_id=req.position_id, exit_price=price, proceeds_sol=proceeds,
                                                         fees_sol=0.0, decision_id=req.decision_id, forced_kind=req.forced_kind, ts=ts)
                        closed = realized is not None
                    return ExecResult(req, True, fr.fill_id, fr.signature, fr.token_delta, fr.lamports_delta, price,
                                      position_id=req.position_id, realized_sol=realized, closed=closed)
        except Exception as e:
            log.error("execution error for %s: %s\n%s", req.mint, e, traceback.format_exc())
            try:
                if self._counts(req):
                    with transaction() as conn:
                        rails.record_failure(conn, f"{type(e).__name__}: {e}")
            except Exception:
                pass
            return ExecResult(req, False, error=f"{type(e).__name__}: {e}")

    @staticmethod
    def _counts(req: ExecRequest) -> bool:
        """Whether a failure of ``req`` counts on the circuit: not a retried buy (its first attempt counted: three tries
        at one token that cannot be bought must not trip every entry) nor a dead-bag sweep (it says nothing of trading)."""
        return not req.retry and req.forced_kind != "dead_bag_sweep"

    def _emptied(self, req: ExecRequest) -> None:
        """The sale left the live book without this mint: its empty token account is the mirror's to close mid-minute
        (agent/fly_live.close_accounts), not this thread's in front of the next order. Not while a buy is queued behind
        the sale: it needs the account."""
        with self.lock:
            if self.inflight.get(req.mint) == ["sell"]:
                self.emptied.add(req.mint)
