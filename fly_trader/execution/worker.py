"""Execution queue for the live book: the caller never blocks on Jupiter.

The caller enqueues ExecRequest objects; one background thread executes them sequentially through
LiveBroker (one in-flight order at a time, balance-verified); results are drained by the caller, which
opens/closes positions accordingly. Mints with an in-flight request are skipped.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..agent import rails
from . import ledger

log = logging.getLogger(__name__)


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


class ExecutionWorker:
    def __init__(self, broker):
        self.broker = broker
        self.q: queue.Queue[ExecRequest] = queue.Queue()
        self.results: queue.Queue[ExecResult] = queue.Queue()
        self.inflight: set[str] = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="runner-exec", daemon=True)   # routed to the runner's log
        self.stop_flag = False
        self.thread.start()

    def submit(self, req: ExecRequest) -> bool:
        with self.lock:
            if req.mint in self.inflight:
                return False
            self.inflight.add(req.mint)
        self.q.put(req)
        return True

    def pending(self) -> set[str]:
        with self.lock:
            return set(self.inflight)

    def drain(self) -> list[ExecResult]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        self.stop_flag = True

    def _loop(self) -> None:
        while not self.stop_flag:
            try:
                req = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            res = self._execute(req)
            with self.lock:
                self.inflight.discard(req.mint)
            self.results.put(res)

    def _execute(self, req: ExecRequest) -> ExecResult:
        try:
            with transaction() as conn:
                fr = self.broker.swap(conn, decision_id=req.decision_id, book="live", side=req.side, mint=req.mint,
                                      amount_in=req.amount_in, slippage_bps=req.slippage_bps,
                                      max_slippage_bps=req.max_slippage_bps, step_bps=config.SLIPPAGE_STEP_BPS)
                if not fr.ok:
                    rails.record_failure(conn, fr.error or fr.status or "swap failed")
                    return ExecResult(req, False, error=fr.error or fr.status)
                rails.record_success(conn)
                ts = datetime.now(timezone.utc)
                if req.side == "buy":
                    cost = -fr.lamports_delta / config.LAMPORTS_PER_SOL
                    price = fr.price_sol or (cost / max(fr.token_delta / (10 ** req.decimals), 1e-12))
                    pid = ledger.open_position(conn, book="live", mint=req.mint, pool=req.pool, qty_raw=fr.token_delta,
                                               cost_sol=cost, entry_price=price, decision_id=req.decision_id,
                                               fees_sol=0.0, ts=ts)
                    rails.record_notional(conn, cost, "entry")
                    return ExecResult(req, True, fr.fill_id, fr.signature, fr.token_delta, fr.lamports_delta, price, position_id=pid)
                else:
                    proceeds = fr.lamports_delta / config.LAMPORTS_PER_SOL
                    pos = conn.execute("SELECT * FROM positions WHERE id=%s FOR UPDATE", (req.position_id,)).fetchone()
                    price = fr.price_sol if fr.price_sol is not None else (proceeds / max(abs(fr.token_delta) / (10 ** req.decimals), 1e-12))
                    remaining = int(pos["qty"]) + fr.token_delta if pos else 0  # token_delta negative
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
                    return ExecResult(req, True, fr.fill_id, fr.signature, fr.token_delta, fr.lamports_delta, price,
                                      position_id=req.position_id, realized_sol=realized)
        except Exception as e:
            log.error("execution error for %s: %s\n%s", req.mint, e, traceback.format_exc())
            try:
                with transaction() as conn:
                    rails.record_failure(conn, f"{type(e).__name__}: {e}")
            except Exception:
                pass
            return ExecResult(req, False, error=f"{type(e).__name__}: {e}")
