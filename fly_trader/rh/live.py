"""The Robinhood Chain fly's live book (``live_rh``): the paper RH fly's minute mirrored on the bot's ETH wallet once
the fly holds the RH seat (handover_rh) and signing is allowed (config.rh_live_prerequisites_missing, rh/guard.py).

``RhLiveMirror`` — agent/fly_live.LiveMirror's shape, in ETH, through rh/exec.RhExecutor, live money first (CLAUDE.md,
agent/live_first.py):
- exits (``exits``: at the minute boundary from the engine's live clock — the RH stream writes a minute seconds to
  minutes after its end, and paper's exit price is that minute's close —; ``minute`` sends them when the clock has
  not): positions held their hold — from their entry decision's minute, as paper counts it — are sold (exit slippage,
  the forced ladder once ``OVERDUE_S`` late); with circuit 3's kill switch on and ``KILL_SWITCH_LIQUIDATE`` every other
  open position is sold at the forced slippage;
- entries (``minute``: the moment the fly has decided, before any paper trade, learning or bookkeeping): the executor
  drained (one intent at a time; each booked by rh/accounting.py from measured balance deltas); this minute's paper RH
  entries, sized from the wallet (native ETH + open positions at cost as the bankroll, native − committed buys as cash,
  the RH gas reserve never spent), one position per token (a buy may queue behind the sale of the same token), never
  more than ``MAX_ENTRY_CHASE`` above its paper fill price (rh/exec.py); blocked by circuit 3 (kill switch, a trip until
  ``CIRCUIT_COOLDOWN_S`` after its last failure, paused entries) or a stale mark of the pool's base asset (a stock
  outside its trading hours). A buy that failed without buying anything is sent again, re-sized, while its paper
  position is open, up to ``ENTRY_RETRY_MIN`` after its decision. Every paper entry the wallet did not take is a
  decisions row of this book (kind 'blocked', the reason as its rail, ``of_decision`` the paper entry);
- bookkeeping (``after``: once every chain's live orders are out): ``DEAD_BAG_S`` past the hold an unsellable position
  is written off (its tokens swept hourly, what they fetch booked onto it); the gap: live trailing its paper mirror by
  more than ``GAP_MAX`` per trade pauses circuit 3's entries;
- between minutes (``idle``), never in front of an intent: the wealth mark — native + positions net of the Pons exit
  cost + base lots → ``wealth_marks`` (book live_rh), read under the wallet lock once no intent is queued or in flight
  and not just before a boundary; the kill switch on its own peak (circuit 3, ``RH_KILL_SWITCH_DRAWDOWN``); mid-minute,
  hourly, dead-bag tokens swept and base lots sold back to ETH, and every ``RECONCILE_S`` the chain reconciled with the
  books (rh/accounting.reconcile).
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config
from ..agent import live_first, rails, sizing
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..markets import RH, RH_CIRCUIT
from . import accounting as A
from .exec import ZERO, RhExecutor, RhRequest

log = logging.getLogger(__name__)
BOOK, MIRROR = RH.live_book, RH.fly_book
DEAD_BAG_S, OVERDUE_S, SWEEP_S, RECONCILE_S = 3 * 3600.0, 600.0, 3600.0, 600.0
GAP_TRADES, GAP_MAX = 50, 0.02
MARK_WAIT_S = 2.0             # how long the wealth mark waits for the wallet lock (an intent between its snapshot and its booking)


def reserved_wei(conn) -> int:
    """ETH owed to vault lockers and held in the trading wallet (never traded): the vault's ETH pot, 0 until it exists."""
    try:
        from ..vault import settle_eth
    except ImportError:
        return 0
    return int(settle_eth.reserved_in_trading(conn))


def pool_of(conn, pool_id: str | None) -> dict | None:
    if not pool_id:
        return None
    r = conn.execute("SELECT pool_id, token, currency0, currency1, fee, tick_spacing, hooks, quote_asset FROM rh_pools WHERE pool_id = %s", (pool_id,)).fetchone()
    return dict(r) if r else None


class RhLiveMirror:
    def __init__(self, horizon_s: float, executor: RhExecutor | None = None):
        if executor is None:
            from .rpc import RhRpc
            from .wallet import RhWallet
            executor = RhExecutor(RhWallet(RhRpc()))
        self.ex, self.H = executor, float(horizon_s)
        self.last_sweep = self.last_reconcile = 0.0
        self.sent: dict[int, dict] = {}       # decision id -> {"e": its paper entry, "ts": its minute, "tries": n}: a buy queued or in flight
        self.failed: dict[int, dict] = {}     # the same for a buy that failed without buying anything: sent again while it may be
        self.exit_lock = threading.Lock()     # the live clock's thread and the engine's both send exits: once per minute
        self.exits_through: datetime | None = None
        self.due_mark: tuple | None = None    # (the minute, its beat id): the wealth mark waiting for an idle executor (``idle``)
        self.last_mark: dict | None = None
        self.last_reconciled: dict | None = None
        self.wallet_eth: float | None = None

    @property
    def wallet(self):
        return self.ex.wallet

    def _decision(self, conn, m1, run_id, beat_id, p, reason: str, forced: bool) -> int:
        return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'fly_exit',%s,%s,%s,%s) RETURNING id",
                                (beat_id, run_id, m1, p["mint"], p["pool"], float(p["cost_sol"]), forced, reason, json.dumps({"book": BOOK}))).fetchone()["id"])

    def _blocked(self, conn, ctx, run_id, beat_id, e: dict, rail: str, reason: str, size: float = 0.0, **detail) -> None:
        """A paper entry the wallet did not take (or not yet): a decisions row of this book, so live and paper can be told apart."""
        conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                     (beat_id, run_id, ctx.m1, e["mint"], e["info"].get("pool"), float(e["score"]), float(size), rail, reason,
                      json.dumps({"book": BOOK, "of_decision": e["decision_id"], **detail}, default=str)))

    def _settled(self, conn, ctx, run_id, beat_id, r) -> None:
        """A drained buy this mirror sent: done when it bought; else recorded, and kept for another try when it bought
        nothing and left no swap open."""
        did = r.request.decision_id
        f = self.sent.pop(did, None) if r.request.kind == "buy" and did is not None else None
        if f is None or r.ok:
            return
        rail = "price ran" if (r.error or "").startswith("price ran") else "entry failed"
        self._blocked(conn, ctx, run_id, beat_id, f["e"], rail, (r.error or "failed")[:300], r.request.amount_in / A.WEI, attempt=f["tries"], retry=bool(r.retryable))
        if r.retryable:
            self.failed[did] = {**f, "error": r.error}

    def _retry_over(self, conn, ctx, did: int, f: dict) -> str | None:
        """Why a failed entry is not tried again (None: it may be)."""
        if (ctx.m1 - f["ts"]).total_seconds() > config.ENTRY_RETRY_MIN * 60.0:
            return f"not filled within {config.ENTRY_RETRY_MIN:g} min of its decision"
        if not conn.execute("SELECT 1 FROM positions WHERE book = %s AND entry_decision_id = %s AND status = 'open'", (MIRROR, did)).fetchone():
            return "its paper position is closed"
        return None

    def _buy(self, e: dict, mint: str, quote: str, pool: dict | None, size: float, retry: int = 0) -> bool:
        return self.ex.submit(RhRequest(kind="buy", token=mint, quote_asset=quote, amount_in=int(size * A.WEI), slippage_bps=config.SLIPPAGE_ENTRY_BPS,
                                        max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS, pool=pool, decision_id=e["decision_id"], hold_s=e.get("hold_s"),
                                        strategy=e.get("strategy"), decimals=int(e["info"].get("decimals") or 18), ref_price=e.get("ref_price"), retry=retry))

    def _sell(self, conn, p: dict, did: int | None, slip: float, forced_kind: str | None = None) -> bool:
        pool = pool_of(conn, p.get("pool"))
        return self.ex.submit(RhRequest(kind="sell", token=p["mint"], quote_asset=(p.get("quote_asset") or (pool or {}).get("quote_asset") or ZERO), amount_in=int(p["qty"]),
                                        slippage_bps=slip, max_slippage_bps=config.SLIPPAGE_FORCED_BPS, pool=pool, decision_id=did, position_id=int(p["id"]),
                                        forced_kind=forced_kind, decimals=int(p.get("decimals") or 18)))

    def _held(self, p: dict, m1: datetime) -> tuple[float, float]:
        """(seconds held, its hold): from its entry decision's minute (paper's clock; an RH fill lands ~10 s after it, a
        lagging minute later still)."""
        return (m1 - (p.get("decision_ts") or p["opened_at"])).total_seconds(), float(p.get("hold_s") or self.H)

    def _dead(self, p: dict, m1: datetime) -> bool:
        held, H = self._held(p, m1)
        return held >= H + DEAD_BAG_S

    def exits(self, conn, *, m1: datetime, run_id: str, beat_id: int | None = None) -> dict:
        """The sales due at minute ``m1``, sent once per minute: by the live clock at the boundary (agent/live_first.py),
        or by ``minute`` when the clock has not. A position past ``DEAD_BAG_S`` is left to ``after`` (written off), one in
        flight to the executor."""
        n_exit = n_liq = 0
        with self.exit_lock:
            if self.exits_through is not None and m1 <= self.exits_through:
                return {"exited": 0, "liquidated": 0}
            self.exits_through = m1
            inflight = self.ex.pending(); liquidate = rails.load_circuit(conn, RH_CIRCUIT).kill_switch and config.KILL_SWITCH_LIQUIDATE
            for p in ledger.open_positions(conn, BOOK):
                if p["mint"].lower() in inflight:
                    continue
                held, H = self._held(p, m1)
                if H <= held < H + DEAD_BAG_S:
                    overdue = held >= H + OVERDUE_S
                    did = self._decision(conn, m1, run_id, beat_id, p, f"held {held / 60:.0f} min" + (" (retrying)" if overdue else ""), False)
                    n_exit += self._sell(conn, p, did, config.SLIPPAGE_FORCED_BPS if overdue else config.SLIPPAGE_EXIT_BPS)
                elif liquidate and held < H:         # the kill switch stops holding, not only entering
                    did = self._decision(conn, m1, run_id, beat_id, p, "kill switch: liquidating", True)
                    n_liq += self._sell(conn, p, did, config.SLIPPAGE_FORCED_BPS)
        if n_liq:
            record_event("error", "rh_live", f"kill switch: liquidating {n_liq} open live RH position(s)", {"positions": n_liq})
        return {"exited": n_exit, "liquidated": n_liq}

    def minute(self, ctx, *, run_id: str, beat_id: int, entries: list) -> dict:
        """The minute's entries, sent the moment the fly has decided (agent/fly_session.FlyBook.on_minute): the executor's
        results first (a failed buy is kept for another try), the minute's sales if the clock has not sent them, then the
        retries and the paper entries. The wallet is read only when there is something to buy; the wealth mark waits for
        ``idle``."""
        conn, m1 = ctx.conn, ctx.m1
        drained = self.ex.drain()
        for r in drained:
            if not r.ok:
                log.warning("live rh %s %s failed: %s", r.request.kind, r.request.token, r.error)
            self._settled(conn, ctx, run_id, beat_id, r)
        ex = self.exits(conn, m1=m1, run_id=run_id, beat_id=beat_id)
        # a token whose only order is its sale is not held: a buy queues behind the sale (paper sold and bought it again)
        opens = ledger.open_positions(conn, BOOK); inflight = self.ex.pending(); selling = self.ex.selling()
        held_mints = {p["mint"].lower() for p in opens if p["mint"].lower() not in selling} | (inflight - selling)
        c = rails.load_circuit(conn, RH_CIRCUIT)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if rails.breaker_blocks(c) else "paused" if c.entries_paused else None
        n_enter = n_retry = 0; skipped = []
        if self.failed or entries:
            reserved = reserved_wei(conn); native_eth = self.wallet.balance() / A.WEI; self.wallet_eth = native_eth
            committed = self.ex.committed_wei() / A.WEI
            bankroll = native_eth + sum(float(p["cost_sol"]) for p in opens if not self._dead(p, m1)) - reserved / A.WEI   # not a dead-bag ``after`` writes off
            cash = native_eth - reserved / A.WEI - committed
            stale = ctx.lag_s > config.ENTRY_MAX_LAG_S
            for did, f in list(self.failed.items()):     # failed entries first (older decisions), each against every rail again
                e = f["e"]; mint = e["mint"].lower(); over = self._retry_over(conn, ctx, did, f)
                if over:
                    self.failed.pop(did); skipped.append((mint, "retries exhausted"))
                    self._blocked(conn, ctx, run_id, beat_id, e, "retries exhausted", f"{over}; last error: {(f.get('error') or '')[:200]}", tries=f["tries"]); continue
                pool = pool_of(conn, e["info"].get("pool")); quote = ((pool or {}).get("quote_asset") or ZERO).lower()
                rail = blocked or ("stale minute" if stale else None) or ("held" if mint in held_mints else None) or ("base mark stale" if A.base_mark(conn, quote)[1] else None)
                size, why = (0.0, "") if rail else sizing.size_position(e["score"], e.get("threshold", 0.0), e.get("table") or [], bankroll, cash,
                                                                        ctx.resqs.get(e["mint"]) or e["info"]["resq"], flat=config.FLY_SIZING == "flat", market=RH)
                if rail or size <= 0:
                    skipped.append((mint, rail or why)); self._blocked(conn, ctx, run_id, beat_id, e, rail or "sizing", f"retry not sent: {rail or why}", size, retry=True); continue
                if self._buy(e, mint, quote, pool, size, retry=f["tries"]):
                    self.failed.pop(did); self.sent[did] = {**f, "tries": f["tries"] + 1}
                    n_enter += 1; n_retry += 1; cash -= size; held_mints.add(mint)
            for e in entries:
                mint = e["mint"].lower()
                if mint in held_mints:
                    skipped.append((mint, "held")); self._blocked(conn, ctx, run_id, beat_id, e, "held", "the wallet holds this token or has an order of it queued"); continue
                pool = pool_of(conn, e["info"].get("pool")); quote = ((pool or {}).get("quote_asset") or ZERO).lower()
                _, stale_mark = A.base_mark(conn, quote)
                if stale_mark:
                    skipped.append((mint, "base mark stale")); self._blocked(conn, ctx, run_id, beat_id, e, "base mark stale", f"no fresh ETH mark of its quote {quote}"); continue
                size, why = sizing.size_position(e["score"], e.get("threshold", 0.0), e.get("table") or [], bankroll, cash, e["info"]["resq"],
                                                 flat=config.FLY_SIZING == "flat", market=RH)
                if blocked or size <= 0:
                    skipped.append((mint, blocked or why)); self._blocked(conn, ctx, run_id, beat_id, e, blocked or "sizing", why, size); continue
                if self._buy(e, mint, quote, pool, size):
                    n_enter += 1; cash -= size; held_mints.add(mint); self.sent[e["decision_id"]] = {"e": e, "ts": m1, "tries": 1}
                    conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), %s) WHERE id = %s", (BOOK, e["decision_id"]))
                else:
                    skipped.append((mint, "held")); self._blocked(conn, ctx, run_id, beat_id, e, "held", "an order of this token is queued or in flight", size)
        self.due_mark = (ctx, beat_id)
        return {"stage": "trading live", "wallet_eth": self.wallet_eth, "entered": n_enter, "retried": n_retry, "exited": ex["exited"], "liquidated": ex["liquidated"],
                "in_flight": len(inflight), "blocked": blocked, "skipped": skipped[:10], "failed_orders": sum(1 for r in drained if not r.ok),
                "awaiting_retry": len(self.failed)}

    def after(self, conn, ctx, *, run_id: str, beat_id: int) -> dict:
        """The minute's bookkeeping, once every chain's live orders are out: dead-bags written off, the gap monitor."""
        n_dead = 0; inflight = self.ex.pending()
        for p in ledger.open_positions(conn, BOOK):
            held, H = self._held(p, ctx.m1)
            if held < H + DEAD_BAG_S or p["mint"].lower() in inflight:
                continue
            did = self._decision(conn, ctx.m1, run_id, beat_id, p, f"dead-bag: unsold {held / 3600:.1f} h after entry", True)
            ledger.close_position(conn, position_id=int(p["id"]), exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=did, forced_kind="dead_bag", ts=ctx.m1)
            record_event("warning", "rh_live", f"dead-bag: {p['mint']} could not be sold; written off, its tokens are swept hourly", {"position": int(p["id"])})
            n_dead += 1
        return {"dead_bags": n_dead, "gap": self.gap(conn), "wealth": (self.last_mark or {}).get("wealth"), "last_mark": self.last_mark,
                "reconcile": self.last_reconciled}

    def idle(self) -> None:
        """Between minutes, never in front of an intent (agent/live_first.py): the waiting wealth mark once the executor has
        nothing queued or in flight; mid-minute, the hourly sweep and the reconcile every ``RECONCILE_S``."""
        now = time.time()
        if self.due_mark is not None and live_first.may_mark(now) and not self.ex.pending():
            self.mark_due()
        if not live_first.mid_minute(now) or self.ex.pending():
            return
        if now - self.last_sweep >= SWEEP_S:
            self.last_sweep = now
            with transaction() as conn:
                self.sweep(conn, datetime.now(timezone.utc))
        elif now - self.last_reconcile >= RECONCILE_S and self.wallet.lock.acquire(blocking=False):
            try:
                self.last_reconcile = now
                with transaction() as conn:
                    self.last_reconciled = A.reconcile(conn, self.wallet)
            finally:
                self.wallet.lock.release()

    def mark_due(self, conn=None) -> dict | None:
        """The waiting minute's wealth mark (rh/accounting.wealth at the minute's prices) and circuit 3's kill switch on the
        book's peak. None when an intent held the wallet lock or the balance could not be read: that minute goes unmarked."""
        if self.due_mark is None:
            return None
        if conn is None:
            with transaction() as c:
                return self.mark_due(c)
        ctx, beat_id = self.due_mark
        ctx = copy.copy(ctx); ctx.conn = conn; m1 = ctx.m1         # the minute's prices, read on this transaction
        ledger.mark_positions(conn, BOOK, ctx.prices, {}, m1)
        w = self._mark(conn, ctx, reserved_wei(conn)); self.due_mark = None
        if w is None:
            return None
        rebase = rails.kill_rebase_at(conn, RH_CIRCUIT)
        pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s AND (%s::timestamptz IS NULL OR ts > %s)", (BOOK, rebase, rebase)).fetchone()
        peak = max(float(pk["pk"] or 0.0), w["wealth"])
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                     "ON CONFLICT (beat_id, book) DO NOTHING",
                     (beat_id, BOOK, m1, w["native"], w["positions"] + w["lots"], w["exit_cost"], w["wealth"], peak, (1.0 - w["wealth"] / peak) if peak > 0 else 0.0,
                      w["exposure"], w["n_open"]))
        killed = rails.check_drawdown(conn, w["wealth"], peak, circuit_id=RH_CIRCUIT, drawdown=RH.kill_drawdown(), unit="ETH")
        self.wallet_eth = w["native"]
        self.last_mark = {"minute": m1.isoformat(), "wealth": w["wealth"], "wallet_eth": w["native"], "killed": bool(killed)}
        return self.last_mark

    def _mark(self, conn, ctx, reserved: int) -> dict | None:
        """The book's wealth (rh/accounting.wealth) with the native balance and the books read together under the wallet
        lock: the executor holds it from an intent's balance snapshot to its booking, and a balance read beside a booking
        in progress counts that trade twice or not at all. Quiet coins are valued at their last known reserve and pool fee.
        None when an intent kept the lock ``MARK_WAIT_S``."""
        if not self.wallet.lock.acquire(timeout=MARK_WAIT_S):
            return None
        try:
            try:
                native = self.wallet.balance()
            except Exception as e:
                log.warning("live rh mark skipped: the wallet balance could not be read: %s", type(e).__name__)
                return None
            return A.wealth(conn, native, reserved, ctx.prices, lambda m: ctx.resqs.get(m) or ctx.last_resq(m), ctx.fee)
        finally:
            self.wallet.lock.release()

    def gap(self, conn) -> dict:
        rows = conn.execute("""SELECT l.realized_sol / NULLIF(l.cost_sol, 0) AS rl, p.realized_sol / NULLIF(p.cost_sol, 0) AS rp
                               FROM positions l JOIN positions p ON p.entry_decision_id = l.entry_decision_id AND p.book = %s AND p.status = 'closed'
                               WHERE l.book = %s AND l.status = 'closed' AND l.entry_decision_id IS NOT NULL ORDER BY l.closed_at DESC LIMIT %s""",
                            (MIRROR, BOOK, GAP_TRADES)).fetchall()
        pairs = [(r["rl"], r["rp"]) for r in rows if r["rl"] is not None and r["rp"] is not None]
        rl = np.array([a for a, _ in pairs], dtype=np.float64); rp = np.array([b for _, b in pairs], dtype=np.float64)
        out = {"trades": int(len(rl)), "live_mean": float(rl.mean()) if len(rl) else None, "mirror_mean": float(rp.mean()) if len(rp) else None}
        if len(rl) >= GAP_TRADES and rl.mean() - rp.mean() < -GAP_MAX:
            if not rails.load_circuit(conn, RH_CIRCUIT).entries_paused:
                conn.execute("UPDATE circuit_state SET entries_paused = true, updated_at = now() WHERE id = %s", (RH_CIRCUIT,))
                conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('gap_pause', %s)", (json.dumps({**out, "circuit": RH_CIRCUIT}),))
                record_event("error", "rh_live", f"RH entries paused: live trails its paper mirror by {(rp.mean() - rl.mean()) * 100:.1f} points per trade", out)
            out["paused"] = True
        return out

    def sweep(self, conn, m1) -> int:
        """Sell the last week's dead-bag tokens (their proceeds are booked onto the latest dead-bag of the token,
        rh/exec.py) and every open base lot (exit legs that stopped half way, stray base). Never a token the book holds
        again: its balance is that position's."""
        n = 0; inflight = self.ex.pending(); held = {p["mint"].lower() for p in ledger.open_positions(conn, BOOK)}
        for r in conn.execute("SELECT DISTINCT mint, pool, quote_asset FROM positions WHERE book = %s AND forced_exit_kind = 'dead_bag' AND closed_at > %s",
                              (BOOK, m1 - timedelta(days=7))).fetchall():
            if r["mint"].lower() in held or r["mint"].lower() in inflight:
                continue
            amt = self.wallet.balance_of(r["mint"])
            if amt > 0:
                n += self.ex.submit(RhRequest(kind="sweep", token=r["mint"], quote_asset=r["quote_asset"] or ZERO, amount_in=amt, slippage_bps=config.SLIPPAGE_FORCED_BPS,
                                              max_slippage_bps=config.SLIPPAGE_FORCED_BPS, pool=pool_of(conn, r["pool"]), forced_kind="dead_bag_sweep"))
        for lot in A.open_lots(conn):
            if lot["asset"] == A.NATIVE or lot["asset"].lower() in inflight:
                continue
            mark, _ = A.base_mark(conn, lot["asset"])
            value = int(lot["qty_raw"]) / 10 ** A._decimals(conn, lot["asset"]) * (mark or 0.0)
            if value < config.RH_MIN_SWEEP_ETH:
                conn.execute("UPDATE rh_base_lots SET status = 'dust', closed_at = now() WHERE id = %s", (lot["id"],)); continue
            n += self.ex.submit(RhRequest(kind="liquidate", token=lot["asset"], amount_in=int(lot["qty_raw"]), slippage_bps=config.SLIPPAGE_EXIT_BPS,
                                          max_slippage_bps=config.SLIPPAGE_FORCED_BPS, lot_id=int(lot["id"])))
        return n

    def stop(self) -> None:
        """Waits for the intent in flight (bounded, rh/exec.STOP_JOIN_S); queued intents are dropped."""
        self.ex.stop()
