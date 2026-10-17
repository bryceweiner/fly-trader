"""Live execution for the fly once it holds the selector's seat: its decisions also trade the bot wallet (book 'live').

The paper book ``paper_fly`` keeps running as the mirror. Only after the handover and while signing is allowed
(LIVE_ENABLED=1 and the live prerequisites, chain/cluster_guard.py), and live money first (CLAUDE.md, agent/live_first.py):
- exits (``exits``: at the minute boundary from the engine's live clock, before the stream has written the minute;
  ``minute`` sends them when the clock has not): live positions held the hold — counted from their entry decision's
  minute, as paper counts it, not from the fill 10–20 s later — are sold (exit slippage, the forced ladder once
  ``OVERDUE_S`` late), retried every minute; while the kill switch is on with ``KILL_SWITCH_LIQUIDATE`` (or a vault
  panic) every other open position is sold at the forced slippage, retried until the book is flat;
- entries (``minute``: the moment the fly has decided, before any paper trade, learning or bookkeeping of the minute):
  the execution worker's results are drained (execution/worker.py books live positions from balance-verified fills);
  every paper entry of this minute is mirrored with the same score, line and certainty bands, sized from the wallet
  (SOL balance + open live positions at cost as the bankroll, the SOL balance as cash; the gas reserve is never spent —
  agent/sizing.py), one position per token (a buy may queue behind the sale of the same token: paper sold and bought it
  again this minute), never more than ``MAX_ENTRY_CHASE`` above its paper fill price, unless the kill switch, the
  circuit breaker (until ``CIRCUIT_COOLDOWN_S`` after its last failure) or paused entries block. A buy that failed
  without moving anything is sent again, re-sized, on the next minutes while its paper position is open, up to
  ``ENTRY_RETRY_MIN`` after its decision. Every paper entry the wallet did not take is a decisions row of this book
  (kind 'blocked', the reason as its rail, ``of_decision`` the paper entry);
- bookkeeping (``after``: once every chain's live orders are out): ``DEAD_BAG_S`` past the hold a position that could
  not be sold is written off (dead-bag: closed at 0, so it leaves wealth); the gap: over the last ``GAP_TRADES`` closed
  live trades, the mean return per trade trailing the paper mirror's same decisions by more than ``GAP_MAX`` pauses
  entries and raises an alert;
- between minutes (``idle``), never in front of an order: the wealth mark — wallet SOL + open positions net of exit cost
  → ``wealth_marks`` (book 'live'), read once the worker has nothing queued or in flight (a balance read beside a
  booking counts that trade twice or not at all) and not just before a boundary; the kill switch on its own peak; and
  every ``SWEEP_S``, mid-minute, the written-off tokens are swept (what they fetch is booked onto their dead-bag).
Learning is unaffected: it learns from market labels and the book's outcomes, not from these orders.
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
from ..chain.balances import snapshot_balances
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..execution.worker import ExecRequest
from ..market.exit_cost import exit_cost_fraction
from . import live_first, rails, sizing

log = logging.getLogger(__name__)
BOOK, MIRROR = "live", "paper_fly"
DEAD_BAG_S, OVERDUE_S, SWEEP_S = 3 * 3600.0, 600.0, 3600.0
GAP_TRADES, GAP_MAX = 50, 0.02
MARK_WAIT_S = 2.0             # how long the wealth mark waits for a swap between its fill and its booking


class LiveMirror:
    def __init__(self, horizon_s: float, broker=None, worker=None):
        if broker is None:
            from ..execution.broker_live import LiveBroker
            broker = LiveBroker()
        if worker is None:
            from ..execution.worker import ExecutionWorker
            worker = ExecutionWorker(broker)
        self.broker, self.worker, self.H = broker, worker, float(horizon_s)
        self.last_sweep = 0.0
        self.sent: dict[int, dict] = {}       # decision id -> {"e": its paper entry, "ts": its minute, "tries": n}: a buy queued or in flight
        self.failed: dict[int, dict] = {}     # the same for a buy that failed without moving anything: sent again while it may be
        self.exit_lock = threading.Lock()     # the live clock's thread and the engine's both send exits: once per minute
        self.exits_through: datetime | None = None
        self.due_mark: tuple | None = None    # (the minute, its beat id): the wealth mark waiting for an idle worker (``idle``)
        self.last_mark: dict | None = None
        self.wallet_sol: float | None = None

    def _decision(self, conn, m1, run_id, beat_id, p, reason: str, forced: bool) -> int:
        return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'fly_exit',%s,%s,%s,%s) RETURNING id",
                                (beat_id, run_id, m1, p["mint"], p["pool"], float(p["cost_sol"]), forced, reason, json.dumps({"book": BOOK}))).fetchone()["id"])

    def _blocked(self, conn, ctx, run_id, beat_id, e: dict, rail: str, reason: str, size: float = 0.0, **detail) -> None:
        """A paper entry the wallet did not take (or not yet): a decisions row of this book, so live and paper can be told apart."""
        conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                     (beat_id, run_id, ctx.m1, e["mint"], e["info"].get("pool"), float(e["score"]), float(size), rail, reason,
                      json.dumps({"book": BOOK, "of_decision": e["decision_id"], **detail}, default=str)))

    def _settled(self, conn, ctx, run_id, beat_id, r) -> None:
        """A drained buy this mirror sent: done when it filled; else recorded, and kept for another try when it moved
        nothing (no fill, no unknown outcome)."""
        did = r.request.decision_id
        f = self.sent.pop(did, None) if r.request.side == "buy" and did is not None else None
        if f is None or r.ok:
            return
        rail = "price ran" if (r.error or "").startswith("price ran") else "entry failed"
        self._blocked(conn, ctx, run_id, beat_id, f["e"], rail, (r.error or "failed")[:300], r.request.size_sol, attempt=f["tries"], retry=bool(r.retryable))
        if r.retryable:
            self.failed[did] = {**f, "error": r.error}

    def _retry_over(self, conn, ctx, did: int, f: dict) -> str | None:
        """Why a failed entry is not tried again (None: it may be)."""
        if (ctx.m1 - f["ts"]).total_seconds() > config.ENTRY_RETRY_MIN * 60.0:
            return f"not filled within {config.ENTRY_RETRY_MIN:g} min of its decision"
        if not conn.execute("SELECT 1 FROM positions WHERE book = %s AND entry_decision_id = %s AND status = 'open'", (MIRROR, did)).fetchone():
            return "its paper position is closed"
        return None

    def _buy(self, e: dict, size: float, retry: int = 0) -> bool:
        return self.worker.submit(ExecRequest(decision_id=e["decision_id"], mint=e["mint"], pool=e["info"]["pool"], side="buy", amount_in=int(size * config.LAMPORTS_PER_SOL),
                                              slippage_bps=config.SLIPPAGE_ENTRY_BPS, max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS, decimals=int(e["info"]["decimals"] or 6),
                                              size_sol=size, hold_s=e.get("hold_s"), strategy=e.get("strategy"), ref_price=e.get("ref_price"), retry=retry))

    def _sell(self, p: dict, did: int, slip: int) -> bool:
        return self.worker.submit(ExecRequest(decision_id=did, mint=p["mint"], pool=p["pool"], side="sell", amount_in=int(p["qty"]), slippage_bps=slip,
                                              max_slippage_bps=config.SLIPPAGE_FORCED_BPS, decimals=int(p.get("decimals") or 6), position_id=int(p["id"])))

    def _held(self, p: dict, m1: datetime) -> tuple[float, float]:
        """(seconds held, its hold): from its entry decision's minute (paper's clock), not from its fill."""
        return (m1 - (p.get("decision_ts") or p["opened_at"])).total_seconds(), float(p.get("hold_s") or self.H)

    def _dead(self, p: dict, m1: datetime) -> bool:
        held, H = self._held(p, m1)
        return held >= H + DEAD_BAG_S

    def exits(self, conn, *, m1: datetime, run_id: str, beat_id: int | None = None) -> dict:
        """The sales due at minute ``m1``, sent once per minute: by the live clock at the boundary (agent/live_first.py),
        or by ``minute`` when the clock has not. A position past ``DEAD_BAG_S`` is left to ``after`` (written off), one in
        flight to the worker."""
        n_exit = n_liq = 0
        with self.exit_lock:
            if self.exits_through is not None and m1 <= self.exits_through:
                return {"exited": 0, "liquidated": 0}
            self.exits_through = m1
            inflight = self.worker.pending(); killed = rails.load_circuit(conn).kill_switch; panic = False
            if killed and config.VAULT_ENABLED:      # Telegram /panic: sell everything whatever KILL_SWITCH_LIQUIDATE says
                from ..vault import state as vault_state
                panic = bool(vault_state.get("panic"))
            liquidate = killed and (config.KILL_SWITCH_LIQUIDATE or panic)
            for p in ledger.open_positions(conn, BOOK):
                if p["mint"] in inflight:
                    continue
                held, H = self._held(p, m1)
                if H <= held < H + DEAD_BAG_S:
                    overdue = held >= H + OVERDUE_S
                    did = self._decision(conn, m1, run_id, beat_id, p, f"held {held / 60:.0f} min" + (" (retrying)" if overdue else ""), False)
                    n_exit += self._sell(p, did, config.SLIPPAGE_FORCED_BPS if overdue else config.SLIPPAGE_EXIT_BPS)
                elif liquidate and held < H:         # the kill switch stops holding, not only entering
                    did = self._decision(conn, m1, run_id, beat_id, p, "kill switch: liquidating", True)
                    n_liq += self._sell(p, did, config.SLIPPAGE_FORCED_BPS)
        if n_liq:
            record_event("error", "fly_live", f"kill switch: liquidating {n_liq} open live position(s)", {"positions": n_liq})
        return {"exited": n_exit, "liquidated": n_liq}

    def minute(self, ctx, *, run_id: str, beat_id: int, entries: list, line: float, table: list | None) -> dict:
        """The minute's entries, sent the moment the fly has decided (agent/fly_session.FlyBook.on_minute): the worker's
        results first (a failed buy is kept for another try), the minute's sales if the clock has not sent them, then the
        retries and the paper entries. The wallet is read only when there is something to buy; the wealth mark waits for
        ``idle``."""
        conn, m1 = ctx.conn, ctx.m1
        drained = self.worker.drain()
        for r in drained:
            if not r.ok:
                log.warning("live %s %s failed: %s", r.request.side, r.request.mint, r.error)
            self._settled(conn, ctx, run_id, beat_id, r)
        ex = self.exits(conn, m1=m1, run_id=run_id, beat_id=beat_id)
        # a token whose only order is its sale is not held: a buy queues behind the sale (paper sold and bought it again)
        opens = ledger.open_positions(conn, BOOK); inflight = self.worker.pending(); selling = self.worker.selling()
        held_mints = {p["mint"] for p in opens if p["mint"] not in selling} | (inflight - selling)
        c = rails.load_circuit(conn)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if rails.breaker_blocks(c) else "paused" if c.entries_paused else None
        n_enter = n_retry = 0; skipped = []
        if self.failed or entries:
            sol_free = self.broker.rpc.get_balance(self.broker.pubkey) / config.LAMPORTS_PER_SOL; self.wallet_sol = sol_free
            reserved = treasury = 0.0
            if config.VAULT_ENABLED:                 # SOL owed to vault lockers is never traded; the treasury's idle SOL is bankroll
                from ..vault import nav as vault_nav
                reserved = vault_nav.reserved_in_trading(conn) / config.LAMPORTS_PER_SOL
                treasury = vault_nav.treasury_free(conn) / config.LAMPORTS_PER_SOL
            # sized on the whole book (not a dead-bag ``after`` writes off this minute), spent only from the trading wallet's own float (cash)
            bankroll = sol_free + sum(float(p["cost_sol"]) for p in opens if not self._dead(p, m1)) - reserved + treasury; cash = sol_free - reserved
            stale = ctx.lag_s > config.ENTRY_MAX_LAG_S
            for did, f in list(self.failed.items()):     # failed entries first (older decisions), each against every rail again
                e = f["e"]; over = self._retry_over(conn, ctx, did, f)
                if over:
                    self.failed.pop(did); skipped.append((e["mint"], "retries exhausted"))
                    self._blocked(conn, ctx, run_id, beat_id, e, "retries exhausted", f"{over}; last error: {(f.get('error') or '')[:200]}", tries=f["tries"]); continue
                rail = blocked or ("stale minute" if stale else None) or ("held" if e["mint"] in held_mints else None)
                size, why = (0.0, "") if rail else sizing.size_position(e["score"], e.get("threshold", line), e.get("table") or table or [], bankroll, cash,
                                                                        ctx.resqs.get(e["mint"]) or e["info"]["resq"], flat=config.FLY_SIZING == "flat")
                if rail or size <= 0:
                    skipped.append((e["mint"], rail or why)); self._blocked(conn, ctx, run_id, beat_id, e, rail or "sizing", f"retry not sent: {rail or why}", size, retry=True); continue
                if self._buy(e, size, retry=f["tries"]):
                    self.failed.pop(did); self.sent[did] = {**f, "tries": f["tries"] + 1}
                    n_enter += 1; n_retry += 1; cash -= size; held_mints.add(e["mint"])
            for e in entries:
                if e["mint"] in held_mints:
                    skipped.append((e["mint"], "held")); self._blocked(conn, ctx, run_id, beat_id, e, "held", "the wallet holds this token or has an order of it queued"); continue
                size, why = sizing.size_position(e["score"], e.get("threshold", line), e.get("table") or table or [], bankroll, cash, e["info"]["resq"],
                                                 flat=config.FLY_SIZING == "flat")
                if blocked or size <= 0:
                    skipped.append((e["mint"], blocked or why)); self._blocked(conn, ctx, run_id, beat_id, e, blocked or "sizing", why, size); continue
                if self._buy(e, size):
                    n_enter += 1; cash -= size; held_mints.add(e["mint"]); self.sent[e["decision_id"]] = {"e": e, "ts": m1, "tries": 1}
                    conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), 'live') WHERE id = %s", (e["decision_id"],))
                else:
                    skipped.append((e["mint"], "held")); self._blocked(conn, ctx, run_id, beat_id, e, "held", "an order of this token is queued or in flight", size)
        self.due_mark = (ctx, beat_id)
        return {"stage": "trading live", "wallet_sol": self.wallet_sol, "entered": n_enter, "retried": n_retry, "exited": ex["exited"], "liquidated": ex["liquidated"],
                "in_flight": len(inflight), "blocked": blocked, "skipped": skipped[:10], "failed_orders": sum(1 for r in drained if not r.ok),
                "awaiting_retry": len(self.failed)}

    def after(self, conn, ctx, *, run_id: str, beat_id: int) -> dict:
        """The minute's bookkeeping, once every chain's live orders are out: dead-bags written off, the gap monitor."""
        n_dead = 0; inflight = self.worker.pending()
        for p in ledger.open_positions(conn, BOOK):
            held, H = self._held(p, ctx.m1)
            if held < H + DEAD_BAG_S or p["mint"] in inflight:
                continue
            did = self._decision(conn, ctx.m1, run_id, beat_id, p, f"dead-bag: unsold {held / 3600:.1f} h after entry", True)
            ledger.close_position(conn, position_id=int(p["id"]), exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=did, forced_kind="dead_bag", ts=ctx.m1)
            record_event("warning", "fly_live", f"dead-bag: {p['mint']} could not be sold; written off, its tokens are swept hourly", {"position": int(p["id"]), "cost_sol": float(p["cost_sol"])})
            n_dead += 1
        return {"dead_bags": n_dead, "gap": self.gap(conn), "wealth": (self.last_mark or {}).get("wealth"), "last_mark": self.last_mark}

    def idle(self) -> None:
        """Between minutes, never in front of an order (agent/live_first.py): the waiting wealth mark once the worker has
        nothing queued or in flight, and every ``SWEEP_S``, mid-minute, the dead-bag sweep."""
        now = time.time()
        if self.due_mark is not None and live_first.may_mark(now) and not self.worker.pending():
            self.mark_due()
        if now - self.last_sweep >= SWEEP_S and live_first.mid_minute(now) and not self.worker.pending():
            self.last_sweep = now
            snap = snapshot_balances(self.broker.rpc, self.broker.pubkey)
            with transaction() as conn:
                self.sweep(conn, snap, self.worker.pending(), datetime.now(timezone.utc))

    def mark_due(self, conn=None) -> dict | None:
        """The waiting minute's wealth mark: wallet SOL + open positions net of exit cost at the minute's prices, and the
        kill switch on the book's peak. None when a swap was between its fill and its booking or the balance could not be
        read: that minute goes unmarked."""
        if self.due_mark is None:
            return None
        if conn is None:
            with transaction() as c:
                return self.mark_due(c)
        ctx, beat_id = self.due_mark
        ctx = copy.copy(ctx); ctx.conn = conn; m1 = ctx.m1         # the minute's prices, read on this transaction
        ledger.mark_positions(conn, BOOK, ctx.prices, {}, m1)
        mark = self._mark(conn, ctx); self.due_mark = None
        if mark is None:
            return None
        rebase = rails.kill_rebase_at(conn)
        lamports, gross_v, exit_cost, exposure, n_open = mark
        sol_now = lamports / config.LAMPORTS_PER_SOL; wealth = sol_now + gross_v - exit_cost
        pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s AND (%s::timestamptz IS NULL OR ts > %s)",
                          (BOOK, rebase, rebase)).fetchone(); peak = max(float(pk["pk"] or 0.0), wealth)
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                     "ON CONFLICT (beat_id, book) DO NOTHING",
                     (beat_id, BOOK, m1, sol_now, gross_v - exit_cost, exit_cost, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure, n_open))
        if config.VAULT_ENABLED:                     # deposits, withdrawals and payouts are not performance
            from ..vault import nav as vault_nav, payout as vault_payout     # trading wallet + treasury are one book: a top-up or sweep is no loss
            vault_nav.mark(conn, m1, lamports + vault_payout.cached_balance(), int(round(gross_v * config.LAMPORTS_PER_SOL)),
                           int(round(exit_cost * config.LAMPORTS_PER_SOL)))
            ix = vault_nav.status(conn, since=rebase)
            killed = rails.check_drawdown(conn, ix["value"], ix["peak"], unit="index")
        else:
            killed = rails.check_drawdown(conn, wealth, peak)
        self.wallet_sol = sol_now
        self.last_mark = {"minute": m1.isoformat(), "wealth": wealth, "wallet_sol": sol_now, "killed": bool(killed)}
        return self.last_mark

    def _mark(self, conn, ctx) -> tuple | None:
        """(wallet lamports, open positions' gross value, their exit cost, exposure at cost, open positions), read while no
        swap is between its fill and its booking (the worker's ``busy``): a balance read beside a booking in progress
        counts that trade twice or not at all. None when the worker stayed busy ``MARK_WAIT_S``."""
        if not self.worker.busy.acquire(timeout=MARK_WAIT_S):
            return None
        try:
            try:
                lamports = int(self.broker.rpc.get_balance(self.broker.pubkey))
            except Exception as e:
                log.warning("live mark skipped: the wallet balance could not be read: %s", type(e).__name__)
                return None
            opens = ledger.open_positions(conn, BOOK); gross_v = exit_cost = exposure = 0.0
            for p in opens:
                px = float(p.get("last_mark_price") or p["entry_price"] or 0.0)
                gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * px
                gross_v += gross; exposure += float(p["cost_sol"])
                exit_cost += gross * exit_cost_fraction(gross, ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"]), ctx.mcap(p["mint"], px), p.get("program_label"), ctx.fee(p["mint"]))
            return lamports, gross_v, exit_cost, exposure, len(opens)
        finally:
            self.worker.busy.release()

    def gap(self, conn) -> dict:
        """Live vs its paper mirror on the same decisions; pauses entries when live trails by more than ``GAP_MAX`` per trade."""
        rows = conn.execute(f"""SELECT l.realized_sol / NULLIF(l.cost_sol, 0) AS rl, p.realized_sol / NULLIF(p.cost_sol, 0) AS rp
                               FROM positions l JOIN positions p ON p.entry_decision_id = l.entry_decision_id AND p.book = '{MIRROR}' AND p.status = 'closed'
                               WHERE l.book = '{BOOK}' AND l.status = 'closed' AND l.entry_decision_id IS NOT NULL ORDER BY l.closed_at DESC LIMIT %s""", (GAP_TRADES,)).fetchall()
        rl = np.array([r["rl"] for r in rows if r["rl"] is not None and r["rp"] is not None], dtype=np.float64)
        rp = np.array([r["rp"] for r in rows if r["rl"] is not None and r["rp"] is not None], dtype=np.float64)
        out = {"trades": int(len(rl)), "live_mean": float(rl.mean()) if len(rl) else None, "mirror_mean": float(rp.mean()) if len(rp) else None}
        if len(rl) >= GAP_TRADES and rl.mean() - rp.mean() < -GAP_MAX:
            c = rails.load_circuit(conn)
            if not c.entries_paused:
                conn.execute("UPDATE circuit_state SET entries_paused = true, updated_at = now() WHERE id = 1")
                conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('gap_pause', %s)", (json.dumps(out),))
                record_event("error", "fly_live", f"entries paused: live trails its paper mirror by {(rp.mean() - rl.mean()) * 100:.1f} points per trade over {len(rl)} trades", out)
            out["paused"] = True
        return out

    def sweep(self, conn, snap, inflight: set, m1: datetime) -> int:
        """Sell whatever is left of the last week's dead-bag tokens (forced slippage; the proceeds are booked onto the
        latest dead-bag of the mint, execution/worker.py). Never a token the book holds again: its balance is that position's."""
        n = 0; held = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
        for r in conn.execute("SELECT DISTINCT mint FROM positions WHERE book = %s AND forced_exit_kind = 'dead_bag' AND closed_at > %s",
                              (BOOK, m1 - timedelta(days=7))).fetchall():
            amt = snap.token(r["mint"])
            if amt > 0 and r["mint"] not in inflight and r["mint"] not in held:
                n += bool(self.worker.submit(ExecRequest(decision_id=None, mint=r["mint"], pool=None, side="sell", amount_in=amt, slippage_bps=config.SLIPPAGE_FORCED_BPS,
                                                         max_slippage_bps=config.SLIPPAGE_FORCED_BPS, decimals=int(snap.decimals.get(r["mint"], 6)), forced_kind="dead_bag_sweep")))
        return n

    def stop(self) -> None:
        """Waits for the order in flight (bounded, execution/worker.STOP_JOIN_S); queued orders are dropped."""
        self.worker.stop()
