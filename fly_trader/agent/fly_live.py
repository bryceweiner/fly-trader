"""Live execution for the fly once it holds the selector's seat: its decisions also trade the bot wallet (book 'live').

The paper book ``paper_fly`` keeps running as the mirror. Each trade minute — only after the handover and only while
signing is allowed (LIVE_ENABLED=1 and the live prerequisites, chain/cluster_guard.py) — ``LiveMirror.minute``:
- drains the execution worker (execution/worker.py books live positions from balance-verified fills);
- exits: live positions held the hold are sold (exit slippage, the forced ladder once ``OVERDUE_S`` late), retried every
  minute; ``DEAD_BAG_S`` past the hold a position that could not be sold is written off (dead-bag: closed at 0, so it
  leaves wealth) and its tokens are swept every ``SWEEP_S``;
- entries: every paper entry of this minute is mirrored with the same score, line and certainty bands, sized from the
  wallet (SOL balance + open live positions at cost as the bankroll, the SOL balance as cash; the gas reserve is never
  spent — agent/sizing.py), one position per token, unless the kill switch, the circuit breaker or paused entries block;
- wealth: wallet SOL + open positions net of exit cost → ``wealth_marks`` (book 'live'); the kill switch on its own peak,
  and with ``KILL_SWITCH_LIQUIDATE`` every open position is sold at the forced slippage while it stays on;
- gap: over the last ``GAP_TRADES`` closed live trades, the mean return per trade trailing the paper mirror's same
  decisions by more than ``GAP_MAX`` pauses entries and raises an alert. Learning is unaffected: it learns from market
  labels, not fills.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config
from ..chain.balances import snapshot_balances
from ..db.apilog import record_event
from ..execution import ledger
from ..execution.worker import ExecRequest
from ..market.exit_cost import exit_cost_fraction
from . import rails, sizing

log = logging.getLogger(__name__)
BOOK, MIRROR = "live", "paper_fly"
DEAD_BAG_S, OVERDUE_S, SWEEP_S = 3 * 3600.0, 600.0, 3600.0
GAP_TRADES, GAP_MAX = 50, 0.02


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

    def _decision(self, conn, ctx, run_id, beat_id, p, reason: str, forced: bool) -> int:
        return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'fly_exit',%s,%s,%s,%s) RETURNING id",
                                (beat_id, run_id, ctx.m1, p["mint"], p["pool"], float(p["cost_sol"]), forced, reason, json.dumps({"book": BOOK}))).fetchone()["id"])

    def minute(self, ctx, *, run_id: str, beat_id: int, entries: list, line: float, table: list | None) -> dict:
        conn, m1 = ctx.conn, ctx.m1
        drained = self.worker.drain()
        for r in drained:
            if not r.ok:
                log.warning("live %s %s failed: %s", r.request.side, r.request.mint, r.error)
        inflight = self.worker.pending()
        snap = snapshot_balances(self.broker.rpc, self.broker.pubkey); sol_free = snap.lamports / config.LAMPORTS_PER_SOL
        n_exit = n_dead = 0
        for p in ledger.open_positions(conn, BOOK):
            held = (m1 - p["opened_at"]).total_seconds(); H = float(p.get("hold_s") or self.H)          # each position's own hold
            if held < H or p["mint"] in inflight:
                continue
            if held >= H + DEAD_BAG_S:
                did = self._decision(conn, ctx, run_id, beat_id, p, f"dead-bag: unsold {held / 3600:.1f} h after entry", True)
                ledger.close_position(conn, position_id=int(p["id"]), exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=did, forced_kind="dead_bag", ts=m1)
                record_event("warning", "fly_live", f"dead-bag: {p['mint']} could not be sold; written off, its tokens are swept hourly", {"position": int(p["id"]), "cost_sol": float(p["cost_sol"])})
                n_dead += 1
                continue
            overdue = held >= H + OVERDUE_S
            did = self._decision(conn, ctx, run_id, beat_id, p, f"held {held / 60:.0f} min" + (" (retrying)" if overdue else ""), False)
            n_exit += bool(self.worker.submit(ExecRequest(decision_id=did, mint=p["mint"], pool=p["pool"], side="sell", amount_in=int(p["qty"]),
                                                          slippage_bps=config.SLIPPAGE_FORCED_BPS if overdue else config.SLIPPAGE_EXIT_BPS,
                                                          max_slippage_bps=config.SLIPPAGE_FORCED_BPS, decimals=int(p.get("decimals") or 6), position_id=int(p["id"]))))
        opens = ledger.open_positions(conn, BOOK); held_mints = {p["mint"] for p in opens} | set(inflight)
        c = rails.load_circuit(conn)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if c.tripped else "paused" if c.entries_paused else None
        reserved = 0.0
        if config.VAULT_ENABLED:                     # SOL owed to vault lockers stays in the wallet but is never traded
            from ..vault import nav as vault_nav
            reserved = vault_nav.reserved_lamports(conn) / config.LAMPORTS_PER_SOL
        bankroll = sol_free + sum(float(p["cost_sol"]) for p in opens) - reserved; cash = sol_free - reserved; n_enter = 0; skipped = []
        for e in entries:
            if e["mint"] in held_mints:
                skipped.append((e["mint"], "held")); continue
            size, why = sizing.size_position(e["score"], e.get("threshold", line), e.get("table") or table or [], bankroll, cash, e["info"]["resq"])
            if blocked or size <= 0:
                skipped.append((e["mint"], blocked or why)); continue
            if self.worker.submit(ExecRequest(decision_id=e["decision_id"], mint=e["mint"], pool=e["info"]["pool"], side="buy", amount_in=int(size * config.LAMPORTS_PER_SOL),
                                              slippage_bps=config.SLIPPAGE_ENTRY_BPS, max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS, decimals=int(e["info"]["decimals"] or 6), size_sol=size)):
                n_enter += 1; cash -= size; held_mints.add(e["mint"])
                conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), 'live') WHERE id = %s", (e["decision_id"],))
        ledger.mark_positions(conn, BOOK, ctx.prices, {}, m1)
        gross_v = exit_cost = exposure = 0.0
        for p in ledger.open_positions(conn, BOOK):
            px = float(p.get("last_mark_price") or p["entry_price"] or 0.0)
            gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * px
            gross_v += gross; exposure += float(p["cost_sol"])
            exit_cost += gross * exit_cost_fraction(gross, ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"]), ctx.mcap(p["mint"], px), p.get("program_label"), ctx.fees.get(p["mint"]))
        wealth = sol_free + gross_v - exit_cost
        rebase = rails.kill_rebase_at(conn)
        pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s AND (%s::timestamptz IS NULL OR ts > %s)",
                          (BOOK, rebase, rebase)).fetchone(); peak = max(float(pk["pk"] or 0.0), wealth)
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                     "ON CONFLICT (beat_id, book) DO NOTHING",
                     (beat_id, BOOK, m1, sol_free, gross_v - exit_cost, exit_cost, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure, len(opens)))
        if config.VAULT_ENABLED:                     # deposits, withdrawals and payouts are not performance
            vault_nav.mark(conn, m1, snap.lamports, int(round(gross_v * config.LAMPORTS_PER_SOL)),
                           int(round(exit_cost * config.LAMPORTS_PER_SOL)))
            ix = vault_nav.status(conn, since=rebase)
            killed = rails.check_drawdown(conn, ix["value"], ix["peak"])
        else:
            killed = rails.check_drawdown(conn, wealth, peak)
        n_liq = self.liquidate(conn, ctx, run_id, beat_id) if killed and config.KILL_SWITCH_LIQUIDATE else 0
        gap = self.gap(conn)
        if ctx.m1_epoch - self.last_sweep >= SWEEP_S:
            self.last_sweep = ctx.m1_epoch; self.sweep(conn, snap, inflight, m1)
        return {"stage": "trading live", "wallet_sol": sol_free, "wealth": wealth, "entered": n_enter, "exited": n_exit, "dead_bags": n_dead, "in_flight": len(inflight),
                "blocked": blocked, "skipped": skipped[:10], "gap": gap, "failed_orders": sum(1 for r in drained if not r.ok), "liquidated": n_liq}

    def liquidate(self, conn, ctx, run_id: str, beat_id: int) -> int:
        """Sell every open live position at the forced slippage while the kill switch is on (``KILL_SWITCH_LIQUIDATE``):
        a book that has lost ``KILL_SWITCH_DRAWDOWN`` of its peak stops holding, not only entering. Called every trade
        minute the switch stays on, so a sell that fails is retried until the book is flat; positions already in flight
        (this minute's scheduled exits included) are left to the worker."""
        n = 0; inflight = self.worker.pending()
        for p in ledger.open_positions(conn, BOOK):
            if p["mint"] in inflight:
                continue
            did = self._decision(conn, ctx, run_id, beat_id, p, "kill switch: liquidating", True)
            n += bool(self.worker.submit(ExecRequest(decision_id=did, mint=p["mint"], pool=p["pool"], side="sell", amount_in=int(p["qty"]),
                                                     slippage_bps=config.SLIPPAGE_FORCED_BPS, max_slippage_bps=config.SLIPPAGE_FORCED_BPS,
                                                     decimals=int(p.get("decimals") or 6), position_id=int(p["id"]))))
        if n:
            record_event("error", "fly_live", f"kill switch: liquidating {n} open live position(s)", {"positions": n})
        return n

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
        """Sell whatever is left of the last week's dead-bag tokens (forced slippage; proceeds reach the wallet)."""
        n = 0
        for r in conn.execute("SELECT DISTINCT mint FROM positions WHERE book = %s AND forced_exit_kind = 'dead_bag' AND closed_at > %s",
                              (BOOK, m1 - timedelta(days=7))).fetchall():
            amt = snap.token(r["mint"])
            if amt > 0 and r["mint"] not in inflight:
                n += bool(self.worker.submit(ExecRequest(decision_id=None, mint=r["mint"], pool=None, side="sell", amount_in=amt, slippage_bps=config.SLIPPAGE_FORCED_BPS,
                                                         max_slippage_bps=config.SLIPPAGE_FORCED_BPS, decimals=int(snap.decimals.get(r["mint"], 6)), forced_kind="dead_bag_sweep")))
        return n

    def stop(self) -> None:
        self.worker.stop()
