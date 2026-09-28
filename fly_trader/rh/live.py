"""The Robinhood Chain fly's live book (``live_rh``): the paper RH fly's minute mirrored on the bot's ETH wallet once
the fly holds the RH seat (handover_rh) and signing is allowed (config.rh_live_prerequisites_missing, rh/guard.py).

``RhLiveMirror.minute`` — agent/fly_live.LiveMirror's shape, in ETH, through rh/exec.RhExecutor:
- drains the executor (one intent at a time; each booked by rh/accounting.py from measured balance deltas);
- exits: positions held their hold are sold (exit slippage, the forced ladder once ``OVERDUE_S`` late); ``DEAD_BAG_S``
  past the hold an unsellable position is written off and its tokens swept hourly;
- entries: this minute's paper RH entries, sized from the wallet (native ETH + open positions at cost as the bankroll,
  native − committed buys as cash, the RH gas reserve never spent), one position per token; blocked by circuit 3
  (kill switch, trip, paused entries) or a stale mark of the pool's base asset (a stock outside its trading hours);
- wealth: native + positions net of the Pons exit cost + base lots → ``wealth_marks`` (book live_rh); the kill switch
  on its own peak (circuit 3, ``RH_KILL_SWITCH_DRAWDOWN``), liquidating with ``KILL_SWITCH_LIQUIDATE``;
- gap: live trailing its paper mirror by more than ``GAP_MAX`` per trade pauses circuit 3's entries;
- hourly: dead-bag tokens swept, base lots sold back to ETH; every ``RECONCILE_S`` the chain is reconciled with the
  books (rh/accounting.reconcile) while no intent is in flight.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import numpy as np

from .. import config
from ..agent import rails, sizing
from ..db.apilog import record_event
from ..execution import ledger
from ..markets import RH, RH_CIRCUIT
from . import accounting as A
from .exec import ZERO, RhExecutor, RhRequest

log = logging.getLogger(__name__)
BOOK, MIRROR = RH.live_book, RH.fly_book
DEAD_BAG_S, OVERDUE_S, SWEEP_S, RECONCILE_S = 3 * 3600.0, 600.0, 3600.0, 600.0
GAP_TRADES, GAP_MAX = 50, 0.02


def minute(book, ctx, st: dict) -> dict:
    """One minute of the live RH mirror for ``book`` (agent/fly_session.FlyBook on chain 'rh')."""
    missing = config.rh_live_prerequisites_missing()
    if missing:
        return {"stage": "live gated: " + ", ".join(missing)}
    if book.mirror is None:
        book.mirror = RhLiveMirror(book.H)
        record_event("info", "rh_live", "the RH fly trades the ETH wallet", {"run_id": book.run_id})
    return book.mirror.minute(ctx, run_id=book.run_id, beat_id=st["beat_id"], entries=st["entries"])


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

    @property
    def wallet(self):
        return self.ex.wallet

    def _decision(self, conn, ctx, run_id, beat_id, p, reason: str, forced: bool) -> int:
        return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'fly_exit',%s,%s,%s,%s) RETURNING id",
                                (beat_id, run_id, ctx.m1, p["mint"], p["pool"], float(p["cost_sol"]), forced, reason, json.dumps({"book": BOOK}))).fetchone()["id"])

    def _sell(self, conn, p: dict, did: int | None, slip: float, forced_kind: str | None = None) -> bool:
        pool = pool_of(conn, p.get("pool"))
        return self.ex.submit(RhRequest(kind="sell", token=p["mint"], quote_asset=(p.get("quote_asset") or (pool or {}).get("quote_asset") or ZERO), amount_in=int(p["qty"]),
                                        slippage_bps=slip, max_slippage_bps=config.SLIPPAGE_FORCED_BPS, pool=pool, decision_id=did, position_id=int(p["id"]),
                                        forced_kind=forced_kind, decimals=int(p.get("decimals") or 18)))

    def minute(self, ctx, *, run_id: str, beat_id: int, entries: list) -> dict:
        conn, m1 = ctx.conn, ctx.m1
        drained = self.ex.drain()
        for r in drained:
            if not r.ok:
                log.warning("live rh %s %s failed: %s", r.request.kind, r.request.token, r.error)
        inflight = self.ex.pending()
        native = self.wallet.balance(); reserved = reserved_wei(conn)
        n_exit = n_dead = 0
        for p in ledger.open_positions(conn, BOOK):
            held = (m1 - p["opened_at"]).total_seconds(); H = float(p.get("hold_s") or self.H)
            if held < H or p["mint"].lower() in inflight:
                continue
            if held >= H + DEAD_BAG_S:
                did = self._decision(conn, ctx, run_id, beat_id, p, f"dead-bag: unsold {held / 3600:.1f} h after entry", True)
                ledger.close_position(conn, position_id=int(p["id"]), exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=did, forced_kind="dead_bag", ts=m1)
                record_event("warning", "rh_live", f"dead-bag: {p['mint']} could not be sold; written off, its tokens are swept hourly", {"position": int(p["id"])})
                n_dead += 1
                continue
            overdue = held >= H + OVERDUE_S
            did = self._decision(conn, ctx, run_id, beat_id, p, f"held {held / 60:.0f} min" + (" (retrying)" if overdue else ""), False)
            n_exit += self._sell(conn, p, did, config.SLIPPAGE_FORCED_BPS if overdue else config.SLIPPAGE_EXIT_BPS)
        opens = ledger.open_positions(conn, BOOK); held_mints = {p["mint"].lower() for p in opens} | inflight
        c = rails.load_circuit(conn, RH_CIRCUIT)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if c.tripped else "paused" if c.entries_paused else None
        native_eth = native / A.WEI; committed = self.ex.committed_wei() / A.WEI
        bankroll = native_eth + sum(float(p["cost_sol"]) for p in opens) - reserved / A.WEI
        cash = native_eth - reserved / A.WEI - committed; n_enter = 0; skipped = []
        for e in entries:
            mint = e["mint"].lower()
            if mint in held_mints:
                skipped.append((mint, "held")); continue
            pool = pool_of(conn, e["info"].get("pool")); quote = ((pool or {}).get("quote_asset") or ZERO).lower()
            _, stale = A.base_mark(conn, quote)
            if stale:
                skipped.append((mint, "base mark stale")); continue
            size, why = sizing.size_position(e["score"], e.get("threshold", 0.0), e.get("table") or [], bankroll, cash, e["info"]["resq"],
                                             flat=config.FLY_SIZING == "flat", market=RH)
            if blocked or size <= 0:
                skipped.append((mint, blocked or why)); continue
            if self.ex.submit(RhRequest(kind="buy", token=mint, quote_asset=quote, amount_in=int(size * A.WEI), slippage_bps=config.SLIPPAGE_ENTRY_BPS,
                                        max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS, pool=pool, decision_id=e["decision_id"], hold_s=e.get("hold_s"),
                                        strategy=e.get("strategy"), decimals=int(e["info"].get("decimals") or 18))):
                n_enter += 1; cash -= size; held_mints.add(mint)
                conn.execute("UPDATE decisions SET book_targets = array_append(COALESCE(book_targets, '{}'), %s) WHERE id = %s", (BOOK, e["decision_id"]))
        ledger.mark_positions(conn, BOOK, ctx.prices, {}, m1)
        w = A.wealth(conn, native, reserved, ctx.prices, {p["mint"]: (ctx.resqs.get(p["mint"]) or ctx.last_resq(p["mint"])) for p in opens}, ctx.fees)
        rebase = rails.kill_rebase_at(conn, RH_CIRCUIT)
        pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s AND (%s::timestamptz IS NULL OR ts > %s)", (BOOK, rebase, rebase)).fetchone()
        peak = max(float(pk["pk"] or 0.0), w["wealth"])
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                     "ON CONFLICT (beat_id, book) DO NOTHING",
                     (beat_id, BOOK, m1, native_eth, w["positions"] + w["lots"], w["exit_cost"], w["wealth"], peak, (1.0 - w["wealth"] / peak) if peak > 0 else 0.0,
                      w["exposure"], w["n_open"]))
        killed = rails.check_drawdown(conn, w["wealth"], peak, circuit_id=RH_CIRCUIT, drawdown=RH.kill_drawdown(), unit="ETH")
        n_liq = self.liquidate(conn, ctx, run_id, beat_id) if killed and config.KILL_SWITCH_LIQUIDATE else 0
        gap = self.gap(conn)
        if ctx.m1_epoch - self.last_sweep >= SWEEP_S:
            self.last_sweep = ctx.m1_epoch; self.sweep(conn, m1)
        rec = None
        if ctx.m1_epoch - self.last_reconcile >= RECONCILE_S and not self.ex.pending() and self.wallet.lock.acquire(blocking=False):
            try:
                self.last_reconcile = ctx.m1_epoch; rec = A.reconcile(conn, self.wallet)
            finally:
                self.wallet.lock.release()
        return {"stage": "trading live", "wallet_eth": native_eth, "wealth": w["wealth"], "entered": n_enter, "exited": n_exit, "dead_bags": n_dead,
                "in_flight": len(inflight), "blocked": blocked, "skipped": skipped[:10], "gap": gap, "failed_orders": sum(1 for r in drained if not r.ok),
                "liquidated": n_liq, "reconcile": rec}

    def liquidate(self, conn, ctx, run_id: str, beat_id: int) -> int:
        n = 0; inflight = self.ex.pending()
        for p in ledger.open_positions(conn, BOOK):
            if p["mint"].lower() in inflight:
                continue
            did = self._decision(conn, ctx, run_id, beat_id, p, "kill switch: liquidating", True)
            n += self._sell(conn, p, did, config.SLIPPAGE_FORCED_BPS)
        if n:
            record_event("error", "rh_live", f"kill switch: liquidating {n} open live RH position(s)", {"positions": n})
        return n

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
        """Sell the last week's dead-bag tokens and every open base lot (exit legs that stopped half way, stray base)."""
        n = 0; inflight = self.ex.pending()
        for r in conn.execute("SELECT DISTINCT mint, pool, quote_asset FROM positions WHERE book = %s AND forced_exit_kind = 'dead_bag' AND closed_at > %s",
                              (BOOK, m1 - timedelta(days=7))).fetchall():
            amt = self.wallet.balance_of(r["mint"])
            if amt > 0 and r["mint"].lower() not in inflight:
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
        self.ex.stop()
