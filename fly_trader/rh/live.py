"""The Robinhood Chain fly's live book (``live_rh``): the paper RH fly's minute mirrored on the bot's ETH wallet once
the fly holds the RH seat (handover_rh) and signing is allowed (config.rh_live_prerequisites_missing, rh/guard.py).

``RhLiveMirror.minute`` — agent/fly_live.LiveMirror's shape, in ETH, through rh/exec.RhExecutor:
- drains the executor (one intent at a time; each booked by rh/accounting.py from measured balance deltas);
- exits: positions held their hold — from their entry decision's minute, as paper counts it — are sold (exit slippage,
  the forced ladder once ``OVERDUE_S`` late); ``DEAD_BAG_S`` past the hold an unsellable position is written off and its
  tokens swept hourly (what they fetch is booked onto it);
- entries: this minute's paper RH entries, sized from the wallet (native ETH + open positions at cost as the bankroll,
  native − committed buys as cash, the RH gas reserve never spent), one position per token (a buy may queue behind the
  sale of the same token), never more than ``MAX_ENTRY_CHASE`` above its paper fill price (rh/exec.py); blocked by
  circuit 3 (kill switch, a trip until ``CIRCUIT_COOLDOWN_S`` after its last failure, paused entries) or a stale mark of
  the pool's base asset (a stock outside its trading hours). A buy that failed without buying anything is sent again,
  re-sized, while its paper position is open, up to ``ENTRY_RETRY_MIN`` after its decision. Every paper entry the wallet
  did not take is a decisions row of this book (kind 'blocked', the reason as its rail, ``of_decision`` the paper entry);
- wealth: native + positions net of the Pons exit cost + base lots → ``wealth_marks`` (book live_rh), read under the
  wallet lock (the executor holds it from an intent's snapshot to its booking; while an intent keeps it the minute is not
  marked); the kill switch on its own peak (circuit 3, ``RH_KILL_SWITCH_DRAWDOWN``), liquidating with
  ``KILL_SWITCH_LIQUIDATE``;
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
MARK_WAIT_S = 2.0             # how long the wealth mark waits for the wallet lock (an intent between its snapshot and its booking)


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
        self.sent: dict[int, dict] = {}       # decision id -> {"e": its paper entry, "ts": its minute, "tries": n}: a buy queued or in flight
        self.failed: dict[int, dict] = {}     # the same for a buy that failed without buying anything: sent again while it may be

    @property
    def wallet(self):
        return self.ex.wallet

    def _decision(self, conn, ctx, run_id, beat_id, p, reason: str, forced: bool) -> int:
        return int(conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'fly_exit',%s,%s,%s,%s) RETURNING id",
                                (beat_id, run_id, ctx.m1, p["mint"], p["pool"], float(p["cost_sol"]), forced, reason, json.dumps({"book": BOOK}))).fetchone()["id"])

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

    def minute(self, ctx, *, run_id: str, beat_id: int, entries: list) -> dict:
        conn, m1 = ctx.conn, ctx.m1
        drained = self.ex.drain()
        for r in drained:
            if not r.ok:
                log.warning("live rh %s %s failed: %s", r.request.kind, r.request.token, r.error)
            self._settled(conn, ctx, run_id, beat_id, r)
        reserved = reserved_wei(conn)
        # the mark first, before this minute's intents, while none is between its swap and its booking (last minute's are
        # done): marked after them, it found the wallet lock taken on most trade minutes; and a kill switch it trips now
        # blocks this minute's entries
        ledger.mark_positions(conn, BOOK, ctx.prices, {}, m1)
        w = self._mark(conn, ctx, reserved); killed = rails.load_circuit(conn, RH_CIRCUIT).kill_switch
        if w is not None:
            rebase = rails.kill_rebase_at(conn, RH_CIRCUIT)
            pk = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s AND (%s::timestamptz IS NULL OR ts > %s)", (BOOK, rebase, rebase)).fetchone()
            peak = max(float(pk["pk"] or 0.0), w["wealth"])
            conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                         "ON CONFLICT (beat_id, book) DO NOTHING",
                         (beat_id, BOOK, m1, w["native"], w["positions"] + w["lots"], w["exit_cost"], w["wealth"], peak, (1.0 - w["wealth"] / peak) if peak > 0 else 0.0,
                          w["exposure"], w["n_open"]))
            killed = rails.check_drawdown(conn, w["wealth"], peak, circuit_id=RH_CIRCUIT, drawdown=RH.kill_drawdown(), unit="ETH")
        inflight = self.ex.pending()
        native = self.wallet.balance()
        n_exit = n_dead = 0
        for p in ledger.open_positions(conn, BOOK):
            # its own hold, from its entry decision's minute (paper's clock; an RH fill lands ~10 s after it, a lagging minute later still)
            held = (m1 - (p.get("decision_ts") or p["opened_at"])).total_seconds(); H = float(p.get("hold_s") or self.H)
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
        # a token whose only order is its sale is not held: a buy queues behind the sale (paper sold and bought it again)
        opens = ledger.open_positions(conn, BOOK); inflight = self.ex.pending(); selling = self.ex.selling()
        held_mints = {p["mint"].lower() for p in opens if p["mint"].lower() not in selling} | (inflight - selling)
        c = rails.load_circuit(conn, RH_CIRCUIT)
        blocked = "kill switch" if c.kill_switch else "circuit tripped" if rails.breaker_blocks(c) else "paused" if c.entries_paused else None
        native_eth = native / A.WEI; committed = self.ex.committed_wei() / A.WEI
        bankroll = native_eth + sum(float(p["cost_sol"]) for p in opens) - reserved / A.WEI
        cash = native_eth - reserved / A.WEI - committed; n_enter = n_retry = 0; skipped = []
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
        return {"stage": "trading live", "wallet_eth": native_eth, "wealth": w["wealth"] if w is not None else None, "entered": n_enter, "retried": n_retry,
                "exited": n_exit, "dead_bags": n_dead, "in_flight": len(inflight), "blocked": blocked, "skipped": skipped[:10], "gap": gap,
                "failed_orders": sum(1 for r in drained if not r.ok), "liquidated": n_liq, "reconcile": rec, "awaiting_retry": len(self.failed),
                **({} if w is not None else {"mark": "skipped: no consistent read of the wallet and the books (an intent mid-booking, or the balance unreadable)"})}

    def _mark(self, conn, ctx, reserved: int) -> dict | None:
        """The book's wealth (rh/accounting.wealth) with the native balance and the books read together under the wallet
        lock: the executor holds it from an intent's balance snapshot to its booking, and a balance read beside a booking
        in progress counts that trade twice or not at all. Quiet coins are valued at their last known reserve and pool fee.
        None when an intent kept the lock ``MARK_WAIT_S``: no mark this minute."""
        if not self.wallet.lock.acquire(timeout=MARK_WAIT_S):
            return None
        try:
            try:                                     # intents went out this minute: a failed read must not undo the minute's rows
                native = self.wallet.balance()
            except Exception as e:
                log.warning("live rh mark skipped: the wallet balance could not be read: %s", type(e).__name__)
                return None
            return A.wealth(conn, native, reserved, ctx.prices, lambda m: ctx.resqs.get(m) or ctx.last_resq(m), ctx.fee)
        finally:
            self.wallet.lock.release()

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
