"""The RH executor: swap intents for the live RH book, one at a time, on the bot's ETH wallet.

An intent (``rh_intents``) is one decision's buy or sell, a sweep of stray tokens, or the liquidation of a base lot.
``execute`` runs it under the wallet lock, so exactly one intent — and one transaction — is in flight:

1. the intent row is written with the request and a snapshot of the wallet's balances (native ETH, the token, the
   quote's base asset) and committed before anything is signed;
2. route: rh/router.plan_swap — KyberSwap atomic (a USDG- or stock-quoted coin is bought ETH→base→meme in one
   transaction), the Uniswap v4 single-pool swap when Kyber has none (ETH-quoted pools only: the pool is the whole
   route). A non-ETH-quoted coin Kyber cannot route takes two legs: ETH→base (Kyber), then base→meme (Kyber or the
   pool); a sale meme→base, then base→ETH;
3. exact approvals and pre-calls, then the swap is simulated (eth_call as the wallet) before it is signed; a sale that
   reverts climbs the slippage ladder (``SLIPPAGE_STEP_BPS`` up to the request's maximum);
4. the result is measured: native and token balance deltas against the snapshot (+ the gas every transaction of the
   intent paid), token deltas cross-checked with the receipt's Transfer logs; rh/accounting.py books it. Base left
   over after a two-leg route that stopped half way becomes a lot (an exit leg's lot stays with its position and is
   liquidated later; an entry leg's is a stray lot).

``recover`` (on start, before the first new intent): every transaction without a receipt applied is settled — mined
(its receipt recorded), superseded at its nonce, rebroadcast (the same bytes, under 5 minutes old) or cancelled at its
nonce — then every unfinished intent is finished from its snapshot: nothing else moved the wallet in between, so the
deltas are that intent's. A crash at any step books each intent exactly once.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..vault.evm import EvmRpcError
from . import accounting as A, kyber as K, router

log = logging.getLogger(__name__)
ZERO = "0x" + "00" * 20
REBROADCAST_S = 300.0
TERMINAL = ("done", "failed", "aborted")


class SwapReverted(RuntimeError):
    pass


class NoReceipt(RuntimeError):
    """Sent, no receipt within the wait: left to ``recover`` (the intent stays open, its token pending)."""


@dataclass
class RhRequest:
    kind: str                                  # 'buy' | 'sell' | 'sweep' | 'liquidate'
    token: str                                 # the memecoin (a base asset for 'liquidate')
    quote_asset: str = ZERO                    # the pool's quote (ZERO: native ETH)
    amount_in: int = 0                         # wei for a buy; raw token units otherwise
    slippage_bps: float = 150.0
    max_slippage_bps: float = 300.0
    pool: dict | None = None                   # rh_pools row: currency0/1, fee, tick_spacing, hooks, pool_id
    decision_id: int | None = None
    position_id: int | None = None
    lot_id: int | None = None
    hold_s: float | None = None
    strategy: str | None = None
    forced_kind: str | None = None
    decimals: int = 18


@dataclass
class RhResult:
    request: RhRequest
    ok: bool
    error: str | None = None
    position_id: int | None = None
    realized: float | None = None
    route: str | None = None
    detail: dict = field(default_factory=dict)


def _native(a: str | None) -> bool:
    return (a or ZERO).lower() in (ZERO, K.NATIVE, A.NATIVE)


class RhExecutor:
    def __init__(self, wallet, planner=None, start: bool = True, recover: bool = True):
        self.wallet = wallet; self.planner = planner or router.plan_swap
        self.q: queue.Queue = queue.Queue(); self.results: list[RhResult] = []; self._pending: dict[str, RhRequest] = {}
        self._mu = threading.Lock(); self._stop = threading.Event(); self.thread = None
        if recover:
            self.recover()
        if start:
            self.thread = threading.Thread(target=self._loop, name="rh-exec", daemon=True); self.thread.start()

    # ---- queue
    def submit(self, req: RhRequest) -> bool:
        key = req.token.lower()
        with self._mu:
            if key in self._pending:
                return False
            self._pending[key] = req
        self.q.put(req)
        return True

    def pending(self) -> set[str]:
        with self._mu:
            return set(self._pending)

    def committed_wei(self) -> int:
        """ETH promised to buys queued or in flight (not yet in the balance)."""
        with self._mu:
            return sum(r.amount_in for r in self._pending.values() if r.kind == "buy")

    def drain(self) -> list[RhResult]:
        with self._mu:
            out, self.results = self.results, []
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                req = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            res = self.execute(req)
            if (res.error or "").startswith("no receipt"):                 # sent, unconfirmed: settle it now (rebroadcast, cancel, or book)
                try:
                    self.recover()
                except Exception:
                    log.exception("rh recovery after a missing receipt failed; the intent stays open until the next start")
                continue
            with self._mu:
                self.results.append(res); self._pending.pop(req.token.lower(), None)

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=30)

    # ---- one intent
    def execute(self, req: RhRequest) -> RhResult:
        with self.wallet.lock:
            iid = self._open_intent(req)
            if iid is None:
                return RhResult(req, False, "this decision already has an intent")
            try:
                if req.kind == "buy":
                    return self._buy(req, iid)
                return self._sell(req, iid)
            except NoReceipt as e:
                return RhResult(req, False, f"no receipt: {e}")
            except Exception as e:                                          # noqa: BLE001 — every failure is booked, then reported
                log.warning("rh %s %s failed: %s", req.kind, req.token, e)
                with transaction() as conn:
                    mined = conn.execute("SELECT count(*) AS n FROM rh_txs WHERE intent_id = %s AND kind = 'swap' AND status = 'mined_ok'", (iid,)).fetchone()["n"]
                if mined:                                                   # part of the route ran: book what the balances show
                    res = self._finish(req, iid, "partial")
                    res.error = res.error or str(e)[:300]
                    return res
                with transaction() as conn:
                    conn.execute("UPDATE rh_intents SET state = 'failed', error = %s, updated_at = now() WHERE id = %s", (str(e)[:300], iid))
                    gas = A.charge_gas(conn, iid, req.position_id if req.kind == "sell" else None)
                    conn.execute("UPDATE rh_txs SET applied_at = now() WHERE intent_id = %s AND applied_at IS NULL AND status IN ('mined_ok', 'reverted')", (iid,))
                return RhResult(req, False, str(e)[:300], detail={"gas_eth": gas})

    def _open_intent(self, req: RhRequest) -> int | None:
        assets = [req.token] + ([] if _native(req.quote_asset) else [req.quote_asset])
        pre = {"native": self.wallet.balance(), **{a.lower(): self.wallet.balance_of(a) for a in assets}}
        committed = req.amount_in if req.kind == "buy" else 0
        with transaction() as conn:
            r = conn.execute("INSERT INTO rh_intents (book, decision_id, position_id, kind, token, quote_asset, amount_in_raw, committed_wei, plan) "
                             "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (decision_id, kind) WHERE decision_id IS NOT NULL DO NOTHING RETURNING id",
                             (A.BOOK, req.decision_id, req.position_id, req.kind, req.token.lower(), (req.quote_asset or ZERO).lower(), req.amount_in, committed,
                              json.dumps({"req": asdict(req), "pre": {k: str(v) for k, v in pre.items()}}))).fetchone()
        return int(r["id"]) if r else None

    def _route(self, iid: int, route: str, state: str | None = None) -> None:
        with transaction() as conn:
            conn.execute("UPDATE rh_intents SET route = %s, state = COALESCE(%s, state), attempts = attempts + 1, updated_at = now() WHERE id = %s",
                         (route, state, iid))

    def _run_plan(self, plan, iid: int, position_id: int | None) -> dict:
        for token, spender, amt in plan.approvals:
            self.wallet.approve_exact(token, spender, amt, intent_id=iid, position_id=position_id)
        for c in plan.pre_calls:
            r = self.wallet.send_and_wait(to=c.to, data=c.data, value=c.value, kind=c.kind, intent_id=iid, position_id=position_id)
            if r.get("ok") is None:
                raise NoReceipt(r["hash"])
            if not r["ok"]:
                raise SwapReverted(f"{c.kind} before the swap reverted")
        try:
            self.wallet.simulate(plan.swap.to, plan.swap.data, plan.swap.value)
        except EvmRpcError as e:
            raise SwapReverted(f"simulation: {str(e)[:200]}") from None
        r = self.wallet.send_and_wait(to=plan.swap.to, data=plan.swap.data, value=plan.swap.value, kind="swap", intent_id=iid, position_id=position_id,
                                      router=plan.route, quote=plan.quote, build=plan.build)
        if r.get("ok") is None:
            raise NoReceipt(r["hash"])
        if not r["ok"]:
            raise SwapReverted(f"swap {r['hash']} reverted")
        return r

    def _plan(self, token_in: str, token_out: str, amount: int, slip: float, pool: dict | None):
        return self.planner(self.wallet.address, token_in, token_out, amount, slip, rpc=self.wallet.rpc, pool=pool)

    def _ladder(self, req: RhRequest) -> list[float]:
        out, s = [], float(req.slippage_bps)
        while s < req.max_slippage_bps:
            out.append(s); s += config.SLIPPAGE_STEP_BPS
        return out + [float(req.max_slippage_bps)]

    def _buy(self, req: RhRequest, iid: int) -> RhResult:
        eth_q = _native(req.quote_asset); route = "atomic"
        try:
            plan = self._plan(K.NATIVE, req.token, req.amount_in, req.slippage_bps, req.pool if eth_q else None)
        except router.NoRoute:
            if eth_q:
                raise
            plan = None
        if plan is not None:
            self._route(iid, plan.route, "leg1"); route = plan.route
            self._run_plan(plan, iid, None)
        else:                                                               # two legs: ETH → base, base → meme
            self._route(iid, "two_leg", "leg1"); route = "two_leg"
            p1 = self._plan(K.NATIVE, req.quote_asset, req.amount_in, req.slippage_bps, None)
            self._run_plan(p1, iid, None)
            got = self.wallet.balance_of(req.quote_asset) - self._pre(iid).get(req.quote_asset.lower(), 0)
            self._route(iid, "two_leg", "leg2")
            try:
                self._run_plan(self._plan(req.quote_asset, req.token, got, req.slippage_bps, req.pool), iid, None)
            except (SwapReverted, router.NoRoute, EvmRpcError) as e:          # the meme leg failed: the base goes back to ETH
                log.warning("rh two-leg buy of %s stopped after the base leg: %s", req.token, e)
                try:
                    self._run_plan(self._plan(req.quote_asset, K.NATIVE, got, config.SLIPPAGE_FORCED_BPS, None), iid, None)
                except (SwapReverted, router.NoRoute, EvmRpcError) as e2:
                    log.warning("rh: returning the base of %s to ETH failed too (%s): it stays a lot", req.token, e2)
        return self._finish(req, iid, route)

    def _sell(self, req: RhRequest, iid: int) -> RhResult:
        eth_q = _native(req.quote_asset); last = None
        if req.kind == "liquidate":                                          # a base lot back to ETH (Kyber)
            eth_q = True
        for slip in self._ladder(req):
            try:
                plan = self._plan(req.token, K.NATIVE, req.amount_in, slip, req.pool if eth_q and req.kind != "liquidate" else None)
            except router.NoRoute:
                if eth_q:
                    raise
                return self._sell_two_leg(req, iid)
            self._route(iid, plan.route, "leg1")
            try:
                self._run_plan(plan, iid, req.position_id)
                return self._finish(req, iid, plan.route)
            except SwapReverted as e:
                last = e; log.info("rh sell %s reverted at %.0f bps: %s", req.token, slip, e)
        raise last or SwapReverted("no attempt")

    def _sell_two_leg(self, req: RhRequest, iid: int) -> RhResult:
        self._route(iid, "two_leg", "leg1")
        self._run_plan(self._plan(req.token, req.quote_asset, req.amount_in, req.max_slippage_bps, req.pool), iid, req.position_id)
        got = self.wallet.balance_of(req.quote_asset) - self._pre(iid).get(req.quote_asset.lower(), 0)
        self._route(iid, "two_leg", "leg2")
        try:
            self._run_plan(self._plan(req.quote_asset, K.NATIVE, got, req.max_slippage_bps, None), iid, req.position_id)
        except (SwapReverted, router.NoRoute, EvmRpcError) as e:               # the base stays an exit lot of this position
            log.warning("rh two-leg sale of %s: the base leg failed (%s); its base is kept as a lot", req.token, e)
        return self._finish(req, iid, "two_leg")

    def _pre(self, iid: int) -> dict:
        with transaction() as conn:
            r = conn.execute("SELECT plan FROM rh_intents WHERE id = %s", (iid,)).fetchone()
        return {k: int(v) for k, v in (r["plan"] or {}).get("pre", {}).items()}

    # ---- measure and book
    def _finish(self, req: RhRequest, iid: int, route: str | None) -> RhResult:
        pre = self._pre(iid); tok = req.token.lower(); base = None if _native(req.quote_asset) or req.kind == "liquidate" else req.quote_asset.lower()
        post_native = self.wallet.balance(); post_tok = self.wallet.balance_of(tok)
        post_base = self.wallet.balance_of(base) if base else 0
        now = datetime.now(timezone.utc)
        with transaction() as conn:
            gas = A.intent_gas_wei(conn, iid)
            eth_delta = post_native - pre["native"] + gas                   # ETH the swaps moved (gas put back)
            tok_delta = post_tok - pre.get(tok, 0)
            base_left = (post_base - pre.get(base, 0)) if base else 0
            swaps = conn.execute("SELECT id, hash FROM rh_txs WHERE intent_id = %s AND kind = 'swap' AND status = 'mined_ok' ORDER BY id", (iid,)).fetchall()
            tx_id = int(swaps[-1]["id"]) if swaps else None
            left_eth = 0.0
            if base_left > 0:
                mark, _ = A.base_mark(conn, base)
                left_eth = base_left / 10 ** A._decimals(conn, base) * (mark or 0.0)
            realized = None; pid = req.position_id
            if req.kind == "buy":
                spent = -eth_delta
                A.add_leg(conn, intent_id=iid, tx_id=tx_id, position_id=None, leg_no=1, asset_in=A.NATIVE, asset_out=tok, amount_in=spent, amount_out=max(tok_delta, 0),
                          eth_in=spent / A.WEI, eth_out=None)
                if base_left > 0:                                          # an entry leg's leftover base: a stray lot, not position cost
                    A.open_lot(conn, asset=base, qty_raw=base_left, basis_eth=left_eth, source="stray", intent_id=iid)
                pid, realized = A.book_buy(conn, intent_id=iid, token=tok, pool=(req.pool or {}).get("pool_id"), quote_asset=(req.quote_asset or ZERO).lower(),
                                           eth_spent_wei=int(spent - left_eth * A.WEI), tokens_raw=max(tok_delta, 0), decision_id=req.decision_id, ts=now,
                                           hold_s=req.hold_s, strategy=req.strategy, decimals=req.decimals)
            else:
                sold = -tok_delta
                A.add_leg(conn, intent_id=iid, tx_id=tx_id, position_id=req.position_id, leg_no=1, asset_in=tok, asset_out=A.NATIVE, amount_in=max(sold, 0),
                          amount_out=max(eth_delta, 0), eth_in=None, eth_out=max(eth_delta, 0) / A.WEI)
                if req.kind == "liquidate" and req.lot_id is not None:
                    realized = A.liquidate_lot(conn, req.lot_id, max(eth_delta, 0) / A.WEI, gas / A.WEI)
                elif req.position_id is not None:
                    pos = next((p for p in ledger.open_positions(conn, A.BOOK) if int(p["id"]) == req.position_id), None)
                    if pos is not None:
                        if base_left > 0:                                  # an exit leg's leftover base stays with the position as a lot
                            A.open_lot(conn, asset=base, qty_raw=base_left, basis_eth=left_eth, source="exit_leg", position_id=req.position_id, intent_id=iid)
                        realized = A.book_sell(conn, intent_id=iid, position=pos, eth_in_wei=max(eth_delta, 0), tokens_sold_raw=max(sold, 0),
                                               decision_id=req.decision_id, ts=now, forced_kind=req.forced_kind, lot_value_eth=left_eth)
            ok = (tok_delta > 0) if req.kind == "buy" else (tok_delta < 0 or req.kind == "liquidate")
            conn.execute("UPDATE rh_intents SET state = %s, position_id = COALESCE(%s, position_id), updated_at = now() WHERE id = %s",
                         ("done" if ok else "aborted", pid, iid))
            conn.execute("UPDATE rh_txs SET applied_at = now() WHERE intent_id = %s AND applied_at IS NULL AND status IN ('mined_ok', 'reverted')", (iid,))
            receipts_tok = self._receipt_tokens(conn, iid, tok)
        if receipts_tok is not None and receipts_tok != tok_delta and req.kind != "liquidate":
            record_event("warning", "rh_live", f"{tok}: balance moved {tok_delta} but the receipts' Transfer logs say {receipts_tok} (fee-on-transfer or a foreign transfer)",
                         {"intent": iid})
        return RhResult(req, ok, None if ok else "no tokens moved", position_id=pid, realized=realized, route=route,
                        detail={"eth_delta_wei": eth_delta, "token_delta": tok_delta, "gas_wei": gas, "base_left": base_left})

    def _receipt_tokens(self, conn, iid: int, token: str) -> int | None:
        rows = conn.execute("SELECT hash FROM rh_txs WHERE intent_id = %s AND kind = 'swap' AND status = 'mined_ok'", (iid,)).fetchall()
        if not rows:
            return None
        n = 0
        for r in rows:
            rc = self.wallet.rpc.receipt(r["hash"])
            if rc is None:
                return None
            n += A.transfer_delta(rc, token, self.wallet.address)
        return n

    # ---- crash recovery
    def recover(self) -> dict:
        """Settle every transaction without an applied receipt, then finish every open intent from its snapshot."""
        out = {"txs": 0, "intents": 0}
        with self.wallet.lock:
            with transaction() as conn:
                txs = [dict(r) for r in conn.execute("SELECT * FROM rh_txs WHERE from_addr = %s AND applied_at IS NULL AND status IN ('signed', 'sent') ORDER BY nonce, id",
                                                     (self.wallet.address,)).fetchall()]
            for tx in txs:
                self._settle_tx(tx); out["txs"] += 1
            with transaction() as conn:
                intents = [dict(r) for r in conn.execute("SELECT * FROM rh_intents WHERE book = %s AND state NOT IN ('done', 'failed', 'aborted') ORDER BY id",
                                                         (A.BOOK,)).fetchall()]
            for it in intents:
                req = RhRequest(**(it["plan"] or {}).get("req", {"kind": it["kind"], "token": it["token"]}))
                with transaction() as conn:
                    mined = conn.execute("SELECT count(*) AS n FROM rh_txs WHERE intent_id = %s AND kind = 'swap' AND status = 'mined_ok'", (it["id"],)).fetchone()["n"]
                if mined:
                    res = self._finish(req, int(it["id"]), it.get("route"))
                else:
                    with transaction() as conn:
                        conn.execute("UPDATE rh_intents SET state = 'failed', error = 'recovered: no swap mined', updated_at = now() WHERE id = %s", (it["id"],))
                        A.charge_gas(conn, int(it["id"]), req.position_id if req.kind == "sell" else None)
                        conn.execute("UPDATE rh_txs SET applied_at = now() WHERE intent_id = %s AND applied_at IS NULL AND status IN ('mined_ok', 'reverted')", (it["id"],))
                    res = RhResult(req, False, "recovered: no swap mined")
                with self._mu:
                    self.results.append(res); self._pending.pop(req.token.lower(), None)
                out["intents"] += 1
        if out["txs"] or out["intents"]:
            record_event("warning", "rh_live", "recovered unfinished RH transactions/intents", out)
        return out

    def _settle_tx(self, tx: dict) -> None:
        settle_tx(self.wallet, tx)


def settle_tx(w, tx: dict) -> None:
    """Settle one of ``w``'s transactions that has no applied receipt (``recover``; rh/withdraw.settle_pending): its
    receipt recorded once mined; 'replaced' when something else took its nonce; under ``REBROADCAST_S`` old, the same
    bytes rebroadcast; otherwise cancelled at its nonce (the original recorded if it wins the race). Under the wallet's lock."""
    rpc = w.rpc
    rc = rpc.receipt(tx["hash"])
    if rc is not None and rc.get("blockNumber"):
        with transaction() as conn:
            w.record_receipt(conn, int(tx["id"]), rc)
        return
    if rpc.tx_count(w.address, "latest") > int(tx["nonce"]):          # something else took the nonce (a replacement or a cancel)
        with transaction() as conn:
            conn.execute("UPDATE rh_txs SET status = 'replaced', applied_at = now(), updated_at = now() WHERE id = %s", (tx["id"],))
        return
    age = (datetime.now(timezone.utc) - tx["created_at"]).total_seconds()
    if age < REBROADCAST_S:
        try:
            rpc.send_raw(tx["raw"])
        except EvmRpcError as e:
            if "known" not in (e.message or "").lower():
                log.warning("rebroadcast of %s failed: %s", tx["hash"], e)
        rc = w.wait(tx["hash"], timeout_s=60.0)
        if rc is not None:
            with transaction() as conn:
                w.record_receipt(conn, int(tx["id"]), rc)
            return
    c = w.cancel(int(tx["nonce"]), int(tx["max_fee_wei"] or 0))
    with transaction() as conn:
        conn.execute("UPDATE rh_txs SET intent_id = %s, position_id = %s WHERE id = %s", (tx["intent_id"], tx["position_id"], c["id"]))
    rc_c = w.wait(c["hash"], timeout_s=120.0)
    rc = rpc.receipt(tx["hash"])
    with transaction() as conn:
        if rc is not None and rc.get("blockNumber"):                   # the original won the race after all
            w.record_receipt(conn, int(tx["id"]), rc)
        else:
            conn.execute("UPDATE rh_txs SET status = 'replaced', applied_at = now(), updated_at = now() WHERE id = %s", (tx["id"],))
        if rc_c is not None:
            w.record_receipt(conn, int(c["id"]), rc_c)
