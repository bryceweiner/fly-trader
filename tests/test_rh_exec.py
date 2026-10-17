"""The live RH executor on a fake chain (the real RhWallet signs; a fake RPC mines): what each intent books is what the
balances moved, gas included; a sale climbs the slippage ladder over on-chain reverts; two-leg routes keep their base in
lots; a crash after broadcast is booked exactly once on recovery; a dropped transaction is cancelled at its nonce; the
reconciler trips circuit 3 on ETH leaving unexplained; the mirror sizes RH entries from the wallet and respects circuit 3."""
import threading
import uuid
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from fly_trader import config
from fly_trader.agent import rails
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.markets import RH_CIRCUIT
from fly_trader.rh import abi, accounting as A, kyber as K, router
from fly_trader.rh.exec import DUPLICATE, RhExecutor, RhRequest, RhResult
from fly_trader.rh.tx import address_of, decode_raw
from fly_trader.rh.wallet import RhWallet
from fly_trader.vault.evm import EvmRpcError

KEY = bytes.fromhex("59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d")
ADDR = address_of(KEY)
TOKEN, BASE, SWAP = "0x" + "ab" * 20, "0x" + "cd" * 20, "0x" + "5a" * 20
GAS_USED, BASE_FEE = 40_000, 10_000_000
FEE = GAS_USED * BASE_FEE                           # wei per mined transaction
WEI = 10 ** 18


class FakeChain:
    chain_id = 4663

    def __init__(self):
        self.native = 10 * WEI; self.tokens = {}; self.allow = {}; self.nonce = 0; self.head = 100; self.receipts = {}
        self.ops = {}; self.no_route = set(); self.revert_onchain = {}; self.drop_next = False; self.hide_receipts = False; self.hidden = {}

    # wallet reads
    def chain(self):
        return self.chain_id

    def get_balance(self, a, tag="latest"):
        return self.native

    def base_fee(self):
        return BASE_FEE

    def max_priority_fee(self):
        return 0

    def estimate_gas(self, tx):
        return 50_000

    def tx_count(self, a, tag="pending"):
        return self.nonce

    def block_number(self, tag="latest"):
        return self.head

    def receipt(self, h):
        return self.receipts.get(h)

    def eth_call(self, to, data, block="latest"):
        d = bytes.fromhex(data[2:])
        if d[:4] == abi.selector("balanceOf(address)"):
            return "0x" + abi.encode(["uint256"], [self.tokens.get(to.lower(), 0)]).hex()
        if d[:4] == abi.selector("allowance(address,address)"):
            _, sp = abi.decode(["address", "address"], d[4:]); return "0x" + abi.encode(["uint256"], [self.allow.get((to.lower(), sp), 0)]).hex()
        raise AssertionError("unexpected eth_call")

    def call(self, method, params):
        to = params[0]["to"].lower(); data = bytes.fromhex(params[0]["data"][2:])
        if to == SWAP:
            op = self.ops[data]
            if op.get("sim_revert"):
                raise EvmRpcError("eth_call", {"message": "execution reverted: slippage"})
        return "0x"

    # mining
    def _log(self, token, frm, to, v):
        t = lambda a: "0x" + "0" * 24 + a[2:]                         # noqa: E731
        return {"address": token, "topics": [A.TRANSFER, t(frm), t(to)], "data": hex(v)}

    def send_raw(self, raw):
        stx = decode_raw(bytes.fromhex(raw[2:])); tx = stx.tx
        if self.drop_next:
            self.drop_next = False; return stx.hash                     # accepted, never mined
        assert tx.nonce == self.nonce, (tx.nonce, self.nonce)
        self.nonce += 1; self.native -= FEE; ok = True; logs = []
        if tx.to == SWAP:
            op = self.ops[tx.data]; pair = (op["tin"], op["tout"])
            if self.revert_onchain.get(pair, 0) > 0:
                self.revert_onchain[pair] -= 1; ok = False
            else:
                if op["tin"] == K.NATIVE:
                    assert tx.value == op["amount"]; self.native -= op["amount"]
                else:
                    self.tokens[op["tin"]] = self.tokens.get(op["tin"], 0) - op["amount"]; logs.append(self._log(op["tin"], ADDR, SWAP, op["amount"]))
                if op["tout"] == K.NATIVE:
                    self.native += op["out"]
                else:
                    self.tokens[op["tout"]] = self.tokens.get(op["tout"], 0) + op["out"]; logs.append(self._log(op["tout"], SWAP, ADDR, op["out"]))
        elif tx.data[:4] == abi.selector("approve(address,uint256)"):
            sp, amt = abi.decode(["address", "uint256"], tx.data[4:]); self.allow[(tx.to, sp)] = amt
        elif tx.to != ADDR:                                             # a plain transfer out (a payout)
            self.native -= tx.value
        rc = {"blockNumber": hex(self.head - 5), "blockHash": "0xbb", "gasUsed": hex(GAS_USED), "effectiveGasPrice": hex(BASE_FEE),
              "status": "0x1" if ok else "0x0", "logs": logs}
        (self.hidden if self.hide_receipts else self.receipts)[stx.hash] = rc
        return stx.hash


class Planner:
    """Routes at fixed rates: ETH→TOKEN 1e6 tokens per wei-unit rate etc. ``rates[(tin, tout)] = out per unit in``."""

    def __init__(self, chain):
        self.c = chain; self.rates = {}

    def __call__(self, account, tin, tout, amount, slip, rpc=None, pool=None):
        tin = K.NATIVE if tin in (K.NATIVE, "0x" + "00" * 20) else tin.lower(); tout = K.NATIVE if tout in (K.NATIVE, "0x" + "00" * 20) else tout.lower()
        if (tin, tout) in self.c.no_route or (tin, tout) not in self.rates:
            raise router.NoRoute(f"{tin}->{tout}")
        out = int(amount * F(self.rates[(tin, tout)]))
        data = f"{tin}{tout}{amount}{slip}".encode()
        self.c.ops[data] = {"tin": tin, "tout": tout, "amount": amount, "out": out, "sim_revert": slip < self.rates.get(("min_slip", tin), 0)}
        approvals = [] if tin == K.NATIVE else [(tin, SWAP, amount)]
        return router.Plan("atomic_kyber", tin, tout, amount, out, out, router.Call(SWAP, data, amount if tin == K.NATIVE else 0), approvals=approvals)


@pytest.fixture
def chain(monkeypatch):
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 3); monkeypatch.setattr(config, "RH_MAX_FEE_GWEI", 5.0)
    monkeypatch.setattr("fly_trader.rh.wallet.RECEIPT_POLL_S", 0.0)
    real_wait = RhWallet.wait
    monkeypatch.setattr(RhWallet, "wait", lambda self, h, timeout_s=120.0, confirmations=None: real_wait(self, h, min(timeout_s, 0.05), confirmations))

    def clean():
        with transaction() as conn:
            for t in ("rh_legs", "rh_base_lots", "rh_intents", "rh_wallet_marks", "rh_wallet_flows"):
                conn.execute(f"DELETE FROM {t}")
            conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))
            conn.execute("DELETE FROM positions WHERE book = 'live_rh'"); conn.execute("DELETE FROM wealth_marks WHERE book = 'live_rh'")
            conn.execute("DELETE FROM rh_base_prices WHERE asset = %s", (BASE,))
            conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, kill_reason=NULL, peak_wealth=NULL, entries_paused=false WHERE id = 3")
    clean()
    c = FakeChain(); p = Planner(c)
    p.rates[(K.NATIVE, TOKEN)] = F(10 ** 6); p.rates[(TOKEN, K.NATIVE)] = F("1.1e-6")
    yield SimpleNamespace(c=c, p=p, ex=lambda: RhExecutor(RhWallet(c, KEY, ADDR), planner=p, start=False))
    clean()


def _pos(pid):
    with transaction() as conn:
        return dict(conn.execute("SELECT * FROM positions WHERE id = %s", (pid,)).fetchone())


def test_round_trip_books_what_the_balances_moved(chain):
    ex = chain.ex(); amt = WEI // 10
    b = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, decision_id=None, hold_s=600.0))
    assert b.ok and b.detail["token_delta"] == amt * 10 ** 6
    p = _pos(b.position_id)
    assert p["chain"] == "rh" and p["qty"] == amt * 10 ** 6 and p["cost_sol"] == pytest.approx((amt + FEE) / WEI) and p["gas_q"] == pytest.approx(FEE / WEI)
    s = ex.execute(RhRequest(kind="sell", token=TOKEN, amount_in=int(p["qty"]), position_id=b.position_id))
    assert s.ok
    p = _pos(b.position_id); eth_in = int(amt * 10 ** 6 * F("1.1e-6"))
    # the sale paid an approval and the swap: both are the exit's gas
    assert p["status"] == "closed" and p["realized_sol"] == pytest.approx((eth_in - 2 * FEE) / WEI - (amt + FEE) / WEI, abs=1e-15)
    assert chain.c.native == 10 * WEI - amt + eth_in - 3 * FEE                         # the chain agrees with the books to the wei
    with transaction() as conn:
        assert A.reconcile(conn, ex.wallet)["ok"]


def test_sale_climbs_the_slippage_ladder_and_pays_every_revert(chain, monkeypatch):
    monkeypatch.setattr(config, "SLIPPAGE_STEP_BPS", 100)
    ex = chain.ex(); b = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=WEI // 100)); q = _pos(b.position_id)["qty"]
    chain.c.revert_onchain[(TOKEN, K.NATIVE)] = 2                                     # two mined reverts, then it fills
    s = ex.execute(RhRequest(kind="sell", token=TOKEN, amount_in=int(q), position_id=b.position_id, slippage_bps=300, max_slippage_bps=500))
    assert s.ok
    with transaction() as conn:
        n = conn.execute("SELECT count(*) AS n, count(*) FILTER (WHERE status = 'reverted') AS r FROM rh_txs WHERE intent_id = "
                         "(SELECT id FROM rh_intents WHERE kind = 'sell')").fetchone()
    assert n["r"] == 2
    p = _pos(b.position_id); eth_in = int(int(q) * F("1.1e-6"))
    assert p["realized_sol"] == pytest.approx((eth_in - n["n"] * FEE) / WEI - (WEI // 100 + FEE) / WEI, abs=1e-15)


def test_failed_entry_gas_is_an_aborted_position(chain):
    ex = chain.ex(); chain.c.revert_onchain[(K.NATIVE, TOKEN)] = 5
    r = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=WEI // 100, slippage_bps=150, max_slippage_bps=150))
    assert not r.ok
    with transaction() as conn:
        rows = conn.execute("SELECT qty, realized_sol, forced_exit_kind FROM positions WHERE book = 'live_rh'").fetchall()
    assert len(rows) == 1 and rows[0]["qty"] == 0 and rows[0]["forced_exit_kind"] == "aborted" and rows[0]["realized_sol"] == pytest.approx(-FEE / WEI)


def test_two_leg_buy_and_a_sale_whose_base_leg_fails_keeps_a_lot(chain):
    c, p = chain.c, chain.p
    p.rates.pop((K.NATIVE, TOKEN)); p.rates.pop((TOKEN, K.NATIVE))
    p.rates[(K.NATIVE, BASE)] = F(2000); p.rates[(BASE, TOKEN)] = F(500); p.rates[(TOKEN, BASE)] = F("0.0022"); p.rates[(BASE, K.NATIVE)] = F("0.0005")
    with transaction() as conn:
        conn.execute("INSERT INTO rh_base_prices (asset, ts, price_eth) VALUES (%s, %s, 0.0005)", (BASE, datetime.now(timezone.utc)))
    ex = chain.ex(); amt = WEI // 100
    b = ex.execute(RhRequest(kind="buy", token=TOKEN, quote_asset=BASE, amount_in=amt))
    assert b.ok and b.route == "two_leg"
    pos = _pos(b.position_id)
    assert pos["qty"] == amt * 2000 * 500 and pos["cost_sol"] == pytest.approx((amt + 3 * FEE) / WEI)   # base approval + two swaps
    c.revert_onchain[(BASE, K.NATIVE)] = 1
    s = ex.execute(RhRequest(kind="sell", token=TOKEN, quote_asset=BASE, amount_in=int(pos["qty"]), position_id=b.position_id, max_slippage_bps=500))
    with transaction() as conn:
        lots = A.open_lots(conn)
    base_got = int(int(pos["qty"]) * F("0.0022"))
    assert s.ok and len(lots) == 1 and lots[0]["source"] == "exit_leg" and int(lots[0]["qty_raw"]) == base_got and lots[0]["position_id"] == b.position_id
    closed = _pos(b.position_id); lot_eth = base_got / WEI * 0.0005
    assert closed["status"] == "closed"
    before = closed["realized_sol"]
    lq = ex.execute(RhRequest(kind="liquidate", token=BASE, amount_in=base_got, lot_id=int(lots[0]["id"])))
    assert lq.ok
    after = _pos(b.position_id)["realized_sol"]
    got = int(base_got * F("0.0005"))
    assert after - before == pytest.approx(got / WEI - FEE / WEI - lot_eth, abs=1e-15)   # the lot sale (its exact approval was already made) settles onto its position
    with transaction() as conn:
        assert A.reconcile(conn, ex.wallet)["ok"]


def test_crash_after_broadcast_is_booked_once(chain):
    c = chain.c; c.hide_receipts = True
    ex = chain.ex()
    r = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=WEI // 100, decision_id=None))
    assert r.error.startswith("no receipt")
    c.receipts.update(c.hidden); c.hide_receipts = False                               # the chain mined it; the process died
    ex2 = chain.ex()                                                                   # a new start recovers
    with transaction() as conn:
        rows = conn.execute("SELECT qty, cost_sol FROM positions WHERE book = 'live_rh'").fetchall()
        st = conn.execute("SELECT state FROM rh_intents").fetchall()
    assert len(rows) == 1 and rows[0]["qty"] == (WEI // 100) * 10 ** 6 and rows[0]["cost_sol"] == pytest.approx((WEI // 100 + FEE) / WEI)
    assert [s["state"] for s in st] == ["done"]
    ex2.recover(); chain.ex()
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book = 'live_rh'").fetchone()["n"] == 1


def test_a_dropped_transaction_is_cancelled_at_its_nonce(chain):
    c = chain.c; c.drop_next = True
    ex = chain.ex()
    r = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=WEI // 100))
    assert r.error.startswith("no receipt")
    with transaction() as conn:
        conn.execute("UPDATE rh_txs SET created_at = now() - interval '10 minutes' WHERE from_addr = %s", (ADDR,))
    chain.ex()
    with transaction() as conn:
        txs = {t["kind"]: t["status"] for t in conn.execute("SELECT kind, status FROM rh_txs WHERE from_addr = %s", (ADDR,)).fetchall()}
        it = conn.execute("SELECT state FROM rh_intents").fetchone()
        pos = conn.execute("SELECT realized_sol, forced_exit_kind FROM positions WHERE book = 'live_rh'").fetchall()
    assert txs == {"swap": "replaced", "cancel": "mined_ok"} and it["state"] == "failed" and c.nonce == 1
    assert len(pos) == 1 and pos[0]["forced_exit_kind"] == "aborted" and pos[0]["realized_sol"] == pytest.approx(-FEE / WEI)


def test_unexplained_outflow_trips_circuit_3(chain):
    ex = chain.ex()
    with transaction() as conn:
        assert A.reconcile(conn, ex.wallet, datetime.now(timezone.utc) - timedelta(minutes=1))["ok"]
    chain.c.native -= WEI // 50                                                       # ETH left without a transaction of ours
    with transaction() as conn:
        out = A.reconcile(conn, ex.wallet)
        cs = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = %s", (RH_CIRCUIT,)).fetchone()
        one = conn.execute("SELECT kill_switch FROM circuit_state WHERE id = 1").fetchone()
    assert not out["ok"] and out["native_gap_eth"] == pytest.approx(-0.02) and cs["kill_switch"] and cs["entries_paused"] and not one["kill_switch"]


def test_mirror_sizes_rh_entries_and_respects_circuit_3(chain, monkeypatch):
    from fly_trader.rh.live import RhLiveMirror
    monkeypatch.setattr(config, "FLY_SIZING", "flat")
    ex = chain.ex(); m = RhLiveMirror(600.0, executor=ex)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    with transaction() as conn:
        run = str(uuid.uuid4()); conn.execute("INSERT INTO runs (run_id, kind, status) VALUES (%s, 'fly_rh', 'running')", (run,))
        beat = conn.execute("INSERT INTO beats (run_id, beat_no) VALUES (%s, 1) RETURNING id", (run,)).fetchone()["id"]
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, kind, size_sol) VALUES (%s, %s, %s, %s, 'enter', 0.01) RETURNING id",
                           (beat, run, now, TOKEN)).fetchone()["id"]
        ctx = SimpleNamespace(conn=conn, m1=now, m1_epoch=now.timestamp(), prices={TOKEN: 1e-6}, resqs={TOKEN: 10.0}, fees={}, last_resq=lambda x: 10.0,
                              fee=lambda x: None, lag_s=0.0)
        e = {"mint": TOKEN, "decision_id": did, "score": 0.1, "threshold": 0.0, "table": None, "hold_s": 600.0, "strategy": "ev",
             "info": {"pool": None, "resq": 10.0, "decimals": 18}}
        out = m.minute(ctx, run_id=run, beat_id=beat, entries=[e])
    assert out["entered"] == 1 and ex.pending() == {TOKEN}
    req = ex.q.get_nowait()
    assert 0 < req.amount_in <= int(config.RH_LABEL_SIZE_ETH * WEI) + 1                  # never bigger than the RH label size
    ex._pending.clear()
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET kill_switch = true WHERE id = %s", (RH_CIRCUIT,))
        ctx.conn = conn
        out = m.minute(ctx, run_id=run, beat_id=beat, entries=[e])
        conn.execute("DELETE FROM decisions WHERE run_id = %s", (run,)); conn.execute("DELETE FROM beats WHERE run_id = %s", (run,))
        conn.execute("DELETE FROM runs WHERE run_id = %s", (run,))
    assert out["entered"] == 0 and out["blocked"] == "kill switch"


def test_a_buy_climbs_the_slippage_ladder_to_the_entry_maximum(chain, monkeypatch):
    """A buy reverted at the entry slippage is not given up: it climbs the ladder to its maximum, as on Solana."""
    monkeypatch.setattr(config, "SLIPPAGE_STEP_BPS", 100)
    chain.c.revert_onchain[(K.NATIVE, TOKEN)] = 1                                   # mined and reverted at 150 bps, filled at 250
    ex = chain.ex(); amt = WEI // 100
    b = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, slippage_bps=150, max_slippage_bps=300))
    with transaction() as conn:
        st = [r["status"] for r in conn.execute("SELECT status FROM rh_txs WHERE from_addr = %s AND kind = 'swap' ORDER BY id", (ADDR,)).fetchall()]
    assert b.ok and st == ["reverted", "mined_ok"] and _pos(b.position_id)["cost_sol"] == pytest.approx((amt + 2 * FEE) / WEI)   # both attempts' gas


def test_a_buy_whose_quote_ran_past_its_paper_fill_is_refused_unsigned(chain):
    """The planner quotes 1e-6 ETH per token: a decision filled at 0.9e-6 would chase it +11 % — refused before anything
    is signed, no circuit failure, sent again later; one filled at 0.98e-6 (+2 %) is bought."""
    ex = chain.ex(); amt = WEI // 100
    r = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, ref_price=0.9e-6))
    with transaction() as conn:
        n = conn.execute("SELECT count(*) AS n FROM rh_txs WHERE from_addr = %s", (ADDR,)).fetchone()["n"]
        it = conn.execute("SELECT state, error FROM rh_intents WHERE kind = 'buy'").fetchone()
        fails = rails.load_circuit(conn, RH_CIRCUIT).fail_count
    assert not r.ok and r.retryable and r.error.startswith("price ran") and n == 0 and chain.c.nonce == 0
    assert it["state"] == "failed" and it["error"].startswith("price ran") and fails == 0
    assert ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, ref_price=0.98e-6)).ok


def test_every_intent_counts_on_circuit_3(chain, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_THRESHOLD", 2)
    chain.c.revert_onchain[(K.NATIVE, TOKEN)] = 2
    ex = chain.ex()
    one = lambda: ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=WEI // 100, slippage_bps=150, max_slippage_bps=150))   # noqa: E731
    assert not one().ok and not one().ok
    with transaction() as conn:
        c = rails.load_circuit(conn, RH_CIRCUIT)
    assert c.tripped and c.fail_count == 2
    assert one().ok
    with transaction() as conn:
        c = rails.load_circuit(conn, RH_CIRCUIT)
    assert not c.tripped and c.fail_count == 0                                       # a success re-arms circuit 3


def test_a_failed_entry_is_sent_again_under_its_decision(chain):
    """A retry's intent leaves the decision's unique slot to the first one and books under the decision; it runs only
    while no intent of the decision bought anything."""
    ex = chain.ex(); amt = WEI // 100
    with transaction() as conn:
        did = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_rh_enter', %s) RETURNING id", (TOKEN,)).fetchone()["id"]
    try:
        chain.c.revert_onchain[(K.NATIVE, TOKEN)] = 1
        first = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, decision_id=did, slippage_bps=150, max_slippage_bps=150))
        assert not first.ok and first.retryable
        assert ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, decision_id=did)).error == DUPLICATE
        again = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, decision_id=did, retry=1))
        assert again.ok and _pos(again.position_id)["entry_decision_id"] == did
        with transaction() as conn:
            assert [r["decision_id"] for r in conn.execute("SELECT decision_id FROM rh_intents WHERE kind = 'buy' ORDER BY id").fetchall()] == [did, None]
            conn.execute("UPDATE positions SET status = 'closed' WHERE id = %s", (again.position_id,))
        assert ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt, decision_id=did, retry=2)).error == DUPLICATE   # it bought: never twice
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM decisions WHERE id = %s", (did,))


def test_a_dead_bag_sweep_is_booked_onto_the_written_off_position(chain):
    ex = chain.ex(); amt = WEI // 100
    b = ex.execute(RhRequest(kind="buy", token=TOKEN, amount_in=amt)); p = _pos(b.position_id)
    with transaction() as conn:
        ledger.close_position(conn, position_id=b.position_id, exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=None, forced_kind="dead_bag")
    s = ex.execute(RhRequest(kind="sweep", token=TOKEN, amount_in=int(p["qty"]), slippage_bps=500, max_slippage_bps=500, forced_kind="dead_bag_sweep"))
    after = _pos(b.position_id); eth_in = int(int(p["qty"]) * F("1.1e-6"))
    assert s.ok and s.position_id == b.position_id
    assert after["realized_sol"] == pytest.approx(-p["cost_sol"] + (eth_in - 2 * FEE) / WEI, abs=1e-15)    # net of its approval and swap
    with transaction() as conn:
        assert A.reconcile(conn, ex.wallet)["ok"]


def test_a_buy_queues_behind_the_sale_of_its_token_and_a_stop_drops_what_never_started(chain):
    ex = chain.ex()
    assert ex.submit(RhRequest(kind="sell", token=TOKEN, amount_in=1, position_id=1)) and ex.selling() == {TOKEN}
    assert ex.submit(RhRequest(kind="buy", token=TOKEN, amount_in=5))
    assert not ex.submit(RhRequest(kind="buy", token=TOKEN, amount_in=7)) and not ex.submit(RhRequest(kind="sell", token=TOKEN, amount_in=1))
    assert ex.selling() == set() and ex.pending() == {TOKEN} and ex.committed_wei() == 5
    ex.stop()
    assert ex.pending() == set() and ex.q.empty() and not ex.submit(RhRequest(kind="sell", token=TOKEN, amount_in=1))


def _minute_rows(conn, now):
    run = str(uuid.uuid4()); conn.execute("INSERT INTO runs (run_id, kind, status) VALUES (%s, 'fly_rh', 'running')", (run,))
    beat = conn.execute("INSERT INTO beats (run_id, beat_no) VALUES (%s, 1) RETURNING id", (run,)).fetchone()["id"]
    ctx = SimpleNamespace(conn=conn, m1=now, m1_epoch=now.timestamp(), prices={TOKEN: 1e-6}, resqs={TOKEN: 10.0}, fees={}, last_resq=lambda x: 10.0,
                          fee=lambda x: None, lag_s=0.0)
    return run, beat, ctx


def _drop_minute_rows(conn, run):
    conn.execute("DELETE FROM decisions WHERE run_id = %s", (run,)); conn.execute("DELETE FROM beats WHERE run_id = %s", (run,))
    conn.execute("DELETE FROM runs WHERE run_id = %s", (run,)); conn.execute("DELETE FROM positions WHERE book = 'paper_rh_fly' AND mint = %s", (TOKEN,))


def test_the_rh_mirror_retries_a_failed_entry_and_records_what_it_did_not_take(chain, monkeypatch):
    from fly_trader.rh.live import RhLiveMirror
    monkeypatch.setattr(config, "FLY_SIZING", "flat"); monkeypatch.setattr(config, "ENTRY_RETRY_MIN", 2.0)
    ex = chain.ex(); m = RhLiveMirror(600.0, executor=ex)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    with transaction() as conn:
        run, beat, ctx = _minute_rows(conn, now)
        did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, kind, size_sol) VALUES (%s, %s, %s, %s, 'fly_rh_enter', 0.01) RETURNING id",
                           (beat, run, now, TOKEN)).fetchone()["id"]
        conn.execute("INSERT INTO positions (book, mint, opened_at, entry_decision_id, qty, cost_sol, entry_price, status, chain) "
                     "VALUES ('paper_rh_fly', %s, %s, %s, 1, 0.01, 1e-6, 'open', 'rh')", (TOKEN, now, did))
        e = {"mint": TOKEN, "decision_id": did, "score": 0.1, "threshold": 0.0, "table": None, "hold_s": 600.0, "strategy": "ev", "ref_price": 1.0e-6,
             "info": {"pool": None, "resq": 10.0, "decimals": 18}}
        assert m.minute(ctx, run_id=run, beat_id=beat, entries=[e])["entered"] == 1
        req = ex.q.get_nowait(); ex._pending.clear()
        ex.results.append(RhResult(req, False, "price ran: quoted 1.1e-06 ETH per token", retryable=True))
        ctx.m1 = now + timedelta(minutes=1); ctx.m1_epoch += 60
        out = m.minute(ctx, run_id=run, beat_id=beat + 1000001, entries=[])
        again = ex.q.get_nowait()
        rows = conn.execute("SELECT rail, detail FROM decisions WHERE run_id = %s AND kind = 'blocked'", (run,)).fetchall()
        _drop_minute_rows(conn, run)
    assert out["retried"] == 1 and again is not req and (again.decision_id, again.retry, again.ref_price) == (did, 1, 1.0e-6)
    assert [(r["rail"], r["detail"]["book"], r["detail"]["of_decision"]) for r in rows] == [("price ran", "live_rh", did)]


def test_the_rh_mark_waits_for_no_intent_between_its_snapshot_and_its_booking(chain, monkeypatch):
    """The mark is bookkeeping, taken after the minute's intents (``mark_due``). The executor holds the wallet lock from an
    intent's snapshot to its booking: a mark read beside it would count that trade twice or not at all, so the minute goes
    unmarked."""
    from fly_trader.rh import live as rh_live
    monkeypatch.setattr(rh_live, "MARK_WAIT_S", 0.05)
    ex = chain.ex(); m = rh_live.RhLiveMirror(600.0, executor=ex); held, go = threading.Event(), threading.Event()

    def intent():
        with ex.wallet.lock:
            held.set(); go.wait(10)
    t = threading.Thread(target=intent); t.start(); assert held.wait(10)
    try:
        with transaction() as conn:
            run, beat, ctx = _minute_rows(conn, datetime.now(timezone.utc).replace(second=0, microsecond=0))
            out = m.minute(ctx, run_id=run, beat_id=beat, entries=[])
            due = m.due_mark is not None
            mark = m.mark_due(conn)
            n = conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book = 'live_rh' AND beat_id = %s", (beat,)).fetchone()["n"]
            _drop_minute_rows(conn, run)
    finally:
        go.set(); t.join(10)
    assert "wealth" not in out and due and mark is None and m.due_mark is None and n == 0
