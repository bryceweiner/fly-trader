"""LiveBroker against fakes: the fake /order builds a real unsigned v0 transaction with the bot
keypair as fee payer, the fake /execute checks that the broker submitted a fully signed transaction
and then applies a scripted on-chain outcome to the fake RPC's balances."""
import base64
import random
import uuid

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from fly_trader import config
from fly_trader.chain.cluster_guard import SigningRefused
from fly_trader.chain.rpc import TOKEN_PROGRAM_ID
from fly_trader.execution import broker_live
from fly_trader.execution.broker_live import LiveBroker

MINT = str(Pubkey.new_unique())
FAKE_DEX = str(Pubkey.new_unique())
SYSTEM = "11111111111111111111111111111111"
BUY_LAMPORTS = 10_000_000
BUY_TOKENS = 1_000_000
BUY_FEES = 7_000


class FakeRpc:
    def __init__(self, lamports: int, tokens: dict | None = None, decimals: dict | None = None):
        self.lamports = lamports
        self.tokens = dict(tokens or {})
        self.decimals = dict(decimals or {})
        self.height = 1_000
        self.landed: set[str] = set()
        self.status_polls = 0

    def get_balance(self, pk):
        return self.lamports

    def get_token_accounts_by_owner(self, pk):
        return [{"address": f"ata-{m[:6]}", "mint": m, "amount": a, "decimals": self.decimals.get(m, 6),
                 "program": TOKEN_PROGRAM_ID} for m, a in self.tokens.items()]

    def get_signature_statuses(self, sigs, search_history=False):
        self.status_polls += 1
        return [({"slot": 5, "confirmations": 12, "err": None, "confirmationStatus": "confirmed"} if s in self.landed else None)
                for s in sigs]

    def get_block_height(self):
        self.height += 40
        return self.height

    def get_transaction(self, sig, encoding="jsonParsed"):
        return {"slot": 5, "meta": {"err": None, "fee": 5000}} if sig in self.landed else None

    def get_mint_decimals(self, mint):
        return None


class FakeJup:
    def __init__(self, keypair: Keypair, rpc: FakeRpc, outcomes: list[str]):
        self.kp = keypair
        self.rpc = rpc
        self.outcomes = list(outcomes)
        self.orders: list[dict] = []
        self.executes: list[tuple] = []

    def order(self, input_mint, output_mint, amount, taker=None, slippage_bps=None, exclude_routers="jupiterz"):
        self.orders.append(dict(input_mint=input_mint, output_mint=output_mint, amount=amount, taker=taker,
                                slippage_bps=slippage_bps, exclude_routers=exclude_routers))
        assert taker == str(self.kp.pubkey())
        ixs = [transfer(TransferParams(from_pubkey=self.kp.pubkey(), to_pubkey=Pubkey.new_unique(), lamports=1)),
               Instruction(Pubkey.from_string(FAKE_DEX), b"\x09", [AccountMeta(self.kp.pubkey(), True, True)])]
        msg = MessageV0.try_compile(self.kp.pubkey(), ixs, [], Hash.new_unique())
        tx = VersionedTransaction.populate(msg, [Signature.default()])
        return {"transaction": base64.b64encode(bytes(tx)).decode(), "requestId": str(uuid.uuid4()),
                "lastValidBlockHeight": str(self.rpc.height + 150), "signatureFeePayer": taker, "taker": taker,
                "router": "metis", "feeBps": 10, "inAmount": str(amount), "outAmount": str(BUY_TOKENS),
                "slippageBps": slippage_bps, "signatureFeeLamports": 5000, "prioritizationFeeLamports": 1000,
                "platformFee": {"feeBps": 10, "feeMint": input_mint}, "_latency_ms": 7}

    def execute(self, signed_transaction_b64, request_id, last_valid_block_height=None):
        tx = VersionedTransaction.from_bytes(base64.b64decode(signed_transaction_b64))
        assert tx.verify_with_results() == [True], "broker must submit a fully signed transaction"
        assert tx.message.account_keys[0] == self.kp.pubkey()
        sig = str(tx.signatures[0])
        outcome = self.outcomes.pop(0)
        self.executes.append((request_id, last_valid_block_height, outcome))
        if outcome == "success_buy":
            self.rpc.lamports -= BUY_LAMPORTS + BUY_FEES
            self.rpc.tokens[MINT] = self.rpc.tokens.get(MINT, 0) + BUY_TOKENS
            self.rpc.landed.add(sig)
            return {"status": "Success", "signature": sig, "slot": "5", "code": 0, "totalInputAmount": str(BUY_LAMPORTS),
                    "totalOutputAmount": str(BUY_TOKENS), "inputAmountResult": "9990000", "outputAmountResult": str(BUY_TOKENS),
                    "swapEvents": [], "_latency_ms": 900}
        if outcome == "success_sell":
            self.rpc.tokens[MINT] = 0
            self.rpc.lamports += 9_990_000 - 6_000
            self.rpc.landed.add(sig)
            return {"status": "Success", "signature": sig, "slot": "6", "code": 0, "totalInputAmount": str(BUY_TOKENS),
                    "totalOutputAmount": "9990000", "_latency_ms": 800}
        if outcome == "success_no_move":
            self.rpc.landed.add(sig)
            return {"status": "Success", "signature": sig, "slot": "5", "code": 0, "totalInputAmount": str(BUY_LAMPORTS),
                    "totalOutputAmount": str(BUY_TOKENS), "_latency_ms": 500}
        if outcome == "failed_but_moved":
            self.rpc.lamports -= BUY_LAMPORTS + BUY_FEES
            self.rpc.tokens[MINT] = self.rpc.tokens.get(MINT, 0) + BUY_TOKENS
            self.rpc.landed.add(sig)
            return {"status": "Failed", "signature": sig, "code": -1001, "error": "unknown", "_latency_ms": 500}
        if outcome == "fail_to_land":
            return {"status": "Failed", "signature": sig, "code": -1000, "error": "Transaction failed to land", "_latency_ms": 30_000}
        raise AssertionError(outcome)

    def program_id_to_label(self):
        return {FAKE_DEX: "FakeDex"}


@pytest.fixture
def gate_open(monkeypatch):
    monkeypatch.setattr(broker_live, "assert_signing_allowed", lambda: None)
    monkeypatch.setattr(broker_live.time, "sleep", lambda s: None)


def _broker(outcomes, lamports=1_000_000_000, tokens=None):
    kp = Keypair()
    rpc = FakeRpc(lamports, tokens, {MINT: 6})
    jup = FakeJup(kp, rpc, outcomes)
    return LiveBroker(rpc=rpc, jup=jup, keypair=kp), rpc, jup


def _swap(broker, conn, decision_id, side="buy", amount=BUY_LAMPORTS, max_slippage=300):
    return broker.swap(conn, decision_id=decision_id, book="live", side=side, mint=MINT, amount_in=amount,
                       slippage_bps=150, max_slippage_bps=max_slippage, step_bps=100)


def _orders(conn, decision_id):
    return conn.execute("SELECT * FROM orders WHERE decision_id = %s ORDER BY attempt", (decision_id,)).fetchall()


def _fills(conn, order_ids):
    return conn.execute("SELECT * FROM fills WHERE order_id = ANY(%s) ORDER BY id", (list(order_ids),)).fetchall()


def test_verified_buy_lands_orders_and_fills_rows(db_conn, gate_open):
    broker, rpc, jup = _broker(["success_buy"])
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert res.ok and res.attempts == 1 and res.status == "confirmed" and res.code == 0 and res.error is None
    assert res.token_delta == BUY_TOKENS and res.lamports_delta == -(BUY_LAMPORTS + BUY_FEES)
    assert res.price_sol == pytest.approx(((BUY_LAMPORTS + BUY_FEES) / 1e9) / (BUY_TOKENS / 1e6))
    assert jup.orders[0]["exclude_routers"] == "jupiterz" and jup.orders[0]["slippage_bps"] == 150
    assert jup.orders[0]["input_mint"] == config.WSOL_MINT and jup.orders[0]["output_mint"] == MINT

    orders = _orders(db_conn, did)
    assert len(orders) == 1
    o = orders[0]
    assert o["id"] == res.order_id and o["status"] == "confirmed" and o["side"] == "buy" and o["book"] == "live"
    assert o["input_mint"] == config.WSOL_MINT and o["output_mint"] == MINT and int(o["amount_in"]) == BUY_LAMPORTS
    assert o["slippage_bps"] == 150 and o["router"] == "metis" and o["fee_bps"] == 10 and o["error_code"] == 0
    assert set(o["program_ids"]) == {SYSTEM, FAKE_DEX}
    assert o["request_id"] == o["order_response"]["requestId"] == jup.executes[0][0]
    assert o["last_valid_block_height"] == int(o["order_response"]["lastValidBlockHeight"]) == jup.executes[0][1]
    assert o["signature"] == res.signature and o["execute_response"]["status"] == "Success"
    assert o["execute_request"]["requestId"] == o["request_id"] and o["latency_ms"] == 900
    assert o["request"]["taker"] == broker.pubkey and "excludeRouters" in o["request"]

    fills = _fills(db_conn, [o["id"]])
    assert len(fills) == 1
    f = fills[0]
    assert f["id"] == res.fill_id and f["verified_by"] == "balance_delta" and f["book"] == "live"
    assert int(f["token_delta"]) == BUY_TOKENS and f["sol_delta_lamports"] == -(BUY_LAMPORTS + BUY_FEES)
    assert f["fee_lamports"] == BUY_FEES and f["mint"] == MINT and f["side"] == "buy" and f["slot"] == 5
    assert f["signature"] == res.signature and f["platform_fee"]["feeBps"] == 10
    assert f["pre_snapshot"]["lamports"] == 1_000_000_000 and MINT not in f["pre_snapshot"]["tokens"]
    assert f["post_snapshot"]["tokens"][MINT] == BUY_TOKENS and f["post_snapshot"]["decimals"][MINT] == 6


def test_verified_sell(db_conn, gate_open):
    broker, rpc, jup = _broker(["success_sell"], tokens={MINT: BUY_TOKENS})
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did, side="sell", amount=BUY_TOKENS)
    assert res.ok and res.token_delta == -BUY_TOKENS and res.lamports_delta == 9_984_000
    assert res.price_sol == pytest.approx(9_984_000 / 1e9 / 1.0)
    assert jup.orders[0]["input_mint"] == MINT and jup.orders[0]["output_mint"] == config.WSOL_MINT
    f = _fills(db_conn, [res.order_id])[0]
    assert f["verified_by"] == "balance_delta" and f["side"] == "sell" and f["fee_lamports"] == 6_000


def test_success_without_movement_is_unverified(db_conn, gate_open):
    broker, rpc, jup = _broker(["success_no_move"])
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert not res.ok and res.attempts == 1 and res.status == "confirmed" and res.token_delta == 0 and res.lamports_delta == 0
    orders = _orders(db_conn, did)
    assert len(orders) == 1 and orders[0]["status"] == "confirmed"
    fills = _fills(db_conn, [orders[0]["id"]])
    assert len(fills) == 1 and fills[0]["verified_by"] == "unverified" and fills[0]["id"] == res.fill_id
    ev = db_conn.execute("SELECT * FROM events WHERE source = 'broker_live' AND (detail->>'order_id')::bigint = %s",
                         (orders[0]["id"],)).fetchall()
    assert len(ev) == 1 and ev[0]["level"] == "warning" and ev[0]["detail"]["tx_found"] is True


def test_failed_status_with_moved_balances_is_a_fill(db_conn, gate_open):
    broker, rpc, jup = _broker(["failed_but_moved"])
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert res.ok and res.attempts == 1 and res.code == -1001
    f = _fills(db_conn, [res.order_id])[0]
    assert f["verified_by"] == "balance_delta_despite_failed" and int(f["token_delta"]) == BUY_TOKENS


def test_failed_to_land_retries_with_higher_slippage(db_conn, gate_open):
    broker, rpc, jup = _broker(["fail_to_land", "success_buy"])
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert res.ok and res.attempts == 2 and res.code == 0
    assert [o["slippage_bps"] for o in jup.orders] == [150, 250]
    orders = _orders(db_conn, did)
    assert [o["attempt"] for o in orders] == [1, 2]
    assert [o["slippage_bps"] for o in orders] == [150, 250]
    assert [o["status"] for o in orders] == ["expired", "confirmed"]
    assert orders[0]["error_code"] == -1000 and "land" in orders[0]["error"]
    assert orders[0]["request_id"] != orders[1]["request_id"]
    assert rpc.status_polls >= 2  # the first attempt was polled until lastValidBlockHeight passed
    fills = _fills(db_conn, [o["id"] for o in orders])
    assert len(fills) == 1 and fills[0]["order_id"] == orders[1]["id"] == res.order_id


def test_retry_ladder_stops_at_max_slippage(db_conn, gate_open):
    broker, rpc, jup = _broker(["fail_to_land", "fail_to_land", "fail_to_land"])
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did, max_slippage=250)
    assert not res.ok and res.attempts == 2 and res.code == -1000 and res.status == "expired"
    assert [o["slippage_bps"] for o in jup.orders] == [150, 250]
    assert len(jup.outcomes) == 1  # third outcome never consumed
    orders = _orders(db_conn, did)
    assert len(orders) == 2 and not _fills(db_conn, [o["id"] for o in orders])


def test_order_error_code_is_recorded_and_not_retried(db_conn, gate_open):
    broker, rpc, jup = _broker([])
    jup.order = lambda *a, **k: {"errorCode": 1, "errorMessage": "Insufficient funds", "transaction": None}
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert not res.ok and res.status == "order_error" and res.code == 1 and "Insufficient" in res.error
    orders = _orders(db_conn, did)
    assert len(orders) == 1 and orders[0]["status"] == "order_error" and orders[0]["error_code"] == 1
    assert orders[0]["order_response"]["errorCode"] == 1


def test_fee_payer_mismatch_refused(db_conn, gate_open):
    broker, rpc, jup = _broker(["success_buy"])
    real_order = jup.order

    def bad_order(*a, **k):
        o = real_order(*a, **k)
        o["signatureFeePayer"] = str(Pubkey.new_unique())
        return o

    jup.order = bad_order
    did = random.randrange(1 << 40)
    res = _swap(broker, db_conn, did)
    assert not res.ok and res.status == "order_error" and "signatureFeePayer" in res.error
    assert jup.executes == []


def test_signing_gate_is_checked_first(db_conn, monkeypatch):
    monkeypatch.setattr(config, "LIVE_ENABLED", False)
    broker, rpc, jup = _broker(["success_buy"])
    with pytest.raises(SigningRefused):
        _swap(broker, db_conn, random.randrange(1 << 40))
    assert jup.orders == []
