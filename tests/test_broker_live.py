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
from fly_trader.db.connection import transaction
from fly_trader.execution import broker_live, ledger
from fly_trader.execution.broker_live import LiveBroker

MINT = str(Pubkey.new_unique())
FAKE_DEX = str(Pubkey.new_unique())
SYSTEM = "11111111111111111111111111111111"
BUY_LAMPORTS = 10_000_000
BUY_TOKENS = 1_000_000
BUY_FEES = 7_000
RENT = 1_513_840            # a Token-2022 account's rent, as a first buy of a pump.fun mint pays it (2026-10-18)


class FakeRpc:
    def __init__(self, lamports: int, tokens: dict | None = None, decimals: dict | None = None):
        self.lamports = lamports
        self.tokens = dict(tokens or {})
        self.decimals = dict(decimals or {})
        self.height = 1_000
        self.landed: set[str] = set()
        self.status_polls = 0
        self.rent: dict[str, int] = {}

    def get_balance(self, pk):
        return self.lamports

    def get_token_accounts_by_owner(self, pk):
        return [{"address": f"ata-{m[:6]}", "mint": m, "amount": a, "decimals": self.decimals.get(m, 6),
                 "program": TOKEN_PROGRAM_ID, "lamports": self.rent.get(m, 0)} for m, a in self.tokens.items()]

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
        if outcome == "success_first_buy":                  # the swap also opens the mint's token account: its rent leaves the wallet
            self.rpc.lamports -= BUY_LAMPORTS + BUY_FEES + RENT
            self.rpc.tokens[MINT] = self.rpc.tokens.get(MINT, 0) + BUY_TOKENS; self.rpc.rent[MINT] = RENT
            self.rpc.landed.add(sig)
            return {"status": "Success", "signature": sig, "slot": "5", "code": 0, "totalInputAmount": str(BUY_LAMPORTS),
                    "totalOutputAmount": str(BUY_TOKENS), "_latency_ms": 900}
        if outcome == "success_buy_beside_a_close":          # a token-account close lands between the swap's two snapshots
            self.rpc.lamports -= BUY_LAMPORTS + BUY_FEES - RENT
            self.rpc.tokens[MINT] = self.rpc.tokens.get(MINT, 0) + BUY_TOKENS
            self.rpc.landed.add(sig)
            return {"status": "Success", "signature": sig, "slot": "5", "code": 0, "totalInputAmount": str(BUY_LAMPORTS),
                    "totalOutputAmount": str(BUY_TOKENS), "_latency_ms": 900}
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


class RawSigner:
    """Stands in for fly_trader/signer here: signs our slot of whatever the fake /order built (the signer's own
    policy has its own tests, tests/test_signer.py)."""

    def __init__(self, kp):
        self.kp = kp

    def call(self, method, **p):
        from fly_trader.chain.signing import sign_transaction_b64, transaction_id
        if method == "pubkeys":
            return {"trading": str(self.kp.pubkey())}
        assert method == "sign_swap" and p["in_amount"] > 0 and p["min_out"] > 0
        signed, _ = sign_transaction_b64(p["tx"], self.kp)
        return {"tx": signed, "signature": transaction_id(signed)}


def _broker(outcomes, lamports=1_000_000_000, tokens=None):
    kp = Keypair()
    rpc = FakeRpc(lamports, tokens, {MINT: 6})
    jup = FakeJup(kp, rpc, outcomes)
    return LiveBroker(rpc=rpc, jup=jup, signer=RawSigner(kp)), rpc, jup


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
    assert jup.orders[0]["exclude_routers"] == "jupiterz,dflow,okx" and jup.orders[0]["slippage_bps"] == 150
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


def test_a_first_buy_keeps_its_token_accounts_rent_apart_from_the_trade(db_conn, gate_open):
    """The first buy of a mint opens its token account: its rent (~0.0015 SOL) is a deposit the account's close returns,
    not the trade. The fill's SOL, price and fee are the swap's own, as paper's are; the deposit is reported apart."""
    broker, rpc, jup = _broker(["success_first_buy"])
    res = _swap(broker, db_conn, random.randrange(1 << 40))
    assert res.ok and res.lamports_delta == -(BUY_LAMPORTS + BUY_FEES) and res.deposit_lamports == RENT
    assert res.price_sol == pytest.approx(((BUY_LAMPORTS + BUY_FEES) / 1e9) / (BUY_TOKENS / 1e6))
    f = _fills(db_conn, [res.order_id])[0]
    assert f["sol_delta_lamports"] == -(BUY_LAMPORTS + BUY_FEES) and f["fee_lamports"] == BUY_FEES
    assert f["post_snapshot"]["deposits"][MINT] == RENT and MINT not in f["pre_snapshot"]["deposits"]


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


def test_a_buy_quoted_past_its_paper_fill_is_refused_before_signing(db_conn, gate_open):
    """The fake /order quotes 1 token (6 decimals) for 0.01 SOL. A decision whose paper fill was 0.0095 SOL per token
    would chase it +5.3 % (over MAX_ENTRY_CHASE): refused with nothing signed or sent; at 0.0098 (+2 %) it is bought."""
    broker, rpc, jup = _broker(["success_buy"])
    did = random.randrange(1 << 40)
    res = broker.swap(db_conn, decision_id=did, book="live", side="buy", mint=MINT, amount_in=BUY_LAMPORTS, slippage_bps=150, max_slippage_bps=300,
                      step_bps=100, ref_price=0.0095, decimals=6)
    assert not res.ok and res.status == "price_ran" and res.error.startswith("price ran") and res.attempts == 1 and res.fill_id is None
    assert jup.executes == [] and len(jup.orders) == 1
    o = _orders(db_conn, did)
    assert len(o) == 1 and o[0]["status"] == "price_ran" and o[0]["order_response"]["outAmount"] == str(BUY_TOKENS)
    res = broker.swap(db_conn, decision_id=did, book="live", side="buy", mint=MINT, amount_in=BUY_LAMPORTS, slippage_bps=150, max_slippage_bps=300,
                      step_bps=100, ref_price=0.0098, decimals=6)
    assert res.ok and res.token_delta == BUY_TOKENS


class _CloseRpc(FakeRpc):
    """The bot's token accounts (``accts``) and close transactions: ``land`` says whether a close lands when sent."""

    def __init__(self, owner, accts, land=True):
        super().__init__(10 ** 9)
        self.owner, self.accts, self.land, self.sent = owner, accts, land, []

    def get_token_accounts_by_owner(self, pk):
        return [dict(a) for a in self.accts]

    def send_transaction(self, tx_b64, skip_preflight=False, max_retries=3):
        sig = f"closesig{len(self.sent)}"; self.sent.append(sig)
        if self.land:
            self.landed.add(sig)
        return sig

    def get_transaction(self, sig, encoding="jsonParsed"):
        gone = [a for a in self.accts if a["amount"] == 0 and a.get("closing")]
        return {"slot": 9, "transaction": {"message": {"accountKeys": [{"pubkey": self.owner}] + [{"pubkey": a["address"]} for a in gone]}},
                "meta": {"err": None, "fee": 5_000, "preBalances": [10 ** 9] + [a["lamports"] for a in gone],
                         "postBalances": [10 ** 9 + sum(a["lamports"] for a in gone) - 5_000] + [0] * len(gone)}}


class _CloseSigner:
    def __init__(self, owner, rpc, on_close=None):
        self.owner, self.rpc, self.on_close, self.closed = owner, rpc, on_close, []

    def call(self, method, **p):
        if method == "pubkeys":
            return {"trading": self.owner}
        assert method == "close_atas"
        self.closed.append(list(p["accounts"]))
        for a in self.rpc.accts:
            a["closing"] = a["address"] in p["accounts"]
        if self.on_close:
            self.on_close()
        return [{"tx": "unsigned-in-tests", "last_valid_block_height": self.rpc.height + 150}]


def _closed_position(conn, mint, cost, proceeds, deposit=0.0):
    pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000, cost_sol=cost, entry_price=1e-7, decision_id=None, fees_sol=0.0, deposit_sol=deposit)
    ledger.close_position(conn, position_id=pid, exit_price=1e-7, proceeds_sol=proceeds, fees_sol=0.0, decision_id=None, forced_kind=None)
    return pid


def test_closing_the_empty_token_accounts_books_each_rent_onto_the_sale_that_emptied_it(db_conn, gate_open):
    """A deposit booked apart comes back and only its share of the fee is its sale's; an account opened before deposits
    were booked apart returns its whole rent to the mint's latest sale; a held or skipped token's account is left alone."""
    owner = str(Keypair().pubkey()); tag = uuid.uuid4().hex[:8]; a, b, h, k = (f"MA{tag}", f"MB{tag}", f"MH{tag}", f"MK{tag}")
    with transaction() as conn:
        pa = _closed_position(conn, a, 0.1, 0.12, deposit=RENT / 1e9)
        pb = _closed_position(conn, b, 0.1 + RENT / 1e9, 0.09)                  # bought before deposits were booked apart: its cost holds the rent
    rpc = _CloseRpc(owner, [{"address": f"acct-{m}", "mint": m, "amount": n, "lamports": RENT} for m, n in ((a, 0), (b, 0), (h, 7), (k, 0))])
    signer = _CloseSigner(owner, rpc)
    try:
        out = broker_live.close_empty_atas(rpc=rpc, signer=signer, skip_mints={k})
        with transaction() as conn:
            ra, rb = (conn.execute("SELECT realized_sol, deposit_refunded FROM positions WHERE id = %s", (i,)).fetchone() for i in (pa, pb))
            ev = conn.execute("SELECT detail FROM wallet_events WHERE kind = 'ata_closed' AND pubkey = %s", (owner,)).fetchone()["detail"]
        assert signer.closed == [[f"acct-{a}", f"acct-{b}"]] and out["closed"] == 2 and out["signatures"] == ["closesig0"]
        assert out["credited"] == pytest.approx((RENT - 5_000) / 1e9)
        assert ra["deposit_refunded"] and ra["realized_sol"] == pytest.approx(0.02 - 2_500 / 1e9)
        assert rb["realized_sol"] == pytest.approx(0.09 - 0.1 - RENT / 1e9 + (RENT - 2_500) / 1e9)
        assert ev["status"] == "confirmed" and ev["booked"][f"acct-{a}"] == {"lamports": RENT - 2_500, "position": pa} and ev["fee"] == 5_000
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM wallet_events WHERE pubkey = %s", (owner,))
            conn.execute("DELETE FROM positions WHERE mint = ANY(%s)", ([a, b, h, k],))


def test_a_close_cut_short_for_an_order_is_settled_once_it_lands_and_never_sent_twice(db_conn, gate_open):
    """An order queued while a close is awaited ends the wait at once (it goes first): the close stays unsettled, its
    accounts are not closed again, and it is booked once seen landed."""
    owner = str(Keypair().pubkey()); mint = f"MU{uuid.uuid4().hex[:8]}"; order = {"queued": False}
    with transaction() as conn:
        pid = _closed_position(conn, mint, 0.1, 0.1, deposit=RENT / 1e9)
    rpc = _CloseRpc(owner, [{"address": "acct-u", "mint": mint, "amount": 0, "lamports": RENT}], land=False)
    signer = _CloseSigner(owner, rpc, on_close=lambda: order.update(queued=True))
    try:
        out = broker_live.close_empty_atas(rpc=rpc, signer=signer, until=lambda: order["queued"])
        assert out["pending"] == ["closesig0"] and out["closed"] == 0 and broker_live.unsettled_closes()
        again = broker_live.close_empty_atas(rpc=rpc, signer=signer)              # not landed yet: its account is not closed twice
        assert again["candidates"] == 0 and len(signer.closed) == 1 and broker_live.unsettled_closes()
        rpc.landed.add("closesig0")
        assert broker_live.settle_closes(rpc) == 0 and not broker_live.unsettled_closes()
        with transaction() as conn:
            p = conn.execute("SELECT realized_sol, deposit_refunded FROM positions WHERE id = %s", (pid,)).fetchone()
        assert p["deposit_refunded"] and p["realized_sol"] == pytest.approx(-5_000 / 1e9)
    finally:
        broker_live._UNSETTLED.clear()
        with transaction() as conn:
            conn.execute("DELETE FROM wallet_events WHERE pubkey = %s", (owner,))
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))


def test_while_a_close_is_unsettled_a_swap_takes_its_own_transactions_deltas(db_conn, gate_open, monkeypatch):
    """A close that may still land would put its rent between a swap's two balance snapshots: the fill is read from the
    swap's own transaction instead."""
    monkeypatch.setattr(broker_live, "tx_deltas", lambda tx, owner, mint: (-(BUY_LAMPORTS + BUY_FEES), BUY_TOKENS))
    monkeypatch.setitem(broker_live._UNSETTLED, "closesig", {"owner": "x", "accounts": [], "last_valid_block_height": 10 ** 9})
    broker, rpc, jup = _broker(["success_buy_beside_a_close"])
    res = _swap(broker, db_conn, random.randrange(1 << 40))
    assert res.ok and res.lamports_delta == -(BUY_LAMPORTS + BUY_FEES) and res.token_delta == BUY_TOKENS


def test_a_swap_takes_its_fill_from_its_own_transaction(db_conn, gate_open, monkeypatch):
    """Several swaps are in flight at once (execution/worker.py): a swap's fill is its own transaction's deltas, never the
    wallet's change between its two snapshots, which another swap may have moved."""
    monkeypatch.setattr(broker_live, "tx_deltas", lambda tx, owner, mint: (-(BUY_LAMPORTS + 1), BUY_TOKENS))
    broker, rpc, jup = _broker(["success_buy"])
    res = _swap(broker, db_conn, random.randrange(1 << 40))
    assert res.ok and res.lamports_delta == -(BUY_LAMPORTS + 1) and res.token_delta == BUY_TOKENS


def test_beside_other_swaps_an_unreadable_transaction_is_an_unknown_fill_never_a_snapshot_guess(db_conn, gate_open, monkeypatch):
    """With another swap in flight the snapshots cannot stand in for a transaction that cannot be read: the fill is unknown
    and entries pause until it is reconciled (verify-fills)."""
    broker, rpc, jup = _broker(["success_buy"])
    monkeypatch.setattr(broker, "_landed", lambda sig: None)
    monkeypatch.setitem(broker_live._SWAPS, "active", 1)                              # another swap in flight meanwhile
    did = random.randrange(1 << 40)
    with transaction() as conn:
        before = conn.execute("SELECT entries_paused FROM circuit_state WHERE id = 1").fetchone()["entries_paused"]
    try:
        with pytest.raises(RuntimeError, match="other swaps"):
            _swap(broker, db_conn, did)
        f = _fills(db_conn, [o["id"] for o in _orders(db_conn, did)])
        with transaction() as conn:
            paused = conn.execute("SELECT entries_paused FROM circuit_state WHERE id = 1").fetchone()["entries_paused"]
        assert [r["verified_by"] for r in f] == ["unknown"] and paused
    finally:
        with transaction() as conn:
            conn.execute("UPDATE circuit_state SET entries_paused = %s WHERE id = 1", (before,))
