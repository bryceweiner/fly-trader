"""The RH wallet on a fake chain: the signed tx is committed before broadcast, nonces never collide or leak, the fee cap
refuses spikes, approvals are exact (reset to 0 only for a token that needs it), receipts store their fee, and the guard
refuses a wrong chain, a wrong key or live trading switched off."""
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh import abi, guard
from fly_trader.rh.tx import address_of, decode_raw, recover_sender
from fly_trader.rh.wallet import FeeTooHigh, RhWallet, SendFailed
from fly_trader.vault.evm import EvmRpcError

KEY = bytes.fromhex("4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318")
ADDR = address_of(KEY)


class FakeRpc:
    chain_id = 4663

    def __init__(self):
        self.pending = 0; self.base = 10_000_000; self.tip = 0; self.sent = []; self.receipts = {}; self.head = 100
        self.allow = {}; self.fail_send = None; self.refuse_nonzero = False; self.seen_rows_at_send = []

    def chain(self):
        return self.chain_id

    def base_fee(self):
        return self.base

    def max_priority_fee(self):
        return self.tip

    def estimate_gas(self, tx):
        return 50_000

    def tx_count(self, a, tag="pending"):
        return self.pending

    def block_number(self, tag="latest"):
        return self.head

    def send_raw(self, raw):
        with transaction() as conn:                                   # what the database holds at the moment of broadcast
            self.seen_rows_at_send.append(conn.execute("SELECT status FROM rh_txs WHERE raw = %s", (raw,)).fetchone())
        if self.fail_send:
            raise EvmRpcError("eth_sendRawTransaction", {"message": self.fail_send})
        stx = decode_raw(bytes.fromhex(raw[2:])); self.sent.append(stx)
        self.receipts[stx.hash] = {"blockNumber": hex(self.head - 5), "blockHash": "0xbb", "gasUsed": hex(40_000), "effectiveGasPrice": hex(self.base),
                                   "status": "0x1", "logs": []}
        if stx.tx.data[:4] == abi.selector("approve(address,uint256)"):
            spender, amt = abi.decode(["address", "uint256"], stx.tx.data[4:]); self.allow[(stx.tx.to, spender)] = amt
        return stx.hash

    def receipt(self, h):
        return self.receipts.get(h)

    def eth_call(self, to, data, block="latest"):
        d = bytes.fromhex(data[2:])
        if d[:4] == abi.selector("allowance(address,address)"):
            _, sp = abi.decode(["address", "address"], d[4:]); return "0x" + abi.encode(["uint256"], [self.allow.get((to.lower(), sp), 0)]).hex()
        raise AssertionError("unexpected eth_call")

    def call(self, method, params):
        assert method == "eth_call"
        d = bytes.fromhex(params[0]["data"][2:])
        if self.refuse_nonzero and d[:4] == abi.selector("approve(address,uint256)"):
            _, amt = abi.decode(["address", "uint256"], d[4:])
            if amt and any(v for (t, _), v in self.allow.items() if t == params[0]["to"].lower()):
                raise EvmRpcError("eth_call", {"message": "execution reverted: approve from non-zero"})
        return "0x"


@pytest.fixture
def w(monkeypatch):
    with transaction() as conn:
        conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 3); monkeypatch.setattr(config, "RH_MAX_FEE_GWEI", 5.0)
    monkeypatch.setattr("fly_trader.rh.wallet.RECEIPT_POLL_S", 0.0)
    yield RhWallet(FakeRpc(), KEY, ADDR)
    with transaction() as conn:
        conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))


def test_signed_tx_is_committed_before_broadcast_and_signs_for_the_right_chain(w):
    r = w.send_and_wait(to="0x" + "11" * 20, data=b"\x01\x02", value=7, kind="swap")
    assert w.rpc.seen_rows_at_send == [{"status": "signed"}]                       # durable before the node ever saw it
    stx = w.rpc.sent[0]
    assert stx.tx.chain_id == 4663 and stx.tx.value == 7 and recover_sender(stx) == ADDR and stx.tx.gas == 50_000 * 13 // 10 + 10_000
    assert r["ok"] is True and r["fee_wei"] == 40_000 * w.rpc.base
    with transaction() as conn:
        row = conn.execute("SELECT status, fee_wei, nonce FROM rh_txs WHERE id = %s", (r["id"],)).fetchone()
    assert row["status"] == "mined_ok" and int(row["fee_wei"]) == 40_000 * w.rpc.base and row["nonce"] == 0


def test_nonces_follow_the_chain_and_our_own_rows_and_a_failed_send_frees_its_nonce(w):
    a = w.send(to="0x" + "11" * 20, kind="swap"); b = w.send(to="0x" + "11" * 20, kind="swap")
    assert (a["nonce"], b["nonce"]) == (0, 1)                                     # the chain still says 0 pending: our rows count
    w.rpc.pending = 5
    assert w.send(to="0x" + "11" * 20, kind="swap")["nonce"] == 5                  # the chain knows more (another signer, a restore)
    w.rpc.fail_send = "insufficient funds for gas"
    with pytest.raises(SendFailed):
        w.send(to="0x" + "11" * 20, kind="swap")
    w.rpc.fail_send = None
    assert w.send(to="0x" + "11" * 20, kind="swap")["nonce"] == 6                  # the dropped attempt's nonce is reused


def test_fee_cap_refuses_before_signing(w):
    w.rpc.base = 6 * 10 ** 9
    with pytest.raises(FeeTooHigh):
        w.send(to="0x" + "11" * 20, kind="swap")
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM rh_txs WHERE from_addr = %s", (ADDR,)).fetchone()["n"] == 0
    w.rpc.base = 2 * 10 ** 9
    s = w.send(to="0x" + "11" * 20, kind="swap")
    assert s["max_fee"] == 4 * 10 ** 9 and w.rpc.sent[-1].tx.max_fee_per_gas <= 5 * 10 ** 9


def test_approvals_are_exact_skipped_when_covered_and_reset_only_when_needed(w):
    tok, sp = "0x" + "aa" * 20, "0x" + "bb" * 20
    with pytest.raises(ValueError):
        w.approve_exact(tok, sp, 2 ** 256 - 1)                                     # never unlimited
    r = w.approve_exact(tok, sp, 1000)
    assert r["ok"] and w.rpc.allow[(tok, sp)] == 1000
    assert w.approve_exact(tok, sp, 900) is None                                   # already covered: nothing signed
    w.rpc.refuse_nonzero = True
    w.approve_exact(tok, sp, 5000)
    amounts = [abi.decode(["address", "uint256"], s.tx.data[4:])[1] for s in w.rpc.sent]
    assert amounts == [1000, 0, 5000]                                              # reset to 0 first, then exactly 5000


def test_cancel_outbids_by_a_quarter(w):
    s = w.send(to="0x" + "11" * 20, kind="swap")
    c = w.cancel(s["nonce"], s["max_fee"])
    tx = w.rpc.sent[-1].tx
    assert c["nonce"] == s["nonce"] and tx.to == ADDR and tx.value == 0 and tx.max_fee_per_gas > s["max_fee"] * 5 // 4


def test_guard(monkeypatch):
    rpc = FakeRpc()
    monkeypatch.setattr(config, "RH_LIVE_ENABLED", False)
    with pytest.raises(guard.SigningRefused, match="RH_LIVE_ENABLED"):
        guard.check(rpc)
    monkeypatch.setattr(config, "RH_LIVE_ENABLED", True); monkeypatch.setenv("RH_BOT_PRIVATE_KEY", "0x" + KEY.hex())
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", "0x" + "00" * 20)
    with pytest.raises(guard.SigningRefused, match="derives to"):
        guard.check(rpc)
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", ADDR.upper().replace("0X", "0x"))
    monkeypatch.setattr(config, "RH_EXPECTED_CHAIN_ID", 46630)
    with pytest.raises(guard.SigningRefused, match="chain 4663"):
        guard.check(rpc)
    monkeypatch.setattr(config, "RH_EXPECTED_CHAIN_ID", 4663)
    assert guard.check(rpc) == (KEY, ADDR)
