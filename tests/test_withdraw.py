"""Withdrawals from the bot wallets to the operator's own wallets (chain/withdraw.py, rh/withdraw.py, ops/wallets.py): only
to listed addresses, without the live switches but with every other guard; what must stay behind (rent, the gas reserve
while live positions are open); the signed transaction recorded before broadcast; an RH withdrawal booked once as a wallet
flow, whichever path records its receipt, so the reconciler still matches the chain; a confirmed withdrawal re-bases its
own kill switch only; the Safety page shows both wallets in USD and sends one withdrawal per dialog."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from fly_trader import config
from fly_trader.agent import rails
from fly_trader.chain import withdraw as sw
from fly_trader.chain.cluster_guard import SigningRefused, assert_withdraw_allowed
from fly_trader.chain.rpc import RpcError
from fly_trader.db.connection import transaction
from fly_trader.markets import RH_CIRCUIT, SOLANA_CIRCUIT
from fly_trader.rh import accounting as A, guard, withdraw as rw
from fly_trader.rh.exec import RhExecutor
from fly_trader.rh.tx import address_of, decode_raw, recover_sender
from fly_trader.rh.wallet import RhWallet

LAM, WEI = config.LAMPORTS_PER_SOL, 10 ** 18
SYSTEM = "11111111111111111111111111111111"
UI = Path(__file__).resolve().parents[1] / "fly_trader" / "ui"


# ---------------------------------------------------------------- Solana
class FakeSol:
    def __init__(self, me: str, lamports: int, to: str):
        self.bal = {me: lamports, to: LAM}; self.height = 100; self.lands = True; self.refuse = None; self.sent = []; self.rows_at_send = []

    def get_balance(self, pk):
        return self.bal.get(str(pk), 0)

    def get_latest_blockhash(self):
        return {"blockhash": str(Hash.new_unique()), "last_valid_block_height": self.height + 150}

    def send_transaction(self, tx_b64, skip_preflight=False, max_retries=3):
        with transaction() as conn:                                      # what the database holds when the network sees it
            self.rows_at_send.append(conn.execute("SELECT detail->>'status' AS s FROM wallet_events WHERE kind = 'withdrawal' ORDER BY id DESC LIMIT 1").fetchone()["s"])
        if self.refuse:
            raise RpcError("sendTransaction", {"code": -32002, "message": self.refuse})
        tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64)); self.sent.append(tx)
        return str(tx.signatures[0])

    def get_signature_statuses(self, sigs, search_history=False):
        return [{"slot": 7, "err": None, "confirmationStatus": "confirmed"} if self.lands else None for _ in sigs]

    def get_block_height(self):
        self.height += 100
        return self.height


@pytest.fixture
def sol(monkeypatch):
    kp, to = Keypair(), str(Keypair().pubkey())
    me = str(kp.pubkey())
    monkeypatch.setattr(config, "FUNDING_ADDRESSES", [to]); monkeypatch.setattr(config, "VAULT_ENABLED", False)
    monkeypatch.setattr(config, "SOLANA_CLUSTER", "mainnet-beta"); monkeypatch.setattr(config, "LIVE_ENABLED", False)
    monkeypatch.setattr("fly_trader.execution.broker_live.CONFIRM_POLL_S", 0.0)

    def clean():
        with transaction() as conn:
            conn.execute("DELETE FROM wallet_events WHERE pubkey = %s", (me,))
            conn.execute("DELETE FROM circuit_events WHERE kind = 'withdrawal'")
            conn.execute("DELETE FROM positions WHERE book = 'live'")
    clean()
    yield SimpleNamespace(kp=kp, me=me, to=to, rpc=FakeSol(me, 2 * LAM, to))
    clean()


def test_sol_plan_leaves_zero_or_rent_and_the_gas_reserve_while_positions_are_open():
    lim = {"balance": 2 * LAM, "fee": sw.FEE_LAMPORTS, "keep": 0, "n_open": 0, "max": 2 * LAM - sw.FEE_LAMPORTS, "to_balance": LAM}
    assert sw.plan(lim, None) == (2 * LAM - 5_000, None)                                  # everything: the wallet ends at 0
    assert sw.plan(lim, 0.5) == (LAM // 2, None)
    assert "leave at least" in sw.plan(lim, 1.999994)[1]                                  # 1,000 lamports of dust: Solana refuses it
    assert "at most" in sw.plan(lim, 2.0)[1] and sw.plan(lim, 0.0)[1] == "enter an amount"
    assert "empty wallet" in sw.plan({**lim, "to_balance": 0}, 0.0001)[1]
    keep = {**lim, "keep": int(0.3 * LAM), "n_open": 2, "max": 2 * LAM - 5_000 - int(0.3 * LAM)}
    assert sw.plan(keep, None) == (int(1.7 * LAM) - 5_000, None)
    assert "2 open live position(s)" in sw.plan(keep, 1.8)[1]


def test_sol_open_live_positions_keep_the_gas_reserve(sol):
    assert sw.limits(sol.rpc, sol.me, sol.to)["keep"] == 0
    with transaction() as conn:
        conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status) VALUES ('live', 'withdraw-test-mint', 1, 0.1, 'open')")
    lim = sw.limits(sol.rpc, sol.me, sol.to)
    assert lim["n_open"] == 1 and lim["keep"] == int(config.GAS_RESERVE_SOL * LAM) and lim["max"] == 2 * LAM - 5_000 - lim["keep"]


def test_sol_gate_needs_a_listed_wallet_but_not_live_trading(sol, monkeypatch):
    assert_withdraw_allowed(sol.to)                                                       # LIVE_ENABLED is off
    with pytest.raises(SigningRefused, match="FUNDING_ADDRESSES"):
        assert_withdraw_allowed(str(Keypair().pubkey()))
    with pytest.raises(SigningRefused, match="FUNDING_ADDRESSES"):
        sw.withdraw(0.1, str(Keypair().pubkey()), rpc=sol.rpc, keypair=sol.kp)
    assert sol.rpc.sent == []
    monkeypatch.setattr(config, "SOLANA_CLUSTER", "devnet")
    with pytest.raises(SigningRefused, match="devnet"):
        assert_withdraw_allowed(sol.to)
    monkeypatch.setattr(config, "SOLANA_CLUSTER", "mainnet-beta"); monkeypatch.setattr(config, "VAULT_ENABLED", True)
    with pytest.raises(SigningRefused, match="treasury"):
        assert_withdraw_allowed(sol.to)


def test_sol_withdrawal_is_recorded_before_broadcast_and_rebases_only_its_kill_switch(sol):
    r = sw.withdraw(0.5, sol.to, rpc=sol.rpc, keypair=sol.kp)
    assert r["status"] == "confirmed" and r["lamports"] == LAM // 2 and sol.rpc.rows_at_send == ["signed"]
    tx = sol.rpc.sent[0]; keys = [str(k) for k in tx.message.account_keys]; ix = tx.message.instructions[0]
    assert keys[0] == sol.me and keys[ix.program_id_index] == SYSTEM and keys[ix.accounts[1]] == sol.to
    assert bytes(ix.data) == (2).to_bytes(4, "little") + (LAM // 2).to_bytes(8, "little")    # SystemProgram transfer of exactly that
    assert tx.signatures[0].verify(sol.kp.pubkey(), to_bytes_versioned(tx.message))
    with transaction() as conn:
        ev = conn.execute("SELECT detail FROM wallet_events WHERE kind = 'withdrawal' AND pubkey = %s", (sol.me,)).fetchone()["detail"]
        assert ev["status"] == "confirmed" and ev["signature"] == r["signature"] and ev["to"] == sol.to
        t1, t3 = rails.kill_rebase_at(conn, SOLANA_CIRCUIT), rails.kill_rebase_at(conn, RH_CIRCUIT)
    assert t1 is not None and (t3 is None or t3 < t1)


def test_sol_refused_or_expired_withdrawals_move_nothing_and_keep_the_peak(sol):
    sol.rpc.refuse = "Transaction simulation failed: insufficient funds"
    with pytest.raises(sw.WithdrawRefused, match="refused"):
        sw.withdraw(0.5, sol.to, rpc=sol.rpc, keypair=sol.kp)
    sol.rpc.refuse = None; sol.rpc.lands = False
    assert sw.withdraw(0.5, sol.to, rpc=sol.rpc, keypair=sol.kp)["status"] == "expired"
    with pytest.raises(sw.WithdrawRefused, match="at most"):
        sw.withdraw(5.0, sol.to, rpc=sol.rpc, keypair=sol.kp)                             # refused before anything is signed
    with transaction() as conn:
        st = [r["s"] for r in conn.execute("SELECT detail->>'status' AS s FROM wallet_events WHERE kind = 'withdrawal' AND pubkey = %s ORDER BY id", (sol.me,))]
        assert st == ["rejected", "expired"]
        assert conn.execute("SELECT count(*) AS n FROM circuit_events WHERE kind = 'withdrawal'").fetchone()["n"] == 0


# ---------------------------------------------------------------- Robinhood Chain
KEY = bytes.fromhex("8da4ef21b864d2cc526dbdb2a120bd2874c36c9d0a1fb7f8c63d7f7a8b41de8f")
ADDR = address_of(KEY)
DEST = "0x" + "d1" * 20
GAS_USED, BASE_FEE = 21_000, 10_000_000
FEE_CAP = (21_000 * 13 // 10 + 10_000) * 2 * BASE_FEE                                    # the estimate's margin at twice the base fee


class FakeRh:
    chain_id = 4663

    def __init__(self, native: int):
        self.native = native; self.nonce = 0; self.head = 100; self.receipts = {}; self.hidden = {}; self.hide = False
        self.sent = []; self.rows_at_send = []

    def chain(self):
        return self.chain_id

    def get_balance(self, a, tag="latest"):
        return self.native

    def base_fee(self):
        return BASE_FEE

    def max_priority_fee(self):
        return 0

    def estimate_gas(self, tx):
        return 21_000

    def tx_count(self, a, tag="pending"):
        return self.nonce

    def block_number(self, tag="latest"):
        return self.head

    def receipt(self, h):
        return self.receipts.get(h)

    def send_raw(self, raw):
        with transaction() as conn:
            self.rows_at_send.append(conn.execute("SELECT status FROM rh_txs WHERE raw = %s", (raw,)).fetchone()["status"])
        stx = decode_raw(bytes.fromhex(raw[2:])); self.sent.append(stx)
        assert stx.tx.nonce == self.nonce
        self.nonce += 1; self.native -= GAS_USED * BASE_FEE + stx.tx.value
        rc = {"blockNumber": hex(self.head - 5), "blockHash": "0xbb", "gasUsed": hex(GAS_USED), "effectiveGasPrice": hex(BASE_FEE), "status": "0x1", "logs": []}
        (self.hidden if self.hide else self.receipts)[stx.hash] = rc
        return stx.hash


@pytest.fixture
def rh(monkeypatch):
    monkeypatch.setattr(config, "RH_FUNDING_ADDRESSES", ["0x" + "D1" * 20]); monkeypatch.setattr(config, "VAULT_ENABLED", False)
    monkeypatch.setattr(config, "RH_LIVE_ENABLED", False); monkeypatch.setenv("RH_BOT_PRIVATE_KEY", "0x" + KEY.hex())
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", ADDR); monkeypatch.setattr(config, "RH_EXPECTED_CHAIN_ID", 4663)
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 3); monkeypatch.setattr(config, "RH_MAX_FEE_GWEI", 5.0)
    monkeypatch.setattr("fly_trader.rh.wallet.RECEIPT_POLL_S", 0.0)
    real_wait = RhWallet.wait
    monkeypatch.setattr(RhWallet, "wait", lambda self, h, timeout_s=120.0, confirmations=None: real_wait(self, h, min(timeout_s, 0.05), confirmations))

    def clean():
        with transaction() as conn:
            for t in ("rh_wallet_flows", "rh_wallet_marks", "rh_base_lots", "rh_intents"):
                conn.execute(f"DELETE FROM {t}")
            conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))
            conn.execute("DELETE FROM positions WHERE book = 'live_rh'")
            conn.execute("DELETE FROM circuit_events WHERE kind = 'withdrawal'")
            conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, kill_reason=NULL, peak_wealth=NULL, entries_paused=false WHERE id = 3")
    clean()
    c = FakeRh(WEI)
    yield SimpleNamespace(c=c, w=lambda: rw.wallet(DEST, c))
    clean()


def _flows() -> list[int]:
    with transaction() as conn:
        return [int(r["wei"]) for r in conn.execute("SELECT wei FROM rh_wallet_flows WHERE kind = 'withdrawal' AND direction = 'out' ORDER BY id")]


def test_rh_guard_needs_a_listed_address_but_not_live_trading(rh, monkeypatch):
    assert guard.check_withdrawal(rh.c, DEST) == (KEY, ADDR)                               # RH_LIVE_ENABLED is off
    with pytest.raises(guard.SigningRefused, match="RH_LIVE_ENABLED"):
        guard.check(rh.c)                                                                  # trading still is not allowed
    with pytest.raises(guard.SigningRefused, match="RH_FUNDING_ADDRESSES"):
        guard.check_withdrawal(rh.c, "0x" + "e2" * 20)
    rh.c.chain_id = 46630
    with pytest.raises(guard.SigningRefused, match="chain 46630"):
        guard.check_withdrawal(rh.c, DEST)
    rh.c.chain_id = 4663
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", "0x" + "00" * 20)
    with pytest.raises(guard.SigningRefused, match="derives to"):
        guard.check_withdrawal(rh.c, DEST)
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", ADDR); monkeypatch.setattr(config, "VAULT_ENABLED", True)
    with pytest.raises(guard.SigningRefused, match="vault"):
        guard.check_withdrawal(rh.c, DEST)


def test_rh_withdrawal_is_booked_once_and_the_books_still_match_the_chain(rh):
    w = rh.w()
    with transaction() as conn:
        assert A.reconcile(conn, w)["ok"]                                                  # the opening balance
    r = rw.withdraw(0.25, DEST.upper().replace("0X", "0x"), w=w)
    assert r["status"] == "mined_ok" and r["wei"] == WEI // 4 and rh.c.rows_at_send == ["signed"]
    stx = rh.c.sent[0]
    assert stx.tx.to == DEST and stx.tx.value == WEI // 4 and stx.tx.data == b"" and recover_sender(stx) == ADDR
    assert _flows() == [WEI // 4]
    with transaction() as conn:
        tx = conn.execute("SELECT id, kind, status, applied_at FROM rh_txs WHERE hash = %s", (r["hash"],)).fetchone()
        assert tx["kind"] == "transfer" and tx["status"] == "mined_ok" and tx["applied_at"] is not None
        w.record_receipt(conn, int(tx["id"]), rh.c.receipts[r["hash"]])                   # recorded again: never booked twice
    assert _flows() == [WEI // 4]
    with transaction() as conn:
        assert A.reconcile(conn, w)["ok"]                                                  # the ETH that left is explained
        assert not rails.load_circuit(conn, RH_CIRCUIT).kill_switch
        t3, t1 = rails.kill_rebase_at(conn, RH_CIRCUIT), rails.kill_rebase_at(conn, SOLANA_CIRCUIT)
    assert t3 is not None and (t1 is None or t1 < t3)


def test_rh_everything_keeps_the_gas_reserve_while_positions_are_open(rh):
    w = rh.w()
    lim = rw.limits(w, DEST)
    assert lim["keep"] == 0 and lim["max"] == WEI - FEE_CAP and rw.plan(lim, None) == (WEI - FEE_CAP, None)
    with transaction() as conn:
        conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status, chain) VALUES ('live_rh', %s, 1, 0.01, 'open', 'rh')", ("0x" + "ab" * 20,))
    lim = rw.limits(w, DEST)
    assert lim["keep"] == int(config.RH_GAS_RESERVE_ETH * WEI) and lim["max"] == WEI - FEE_CAP - lim["keep"]
    assert "1 open live position(s)" in rw.plan(lim, 0.9999)[1]
    r = rw.withdraw(None, DEST, w=w)
    assert r["wei"] == lim["max"] and rh.c.native >= lim["keep"]


def test_rh_withdrawal_without_a_receipt_is_booked_before_the_next_one(rh):
    w = rh.w(); rh.c.hide = True
    assert rw.withdraw(0.1, DEST, w=w)["status"] == "pending" and _flows() == []
    rh.c.receipts.update(rh.c.hidden); rh.c.hide = False
    assert rw.withdraw(0.1, DEST, w=w)["status"] == "mined_ok"
    assert _flows() == [WEI // 10, WEI // 10]


def test_the_executors_recovery_books_a_pending_withdrawal_too(rh):
    w = rh.w(); rh.c.hide = True
    rw.withdraw(0.1, DEST, w=w)
    rh.c.receipts.update(rh.c.hidden)
    RhExecutor(w, start=False)                                                             # recover() on start
    assert _flows() == [WEI // 10]


# ---------------------------------------------------------------- the console page
def _summary(chain, address, native, px):
    return {"chain": chain, "unit": "SOL" if chain == "sol" else "ETH", "address": address, "native": native, "error": None, "n_open": 0,
            "positions": 0.0, "marked_at": None, "price_usd": px, "price_at": None, "native_usd": native * px, "positions_usd": 0.0, "total_usd": native * px}


def test_safety_page_shows_both_wallets_in_usd_and_sends_one_withdrawal_per_dialog(db_conn, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from fly_trader.ops import wallets
    st.cache_data.clear()
    sol_addr, to = str(Keypair().pubkey()), str(Keypair().pubkey())
    fake = {"sol": _summary("sol", sol_addr, 2.0, 100.0), "rh": _summary("rh", "0x" + "ab" * 20, 0.5, 2600.0)}
    sent = []
    monkeypatch.setattr(config, "RH_ENABLED", True); monkeypatch.setattr(config, "FUNDING_ADDRESSES", [])
    monkeypatch.setattr(wallets, "summary", lambda chain: fake[chain])
    monkeypatch.setattr(wallets, "address", lambda chain: fake[chain]["address"])
    at = AppTest.from_file(str(UI / "app_pages" / "safety.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    shown = {(m.label, m.value) for m in at.metric}
    assert {("Balance", "$200.00"), ("Balance", "$1,300.00")} <= shown
    assert at.button(key="sol_withdraw").disabled                                          # no FUNDING_ADDRESSES: nowhere to send
    monkeypatch.setattr(config, "FUNDING_ADDRESSES", [to])
    monkeypatch.setattr(wallets, "withdraw_blockers", lambda chain: [])
    monkeypatch.setattr(wallets, "limits", lambda chain, dest: {"balance": 2 * LAM, "fee": 5_000, "keep": 0, "n_open": 0, "max": 2 * LAM - 5_000, "to_balance": LAM})
    monkeypatch.setattr(wallets, "withdraw", lambda chain, amt, dest: sent.append((chain, amt, dest)) or {"status": "confirmed", "tx": "sig", "amount": amt, "to": dest})
    at.run()
    at.button(key="sol_withdraw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="sol_wd_to").options == [to] and at.button(key="sol_wd_go").disabled  # no amount yet
    at.number_input(key="sol_wd_amt").set_value(0.5).run()
    assert not at.button(key="sol_wd_go").disabled and at.button(key="sol_wd_go").label == "Withdraw 0.500000 SOL"
    at.button(key="sol_wd_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert sent == [("sol", 0.5, to)] and any("Sent 0.500000 SOL" in s.value for s in at.success)
    assert not [b for b in at.button if b.key == "sol_wd_go"]                              # the result replaces the form: no second send
    at.button(key="sol_wd_close").click().run()
    assert "withdraw_open" not in at.session_state and not [b for b in at.button if b.key == "sol_wd_close"]
