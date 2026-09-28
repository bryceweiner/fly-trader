"""The vault's ETH pot: Robinhood Chain profit shared by the same locker weights as the SOL pot, losses carried forward,
never more than the wallet holds liquid; the owed ETH is reserved from live RH trading; a claim's ETH leg is fixed with
its SOL amount, paid once in native ETH to the claim's EVM address (a dropped payout is cancelled and re-sent, never
paid twice), and an ETH-only claim needs no SOL owed."""
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh.wallet import RhWallet
from fly_trader.vault import claims, claims_eth, settle_eth, state
from tests.test_rh_exec import ADDR, KEY, WEI, FakeChain

A1, A2 = "0x" + "11" * 20, "0x" + "22" * 20
T1 = datetime(2026, 9, 21, tzinfo=timezone.utc); T0 = T1 - timedelta(days=7)


def _clean():
    with transaction() as conn:
        for t in ("vault_eth_allocations", "vault_eth_settlements", "vault_allocations", "vault_claims", "vault_settlements", "rh_wallet_flows", "rh_intents"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("DELETE FROM vault_events WHERE account IN (%s, %s)", (A1, A2))
        conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))
        conn.execute("DELETE FROM positions WHERE book IN ('paper_rh_fly', 'live_rh')")
        conn.execute("DELETE FROM wealth_marks WHERE book IN ('paper_rh_fly', 'live_rh')")
        conn.execute("DELETE FROM vault_kv WHERE key LIKE 'settlement_eth_inputs:%%'")


@pytest.fixture
def vault(monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    _clean()
    with transaction() as conn:
        # A1 locks 3 for the whole week, A2 locks 1 from mid-week: weights 3·7d : 1·3.5d
        conn.execute("INSERT INTO vault_events (chain_id, block_number, log_index, tx_hash, block_time, event, account, locked_after) VALUES "
                     "(%s, 1, 0, '0x1', %s, 'Locked', %s, 3), (%s, 2, 0, '0x2', %s, 'Locked', %s, 1)",
                     (config.RH_CHAIN_ID, int(T0.timestamp()) - 10, A1, config.RH_CHAIN_ID, int((T0 + timedelta(days=3.5)).timestamp()), A2))
    state.put("vault_started_at", int(T0.timestamp()) - 100)
    yield
    _clean()


def _sol_settlement(t0, t1) -> int:
    with transaction() as conn:
        return int(conn.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) VALUES (%s, %s, 'allocated', 0, 0) RETURNING id",
                                (t0, t1)).fetchone()["id"])


def _closed(book, realized_eth, closed_at):
    with transaction() as conn:
        conn.execute("INSERT INTO positions (book, mint, opened_at, closed_at, qty, cost_sol, entry_price, realized_sol, fees_sol, status, chain) "
                     "VALUES (%s, '0xaa', %s, %s, 0, 0.01, 1, %s, 0, 'closed', 'rh')", (book, closed_at - timedelta(minutes=30), closed_at, realized_eth))


def test_paper_eth_pot_uses_the_same_weights_and_carries_losses(vault, monkeypatch):
    monkeypatch.setattr(config, "VAULT_BOOK", "paper_fly")
    _closed("paper_rh_fly", 0.03, T1 - timedelta(days=1))
    with transaction() as conn:
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) "
                     "VALUES (%s, 'paper_rh_fly', %s, 1.0, 0, 0, 1.0, 1.0, 0, 0, 0)", (beat, T1))
    sid = _sol_settlement(T0, T1)
    out = settle_eth.run_once()
    assert out[sid]["pot"] == 3 * 10 ** 16
    with transaction() as conn:
        al = {r["evm"]: int(r["wei"]) for r in conn.execute("SELECT evm, wei FROM vault_eth_allocations").fetchall()}
    assert al[A1] + al[A2] <= 3 * 10 ** 16 and al[A1] == pytest.approx(6 * al[A2], rel=1e-9)          # 3 × 7 d : 1 × 3.5 d
    _closed("paper_rh_fly", -0.05, T1 + timedelta(days=1))                                          # a losing week: nothing new, carried
    sid2 = _sol_settlement(T1, T1 + timedelta(days=7))
    assert settle_eth.run_once()[sid2]["pot"] == 0
    _closed("paper_rh_fly", 0.04, T1 + timedelta(days=9))                                         # R back to 0.02: still below what was allocated
    sid3 = _sol_settlement(T1 + timedelta(days=7), T1 + timedelta(days=14))
    assert settle_eth.run_once()[sid3]["pot"] == 0
    _closed("paper_rh_fly", 0.02, T1 + timedelta(days=16))                                        # R 0.04: only the profit above all allocated
    sid4 = _sol_settlement(T1 + timedelta(days=14), T1 + timedelta(days=21))
    assert settle_eth.run_once()[sid4]["pot"] == pytest.approx(4 * 10 ** 16 - (al[A1] + al[A2]), abs=100)   # paper realized is float ETH
    assert settle_eth.run_once() == {}                                                           # each SOL settlement gets one ETH row


def test_live_eth_pot_is_liquid_and_reserved_from_trading(vault, monkeypatch):
    monkeypatch.setattr(config, "VAULT_BOOK", "live"); monkeypatch.setattr(config, "VAULT_RH_GAS_RESERVE_WEI", 10 ** 15)
    with transaction() as conn:
        conn.execute("INSERT INTO rh_wallet_flows (direction, kind, wei) VALUES ('in', 'deposit', %s)", (WEI // 5,))
    sid = _sol_settlement(T0, T1)
    out = settle_eth.run_once(lambda: WEI // 5 + 5 * 10 ** 16)                                    # the wallet made 0.05 ETH
    assert out[sid]["pot"] == 5 * 10 ** 16 and out[sid]["allocated"] <= 5 * 10 ** 16
    with transaction() as conn:
        assert settle_eth.reserved_in_trading(conn) == out[sid]["allocated"]
        from fly_trader.rh.live import reserved_wei
        assert reserved_wei(conn) == out[sid]["allocated"]


def test_live_eth_pot_waits_for_an_intent_in_flight(vault, monkeypatch):
    monkeypatch.setattr(config, "VAULT_BOOK", "live")
    with transaction() as conn:
        conn.execute("INSERT INTO rh_intents (book, kind, token, state) VALUES ('live_rh', 'buy', '0xaa', 'leg1')")
    sid = _sol_settlement(T0, T1)
    assert "waiting" in settle_eth.run_once(lambda: WEI)[sid]


class _Relay:
    def pending_claims(self, after=0):
        return []

    def report(self, rows):
        pass


def _claim(evm, status="verified"):
    with transaction() as conn:
        return int(conn.execute("INSERT INTO vault_claims (evm, sol, nonce, status) VALUES (%s, 'SoLaddr', %s, %s) RETURNING id",
                                (evm, f"n-{evm[-4:]}", status)).fetchone()["id"])


def _eth_alloc(evm, wei):
    with transaction() as conn:
        sid = conn.execute("INSERT INTO vault_settlements (period_start, period_end, status) VALUES (%s, %s, 'allocated') RETURNING id",
                           (T0 - timedelta(days=70), T1 - timedelta(days=70))).fetchone()["id"]
        eid = conn.execute("INSERT INTO vault_eth_settlements (settlement_id, period_start, period_end, status, book, allocated) VALUES (%s, %s, %s, 'allocated', 'live_rh', %s) RETURNING id",
                           (sid, T0, T1, wei)).fetchone()["id"]
        conn.execute("INSERT INTO vault_eth_allocations (eth_settlement_id, evm, weight, wei) VALUES (%s, %s, 1, %s)", (eid, evm, wei))


@pytest.fixture
def chain(monkeypatch):
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 3); monkeypatch.setattr("fly_trader.rh.wallet.RECEIPT_POLL_S", 0.0)
    real_wait = RhWallet.wait
    monkeypatch.setattr(RhWallet, "wait", lambda self, h, timeout_s=120.0, confirmations=None: real_wait(self, h, min(timeout_s, 0.05), confirmations))
    monkeypatch.setattr("fly_trader.vault.alerts.send", lambda *a, **k: None)
    c = FakeChain()
    return c, RhWallet(c, KEY, ADDR)


def test_eth_only_claim_is_paid_once_in_eth(vault, chain):
    c, w = chain
    _eth_alloc(A1, 3 * 10 ** 16)
    cid = _claim(A1)
    out = claims.process(_Relay(), None, None)                           # no SOL owed: the SOL leg closes, the ETH leg waits for RH signing
    with transaction() as conn:
        row = dict(conn.execute("SELECT * FROM vault_claims WHERE id = %s", (cid,)).fetchone())
    assert row["status"] == "paid" and int(row["lamports"]) == 0 and row["eth_status"] == "owed" and int(row["eth_wei"]) == 3 * 10 ** 16
    assert out["eth"]["waiting"] == 1
    from fly_trader.rh import accounting as A
    with transaction() as conn:
        conn.execute("DELETE FROM rh_wallet_marks")
        assert A.reconcile(conn, w, datetime.now(timezone.utc) - timedelta(seconds=5))["ok"]            # the opening balance
    before = c.native
    assert claims_eth.process(w)["paid"] == 1
    with transaction() as conn:
        assert A.reconcile(conn, w)["ok"]                                                             # payout + gas are booked flows
    with transaction() as conn:
        row = dict(conn.execute("SELECT * FROM vault_claims WHERE id = %s", (cid,)).fetchone())
        owed = claims_eth.owed(conn, A1)
        flow = conn.execute("SELECT kind, wei FROM rh_wallet_flows WHERE kind = 'payout'").fetchone()
    assert row["eth_status"] == "paid" and owed == 0 and int(flow["wei"]) == 3 * 10 ** 16
    assert before - c.native == 3 * 10 ** 16 + 40_000 * 10_000_000
    assert claims_eth.process(w) == {"paid": 0, "waiting": 0}                   # nothing left: never paid twice


def test_dropped_payout_is_cancelled_then_sent_again(vault, chain):
    c, w = chain
    _eth_alloc(A2, 10 ** 16)
    cid = _claim(A2)
    with transaction() as conn:
        assert claims_eth.fix_amount(conn, cid, A2)
    c.drop_next = True
    assert claims_eth.pay_one(w, _row(cid)) == "sending"
    with transaction() as conn:
        conn.execute("UPDATE rh_txs SET created_at = now() - interval '10 minutes' WHERE from_addr = %s", (ADDR,))
    before = c.native
    st = claims_eth.pay_one(w, _row(cid))                                 # the dead payout's nonce is cancelled, then re-sent
    assert st == "paid" and c.native == before - 10 ** 16 - 2 * 40_000 * 10_000_000
    with transaction() as conn:
        kinds = sorted(r["kind"] + ":" + r["status"] for r in conn.execute("SELECT kind, status FROM rh_txs WHERE from_addr = %s", (ADDR,)).fetchall())
    assert kinds == ["cancel:mined_ok", "payout:mined_ok", "payout:replaced"]


def _row(cid):
    with transaction() as conn:
        return dict(conn.execute("SELECT * FROM vault_claims WHERE id = %s", (cid,)).fetchone())
