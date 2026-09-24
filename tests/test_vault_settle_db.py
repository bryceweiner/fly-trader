"""A weekly settlement against the test database: snapshot (fake Solana RPC) then allocation from indexed events."""
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import settle, state

T1 = 345600 + 2921 * 604800          # a Monday 00:00 UTC
T0 = T1 - 604800
A, B = "0x" + "aa" * 20, "0x" + "bb" * 20
E18 = 10 ** 18


class Rpc:
    def __init__(self, native):
        self.native = native
    def call(self, method, params=None):
        assert method == "getSlot"
        return 1000
    def get_balance_ctx(self, wallet, commitment="confirmed", min_context_slot=None):
        return self.native, 1000
    def get_token_accounts_by_owner(self, wallet, commitment="confirmed"):
        return [{"mint": "EMPTY", "amount": 0, "lamports": 2_000_000}]
    def get_signatures_for_address(self, *a, **k):
        return []


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(config, "RH_CHAIN_ID", 777)
    monkeypatch.setattr(config, "VAULT_PERIOD_S", 604800)
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3)
    monkeypatch.setattr(config, "VAULT_IMPL_CODEHASHES", [])
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv", "vault_events"):
            c.execute(f"DELETE FROM {t}")
        c.execute("DELETE FROM vault_scan")
        c.execute("DELETE FROM positions WHERE book = 'live'")
        c.execute("DELETE FROM fills WHERE book = 'live'")
        # 5 SOL deposit and a 0.2 SOL gift, both before the snapshot slot; one open position that cost 1 SOL
        c.execute("INSERT INTO vault_flows (signature, slot, direction, kind, counterparty, lamports) VALUES "
                  "('dep', 10, 'in', 'deposit', 'F', 5000000000), ('gift', 20, 'in', 'profit', 'X', 200000000)")
        c.execute("INSERT INTO positions (book, mint, qty, cost_sol, status) VALUES ('live', 'M1', 1, 1.0, 'open')")
        # A holds 100 all week; B locks 100 halfway through; A requests 50 at 3/4
        for n, (t, who, after) in enumerate([(T0 - 10, A, 100), (T0 + 302400, B, 100), (T0 + 453600, A, 50)]):
            c.execute("INSERT INTO vault_events (chain_id, block_number, log_index, tx_hash, block_time, event, account, locked_after) "
                      "VALUES (777, %s, 0, 'h', %s, 'x', %s, %s)", (n + 1, t, who, after * E18))
    state.put("vault_started_at", T0 - 100)
    yield
    with transaction() as c:
        c.execute("DELETE FROM positions WHERE book = 'live'")


def test_weekly_settlement(monkeypatch):
    monkeypatch.setattr("fly_trader.vault.flows.scan", lambda rpc, wallet: {"through_slot": 1000})
    monkeypatch.setattr("fly_trader.vault.flows.scanned_through", lambda: 1000)
    # the wallet made +0.8 SOL closed profit: 5 deposit + 0.2 gift + 0.8 - 1.0 open cost = 5.0 native
    native = 5_000_000_000
    assert settle.due(T1 + 60) == (T0, T1)
    sid = settle.take_snapshot(Rpc(native), "W", T0, T1)
    with transaction() as c:
        s = c.execute("SELECT * FROM vault_settlements WHERE id = %s", (sid,)).fetchone()
    # R = N + K + C - D = 5.0 + 0.002 + 1.0 - 5.0 = 1.002 SOL (the gift counts as profit)
    assert s["realized"] == 1_002_000_000 and s["status"] == "snapshotted"
    assert settle.allocate_settlement(sid, {"through_time": T1 - 1})["waiting"]
    res = settle.allocate_settlement(sid, {"through_time": T1 + 1, "through_block": 99})
    # liquid cap: 5.0 - 0 reserved - 0.3 gas = 4.7 > pot 1.002, so the whole pot goes out
    assert res["pot"] == 1_002_000_000 and res["allocated"] <= res["pot"] and res["pot"] - res["allocated"] < 3
    with transaction() as c:
        al = {r["evm"]: int(r["lamports"]) for r in c.execute("SELECT evm, lamports FROM vault_allocations")}
    # weights: A = 100*0.75 + 50*0.25 = 87.5 ; B = 100*0.5 = 50  (week units)
    assert abs(al[A] / al[B] - 87.5 / 50) < 1e-6
    assert settle.due(T1 + 60) is None                       # settled once
    sid2 = settle.take_snapshot(Rpc(native - 300_000_000), "W", T1, T1 + 604800)   # a 0.3 SOL loss next week
    res2 = settle.allocate_settlement(sid2, {"through_time": T1 + 604801})
    assert res2["pot"] == 0 and res2["allocated"] == 0      # nothing until the loss is earned back
