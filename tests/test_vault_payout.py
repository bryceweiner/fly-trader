"""The treasury and the trading float: top-ups and sweeps move SOL inside one book without being profit, loss or a flow."""
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import flows, nav, payout, settle, state

from test_signer import L1, ME, TREASURY, limit, vault_signer

SOL = 10**9
RENT = payout.TREASURY_RENT


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {"VAULT_ENABLED": True, "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": "http://devnet.invalid",
                 "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": False, "GAS_RESERVE_SOL": 0.3, "TELEGRAM_BOT_TOKEN": None,
                 "TREASURY_FLOAT_SOL": 2.0, "TOPUP_MIN_SOL": 0.1, "VAULT_BOOK": "live"}.items():
        monkeypatch.setattr(config, k, v)
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv"):
            c.execute(f"DELETE FROM {t}")
        c.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) "
                  "VALUES (now() - interval '7 days', now(), 'allocated', 400000000, 250000000)")
    yield


def test_plan():
    g, m = int(0.3 * SOL), int(0.1 * SOL)
    assert payout.plan(1 * SOL, 5 * SOL, 0, 2 * SOL, g, m) == ("topup", 1 * SOL)                  # refill to the float
    assert payout.plan(3 * SOL, 5 * SOL, 0, 2 * SOL, g, m) == ("sweep", 1 * SOL)                  # excess back
    assert payout.plan(int(1.95 * SOL), 5 * SOL, 0, 2 * SOL, g, m) == (None, 0)                  # within the dust band
    assert payout.plan(1 * SOL, int(1.5 * SOL), 1 * SOL, 2 * SOL, g, m) == ("topup", SOL // 2 - RENT)   # never lends what holders are owed
    # the treasury cannot pay what is owed: the float gives back what it can spare above gas, even below its target
    assert payout.plan(1 * SOL, 0, int(0.5 * SOL), 2 * SOL, g, m) == ("sweep", int(0.5 * SOL) + RENT)
    assert payout.plan(int(0.35 * SOL), 0, 1 * SOL, 2 * SOL, g, m) == ("sweep", int(0.05 * SOL) - payout.FEE)
    assert payout.plan(int(0.3 * SOL), 0, 1 * SOL, 2 * SOL, g, m) == (None, 0)                   # never below gas


class Rpc:
    def __init__(self, trading, treasury):
        self.bal = {str(ME): trading, str(TREASURY): treasury}
        self.sent = []

    def get_balance(self, pk):
        return self.bal[str(pk)]

    def send_transaction(self, b64):
        self.sent.append(b64)


def _confirm(monkeypatch, slot=55):
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": slot}))


def test_topup_and_sweep_are_internal_and_known(monkeypatch):
    _confirm(monkeypatch)
    sg = vault_signer()
    out = payout.rebalance(Rpc(1 * SOL, 5 * SOL), sg, str(ME), str(TREASURY))
    assert out["action"] == "topup" and out["lamports"] == 1 * SOL
    out = payout.rebalance(Rpc(3 * SOL, 5 * SOL), sg, str(ME), str(TREASURY))
    assert out["action"] == "sweep" and out["lamports"] == 1 * SOL
    with transaction() as c:
        rows = c.execute("SELECT kind, direction, slot, lamports, signature FROM vault_flows ORDER BY id").fetchall()
        totals = settle.flow_totals(c, 2**62)
    assert [(r["kind"], r["direction"], r["slot"], r["lamports"]) for r in rows] == [("topup", "in", 55, SOL), ("sweep", "out", 55, SOL)]
    assert totals == {"deposits": 0, "withdrawals": 0, "payouts": 0, "gifts": 0}              # neither is a flow
    with transaction() as c:
        assert {r["signature"] for r in rows} <= flows.known_ours(c, [r["signature"] for r in rows])   # no halt when scanned


def test_refused_topup_alerts_and_writes_nothing(monkeypatch):
    _confirm(monkeypatch)
    sent = []
    monkeypatch.setattr(payout.alerts, "send", lambda text, **k: sent.append(text))
    sg = vault_signer()
    del sg.chain.acc[L1]                                                   # the owner revoked L1
    out = payout.rebalance(Rpc(1 * SOL, 5 * SOL), sg, str(ME), str(TREASURY))
    assert out["code"] == "cap" and sent and "cannot refill" in sent[0]
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM vault_flows").fetchone()["n"] == 0


def test_bankroll_counts_the_treasury_but_reserves_what_is_owed():
    state.put("treasury_balance", {"lamports": 3 * SOL})
    with transaction() as c:
        assert nav.reserved_lamports(c) == 250_000_000
        assert nav.reserved_in_trading(c) == 0                             # the treasury holds all that is owed
        assert nav.treasury_free(c) == 3 * SOL - 250_000_000
    state.put("treasury_balance", {"lamports": 100_000_000})
    with transaction() as c:
        assert nav.reserved_in_trading(c) == 150_000_000 and nav.treasury_free(c) == 0
