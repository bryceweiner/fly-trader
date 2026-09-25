"""The payout wallet: sweeps move owed SOL out of the trading wallet without being profit, loss or a flow."""
import pytest
from solders.keypair import Keypair

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import nav, payout, settle, state

TRADING = Keypair.from_seed(bytes([3]) * 32)
PAYOUT = Keypair.from_seed(bytes([4]) * 32)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {"VAULT_ENABLED": True, "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": "http://devnet.invalid",
                 "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": False, "GAS_RESERVE_SOL": 0.3, "TELEGRAM_BOT_TOKEN": None,
                 "PAYOUT_FEE_BUFFER_LAMPORTS": 10_000_000, "PAYOUT_MIN_SWEEP_LAMPORTS": 1_000_000, "VAULT_BOOK": "live"}.items():
        monkeypatch.setattr(config, k, v)
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv"):
            c.execute(f"DELETE FROM {t}")
        c.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) "
                  "VALUES (now() - interval '7 days', now(), 'allocated', 400000000, 250000000)")
    yield


def test_sweep_amount():
    assert payout.sweep_amount(250_000_000, 0, 10**12) == 260_000_000              # owed + fee buffer
    assert payout.sweep_amount(250_000_000, 255_000_000, 10**12) == 5_000_000      # top up the buffer
    assert payout.sweep_amount(250_000_000, 259_500_000, 10**12) == 0              # dust top-ups are skipped
    assert payout.sweep_amount(250_000_000, 0, 100_000_000) == 100_000_000         # never beyond what trading can spare


class Rpc:
    def __init__(self, trading, payout_):
        self.bal = {str(TRADING.pubkey()): trading, str(PAYOUT.pubkey()): payout_}
        self.sent = []

    def get_balance(self, pk):
        return self.bal[str(pk)]

    def get_latest_blockhash(self):
        return {"blockhash": "11111111111111111111111111111111", "last_valid_block_height": 10}

    def send_transaction(self, b64):
        self.sent.append(b64)


def test_sweep_moves_owed_and_keeps_gas_and_realized(monkeypatch):
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": 55}))
    rpc = Rpc(trading=5_000_000_000, payout_=0)
    before = settle.realized(5_000_000_000 + 0, 0, 0, 0, 0, 0)
    out = payout.sweep(rpc, TRADING, str(PAYOUT.pubkey()))
    assert out["swept"] == 260_000_000 and len(rpc.sent) == 1
    with transaction() as c:
        f = c.execute("SELECT kind, slot, lamports FROM vault_flows").fetchone()
        totals = settle.flow_totals(c, 2**62)
    assert (f["kind"], f["slot"], f["lamports"]) == ("sweep", 55, 260_000_000)
    assert totals == {"deposits": 0, "withdrawals": 0, "payouts": 0, "gifts": 0}   # a sweep is none of these
    # consolidated: trading lost what payout gained (minus the fee), so R moves only by the fee
    after = settle.realized((5_000_000_000 - 260_000_000 - 5000) + 260_000_000, 0, 0, 0, 0, 0)
    assert before - after == 5000
    # the trading bankroll no longer reserves what now sits in the payout wallet
    state.put("payout_balance", {"lamports": 260_000_000})
    with transaction() as c:
        assert nav.reserved_lamports(c) == 250_000_000 and nav.reserved_in_trading(c) == 0


def test_sweep_never_touches_gas_reserve(monkeypatch):
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": 1}))
    rpc = Rpc(trading=400_000_000, payout_=0)                                       # 0.4 SOL, 0.3 is gas reserve
    out = payout.sweep(rpc, TRADING, str(PAYOUT.pubkey()))
    assert out["swept"] == 400_000_000 - 300_000_000 - 5000


def test_sweep_is_known_to_the_flow_scanner(monkeypatch):
    import fly_trader.execution.broker_live as bl
    from fly_trader.vault import flows
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": 1}))
    payout.sweep(Rpc(5_000_000_000, 0), TRADING, str(PAYOUT.pubkey()))
    with transaction() as c:
        sig = c.execute("SELECT signature FROM vault_flows WHERE kind = 'sweep'").fetchone()["signature"]
        assert sig in flows.known_ours(c, [sig])                                    # not an unknown outbound: no halt
