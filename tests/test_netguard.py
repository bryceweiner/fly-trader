"""No money moves from a test run (fly_trader/netguard.py): every path that broadcasts a transaction or changes a live
account refuses a real network while pytest runs -- the Solana RPC (swaps' fallback, token-account closes, withdrawals,
vault payouts and claims), Jupiter's /execute, the Robinhood Chain and vault EVM RPCs, Kalshi's order endpoints -- and
only a loopback test chain gets through. The signing keys and live switches are off, and a non-test database is refused."""
import os

import pytest

from fly_trader import config
from fly_trader.netguard import BroadcastUnderTest, refuse_broadcast, under_test

REAL = "https://guard.invalid/rpc"          # .invalid never resolves: even a failed guard could not reach anything


def _senders():
    from fly_trader.chain.jupiter_swap import JupiterSwap
    from fly_trader.chain.rpc import HttpSolanaRpc
    from fly_trader.kalshi.client import KalshiRest
    from fly_trader.rh.rpc import RhRpc
    from fly_trader.vault.evm import EvmRpc
    return {
        "solana sendTransaction": lambda: HttpSolanaRpc(REAL).send_transaction("AA=="),
        "solana batch": lambda: HttpSolanaRpc(REAL).batch([("getSlot", None), ("sendTransaction", ["AA=="])]),
        "jupiter execute": lambda: JupiterSwap(api_key="test").execute("AA==", "request-id"),
        "rh send_raw": lambda: RhRpc(REAL).send_raw("0x00"),
        "rh batch": lambda: RhRpc(REAL).batch([("eth_sendRawTransaction", ["0x00"])]),
        "vault evm": lambda: EvmRpc(REAL).call("eth_sendRawTransaction", ["0x00"]),
        "kalshi order": lambda: KalshiRest(base_url=REAL).request("POST", "/portfolio/orders", body={"ticker": "T"}),
    }


@pytest.mark.parametrize("name", list(_senders()))
def test_every_send_path_refuses_a_real_network(name):
    with pytest.raises(BroadcastUnderTest):
        _senders()[name]()


def test_only_a_loopback_chain_gets_through():
    assert under_test()
    for url in ("http://127.0.0.1:8545", "http://localhost:8899", "http://[::1]:8545"):
        refuse_broadcast(url, "a test transaction")
    for url in (REAL, "https://api.jup.ag/swap/v2", "https://mainnet.helius-rpc.com", None):
        with pytest.raises(BroadcastUnderTest):
            refuse_broadcast(url, "a transaction")


def test_the_operators_keys_and_live_switches_are_off():
    from fly_trader.chain.cluster_guard import signing_allowed
    assert all(os.environ.get(k) == "" for k in ("BOT_PRIVATE_KEY", "BOT_PRIVATE_KEY_FILE", "RH_BOT_PRIVATE_KEY", "SIGNER_SOCKET"))
    assert not (config.LIVE_ENABLED or config.RH_LIVE_ENABLED or config.KALSHI_LIVE_ENABLED or config.LIVE_WITHOUT_HANDOVER)
    assert not signing_allowed() and "RH_BOT_PRIVATE_KEY" in config.rh_live_prerequisites_missing()


def test_a_non_test_database_is_refused():
    from fly_trader.db.connection import ProductionDatabaseUnderPytest, _refuse_production_under_pytest
    with pytest.raises(ProductionDatabaseUnderPytest):
        _refuse_production_under_pytest("postgresql:///fly_trader")
    _refuse_production_under_pytest("postgresql:///fly_trader_test")
