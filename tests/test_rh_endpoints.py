"""Robinhood Chain endpoints: the QuickNode URL (its key appended only when the URL lacks it) serves transactions, log
readers keep their own endpoint (QuickNode's free plan refuses log ranges over 5 blocks), and the key never reaches a log."""
from fly_trader import config
from fly_trader.logging_setup import scrub
from fly_trader.rh import rpc

KEY = "0123456789abcdef0123456789abcdef01234567"


def test_quicknode_url(monkeypatch):
    monkeypatch.delenv("QUICKNODE_RH_API_URL", raising=False); monkeypatch.delenv("QUICKNODE_RH_API_KEY", raising=False)
    assert config.quicknode_rh_url() is None
    monkeypatch.setenv("QUICKNODE_RH_API_KEY", KEY)
    assert config.quicknode_rh_url() is None                                   # a key without a URL is no endpoint
    monkeypatch.setenv("QUICKNODE_RH_API_URL", f"https://x.robinhood-mainnet.quiknode.pro/{KEY}")
    assert config.quicknode_rh_url() == f"https://x.robinhood-mainnet.quiknode.pro/{KEY}"
    monkeypatch.setenv("QUICKNODE_RH_API_URL", "https://x.robinhood-mainnet.quiknode.pro/")
    assert config.quicknode_rh_url() == f"https://x.robinhood-mainnet.quiknode.pro/{KEY}"
    monkeypatch.delenv("QUICKNODE_RH_API_KEY")
    assert config.quicknode_rh_url() == "https://x.robinhood-mainnet.quiknode.pro/"


def test_tests_never_reach_quicknode():
    assert config.RH_RPC_URL_TX is None                                        # tests/conftest.py blanks it before import


def test_default_client_is_the_transaction_endpoint(monkeypatch):
    monkeypatch.setattr(config, "RH_RPC_URL_TX", "https://tx.example")
    assert rpc.RhRpc().url == "https://tx.example"
    assert rpc.RhRpc("https://explicit.example").url == "https://explicit.example"
    monkeypatch.setattr(config, "RH_RPC_URL_TX", None)
    assert rpc.RhRpc().url == config.RH_RPC_URL


def test_log_readers_never_use_the_transaction_endpoint(monkeypatch):
    monkeypatch.setattr(config, "RH_RPC_URL_TX", "https://tx.example"); monkeypatch.setattr(config, "RH_RPC_URL_LOGS", None)
    monkeypatch.delenv("ENVIO_API_TOKEN", raising=False)
    assert rpc.logs_rpc().url == config.RH_RPC_URL
    monkeypatch.setenv("ENVIO_API_TOKEN", "token")
    assert rpc.logs_rpc().url == config.RH_RPC_URL                              # HyperSync's RPC side (beyond its height)


def test_key_is_scrubbed(monkeypatch):
    monkeypatch.setenv("QUICKNODE_RH_API_KEY", KEY)
    assert KEY not in scrub(f"429 Too Many Requests for url 'https://x.robinhood-mainnet.quiknode.pro/{KEY}'")
