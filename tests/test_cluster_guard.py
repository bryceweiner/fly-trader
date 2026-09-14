import pytest

from fly_trader import config
from fly_trader.chain.cluster_guard import SigningRefused, assert_signing_allowed, signing_allowed


def _open_gate(monkeypatch):
    monkeypatch.setattr(config, "SOLANA_CLUSTER", "mainnet-beta")
    monkeypatch.setattr(config, "LIVE_ENABLED", True)
    monkeypatch.setattr(config, "live_prerequisites_missing", lambda: [])


def test_default_deny(monkeypatch):
    monkeypatch.setattr(config, "LIVE_ENABLED", False)
    with pytest.raises(SigningRefused, match="LIVE_ENABLED"):
        assert_signing_allowed()
    assert signing_allowed() is False


def test_wrong_cluster_refused_even_when_live(monkeypatch):
    _open_gate(monkeypatch)
    monkeypatch.setattr(config, "SOLANA_CLUSTER", "devnet")
    with pytest.raises(SigningRefused, match="devnet"):
        assert_signing_allowed()


def test_missing_prerequisite_refused(monkeypatch):
    _open_gate(monkeypatch)
    monkeypatch.setattr(config, "live_prerequisites_missing", lambda: ["BOT_PRIVATE_KEY"])
    with pytest.raises(SigningRefused, match="BOT_PRIVATE_KEY"):
        assert_signing_allowed()


def test_allowed_only_under_the_full_conjunction(monkeypatch):
    _open_gate(monkeypatch)
    assert assert_signing_allowed() is None
    assert signing_allowed() is True
