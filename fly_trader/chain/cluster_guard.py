"""Default-deny signing gate (plan §9; pattern from better_bot code/solana_client.py:28-52).

Every path that signs a transaction calls ``assert_signing_allowed()`` first. Reads (balances,
quotes, signature statuses) never do. Signing is allowed only when ALL of these hold:
``SOLANA_CLUSTER == 'mainnet-beta'`` (the only cluster where the graduated-memecoin universe
exists), ``LIVE_ENABLED=1``, and ``config.live_prerequisites_missing()`` is empty (capital, position
cap, gas reserve, bot key and API keys all set). ``config`` attributes are read at call time so a
test can monkeypatch them.

The one exception is a withdrawal to the operator's own wallet (``assert_withdraw_allowed``,
chain/withdraw.py): it takes money out of the fly's reach and never trades, so it needs no LIVE_ENABLED.
"""
from __future__ import annotations

from .. import config


class SigningRefused(RuntimeError):
    """Raised instead of signing when the live gate is not fully open."""


def assert_signing_allowed() -> None:
    cluster = config.SOLANA_CLUSTER
    if cluster != "mainnet-beta":
        raise SigningRefused(f"refusing to sign on cluster {cluster!r}: only mainnet-beta is allowed")
    if not config.LIVE_ENABLED:
        raise SigningRefused("refusing to sign: LIVE_ENABLED is not set (default-deny)")
    missing = config.live_prerequisites_missing()
    if missing:
        raise SigningRefused("refusing to sign: live prerequisites missing: " + ", ".join(missing))


def signing_allowed() -> bool:
    try:
        assert_signing_allowed()
        return True
    except SigningRefused:
        return False


def assert_withdraw_allowed(dest: str) -> None:
    """A withdrawal of the bot's SOL (chain/withdraw.py): mainnet, and only to a wallet listed in FUNDING_ADDRESSES — the
    list lives in .env and is never typed into the console, so whoever reaches the console can at most send the bot's SOL
    to its owner. Never on the hosted vault fly: its SOL leaves through the treasury (``fly-trader vault withdraw``)."""
    if config.VAULT_ENABLED:
        raise SigningRefused("refusing: the vault fly's SOL leaves through the treasury (fly-trader vault withdraw), not from the trading wallet")
    if config.SOLANA_CLUSTER != "mainnet-beta":
        raise SigningRefused(f"refusing to withdraw on cluster {config.SOLANA_CLUSTER!r}: the bot wallet is on mainnet-beta")
    if dest not in config.FUNDING_ADDRESSES:
        raise SigningRefused(f"refusing: {dest} is not one of your FUNDING_ADDRESSES (.env)")


def assert_vault_signing_allowed() -> None:
    """The vault's own signing (claim payouts, principal withdrawals, token-account closes). On mainnet it is the same
    gate as trading. The devnet rehearsal (``VAULT_CLUSTER=devnet``) may sign with a throwaway key that is not a
    mainnet trading key: trading stays refused there because SOLANA_CLUSTER is not mainnet-beta."""
    if not config.VAULT_ENABLED:
        raise SigningRefused("refusing to sign: VAULT_ENABLED is not set")
    if config.VAULT_CLUSTER == "mainnet-beta":
        assert_signing_allowed()
        return
    if config.VAULT_CLUSTER != "devnet":
        raise SigningRefused(f"refusing to sign: VAULT_CLUSTER {config.VAULT_CLUSTER!r} is not mainnet-beta or devnet")
    if config.SOLANA_CLUSTER == "mainnet-beta" and config.LIVE_ENABLED:
        raise SigningRefused("refusing to sign on devnet while live mainnet trading is enabled in the same process")
    if not config.VAULT_SOLANA_RPC_URL:
        raise SigningRefused("refusing to sign on devnet without VAULT_SOLANA_RPC_URL pointing at devnet")
