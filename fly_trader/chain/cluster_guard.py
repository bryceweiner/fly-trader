"""Default-deny signing gate (plan §9; pattern from better_bot code/solana_client.py:28-52).

Every path that signs a transaction calls ``assert_signing_allowed()`` first. Reads (balances,
quotes, signature statuses) never do. Signing is allowed only when ALL of these hold:
``SOLANA_CLUSTER == 'mainnet-beta'`` (the only cluster where the graduated-memecoin universe
exists), ``LIVE_ENABLED=1``, and ``config.live_prerequisites_missing()`` is empty (capital, position
cap, gas reserve, bot key and API keys all set). ``config`` attributes are read at call time so a
test can monkeypatch them.
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
