"""No money moves from a test run: every place that broadcasts a transaction or changes a live account refuses while
pytest runs, unless it talks to a local test chain (anvil, a fake on 127.0.0.1).

Why it exists: on 2026-10-17 the test suite ran while the operator's .env had RH_LIVE_ENABLED=1 and the real RH bot key.
A vault test expected Robinhood Chain signing to be off; it was on, so the claim processor built the real wallet
(RhWallet(RhRpc())) and paid two test claims of 0.03 ETH to the test's placeholder address 0x1111…1111 on mainnet —
0.06 ETH burned. The database already refuses production under pytest (db/connection.py); this does the same for chains
and exchanges, at the lowest layer, so no configuration or code path above it can get around it.

A test run is recognised by PYTEST_CURRENT_TEST (set by pytest for every test's setup, call and teardown) or
FLY_TRADER_TESTING=1 (set by tests/conftest.py before anything is imported). Neither is ever set in production.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class BroadcastUnderTest(RuntimeError):
    """A test tried to move real money."""


def under_test() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("FLY_TRADER_TESTING") == "1"


def refuse_broadcast(url: str | None, what: str) -> None:
    """Raise when a test is about to send ``what`` to a real network (anything but a loopback host)."""
    if not under_test():
        return
    host = urlsplit(url or "").hostname or ""
    if host not in LOOPBACK:
        raise BroadcastUnderTest(f"refusing to send {what} to {host or 'an unknown host'} from a test run: tests never move real money")
