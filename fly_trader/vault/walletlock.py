"""The wallet lock: one Postgres advisory lock that serializes everything which moves the trading wallet's SOL.

Held exclusively around a swap and its ledger commit, a claim payout, a principal withdrawal, a token-account close and
the settlement snapshot, so a snapshot never sees SOL gone without its position (or a payout without its row). The
per-minute NAV mark takes it shared with try semantics and marks itself inconsistent when it is busy.
Session-level lock on a dedicated autocommit connection (pattern: agent/runner.acquire_runner_lock).
"""
from __future__ import annotations

import contextlib
import time

from ..db.connection import connect

WALLET_LOCK_KEY = 0x666C795F77616C  # "fly_wal"


class WalletLockTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def exclusive(timeout_s: float = 120.0, poll_s: float = 0.2):
    conn = connect(autocommit=True)
    try:
        deadline = time.monotonic() + timeout_s
        while not conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (WALLET_LOCK_KEY,)).fetchone()["ok"]:
            if time.monotonic() > deadline:
                raise WalletLockTimeout("wallet lock busy")
            time.sleep(poll_s)
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (WALLET_LOCK_KEY,))
    finally:
        conn.close()


@contextlib.contextmanager
def try_shared():
    """Yields True when the shared lock was taken (no exclusive holder), False otherwise; never waits."""
    conn = connect(autocommit=True)
    try:
        ok = bool(conn.execute("SELECT pg_try_advisory_lock_shared(%s) AS ok", (WALLET_LOCK_KEY,)).fetchone()["ok"])
        try:
            yield ok
        finally:
            if ok:
                conn.execute("SELECT pg_advisory_unlock_shared(%s)", (WALLET_LOCK_KEY,))
    finally:
        conn.close()


@contextlib.contextmanager
def maybe_exclusive(enabled: bool, timeout_s: float = 120.0):
    """The lock only when the vault is on (the Mac research fly never takes it)."""
    if not enabled:
        yield
        return
    with exclusive(timeout_s=timeout_s):
        yield
