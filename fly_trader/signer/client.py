"""The trading engine's side of the signer: the Signer runs in-process from the usual key settings."""
from __future__ import annotations

import threading

from .policy import PolicyError


class SignerUnavailable(RuntimeError):
    pass


class LocalSigner:
    """The Signer in this process; one call at a time."""

    def __init__(self, signer):
        self.signer, self.lock = signer, threading.Lock()

    def call(self, method: str, **params):
        from .core import SERVED
        if method not in SERVED:
            raise PolicyError(f"unknown method {method!r}")
        with self.lock:
            return getattr(self.signer, method)(**params)


_client = None
_lock = threading.Lock()


def get():
    """The process-wide signer client: an in-process signer built from the key settings (BOT_PRIVATE_KEY / _FILE)."""
    global _client
    with _lock:
        if _client is None:
            _client = LocalSigner(local_signer())
        return _client


def local_signer():
    import os
    from .. import config
    from ..chain.keys import load_keypair
    from .core import Signer, SignerConfig
    from .ledger import Ledger
    from .rpc import Rpc
    url = config.SIGNER_RPC_URL or config.helius_http_url()
    return Signer(load_keypair(), Rpc(url), SignerConfig.from_env(os.environ),
                  Ledger(os.environ.get("SIGNER_DB") or config.DATA_DIR / "signer" / "ledger.sqlite"))


def reset() -> None:
    """Tests: forget the cached client."""
    global _client
    with _lock:
        _client = None
