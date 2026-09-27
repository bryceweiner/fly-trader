"""The brain's side of the signer. On the vault server it talks to the signer container over its Unix socket
(``SIGNER_SOCKET``); anywhere else (the Mac, tests) the same Signer runs in-process from the usual key settings, so
there is one code path and one policy."""
from __future__ import annotations

import json
import socket
import threading

from .policy import PolicyError


class SignerUnavailable(RuntimeError):
    pass


class SocketSigner:
    def __init__(self, path: str, timeout: float = 90.0):
        self.path, self.timeout = path, timeout

    def call(self, method: str, **params):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.path)
            s.sendall(json.dumps({"method": method, "params": params}).encode() + b"\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        except OSError as e:
            raise SignerUnavailable(f"signer socket {self.path}: {e}") from e
        finally:
            s.close()
        if not buf:
            raise SignerUnavailable("the signer closed the connection without a reply")
        r = json.loads(buf)
        if r.get("ok"):
            return r["result"]
        if r.get("code") == "error":
            raise SignerUnavailable(r.get("error"))
        raise PolicyError(r.get("error") or "refused", code=r.get("code") or "policy")


class LocalSigner:
    """The same Signer in this process (no socket); one call at a time, like the server."""

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
    """The process-wide signer client: the socket when SIGNER_SOCKET is set, else an in-process signer built from the
    key settings (BOT_PRIVATE_KEY / _FILE, PAYOUT_PRIVATE_KEY_FILE)."""
    global _client
    with _lock:
        if _client is None:
            from .. import config
            if config.SIGNER_SOCKET:
                _client = SocketSigner(config.SIGNER_SOCKET)
            else:
                _client = LocalSigner(local_signer())
        return _client


def local_signer():
    import os
    from .. import config
    from ..chain.keys import load_keypair
    from .core import Signer, SignerConfig
    from .ledger import Ledger
    from .rpc import Rpc
    payout = None
    if config.PAYOUT_PRIVATE_KEY_FILE:
        from ..chain.keys import _keypair_from
        payout = _keypair_from(open(config.PAYOUT_PRIVATE_KEY_FILE).read())
    url = config.SIGNER_RPC_URL or config.vault_solana_rpc_url()
    return Signer(load_keypair(), payout, Rpc(url), SignerConfig.from_env({**os.environ, **config.signer_env()}),
                  Ledger(os.environ.get("SIGNER_DB") or config.DATA_DIR / "signer" / "ledger.sqlite"))


def reset() -> None:
    """Tests: forget the cached client."""
    global _client
    with _lock:
        _client = None
