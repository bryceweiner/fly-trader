"""The signer container's entry point (``python -m fly_trader.signer``), standard library + solders + httpx only: it
imports nothing else of fly_trader, so its image carries none of the brain's dependencies.

    python -m fly_trader.signer serve          # the service (SIGNER_SOCKET, SIGNER_DB, key files, SIGNER_RPC_URL)
    python -m fly_trader.signer status
    python -m fly_trader.signer ping           # the container's health check: the socket answers
    python -m fly_trader.signer clear-panic    # only from a shell on the server: the brain cannot call this
"""
from __future__ import annotations

import json
import logging
import os
import sys

from .core import Signer, SignerConfig
from .ledger import Ledger
from .rpc import Rpc


def _keypair(path: str):
    import base58
    from solders.keypair import Keypair
    raw = open(path).read().strip()
    b = bytes(json.loads(raw)) if raw.startswith("[") else base58.b58decode(raw)
    return Keypair.from_seed(b) if len(b) == 32 else Keypair.from_bytes(b)


def build() -> Signer:
    env = os.environ
    payout = env.get("PAYOUT_PRIVATE_KEY_FILE")
    return Signer(_keypair(env["BOT_PRIVATE_KEY_FILE"]), _keypair(payout) if payout else None, Rpc(env.get("SIGNER_RPC_URL", "")),
                  SignerConfig.from_env(env), Ledger(env.get("SIGNER_DB", "/srv/signer/ledger.sqlite")))


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cmd = argv[1] if len(argv) > 1 else "serve"
    if cmd == "ping":                               # no keys needed: ask the running server over its socket
        from .client import SocketSigner
        SocketSigner(os.environ.get("SIGNER_SOCKET", "/run/fly-signer/sock"), timeout=5).call("pubkeys")
        return
    s = build()
    if cmd == "serve":
        from .server import serve
        serve(s, os.environ.get("SIGNER_SOCKET", "/run/fly-signer/sock"))
    elif cmd == "status":
        print(json.dumps(s.status(), indent=1))
    elif cmd == "clear-panic":
        print(json.dumps(s.clear_panic()))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
