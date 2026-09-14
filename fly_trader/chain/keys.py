"""Bot wallet keys (plan §9; pattern lifted from VOC dexlp/keys.py).

The secret lives ONLY in the gitignored ``.env`` as ``BOT_PRIVATE_KEY`` — base58 of the 64-byte
secret key (what ``create_wallet`` writes) or a JSON int array (what ``solana-keygen`` writes). It
is never logged, never printed and never placed in an exception message: errors name the source
and the failure type only. Note that ``str(Keypair)`` IS the base58 secret, so no Keypair object
is ever formatted, repr'd or passed to a logger in this module.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import base58
from psycopg.types.json import Jsonb
from solders.keypair import Keypair

from .. import config
from ..db.connection import transaction

log = logging.getLogger(__name__)

ENV_NAME = "BOT_PRIVATE_KEY"
ENV_PATH = config.REPO_ROOT / ".env"
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?BOT_PRIVATE_KEY\s*=")


class WalletKeyError(RuntimeError):
    """Missing or malformed bot key. The message never includes secret material."""


class WalletExists(RuntimeError):
    """``create_wallet`` refused because a BOT_PRIVATE_KEY already exists."""


def _keypair_from(raw: str | None) -> Keypair:
    raw = (raw or "").strip()
    if not raw:
        raise WalletKeyError(f"{ENV_NAME} not set")
    try:
        if raw.startswith("["):
            arr = json.loads(raw)
            if not isinstance(arr, list):
                raise ValueError("not a list")
            data = bytes(int(x) & 0xFF for x in arr)
        else:
            data = base58.b58decode(raw)
    except Exception as e:  # never surface the value
        raise WalletKeyError(f"could not decode {ENV_NAME}: {type(e).__name__}") from None
    try:
        if len(data) == 64:
            return Keypair.from_bytes(data)
        if len(data) == 32:  # a bare seed (solana-keygen never writes this, but accept it)
            return Keypair.from_seed(data)
    except Exception as e:
        raise WalletKeyError(f"invalid {ENV_NAME} bytes: {type(e).__name__}") from None
    raise WalletKeyError(f"{ENV_NAME} must decode to 64 bytes (got {len(data)})")


def load_keypair(raw: str | None = None) -> Keypair:
    """Keypair from ``BOT_PRIVATE_KEY`` (or ``raw``). Accepts base58 or a JSON int array."""
    return _keypair_from(raw if raw is not None else os.environ.get(ENV_NAME))


def bot_pubkey(raw: str | None = None) -> str:
    return str(load_keypair(raw).pubkey())


def key_available() -> bool:
    try:
        load_keypair()
        return True
    except WalletKeyError:
        return False


def env_file_has_key(path: Path | None = None) -> bool:
    path = Path(path) if path else ENV_PATH
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return any(_ENV_LINE.match(line) for line in f)


def create_wallet(env_path: Path | None = None, url: str | None = None) -> str:
    """Generate the bot keypair once and append it to ``.env`` (mode 600). Returns the pubkey.

    Refuses if ``BOT_PRIVATE_KEY`` is already present in the process environment or in the file.
    A preceding newline is written when the file does not end with one. Only the pubkey is
    logged; a ``wallet_events(kind='created')`` row records the event.
    """
    path = Path(env_path) if env_path else ENV_PATH
    if os.environ.get(ENV_NAME, "").strip():
        raise WalletExists(f"{ENV_NAME} is already set in the environment; refusing to create another wallet")
    if env_file_has_key(path):
        raise WalletExists(f"{path} already contains {ENV_NAME}; refusing to overwrite")
    kp = Keypair()
    pubkey = str(kp.pubkey())
    secret_b58 = base58.b58encode(bytes(kp)).decode()
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(f"{ENV_NAME}={secret_b58}\n")
    finally:
        os.chmod(path, 0o600)
    del secret_b58, kp
    try:
        with transaction(url) as conn:
            conn.execute(
                "INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('created', %s, %s)",
                (pubkey, Jsonb({"env_path": str(path)})),
            )
    except Exception as e:  # the key is already safely on disk; do not lose it over a DB hiccup
        log.warning("wallet_events insert failed after wallet creation: %s", type(e).__name__)
    log.info("bot wallet created: %s", pubkey)
    return pubkey


def show_wallet(rpc=None, url: str | None = None) -> str:
    """Pubkey + SOL balance (read-only RPC); records ``wallet_events(kind='balance_snapshot')``."""
    pubkey = bot_pubkey()
    if rpc is None:
        from .rpc import HttpSolanaRpc
        rpc = HttpSolanaRpc()
    lamports = int(rpc.get_balance(pubkey))
    sol = lamports / config.LAMPORTS_PER_SOL
    try:
        with transaction(url) as conn:
            conn.execute(
                "INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('balance_snapshot', %s, %s)",
                (pubkey, Jsonb({"lamports": lamports, "sol": sol})),
            )
    except Exception as e:
        log.warning("wallet_events insert failed: %s", type(e).__name__)
    return f"pubkey={pubkey} sol={sol:.9f} lamports={lamports}"
