"""Nothing signs an RH transaction unless every check here passes, every time (not once at start):

- ``RH_LIVE_ENABLED`` is on;
- the RPC answers the chain id this install expects (4663, or 46630 with ``RH_TESTNET``) — a testnet key never signs
  on mainnet and the reverse;
- ``RH_BOT_PRIVATE_KEY`` is set and derives to ``RH_BOT_ADDRESS`` (a pasted wrong key never trades).
"""
from __future__ import annotations

from .. import config
from .tx import address_of


class SigningRefused(RuntimeError):
    pass


def load_key() -> bytes:
    raw = (config.env_str("RH_BOT_PRIVATE_KEY") or "").strip()
    if not raw:
        raise SigningRefused("RH_BOT_PRIVATE_KEY is not set")
    h = raw[2:] if raw.lower().startswith("0x") else raw
    try:
        key = bytes.fromhex(h)
    except ValueError:
        raise SigningRefused("RH_BOT_PRIVATE_KEY is not hex") from None
    if len(key) != 32:
        raise SigningRefused("RH_BOT_PRIVATE_KEY must be 32 bytes")
    return key


def check(rpc) -> tuple[bytes, str]:
    """(private key, address) when signing is allowed; raises ``SigningRefused`` with the reason otherwise."""
    if not config.RH_LIVE_ENABLED:
        raise SigningRefused("RH_LIVE_ENABLED is off")
    key = load_key(); addr = address_of(key)
    want = (config.RH_BOT_ADDRESS or "").lower()
    if not want:
        raise SigningRefused("RH_BOT_ADDRESS is not set: the key's address must be pinned")
    if addr != want:
        raise SigningRefused(f"RH_BOT_PRIVATE_KEY derives to {addr}, not RH_BOT_ADDRESS")
    cid = rpc.chain()
    if cid != config.RH_EXPECTED_CHAIN_ID:
        raise SigningRefused(f"the RPC is chain {cid}; this install signs only on {config.RH_EXPECTED_CHAIN_ID}")
    return key, addr
