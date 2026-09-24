"""Minimal EVM plumbing for Robinhood Chain: keccak-256, EIP-55 addresses, ABI words, JSON-RPC and log decoding.

No web3 dependency: the vault only needs ``eth_getLogs``, ``eth_getBlockByNumber``, ``eth_call``, ``eth_getCode`` and
``eth_getStorageAt``, and the events it decodes have static argument types (address, uint256, uint64).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx
from Crypto.Hash import keccak as _keccak

from ..db.apilog import record_api_call
from ..logging_setup import scrub

log = logging.getLogger(__name__)

ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
RPC_TIMEOUT_S = 20.0
# ERC-1967 implementation slot: bytes32(uint256(keccak256("eip1967.proxy.implementation")) - 1)
IMPLEMENTATION_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


def keccak256(data: bytes) -> bytes:
    h = _keccak.new(digest_bits=256)
    h.update(data)
    return h.digest()


def is_address(s: str) -> bool:
    return bool(ADDRESS_RE.match(s or ""))


def to_checksum(address: str) -> str:
    """EIP-55 mixed-case form of a 20-byte hex address (any case in)."""
    if not is_address(address):
        raise ValueError("not a 20-byte hex address")
    a = address[2:].lower()
    h = keccak256(a.encode()).hex()
    return "0x" + "".join(c.upper() if c.isalpha() and int(h[i], 16) >= 8 else c for i, c in enumerate(a))


def is_checksummed(address: str) -> bool:
    return is_address(address) and to_checksum(address) == address


def topic(signature: str) -> str:
    """Event topic0 / function selector source: keccak of the canonical signature, 0x-hex."""
    return "0x" + keccak256(signature.encode()).hex()


def selector(signature: str) -> str:
    return topic(signature)[:10]


def word_to_int(word: str) -> int:
    return int(word, 16)


def word_to_address(word: str) -> str:
    return to_checksum("0x" + word[-40:])


def data_words(data: str) -> list[str]:
    d = data[2:] if data.startswith("0x") else data
    return [d[i:i + 64] for i in range(0, len(d), 64)]


def encode_uint(n: int) -> str:
    return format(int(n), "064x")


def encode_bytes32(b: bytes) -> str:
    if len(b) != 32:
        raise ValueError("bytes32 must be 32 bytes")
    return b.hex()


def encode_dynamic_bytes(b: bytes) -> str:
    """ABI tail for a ``bytes`` argument: length word + right-padded data."""
    padded = b + b"\x00" * ((32 - len(b) % 32) % 32)
    return encode_uint(len(b)) + padded.hex()


class EvmRpcError(RuntimeError):
    def __init__(self, method: str, error: Any):
        err = error if isinstance(error, dict) else {"message": str(error)}
        self.method = method
        self.code = err.get("code")
        self.message = str(err.get("message", ""))
        super().__init__(f"evm rpc {method} error {self.code}: {scrub(self.message)}")


class EvmRpc:
    """JSON-RPC client for one EVM chain. Public RPCs carry no key, but URLs are still scrubbed from errors."""

    def __init__(self, url: str, chain_id: int | None = None, service: str = "rh_rpc", timeout: float = RPC_TIMEOUT_S):
        self.url = url
        self.chain_id = chain_id
        self.service = service
        self._client = httpx.Client(timeout=timeout, headers={"Content-Type": "application/json"})
        self._lock = threading.Lock()
        self._id = 0

    def call(self, method: str, params: list | None = None) -> Any:
        with self._lock:
            self._id += 1
            rid = self._id
        t0 = time.monotonic()
        status, ok, err = None, False, None
        try:
            r = self._client.post(self.url, json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params or []})
            status = r.status_code
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                err = str((body["error"] or {}).get("message", ""))[:200]
                raise EvmRpcError(method, body["error"])
            ok = True
            return body.get("result")
        except httpx.HTTPError as e:
            err = type(e).__name__
            raise EvmRpcError(method, {"message": type(e).__name__ + (f" {status}" if status else "")}) from None
        finally:
            record_api_call(self.service, method, "POST", status, int((time.monotonic() - t0) * 1000), ok, err)

    def close(self) -> None:
        self._client.close()

    # ---- helpers ----
    def chain(self) -> int:
        return int(self.call("eth_chainId"), 16)

    def block(self, tag: str | int, full: bool = False) -> dict | None:
        t = hex(tag) if isinstance(tag, int) else tag
        return self.call("eth_getBlockByNumber", [t, full])

    def block_number(self, tag: str = "latest") -> int:
        b = self.block(tag)
        if b is None:
            raise EvmRpcError("eth_getBlockByNumber", {"message": f"no {tag} block"})
        return int(b["number"], 16)

    def get_logs(self, address: str, from_block: int, to_block: int, topics: list | None = None) -> list[dict]:
        flt = {"address": address, "fromBlock": hex(from_block), "toBlock": hex(to_block)}
        if topics:
            flt["topics"] = topics
        return self.call("eth_getLogs", [flt]) or []

    def eth_call(self, to: str, data: str, block: str = "latest", gas: int | None = None) -> str:
        tx = {"to": to, "data": data}
        if gas:
            tx["gas"] = hex(gas)
        return self.call("eth_call", [tx, block])

    def code(self, address: str, block: str = "latest") -> str:
        return self.call("eth_getCode", [address, block]) or "0x"

    def storage(self, address: str, slot: str, block: str = "latest") -> str:
        return self.call("eth_getStorageAt", [address, slot, block])

    def implementation(self, proxy: str, block: str = "latest") -> str:
        """The ERC-1967 implementation behind a proxy."""
        return word_to_address(self.storage(proxy, IMPLEMENTATION_SLOT, block)[2:].rjust(64, "0"))

    def code_hash(self, address: str, block: str = "latest") -> str:
        c = self.code(address, block)
        return "0x" + keccak256(bytes.fromhex(c[2:])).hex()
