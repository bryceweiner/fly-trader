"""Minimal EVM plumbing for Robinhood Chain: keccak-256, address checks and JSON-RPC.

No web3 dependency: the Robinhood Chain path needs only plain JSON-RPC.
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


def keccak256(data: bytes) -> bytes:
    h = _keccak.new(digest_bits=256)
    h.update(data)
    return h.digest()


def is_address(s: str) -> bool:
    return bool(ADDRESS_RE.match(s or ""))


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
        if method == "eth_sendRawTransaction":
            from ..netguard import refuse_broadcast
            refuse_broadcast(self.url, "a transaction")             # never from a test run (fly_trader/netguard.py)
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
