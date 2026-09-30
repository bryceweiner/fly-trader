"""JSON-RPC for the RH trading path: fly_trader/vault/evm.EvmRpc (reads, logged and scrubbed) plus what sending and
indexing need — raw transactions, receipts, nonces, balances, fee history, gas estimates, multi-address logs and
batched requests. vault/evm.py is left untouched (the vault is edited concurrently on master).
"""
from __future__ import annotations

import time
from typing import Any

from .. import config
from ..db.apilog import record_api_call
from ..vault.evm import EvmRpc, EvmRpcError


RATE_LIMIT_TRIES = 6                # 1, 2, 4, 8, 16 s: the public endpoint's 429s clear within seconds


def _rate_limited(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "Too Many" in s or "rate limit" in s.lower()


class RhRpc(EvmRpc):
    def __init__(self, url: str | None = None, chain_id: int | None = None, service: str = "rh_rpc", timeout: float = 20.0):
        super().__init__(url or config.RH_RPC_URL, chain_id if chain_id is not None else config.RH_EXPECTED_CHAIN_ID, service, timeout)

    def call(self, method: str, params: list | None = None) -> Any:
        """``EvmRpc.call`` that waits out a rate limit (429) with exponential backoff; any other error raises at once.
        Never retried: eth_sendRawTransaction (the wallet decides what a failed broadcast means)."""
        for i in range(RATE_LIMIT_TRIES):
            try:
                return super().call(method, params)
            except EvmRpcError as e:
                if method == "eth_sendRawTransaction" or not _rate_limited(e) or i == RATE_LIMIT_TRIES - 1:
                    raise
                time.sleep(2 ** i)

    # ---- batching: many reads in one POST (tx lookups while indexing, balances while reconciling)
    def batch(self, calls: list[tuple[str, list]]) -> list[Any]:
        """Results in call order; an individual error comes back as an ``EvmRpcError`` instance (not raised). A rate-limited
        batch waits and retries like ``call``."""
        for i in range(RATE_LIMIT_TRIES):
            try:
                return self._batch(calls)
            except EvmRpcError as e:
                if not _rate_limited(e) or i == RATE_LIMIT_TRIES - 1:
                    raise
                time.sleep(2 ** i)
        return []

    def _batch(self, calls: list[tuple[str, list]]) -> list[Any]:
        if not calls:
            return []
        with self._lock:
            base = self._id + 1; self._id += len(calls)
        body = [{"jsonrpc": "2.0", "id": base + i, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
        t0 = time.monotonic(); status, ok, err = None, False, None
        try:
            r = self._client.post(self.url, json=body)
            status = r.status_code; r.raise_for_status()
            got = {x.get("id"): x for x in (r.json() or [])}
            out = []
            for i, (m, _) in enumerate(calls):
                x = got.get(base + i)
                if x is None:
                    out.append(EvmRpcError(m, {"message": "missing from batch response"}))
                elif x.get("error"):
                    out.append(EvmRpcError(m, x["error"]))
                else:
                    out.append(x.get("result"))
            ok = True
            return out
        except Exception as e:
            err = type(e).__name__
            raise EvmRpcError("batch", {"message": err + (f" {status}" if status else "")}) from None
        finally:
            record_api_call(self.service, f"batch[{len(calls)}]", "POST", status, int((time.monotonic() - t0) * 1000), ok, err)

    # ---- sending
    def send_raw(self, raw_hex: str) -> str:
        return self.call("eth_sendRawTransaction", [raw_hex])

    def receipt(self, tx_hash: str) -> dict | None:
        return self.call("eth_getTransactionReceipt", [tx_hash])

    def tx_by_hash(self, tx_hash: str) -> dict | None:
        return self.call("eth_getTransactionByHash", [tx_hash])

    def tx_count(self, address: str, tag: str = "pending") -> int:
        return int(self.call("eth_getTransactionCount", [address, tag]), 16)

    def get_balance(self, address: str, tag: str = "latest") -> int:
        return int(self.call("eth_getBalance", [address, tag]), 16)

    def estimate_gas(self, tx: dict) -> int:
        return int(self.call("eth_estimateGas", [tx]), 16)

    def base_fee(self) -> int:
        b = self.block("latest")
        return int(b.get("baseFeePerGas") or "0x0", 16)

    def max_priority_fee(self) -> int:
        try:
            return int(self.call("eth_maxPriorityFeePerGas", []), 16)
        except EvmRpcError:
            return 0                                   # Arbitrum-style chains order by arrival; the tip is ignored

    def get_logs_multi(self, addresses: list[str] | None, from_block: int, to_block: int, topics: list | None = None) -> list[dict]:
        """``eth_getLogs`` over several emitters (or any emitter when ``addresses`` is None)."""
        flt: dict = {"fromBlock": hex(from_block), "toBlock": hex(to_block)}
        if addresses:
            flt["address"] = addresses if len(addresses) > 1 else addresses[0]
        if topics:
            flt["topics"] = topics
        return self.call("eth_getLogs", [flt]) or []


def logs_rpc() -> RhRpc:
    """The endpoint for heavy reads (getLogs, tx lookups): ``RH_RPC_URL_LOGS`` when set, else the public RPC. Its ``blocks``
    attribute is the endpoint block headers are read from (``RH_RPC_URL_BLOCKS``)."""
    if config.env_str("ENVIO_API_TOKEN"):                         # bulk history from HyperSync (rh/hypersync.py)
        from .hypersync import HyperRpc, HyperSync
        r = HyperRpc(HyperSync(config.env_str("ENVIO_API_TOKEN")), url=config.RH_RPC_URL_LOGS or config.RH_RPC_URL, service="rh_logs")
    else:
        r = RhRpc(config.RH_RPC_URL_LOGS or config.RH_RPC_URL, service="rh_logs")
    r.blocks = RhRpc(config.RH_RPC_URL_BLOCKS, service="rh_blocks") if config.RH_RPC_URL_BLOCKS else r
    return r
