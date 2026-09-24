"""Solana JSON-RPC over Helius using httpx + solders only (no solana-py). Pattern: VOC
dexlp/ingest/helius_swaps.py HttpSolanaRpc.

The RPC URL carries the Helius API key: it is never logged, never stored in ``api_calls`` and never
placed in an exception message (httpx error text is scrubbed and re-raised as our own types).
Every call is persisted to ``api_calls`` (service='helius', endpoint=<method>) through
``fly_trader.db.apilog.record_api_call``, which swallows DB errors so a database hiccup never
breaks an RPC call.

Response shapes verified against mainnet on 2026-09-12:
  getBalance            -> {"context": {"slot", "apiVersion"}, "value": <int lamports>}
  getTokenAccountsByOwner (jsonParsed) -> {"context", "value": [{"pubkey", "account": {"lamports",
                           "owner": <program>, "data": {"program": "spl-token"|"spl-token-2022",
                           "parsed": {"type": "account", "info": {"mint", "owner", "state", "isNative",
                           "tokenAmount": {"amount": "<str>", "decimals", "uiAmount", "uiAmountString"},
                           "extensions": [...] (token-2022 only)}}}}}]}
  getSignatureStatuses  -> {"context", "value": [{"slot", "confirmations", "err", "status",
                           "confirmationStatus": processed|confirmed|finalized} | null]}
  getLatestBlockhash    -> {"context", "value": {"blockhash", "lastValidBlockHeight": <int>}}
  getBlockHeight        -> <int>   (same units as lastValidBlockHeight)
  getTransaction        -> {"transaction", "meta": {"err", "status", "fee", "preBalances",
                           "postBalances", "preTokenBalances", "postTokenBalances", "innerInstructions",
                           "logMessages", "computeUnitsConsumed", ...}, "version", "slot", "blockTime"}
  sendTransaction       -> "<signature>"; malformed input returns error {"code": -32602, ...}
  JSON-RPC batches are supported (array in, array out, matched by id).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from .. import config
from ..db.apilog import record_api_call
from ..logging_setup import scrub

log = logging.getLogger(__name__)

TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
TOKEN_PROGRAM_IDS = (TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID)
RPC_TIMEOUT_S = 15.0
COMMITMENT = "confirmed"


class RpcError(RuntimeError):
    """A JSON-RPC level error ({"error": {code, message, data}})."""

    def __init__(self, method: str, error: dict | Any):
        err = error if isinstance(error, dict) else {"message": str(error)}
        self.method = method
        self.code = err.get("code")
        self.message = str(err.get("message", ""))
        self.data = err.get("data")
        super().__init__(f"rpc {method} error {self.code}: {scrub(self.message)}")


class RpcTransportError(RuntimeError):
    """HTTP/transport failure. Message never contains the URL."""


def _request_summary(method: str, params: list | None) -> dict:
    if method == "sendTransaction" and params:
        opts = params[1] if len(params) > 1 and isinstance(params[1], dict) else {}
        return {"tx_b64_len": len(params[0]) if isinstance(params[0], str) else None, **opts}
    if method == "getSignatureStatuses" and params:
        # full signatures are 88 base58 chars and the log scrubber would redact them as secrets
        return {"n": len(params[0]), "signature_prefixes": [str(s)[:12] for s in params[0]]}
    if method == "getTransaction" and params:
        return {"signature_prefix": str(params[0])[:12]}
    if method == "getSignaturesForAddress" and params:
        opts = dict(params[1]) if len(params) > 1 and isinstance(params[1], dict) else {}
        for k in ("before", "until"):
            if opts.get(k):
                opts[k] = str(opts[k])[:12]
        return {"address": str(params[0]), **opts}
    return {"params": params}


class HttpSolanaRpc:
    def __init__(self, url: str | None = None, timeout_s: float = RPC_TIMEOUT_S):
        self._url = url or config.helius_http_url()
        self._client = httpx.Client(timeout=timeout_s)
        self._id = 0
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _post(self, payload: dict | list, endpoint: str, summary: dict | None):
        t0 = time.monotonic()
        status: int | None = None
        err: str | None = None
        nbytes: int | None = None
        try:
            try:
                r = self._client.post(self._url, json=payload)
            except httpx.HTTPError as e:
                raise RpcTransportError(f"{endpoint}: {type(e).__name__}: {scrub(str(e))}") from None
            status = r.status_code
            nbytes = len(r.content)
            if status >= 400:
                raise RpcTransportError(f"{endpoint}: HTTP {status}: {scrub(r.text[:200])}")
            try:
                body = r.json()
            except ValueError:
                raise RpcTransportError(f"{endpoint}: non-JSON response ({nbytes} bytes)") from None
            if isinstance(payload, dict):
                if not isinstance(body, dict):
                    raise RpcTransportError(f"{endpoint}: unexpected response shape {type(body).__name__}")
                if body.get("error"):
                    raise RpcError(endpoint, body["error"])
                return body.get("result")
            if not isinstance(body, list):
                raise RpcTransportError(f"{endpoint}: batch response is not a list")
            by_id = {item.get("id"): item for item in body if isinstance(item, dict)}
            results = []
            for req in payload:
                item = by_id.get(req["id"])
                if item is None:
                    raise RpcTransportError(f"{endpoint}: missing batch response for id {req['id']}")
                if item.get("error"):
                    raise RpcError(req["method"], item["error"])
                results.append(item.get("result"))
            return results
        except Exception as e:
            err = str(e)
            raise
        finally:
            record_api_call("helius", endpoint, "POST", status, int((time.monotonic() - t0) * 1000),
                            err is None, err, request=summary, response_bytes=nbytes)

    # ---- generic ----
    def call(self, method: str, params: list | None = None):
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or []}
        return self._post(payload, method, _request_summary(method, params))

    def batch(self, calls: list[tuple[str, list | None]]) -> list:
        """Send several calls in one HTTP request; results in call order. Raises on the first error."""
        if not calls:
            return []
        payload = [{"jsonrpc": "2.0", "id": self._next_id(), "method": m, "params": p or []} for m, p in calls]
        methods = [m for m, _ in calls]
        return self._post(payload, "batch:" + ",".join(methods), {"methods": methods})

    # ---- typed helpers ----
    def get_balance(self, pubkey) -> int:
        res = self.call("getBalance", [str(pubkey), {"commitment": COMMITMENT}])
        return int(res["value"])

    def get_balance_ctx(self, pubkey, commitment: str = COMMITMENT, min_context_slot: int | None = None) -> tuple[int, int]:
        """(lamports, context slot). ``min_context_slot`` makes a lagging node refuse instead of answering stale."""
        opts: dict = {"commitment": commitment}
        if min_context_slot:
            opts["minContextSlot"] = int(min_context_slot)
        res = self.call("getBalance", [str(pubkey), opts])
        return int(res["value"]), int(res["context"]["slot"])

    def get_signatures_for_address(self, pubkey, before: str | None = None, until: str | None = None,
                                   limit: int = 1000, commitment: str = "finalized") -> list[dict]:
        """Newest first: ``{signature, slot, err, blockTime, confirmationStatus, memo}``."""
        opts: dict = {"limit": int(limit), "commitment": commitment}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        return self.call("getSignaturesForAddress", [str(pubkey), opts]) or []

    def get_token_accounts_by_owner(self, pubkey, commitment: str = COMMITMENT) -> list[dict]:
        """All token accounts of ``pubkey`` under BOTH token programs, as
        ``{address, mint, amount:int, decimals, program, lamports, is_native, state}``."""
        out: list[dict] = []
        for program in TOKEN_PROGRAM_IDS:
            res = self.call("getTokenAccountsByOwner",
                            [str(pubkey), {"programId": program}, {"encoding": "jsonParsed", "commitment": commitment}])
            for item in (res or {}).get("value") or []:
                try:
                    info = item["account"]["data"]["parsed"]["info"]
                    amt = info["tokenAmount"]
                    out.append({"address": item["pubkey"], "mint": info["mint"], "amount": int(amt["amount"]),
                                "decimals": int(amt["decimals"]), "program": program,
                                "lamports": int(item["account"].get("lamports") or 0),
                                "is_native": bool(info.get("isNative")), "state": info.get("state")})
                except (KeyError, TypeError, ValueError):
                    log.warning("unparsed token account %s", item.get("pubkey") if isinstance(item, dict) else "?")
        return out

    def get_signature_statuses(self, sigs: list[str], search_history: bool = False) -> list[dict | None]:
        res = self.call("getSignatureStatuses", [list(sigs), {"searchTransactionHistory": bool(search_history)}])
        value = (res or {}).get("value")
        if not isinstance(value, list):
            return [None] * len(sigs)
        return value

    def get_transaction(self, sig: str, encoding: str = "jsonParsed") -> dict | None:
        return self.call("getTransaction", [sig, {"encoding": encoding, "maxSupportedTransactionVersion": 0,
                                                  "commitment": COMMITMENT}])

    def get_latest_blockhash(self) -> dict:
        """``{"blockhash": str, "last_valid_block_height": int}``."""
        v = self.call("getLatestBlockhash", [{"commitment": COMMITMENT}])["value"]
        return {"blockhash": v["blockhash"], "last_valid_block_height": int(v["lastValidBlockHeight"])}

    def get_block_height(self) -> int:
        return int(self.call("getBlockHeight", [{"commitment": COMMITMENT}]))

    def get_account_info(self, pubkey) -> dict | None:
        res = self.call("getAccountInfo", [str(pubkey), {"encoding": "jsonParsed", "commitment": COMMITMENT}])
        return (res or {}).get("value")

    def get_mint_decimals(self, mint: str) -> int | None:
        try:
            info = self.get_account_info(mint)
            return int(info["data"]["parsed"]["info"]["decimals"])
        except Exception:
            return None

    def send_transaction(self, tx_b64: str, skip_preflight: bool = False, max_retries: int = 3) -> str:
        sig = self.call("sendTransaction", [tx_b64, {"encoding": "base64", "skipPreflight": bool(skip_preflight),
                                                     "preflightCommitment": COMMITMENT, "maxRetries": int(max_retries)}])
        return str(sig)
