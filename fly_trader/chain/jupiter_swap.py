"""Jupiter Swap API v2 client: ``GET /order``, ``POST /execute``, ``GET /program-id-to-label``.

Base ``config.JUPITER_SWAP_BASE`` (https://api.jup.ag/swap/v2), header ``x-api-key``. Token-bucket
rate limits: one bucket (``config.JUPITER_RPS``) for ``/order`` and a separate one for ``/execute``
(Jupiter meters ``/execute`` separately). Every call is persisted to ``api_calls``
(service='jupiter'); the key is never stored (the request jsonb holds query params / a summary).

``GET /order`` verified 2026-09-12 against mainnet (0.01 SOL -> USDC):
  quote-only (no taker) -> HTTP 200 with fields: swapType ("aggregator"), inAmount, outAmount,
    otherAmountThreshold, swapMode ("ExactIn"), slippageBps (0 when not requested), priceImpactPct
    (string), routePlan [{percent, bps, usdValue, swapInfo {ammKey, label, inputMint, outputMint,
    inAmount, outAmount}}], feeMint, feeBps (2 for SOL/USDC), platformFee {feeBps, feeMint},
    signatureFeeLamports (0), signatureFeePayer (null), prioritizationFeeLamports (0),
    prioritizationFeePayer (null), rentFeeLamports, rentFeePayer, transaction (null), gasless,
    jitOptimized, taker (null), inputMint, outputMint, router ("metis"), guaranteedPrice, requestId,
    inUsdValue, outUsdValue, swapUsdValue, priceImpact (number, percentage points), mode ("ultra" and
    "manual" both observed), totalTime. No lastValidBlockHeight / expireAt / errorCode in the
    quote-only response. feeBps was 2 for SOL/USDC (docs: 10 bps most pairs, 50 bps for tokens < 24 h).
  with taker -> adds transaction (base64 v0, taker = fee payer, 1 required signature, no
    pre-filled signatures), lastValidBlockHeight (STRING, e.g. "424496168"), signatureFeePayer ==
    taker, signatureFeeLamports 5000, prioritizationFeeLamports/Payer, rentFeeLamports/Payer,
    mode "manual". Documented error fields when the order cannot be built: errorCode
    (aggregator: 1 insufficient funds, 2 insufficient SOL for gas, 3 below gasless minimum;
    jupiterz: 1 insufficient balance, 2 missing ATA, 3 quote could not be built) + errorMessage.
  Rate-limit headers: x-ratelimit-remaining / x-ratelimit-current / x-ratelimit-reset.

``POST /execute`` (per the v2 reference; never exercised here): body {signedTransaction, requestId,
  lastValidBlockHeight?(string)} -> {status "Success"|"Failed", signature, slot (string), code
  (0 ok; -1 missing cached order, -2 invalid signed tx, -3 invalid message bytes; -1000 failed to
  land, -1001 unknown, -1002 invalid tx, -1003 not fully signed, -1004 invalid block height;
  -2000 RFQ failed to land, -2001 unknown, -2002 invalid payload, -2003 quote expired, -2004 swap
  rejected), error, totalInputAmount, totalOutputAmount, inputAmountResult, outputAmountResult,
  swapEvents [{inputMint, inputAmount, outputMint, outputAmount}]}. The same signedTransaction +
  requestId is idempotent for ~2 minutes.

``GET /program-id-to-label`` verified keyless at https://api.jup.ag/swap/v2/program-id-to-label
(107 entries; /swap/v1 and lite-api.jup.ag serve the same map). Cached to
``data/jupiter_program_labels.json``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from .. import config
from ..db.apilog import record_api_call
from ..logging_setup import scrub
from .jupiter_tokens import TokenBucket

log = logging.getLogger(__name__)

JUPITER_EXECUTE_RPS = config.JUPITER_RPS  # separate bucket, same rate
JUPITER_LABELS_URL = "https://api.jup.ag/swap/v2/program-id-to-label"
JUPITER_LABELS_CACHE = config.DATA_DIR / "jupiter_program_labels.json"
JUPITER_LABELS_MAX_AGE_S = 86400.0
DEFAULT_EXCLUDE_ROUTERS = "jupiterz"


class JupiterError(RuntimeError):
    def __init__(self, endpoint: str, status: int | None, detail: str = ""):
        self.endpoint = endpoint
        self.status = status
        self.detail = detail
        super().__init__(f"jupiter {endpoint}: HTTP {status}: {detail}")


def _read_labels_cache(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("labels"), dict):
            return data
    except (OSError, ValueError):
        pass
    return None


def fetch_program_labels(cache_path: Path | None = None, refresh: bool = False,
                         max_age_s: float = JUPITER_LABELS_MAX_AGE_S) -> dict[str, str]:
    """Keyless ``GET /program-id-to-label`` with a JSON file cache; stale cache on network failure."""
    path = Path(cache_path) if cache_path else JUPITER_LABELS_CACHE
    cached = _read_labels_cache(path)
    if cached and not refresh and time.time() - float(cached.get("fetched_at", 0)) < max_age_s:
        return cached["labels"]
    t0 = time.monotonic()
    status = None
    err = None
    nbytes = None
    try:
        r = httpx.get(JUPITER_LABELS_URL, timeout=20)
        status = r.status_code
        nbytes = len(r.content)
        body = r.json() if status < 400 else None
        if status >= 400 or not isinstance(body, dict):
            raise JupiterError("/program-id-to-label", status, scrub(r.text[:200]))
        labels = {str(k): str(v) for k, v in body.items()}
    except (JupiterError, httpx.HTTPError, ValueError) as e:
        err = f"{type(e).__name__}: {scrub(str(e))}"
        if cached:
            log.warning("program-id-to-label fetch failed (%s); using stale cache", type(e).__name__)
            return cached["labels"]
        raise
    finally:
        record_api_call("jupiter", "swap/program-id-to-label", "GET", status,
                        int((time.monotonic() - t0) * 1000), err is None, err, response_bytes=nbytes)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "url": JUPITER_LABELS_URL, "labels": labels}, f, indent=0)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("could not write %s: %s", path, type(e).__name__)
    return labels


class JupiterSwap:
    def __init__(self, api_key: str | None = None, rps: float | None = None, timeout_s: float = 30.0):
        self.api_key = api_key or config.JUPITER_API_KEY
        if not self.api_key:
            raise RuntimeError("JUPITER_API_KEY is not set")
        self.base = config.JUPITER_SWAP_BASE
        self.order_bucket = TokenBucket(rps or config.JUPITER_RPS)
        self.execute_bucket = TokenBucket(rps or JUPITER_EXECUTE_RPS)
        self.client = httpx.Client(timeout=timeout_s, headers={"x-api-key": self.api_key})

    def _request(self, method: str, path: str, *, bucket: TokenBucket, params: dict | None = None,
                 json_body: dict | None = None, log_request: dict | None = None, accept_error_body,
                 retries: int = 3) -> tuple[int, object, int]:
        """One rate-limited request. Returns ``(http_status, parsed_body, latency_ms)``.
        Raises ``JupiterError`` on transport errors and on HTTP >= 400 unless ``accept_error_body(body)``
        says the JSON body is a structured Jupiter result the caller wants to see."""
        bucket.acquire()
        t0 = time.monotonic()
        status: int | None = None
        err: str | None = None
        nbytes: int | None = None
        ok = False
        try:
            for attempt in range(retries + 1):
                try:
                    r = self.client.request(method, self.base + path, params=params, json=json_body)
                except httpx.HTTPError as e:
                    raise JupiterError(path, None, f"{type(e).__name__}: {scrub(str(e))}") from None
                status = r.status_code
                nbytes = len(r.content)
                if status == 429 and attempt < retries:
                    time.sleep(1.0 + attempt)
                    bucket.acquire()
                    continue
                break
            try:
                body = r.json()
            except ValueError:
                body = None
            latency = int((time.monotonic() - t0) * 1000)
            if status >= 400 and not accept_error_body(body):
                snippet = json.dumps(body)[:300] if body is not None else r.text[:300]
                raise JupiterError(path, status, scrub(snippet))
            if isinstance(body, dict):
                if body.get("errorCode") is not None:
                    err = f"errorCode {body.get('errorCode')}: {body.get('errorMessage') or body.get('error') or ''}"
                elif body.get("code") not in (None, 0):
                    err = f"code {body.get('code')}: {body.get('error') or body.get('status') or ''}"
            ok = status < 400 and err is None
            return status, body, latency
        except JupiterError as e:
            err = str(e)
            raise
        finally:
            record_api_call("jupiter", f"swap{path}", method, status, int((time.monotonic() - t0) * 1000),
                            ok, scrub(err) if err else None, request=log_request, response_bytes=nbytes)

    def order(self, input_mint: str, output_mint: str, amount: int, taker: str | None = None,
              slippage_bps: int | None = None, exclude_routers: str | None = DEFAULT_EXCLUDE_ROUTERS,
              exclude_dexes: str | None = None) -> dict:
        """``GET /order``. Without ``taker`` this is a quote (no transaction is built). Returns the
        parsed JSON plus ``_latency_ms`` and ``_http_status``; a body carrying ``errorCode`` is returned
        as-is (the caller decides), other HTTP errors raise ``JupiterError``."""
        params: dict = {"inputMint": input_mint, "outputMint": output_mint, "amount": str(int(amount))}
        if taker:
            params["taker"] = str(taker)
        if slippage_bps is not None:
            params["slippageBps"] = int(slippage_bps)
        if exclude_routers:
            params["excludeRouters"] = exclude_routers
        if exclude_dexes:
            params["excludeDexes"] = exclude_dexes
        status, body, latency = self._request(
            "GET", "/order", bucket=self.order_bucket, params=params, log_request={"params": params},
            accept_error_body=lambda b: isinstance(b, dict) and "errorCode" in b)
        if not isinstance(body, dict):
            raise JupiterError("/order", status, "non-JSON response")
        body["_latency_ms"] = latency
        body["_http_status"] = status
        return body

    def execute(self, signed_transaction_b64: str, request_id: str, last_valid_block_height: int | None = None) -> dict:
        """``POST /execute``. Returns the parsed JSON plus ``_latency_ms``/``_http_status``; a body with
        ``status``/``code`` is returned even on HTTP errors so the broker can see the code."""
        payload: dict = {"signedTransaction": signed_transaction_b64, "requestId": request_id}
        if last_valid_block_height is not None:
            payload["lastValidBlockHeight"] = str(int(last_valid_block_height))
        summary = {"requestId": request_id, "lastValidBlockHeight": payload.get("lastValidBlockHeight"),
                   "signedTransactionBytes": len(signed_transaction_b64)}
        status, body, latency = self._request(
            "POST", "/execute", bucket=self.execute_bucket, json_body=payload, log_request=summary,
            accept_error_body=lambda b: isinstance(b, dict) and ("code" in b or "status" in b))
        if not isinstance(body, dict):
            raise JupiterError("/execute", status, "non-JSON response")
        body["_latency_ms"] = latency
        body["_http_status"] = status
        return body

    def program_id_to_label(self, refresh: bool = False) -> dict[str, str]:
        return fetch_program_labels(refresh=refresh)
