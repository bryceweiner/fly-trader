"""Jupiter Tokens API v2 client (https://api.jup.ag/tokens/v2).

Endpoint (verified 2026-09-12 with the paid key): /search?query=<csv of up to 100 mints>. Tokens carry organicScore,
holderCount, liquidity, mcap, stats5m/1h/6h/24h and audit: mintAuthorityDisabled,
freezeAuthorityDisabled, topHoldersPercentage, devBalancePercentage, devMints, devMigrations (isSus is
absent unless flagged). Rate limit headers: x-ratelimit-remaining/current/reset.
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

from .. import config
from ..db.apilog import record_api_call

log = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int | None = None):
        self.rate = rate_per_s
        self.capacity = burst or max(1, int(rate_per_s))
        self.tokens = float(self.capacity)
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class JupiterTokens:
    def __init__(self, api_key: str | None = None, rps: float | None = None):
        self.api_key = api_key or config.JUPITER_API_KEY
        if not self.api_key:
            raise RuntimeError("JUPITER_API_KEY is not set")
        self.base = config.JUPITER_TOKENS_BASE
        self.bucket = TokenBucket(rps or config.JUPITER_RPS)
        self.client = httpx.Client(timeout=30, headers={"x-api-key": self.api_key})

    def _get(self, path: str, params: dict | None = None) -> list[dict]:
        self.bucket.acquire()
        t0 = time.monotonic()
        status = None
        err = None
        try:
            for attempt in range(4):
                r = self.client.get(self.base + path, params=params)
                status = r.status_code
                if r.status_code == 429:
                    time.sleep(1.0 + attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else []
            raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            record_api_call("jupiter", f"tokens{path}", "GET", status, int((time.monotonic() - t0) * 1000),
                            err is None, err, request={"params": params} if params else None)

    def search(self, query: str) -> list[dict]:
        return self._get("/search", {"query": query})

    def search_many(self, mints: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(mints), 100):
            out.extend(self.search(",".join(mints[i:i + 100])))
        return out
