"""USD prices for the public stats: SOL from Jupiter Price v3, $FLY from its GeckoTerminal pool. Cached 60 s; a
failed fetch keeps the last good value (and its timestamp, so the page can show its age)."""
from __future__ import annotations

import logging
import threading
import time

import httpx

from .. import config

log = logging.getLogger(__name__)
TTL_S = 60.0
_cache: dict[str, tuple[float, float]] = {}
_lock = threading.Lock()


def _cached(key: str, fetch) -> tuple[float, float]:
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[1] < TTL_S:
        return hit
    try:
        val = float(fetch())
        if val > 0:
            hit = (val, time.time())
            with _lock:
                _cache[key] = hit
    except Exception as e:
        log.info("price %s unavailable: %s", key, type(e).__name__)
    return hit or (0.0, 0.0)


def _sol_usd() -> float:
    headers = {"x-api-key": config.JUPITER_API_KEY} if config.JUPITER_API_KEY else {}
    base = "https://api.jup.ag/price/v3" if config.JUPITER_API_KEY else "https://lite-api.jup.ag/price/v3"
    r = httpx.get(base, params={"ids": config.WSOL_MINT}, headers=headers, timeout=10)
    r.raise_for_status()
    return float(r.json()[config.WSOL_MINT]["usdPrice"])


def _fly_usd() -> float:
    r = httpx.get(f"https://api.geckoterminal.com/api/v2/networks/{config.GECKO_NETWORK}/pools/{config.FLY_POOL}",
                  headers={"Accept": "application/json"}, timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["attributes"]["base_token_price_usd"])


def sol_usd() -> tuple[float, float]:
    return _cached("sol", _sol_usd)


def fly_usd() -> tuple[float, float]:
    return _cached("fly", _fly_usd)


def snapshot() -> dict:
    s, st = sol_usd()
    f, ft = fly_usd()
    return {"sol_usd": s, "fly_usd": f, "ts": int(min(t for t in (st, ft) if t) if (st or ft) else 0)}
