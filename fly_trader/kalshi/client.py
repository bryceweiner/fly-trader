"""fly-trader's Kalshi REST client: one persistent, authenticated SDK client on a private event-loop thread, paced under
the account's rate tier, every call logged to ``api_calls`` (service 'kalshi'; the key never appears).

The vendored better_bot client (``kalshi/vendor/kalshi_client.py``) opens and closes an SDK client per call, which is
right for a daily batch and wrong for a history pull of tens of thousands of candlestick requests or a trading loop.
This wrapper keeps the same authentication (``Configuration.api_key_id`` / ``private_key_pem``, RSA-PSS by the SDK) and
the same ``call_api`` transport, adds the endpoints better_bot never needed — subaccounts, resting (GTC, post-only)
orders, cancel/amend, the historical tier, series and events — and passes ``KALSHI_SUBACCOUNT`` on every portfolio and
order call. Prices cross this boundary in cents (ints) and contracts as floats; the exchange's fixed-point dollar
strings are converted here and nowhere else. Paths and body fields verified against kalshi-python-async 3.10 on
2026-09-22 (``docs.kalshi.com``): V2 orders are ``POST /portfolio/orders`` with ``yes_price``/``no_price`` in cents.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from urllib.parse import urlencode

from .. import config
from ..chain.jupiter_tokens import TokenBucket
from ..db.apilog import record_api_call
from ..logging_setup import scrub

log = logging.getLogger(__name__)
RETRIES_429 = 5


class KalshiApiError(RuntimeError):
    def __init__(self, method: str, path: str, status: int | None, body):
        self.method, self.path, self.status, self.body = method, path, status, body
        err = (body or {}).get("error") if isinstance(body, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else None) or (body.get("message") if isinstance(body, dict) else None) or str(body)[:200]
        super().__init__(f"kalshi {method} {path}: HTTP {status}: {scrub(str(msg))}")


def _f(x) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def cents(dollars) -> int | None:
    """A fixed-point dollar string ('0.5600') as whole cents; None when absent."""
    v = _f(dollars)
    return int(round(v * 100)) if v is not None else None


def _summary(params: dict | None, body: dict | None) -> dict | None:
    out = {}
    if params:
        out["params"] = {k: v for k, v in params.items()}
    if body:
        out["body"] = {k: v for k, v in body.items() if k not in ("signedTransaction",)}
    return out or None


class KalshiRest:
    """Synchronous facade over the async SDK client (one per process is enough: MPS and the minute loop are single-threaded)."""

    def __init__(self, rps: float | None = None, subaccount: int | None = None, base_url: str | None = None):
        self.base = (base_url or config.KALSHI_API_BASE_URL).rstrip("/")
        self.subaccount = config.KALSHI_SUBACCOUNT if subaccount is None else int(subaccount)
        self.bucket = TokenBucket(rps or config.KALSHI_RPS)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client = None
        self._lock = threading.Lock()

    # ---- lifecycle ----
    def _ensure(self):
        with self._lock:
            if self._client is not None:
                return self._client
            if not config.KALSHI_API_KEY_ID or not config.KALSHI_PRIVATE_KEY_PATH:
                raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH are not set")
            from kalshi_python_async import Configuration, KalshiClient
            cfg = Configuration(); cfg.host = self.base
            cfg.api_key_id = config.KALSHI_API_KEY_ID
            with open(config.KALSHI_PRIVATE_KEY_PATH) as f:
                cfg.private_key_pem = f.read()
            try:
                import certifi
                cfg.ssl_ca_cert = certifi.where()
            except ImportError:
                pass
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name=f"{threading.current_thread().name}-kalshi-io", daemon=True)
            self._thread.start()
            self._client = asyncio.run_coroutine_threadsafe(self._make(cfg, KalshiClient), self._loop).result(30)
            return self._client

    @staticmethod
    async def _make(cfg, KalshiClient):
        return KalshiClient(configuration=cfg)

    def close(self) -> None:
        with self._lock:
            if self._client is not None and self._loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(10)
                except Exception:
                    pass
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._client = None

    # ---- transport ----
    async def _call(self, method: str, url: str, body: dict | None):
        raw = await self._client.call_api(method, url, body=body, header_params={"Content-Type": "application/json"} if body is not None else None)
        data = await raw.read()
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        start = text.find("{")
        try:
            parsed = json.loads(text[start:]) if start >= 0 else {}
        except ValueError:
            parsed = {"raw": text[:300]}
        return int(raw.status), parsed

    def request(self, method: str, path: str, params: dict | None = None, body: dict | None = None, timeout: float = 60.0) -> dict:
        """One paced, logged call. Raises ``KalshiApiError`` on HTTP >= 400 (429 is retried with backoff first)."""
        client = self._ensure()
        q = {k: v for k, v in (params or {}).items() if v is not None}
        url = self.base + path + (("?" + urlencode(q)) if q else "")
        t0 = time.monotonic(); status = None; err = None; parsed = None
        try:
            for attempt in range(RETRIES_429 + 1):
                self.bucket.acquire()
                status, parsed = asyncio.run_coroutine_threadsafe(self._call(method, url, body), self._loop).result(timeout)
                if status == 429 and attempt < RETRIES_429:
                    time.sleep(1.0 + attempt)
                    continue
                break
            if status >= 400:
                raise KalshiApiError(method, path, status, parsed)
            return parsed
        except KalshiApiError as e:
            err = str(e); raise
        except Exception as e:
            err = f"{type(e).__name__}: {scrub(str(e))}"; raise
        finally:
            record_api_call("kalshi", path, method, status, int((time.monotonic() - t0) * 1000), err is None, err,
                            request=_summary(q, body), response_bytes=len(json.dumps(parsed)) if parsed is not None else None)

    def get(self, path: str, **params) -> dict:
        return self.request("GET", path, params)

    def pages(self, path: str, key: str, params: dict | None = None, limit: int = 1000, max_pages: int = 10_000):
        """Cursor walk: yields each page's ``key`` list until the cursor ends or a page is empty."""
        cursor = None
        for _ in range(max_pages):
            data = self.request("GET", path, {**(params or {}), "limit": limit, "cursor": cursor})
            items = data.get(key) or []
            if items:
                yield items
            cursor = data.get("cursor")
            if not cursor or not items:
                break

    # ---- exchange / catalogue ----
    def exchange_status(self) -> dict:
        return self.get("/exchange/status")

    def markets(self, **params):
        """Pages of market rows (``status`` one of unopened|open|closed|settled|...; ``min_settled_ts`` etc.)."""
        yield from self.pages("/markets", "markets", params)

    def historical_markets(self, **params):
        yield from self.pages("/historical/markets", "markets", params)

    def historical_cutoff(self) -> dict:
        return self.get("/historical/cutoff")

    def market(self, ticker: str) -> dict:
        return (self.get(f"/markets/{ticker}") or {}).get("market") or {}

    def historical_market(self, ticker: str) -> dict:
        return (self.get(f"/historical/markets/{ticker}") or {}).get("market") or {}

    def events(self, **params):
        """Pages of events (``with_nested_markets=true`` nests their markets); limit is capped at 200 by the exchange."""
        yield from self.pages("/events", "events", params, limit=200)

    def event(self, event_ticker: str, with_nested_markets: bool = True) -> dict:
        return self.get(f"/events/{event_ticker}", with_nested_markets=str(with_nested_markets).lower())

    def series_list(self, category: str | None = None, **params) -> list[dict]:
        return (self.get("/series", category=category, **params) or {}).get("series") or []

    def series(self, series_ticker: str) -> dict:
        return (self.get(f"/series/{series_ticker}") or {}).get("series") or {}

    def orderbook(self, ticker: str, depth: int | None = None) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", depth=depth)

    def candlesticks(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1) -> list[dict]:
        """1/60/1440-minute candles; a market archived past the historical cutoff answers on the historical path."""
        params = {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": int(period_interval)}
        try:
            return (self.request("GET", f"/series/{series_ticker}/markets/{ticker}/candlesticks", params) or {}).get("candlesticks") or []
        except KalshiApiError as e:
            if e.status != 404:
                raise
        return (self.request("GET", f"/historical/markets/{ticker}/candlesticks", params) or {}).get("candlesticks") or []

    def trades(self, ticker: str, min_ts: int | None = None, max_ts: int | None = None, historical: bool = False):
        yield from self.pages("/historical/trades" if historical else "/markets/trades", "trades", {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts})

    # ---- portfolio (always on this process's subaccount unless told otherwise) ----
    def _sub(self, subaccount) -> int:
        return self.subaccount if subaccount is None else int(subaccount)

    def balance(self, subaccount: int | None = None) -> dict:
        """``{"balance_cents", "portfolio_value_cents"}`` of one subaccount (0 = primary)."""
        d = self.get("/portfolio/balance", subaccount=self._sub(subaccount))
        return {"balance_cents": int(d.get("balance") or 0), "portfolio_value_cents": int(d.get("portfolio_value") or 0), "updated_ts": d.get("updated_ts")}

    def subaccount_balances(self) -> list[dict]:
        return (self.get("/portfolio/subaccounts/balances") or {}).get("subaccount_balances") or []

    def create_subaccount(self) -> int:
        d = self.request("POST", "/portfolio/subaccounts", body={})
        return int(d["subaccount_number"])

    def transfer_between_subaccounts(self, from_subaccount: int, to_subaccount: int, amount_cents: int) -> dict:
        body = {"client_transfer_id": str(uuid.uuid4()), "from_subaccount": int(from_subaccount), "to_subaccount": int(to_subaccount), "amount_cents": int(amount_cents)}
        return self.request("POST", "/portfolio/subaccounts/transfer", body=body)

    def positions(self, subaccount: int | None = None, **params) -> list[dict]:
        out = []
        for page in self.pages("/portfolio/positions", "market_positions", {**params, "subaccount": self._sub(subaccount)}, limit=1000):
            out.extend(page)
        return out

    def orders(self, status: str | None = None, subaccount: int | None = None, **params) -> list[dict]:
        out = []
        for page in self.pages("/portfolio/orders", "orders", {**params, "status": status, "subaccount": self._sub(subaccount)}, limit=1000):
            out.extend(page)
        return out

    def order(self, order_id: str) -> dict:
        return (self.get(f"/portfolio/orders/{order_id}") or {}).get("order") or {}

    def fills(self, subaccount: int | None = None, **params) -> list[dict]:
        out = []
        for page in self.pages("/portfolio/fills", "fills", {**params, "subaccount": self._sub(subaccount)}, limit=1000):
            out.extend(page)
        return out

    def settlements(self, subaccount: int | None = None, **params) -> list[dict]:
        out = []
        for page in self.pages("/portfolio/settlements", "settlements", {**params, "subaccount": self._sub(subaccount)}, limit=1000):
            out.extend(page)
        return out

    # ---- orders ----
    def create_order(self, ticker: str, side: str, price_cents: int, count: float, *, tif: str = "immediate_or_cancel", post_only: bool = False,
                     expiration_ts: int | None = None, client_order_id: str | None = None, subaccount: int | None = None, action: str = "buy") -> dict:
        """Buy (or sell) ``count`` contracts of ``side`` ('yes'|'no') at ``price_cents`` each. ``tif``: immediate_or_cancel |
        good_till_canceled | fill_or_kill; a resting order is GTC (+ ``post_only`` never crosses, ``expiration_ts`` ends it).
        Returns the exchange's order object (``order_id``, ``status``, ``fill_count``/``remaining_count`` ...)."""
        if side not in ("yes", "no"):
            raise ValueError(f"side must be yes|no, got {side!r}")
        if not 1 <= int(price_cents) <= 99:
            raise ValueError(f"price {price_cents} outside 1..99 cents")
        body = {"ticker": ticker, "client_order_id": client_order_id or f"fly-{uuid.uuid4().hex[:16]}", "side": side, "action": action,
                "count_fp": f"{float(count):.2f}", ("yes_price" if side == "yes" else "no_price"): int(price_cents), "time_in_force": tif,
                "post_only": bool(post_only), "self_trade_prevention_type": "maker" if post_only else "taker_at_cross",
                "subaccount": self._sub(subaccount)}
        if expiration_ts is not None:
            body["expiration_ts"] = int(expiration_ts)
        d = self.request("POST", "/portfolio/orders", body=body)
        return d.get("order") or d

    def cancel_order(self, order_id: str, subaccount: int | None = None) -> dict:
        d = self.request("DELETE", f"/portfolio/orders/{order_id}", {"subaccount": self._sub(subaccount)})
        return d.get("order") or d

    def amend_order(self, order_id: str, ticker: str, side: str, price_cents: int, count: float, client_order_id: str, subaccount: int | None = None,
                    action: str = "buy") -> dict:
        body = {"ticker": ticker, "side": side, "action": action, "client_order_id": client_order_id, "updated_client_order_id": f"fly-{uuid.uuid4().hex[:16]}",
                ("yes_price" if side == "yes" else "no_price"): int(price_cents), "count_fp": f"{float(count):.2f}", "subaccount": self._sub(subaccount)}
        d = self.request("POST", f"/portfolio/orders/{order_id}/amend", body=body)
        return d.get("order") or d


_REST: KalshiRest | None = None
_REST_LOCK = threading.Lock()


def rest() -> KalshiRest:
    """The process-wide client (built on first use)."""
    global _REST
    with _REST_LOCK:
        if _REST is None:
            _REST = KalshiRest()
        return _REST


# ---- operator commands (fly-trader kalshi-subaccount / kalshi-fund / kalshi-status) ----
def subaccount_create(write_env: bool = True) -> int:
    """Create the dedicated subaccount and record it in .env (``KALSHI_SUBACCOUNT``); refuses when one is already set."""
    import re
    from ..db.apilog import record_event
    if config.KALSHI_SUBACCOUNT > 0:
        raise RuntimeError(f"KALSHI_SUBACCOUNT={config.KALSHI_SUBACCOUNT} is already set; refusing to create another")
    n = KalshiRest(subaccount=0).create_subaccount()
    if write_env:
        path = config.REPO_ROOT / ".env"; text = path.read_text() if path.exists() else ""
        text = re.sub(r"^KALSHI_SUBACCOUNT=.*$", f"KALSHI_SUBACCOUNT={n}", text, flags=re.M) if re.search(r"^KALSHI_SUBACCOUNT=", text, re.M) else text.rstrip("\n") + f"\nKALSHI_SUBACCOUNT={n}\n"
        path.write_text(text)
    record_event("info", "kalshi", f"subaccount {n} created", {"subaccount": n})
    return n


def fund(usd: float, to_subaccount: int | None = None) -> dict:
    """Move ``usd`` from the primary account into the fly's subaccount."""
    from ..db.apilog import record_event
    to = config.KALSHI_SUBACCOUNT if to_subaccount is None else int(to_subaccount)
    if to <= 0:
        raise RuntimeError("no subaccount: run fly-trader kalshi-subaccount create first")
    out = KalshiRest(subaccount=0).transfer_between_subaccounts(0, to, int(round(usd * 100)))
    record_event("info", "kalshi", f"funded subaccount {to} with ${usd:.2f}", {"transfer": out})
    return out


def status() -> dict:
    r = rest()
    out = {"exchange": r.exchange_status(), "subaccount": r.subaccount, "balances": r.subaccount_balances(),
           "prerequisites_missing": config.kalshi_live_prerequisites_missing()}
    try:
        out["balance"] = r.balance()
        out["resting_orders"] = r.orders(status="resting")
        out["positions"] = [p for p in r.positions() if _f(p.get("position_fp") or p.get("position")) not in (None, 0.0)]
        out["settlements_24h"] = r.settlements(min_ts=int(time.time()) - 86400)
    except KalshiApiError as e:
        out["error"] = str(e)
    return out
