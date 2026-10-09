# Vendored verbatim from better_bot code/kalshi_client.py by tools/vendor_kalshi.py; sports sections removed. Do not edit.
import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import (
    KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_API_BASE_URL,
    KALSHI_FEE_FACTOR,
)
from .gate_register import can_use

logger = logging.getLogger(__name__)


#: Per-series fee configuration, read from the exchange and memoized for the
#: process: ``series_ticker -> {"fee_type": str, "fee_multiplier": float}``.
#:
#: Never populated lazily from inside the fee arithmetic. The fee functions run
#: inside orderbook walks, wager writes and the coverage gate's replay, and a
#: network call in any of those would make sizing depend on reachability and the
#: gate's verdict depend on the day it ran. Warm it explicitly with
#: :func:`prefetch_series_fees` before a cycle that prices anything.
_series_fee_cache = {}

#: The multiplier the exchange itself defaults to, and therefore what an
#: unknown series is charged at. Deliberately the CONSERVATIVE direction: every
#: real multiplier we have seen is 1.0 or 0.5, so assuming 1.0 can only
#: over-state the fee and under-state edge. A missing lookup must never make a
#: wager look cheaper than it is.
DEFAULT_FEE_MULTIPLIER = 1.0


def series_ticker_of(ticker):
    """The series segment of a Kalshi ticker, or ``None``.

    Kalshi tickers are ``SERIES-EVENT-OUTCOME``, and a combo market's is
    ``KXMVECROSSCATEGORY-S2026...-F5F8...``, so the first dash-part is the
    series for market, event and multivariate tickers alike.
    """
    if not ticker:
        return None
    return str(ticker).split("-")[0] or None


def series_fee_multiplier(ticker=None):
    """The fee multiplier for a market's series, defaulting to 1.0.

    MLB is the reason this exists: every ``KXMLB*`` series -- game, spread,
    total, first five, and all eight player-prop ladders -- carries
    ``fee_multiplier: 0.5``, so charging the flat factor over-stated the fee by
    2x and discarded ~0.8pp of edge on every MLB wager, including at the
    selection gate that rejects candidates for falling short of break-even.
    """
    entry = _series_fee_cache.get(series_ticker_of(ticker))
    if not entry:
        return DEFAULT_FEE_MULTIPLIER
    return entry["fee_multiplier"]


def series_fee_type(ticker=None):
    """``'quadratic'`` or ``'quadratic_with_maker_fees'`` for a market's series.

    Only ``quadratic_with_maker_fees`` charges a resting order. Nothing sizes on
    this yet -- every order the bot places is IOC, so it is always the taker --
    but it is the difference between a free and a charged quote on the RFQ path,
    and it arrives in the same response as the multiplier.
    """
    entry = _series_fee_cache.get(series_ticker_of(ticker))
    return entry["fee_type"] if entry else None


def kalshi_fee_rate_cents(price_cents, ticker=None):
    """Marginal Kalshi entry fee per contract, in cents, NOT rounded.

    ``factor * multiplier * p * (1-p) * 100``. Quadratic in price and maximal at
    50c (1.75c/contract at the default 7% factor and multiplier 1, versus 1.12c
    at 80c) -- so a market trading near even money costs ~3.5% of stake to
    enter, against ~1.4% at 80c. Fee as a fraction of STAKE is
    ``factor * multiplier * (1-p)``, which falls monotonically as price rises:
    a 5c contract pays 6.7% of stake, a 50c one 3.5%.

    ``ticker`` is a market, event or series ticker and selects the series'
    multiplier; omitting it charges the exchange default of 1.0. Pass it
    wherever it is known -- an MLB market charged without it pays double.

    Use this for **sizing** (Kelly, edge, break-even), where what matters is the
    marginal cost of the next contract. Use :func:`kalshi_trading_fee_cents` for
    **accounting** an actual order, where the exchange ceils the whole order to a
    cent. Sizing must not use the ceiled single-contract figure: the exchange
    ceils per order, so ``kalshi_trading_fee_cents(price, 1)`` is 2c at every
    price and overstates the true marginal rate by 14% at 50c and 79% at 80c.
    """
    p = price_cents / 100.0
    return KALSHI_FEE_FACTOR * series_fee_multiplier(ticker) * p * (1 - p) * 100


def effective_price_cents(price_cents, ticker=None):
    """Fee-inclusive cost basis per contract, in cents.

    The entry fee is part of what a contract costs, so every profitability
    question -- break-even, edge, Kelly -- is asked against price + fee, never
    the raw ask. Break-even follows directly: EV = p(100 - P) - (1 - p)P - f
    = 100p - P - f, so EV > 0 iff p > (P + f)/100.

    At 50c and multiplier 1 that is p* = 0.5175, not 0.50: 1.75 points of edge
    are consumed by the fee before a wager breaks even. On MLB's half-rate
    series the same contract breaks even at 0.5088.
    """
    return price_cents + kalshi_fee_rate_cents(price_cents, ticker)


def kalshi_trading_fee_cents(price_cents, quantity, ticker=None):
    """Modeled Kalshi trading fee in cents: ceil(factor * mult * C * p * (1-p)).

    Real fills report the fee via the API; --paper simulated fills have no
    exchange record, so this models the entry fee for realistic paper P&L.
    """
    # Round before ceil so float artifacts don't bump an exact-cent fee up a cent
    # (e.g. 0.07*100*0.5*0.5*100 = 175.0 must not ceil to 176).
    return math.ceil(
        round(kalshi_fee_rate_cents(price_cents, ticker) * quantity, 6))


_client_instance = None


def _kalshi_available():
    return bool(
        can_use("kalshi_calibration")
        and KALSHI_API_KEY_ID
        and KALSHI_PRIVATE_KEY_PATH
    )


def _load_private_key():
    key_path = KALSHI_PRIVATE_KEY_PATH
    if not key_path or not os.path.exists(key_path):
        return None
    with open(key_path, "r") as f:
        return f.read()


def _build_client():
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    if not _kalshi_available():
        return None
    try:
        from kalshi_python_async import KalshiClient, Configuration
        private_key_pem = _load_private_key()
        if not private_key_pem:
            logger.warning("Kalshi private key file not found or empty")
            return None
        config = Configuration()
        config.host = KALSHI_API_BASE_URL
        config.api_key_id = KALSHI_API_KEY_ID
        config.private_key_pem = private_key_pem
        try:
            import certifi
            config.ssl_ca_cert = certifi.where()
        except ImportError:
            logger.debug("certifi not available, using default CA bundle")
        _client_instance = KalshiClient(configuration=config)
        return _client_instance
    except Exception as e:
        logger.warning(f"Failed to initialize Kalshi client: {e}")
        return None


async def _close_client():
    global _client_instance
    if _client_instance is not None:
        try:
            await _client_instance.close()
        except Exception as e:
            logger.debug("kalshi client close failed: %s", e)
        _client_instance = None


def _market_implied_probability(market):
    try:
        yes_bid = float(market.yes_bid_dollars or "0")
        yes_ask = float(market.yes_ask_dollars or "0")
        if yes_bid > 0 and yes_ask > 0:
            midpoint = (yes_bid + yes_ask) / 2.0
            return round(midpoint * 100.0, 2)
        last_price = float(market.last_price_dollars or "0")
        if last_price > 0:
            return round(last_price * 100.0, 2)
    except (ValueError, TypeError):
        pass
    return None


def market_quote(market):
    """``{implied_prob, volume, ticker, yes_ask}`` for one market, or ``None``.

    One reader for the four values every consumer of a Kalshi market wants, so
    the ``volume``/``volume_fp`` fallback and the dollars-to-cents conversion
    live in one place instead of being repeated at each call site.
    """
    implied = _market_implied_probability(market)
    if implied is None:
        return None
    quote = {
        "market_prob": implied,
        "volume": int(float(getattr(market, "volume", None)
                                   or getattr(market, "volume_fp", None) or 0)),
        "market_id": market.ticker,
        "yes_ask_cents": None,
    }
    try:
        ask = float(market.yes_ask_dollars or "0")
        if ask > 0:
            quote["yes_ask_cents"] = round(ask * 100.0)
    except (ValueError, TypeError):
        pass
    return quote


def _run_sync(coro):
    """Run an async coroutine synchronously, handling existing event loops."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return asyncio.run(coro)
    except Exception as e:
        logger.warning(f"Kalshi sync wrapper failed: {e}")
        return None


def _kalshi_sync(async_fn, default=None, error_msg="Kalshi operation failed",
                 log_level="error"):
    """Run an async Kalshi operation with standard availability check and error handling.

    async_fn receives the built client as its sole argument and returns the result.
    """
    if not _kalshi_available():
        return default

    async def _run():
        client = _build_client()
        if not client:
            return default
        try:
            return await async_fn(client)
        except Exception as e:
            getattr(logger, log_level)(f"{error_msg}: {e}")
            return default
        finally:
            await _close_client()

    result = _run_sync(_run())
    return result if result is not None else default


def get_orderbook_sync(ticker, depth=0):
    """Full resting depth for one market, or ``None`` when unavailable.

    ``depth`` of 0 requests every level. Returns the raw payload; parsing and the
    Yes/No side convention live in ``book_sizing`` so the network boundary stays
    thin and the sizing logic stays a pure function.

    ``None`` (rather than an empty book) on failure is deliberate: the caller
    must be able to tell "the market has no depth" from "we could not ask", and
    fall back to top-of-book sizing rather than skipping a wager because a fetch
    errored.
    """
    async def _fetch(client):
        # Called through call_api rather than the SDK's MarketApi class, matching
        # how every other endpoint here reaches the API and avoiding a dependency
        # on the generated model layer for a two-field payload.
        url = f"{KALSHI_API_BASE_URL}/markets/{ticker}/orderbook"
        if depth:
            url = f"{url}?depth={int(depth)}"
        raw = await client.call_api("GET", url)
        return _read_json_body(await raw.read())

    return _kalshi_sync(_fetch, default=None,
                        error_msg=f"Kalshi orderbook fetch failed for {ticker}",
                        log_level="warning")


#: Kalshi caps a /markets page at 1000; 200 keeps a single prop series to one
#: or two round trips without risking a rejected limit.
_MARKETS_PAGE_LIMIT = 200
#: Refuses to walk a series forever if the cursor never terminates. A prop
#: series for one league-day is a few hundred markets, so this is far above any
#: real answer and only bounds a pathological response.
_MARKETS_MAX_PAGES = 25


def get_markets_sync(series_ticker=None, event_ticker=None, status="open",
                     max_pages=_MARKETS_MAX_PAGES):
    """Every market in a series (or event), following Kalshi's cursor.

    This is the endpoint prop discovery needs and the one that was missing: only
    ``GET /events?with_nested_markets`` was wired, which reaches the single
    winner market per game and cannot enumerate a prop ladder.

    Returns ``[]`` on failure rather than ``None`` -- unlike an orderbook, an
    empty series and an unreachable one lead to the same action (price nothing),
    and every caller would otherwise repeat the same None check.
    """
    if not series_ticker and not event_ticker:
        raise ValueError("get_markets_sync needs a series_ticker or an "
                         "event_ticker; an unfiltered /markets walk would page "
                         "through the whole exchange")

    async def _fetch(client):
        collected = []
        cursor = None
        for _ in range(max_pages):
            params = [f"limit={_MARKETS_PAGE_LIMIT}"]
            if series_ticker:
                params.append(f"series_ticker={series_ticker}")
            if event_ticker:
                params.append(f"event_ticker={event_ticker}")
            if status:
                params.append(f"status={status}")
            if cursor:
                params.append(f"cursor={cursor}")
            raw = await client.call_api(
                "GET", f"{KALSHI_API_BASE_URL}/markets?" + "&".join(params))
            if raw.status >= 400:
                break
            data = _read_json_body(await raw.read())
            page = data.get("markets") or []
            collected.extend(page)
            cursor = data.get("cursor")
            # Kalshi returns the same cursor with an empty page at the end of a
            # series, so stop on either signal rather than on the cursor alone.
            if not cursor or not page:
                break
        return collected

    label = series_ticker or event_ticker
    return _kalshi_sync(_fetch, default=[],
                        error_msg=f"Kalshi market listing failed for {label}",
                        log_level="warning")


#: Kalshi's ``/markets?tickers=`` accepts a comma-joined list; keep the batch
#: well inside any URL length limit rather than discovering it in production.
_TICKERS_PER_REQUEST = 40


def get_markets_by_ticker_sync(tickers):
    """``{ticker: market}`` for specific markets, batched.

    The complement to :func:`get_markets_sync`: that one enumerates a series,
    this one reads a known set. Combo legs come as tickers scattered across many
    series, so listing each series to find one market would be hundreds of
    requests for a handful of rows.

    A ticker the exchange does not return is simply absent from the result. The
    caller decides what a missing market means -- for a combo leg it means the
    whole structure is unmeasurable, which is not an error.
    """
    wanted = sorted({t for t in tickers or () if t})
    if not wanted:
        return {}

    async def _fetch(client):
        collected = {}
        for start in range(0, len(wanted), _TICKERS_PER_REQUEST):
            batch = wanted[start:start + _TICKERS_PER_REQUEST]
            raw = await client.call_api(
                "GET", f"{KALSHI_API_BASE_URL}/markets"
                       f"?limit={_MARKETS_PAGE_LIMIT}&tickers=" + ",".join(batch))
            if raw.status >= 400:
                continue
            for market in _read_json_body(await raw.read()).get("markets") or []:
                collected[market.get("ticker")] = market
        return collected

    return _kalshi_sync(_fetch, default={},
                        error_msg=f"Kalshi market lookup failed for "
                                  f"{len(wanted)} ticker(s)",
                        log_level="warning")


def prefetch_series_fees(tickers):
    """Warm the per-series fee cache for every series among ``tickers``.

    Accepts market, event or series tickers interchangeably and fetches each
    distinct series once. Returns the number of series newly resolved.

    A series that cannot be read is left ABSENT from the cache rather than
    recorded at the default, so a later call can retry it. Recording the
    fallback would make one unlucky request permanently charge that series the
    full rate for the rest of the process.
    """
    wanted = {series_ticker_of(t) for t in tickers or ()}
    wanted.discard(None)
    missing = sorted(wanted - _series_fee_cache.keys())
    if not missing:
        return 0

    async def _fetch(client):
        resolved = {}
        for series in missing:
            raw = await client.call_api(
                "GET", f"{KALSHI_API_BASE_URL}/series/{series}")
            if raw.status >= 400:
                continue
            body = _read_json_body(await raw.read()).get("series") or {}
            multiplier = body.get("fee_multiplier")
            if multiplier is None:
                continue
            resolved[series] = {"fee_type": body.get("fee_type"),
                                "fee_multiplier": float(multiplier)}
        return resolved

    resolved = _kalshi_sync(_fetch, default={},
                            error_msg="Kalshi series fee lookup failed",
                            log_level="warning")
    _series_fee_cache.update(resolved)
    discounted = sorted(s for s, v in resolved.items()
                        if v["fee_multiplier"] != DEFAULT_FEE_MULTIPLIER)
    if discounted:
        logger.info("kalshi fees: %d series below the default multiplier (%s)",
                    len(discounted), ", ".join(discounted[:8]))
    return len(resolved)


def clear_series_fee_cache():
    """Forget every memoized series fee. For tests and long-lived processes."""
    _series_fee_cache.clear()


def market_ask_cents(market, side):
    """Best quoted ask for ``side`` on a raw ``/markets`` row, or ``None``.

    Kalshi publishes both ``yes_ask`` (cent integer) and ``yes_ask_dollars``
    (fixed-point string). Which key the value came from decides the scale --
    never the value itself, since ``1`` is a legitimate reading of both one cent
    and one dollar and guessing wrong is a silent 100x. Same discipline as
    ``book_sizing.parse_kalshi_orderbook``.

    This is top-of-book only, and is a *filter* rather than a sizing input: the
    order book proper is fetched for the few candidates that survive it, because
    fetching depth for every rung of every ladder on a slate is thousands of
    round trips to answer a question a quote already answers.
    """
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    dollars, cents = market.get(f"{side}_ask_dollars"), market.get(f"{side}_ask")
    try:
        if dollars not in (None, ""):
            value = int(round(float(dollars) * 100))
        elif cents not in (None, ""):
            value = int(cents)
        else:
            return None
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 99 else None


def get_market_sync(ticker):
    """One market's current row, or ``None``.

    Settlement reads the exchange's own ``result`` through this. A prop resolves
    on the exchange's rules -- which player was credited with the assist, how a
    postponement is handled -- and re-deriving that from our box scores would be
    inventing a second, disagreeing authority for something Kalshi publishes.
    """
    async def _fetch(client):
        raw = await client.call_api(
            "GET", f"{KALSHI_API_BASE_URL}/markets/{ticker}")
        if raw.status >= 400:
            return None
        return (_read_json_body(await raw.read()) or {}).get("market")

    return _kalshi_sync(_fetch, default=None,
                        error_msg=f"Kalshi market fetch failed for {ticker}",
                        log_level="warning")


#: Candlestick periods Kalshi accepts, in minutes. Not a free integer.
CANDLE_PERIODS = (1, 60, 1440)


def get_candlesticks_sync(series_ticker, ticker, start_ts, end_ts,
                          period_interval=1):
    """Quote history for one market, oldest first, or ``[]``.

    The only way to recover what the exchange was asking at a moment that has
    passed. A finalized market reports ``yes_ask`` as ``None`` on ``/markets``
    -- it has settled, so there is no quote left to read -- which is why the
    calibration log has to capture the price when it prices a rung, or come
    back here for it.

    Note the path: candlesticks hang off the SERIES, not off ``/markets``, and
    ``/markets/{ticker}/candlesticks`` is a plain 404 rather than an error that
    explains itself. The ``/historical/`` prefix in the current API reference
    resolves the ticker and then reports it not found for a settled prop.

    Prices come back as fixed-point dollar strings (``close_dollars``), never
    cent integers, so the scale is decided by the key -- see
    :func:`market_ask_cents` for why guessing it is a silent 100x.
    """
    if period_interval not in CANDLE_PERIODS:
        raise ValueError(f"period_interval must be one of {CANDLE_PERIODS}, "
                         f"got {period_interval}")

    async def _fetch(client):
        raw = await client.call_api(
            "GET", f"{KALSHI_API_BASE_URL}/series/{series_ticker}/markets/"
                   f"{ticker}/candlesticks?start_ts={int(start_ts)}"
                   f"&end_ts={int(end_ts)}&period_interval={period_interval}")
        if raw.status >= 400:
            return []
        return (_read_json_body(await raw.read()) or {}).get("candlesticks") or []

    return _kalshi_sync(_fetch, default=[],
                        error_msg=f"Kalshi candlesticks failed for {ticker}",
                        log_level="debug")


def candle_ask_cents(candle, side="yes"):
    """Closing ask for one side of a candlestick, in cents, or ``None``.

    A candle with no trades still carries a quote, so this reads the book rather
    than ``price`` -- the latter is null whenever nothing changed hands and would
    drop exactly the quiet rungs the calibration log is full of.

    A candle quotes only the Yes side, and **an ask on one side is a bid on the
    other**: buying No consumes Yes bids, so a No ask is ``100 - yes_bid``. Same
    identity ``prop_betting._quoted_book`` uses to build a book from a listing's
    two quotes, and reading ``yes_ask`` for both sides instead is an error worth
    most of a dollar on a rung the market prices at 0.94.
    """
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    key = "yes_ask" if side == "yes" else "yes_bid"
    dollars = (candle.get(key) or {}).get("close_dollars")
    if dollars in (None, ""):
        return None
    try:
        value = int(round(float(dollars) * 100))
    except (TypeError, ValueError):
        return None
    if side == "no":
        value = 100 - value
    return value if 1 <= value <= 99 else None


async def _get_balance(client):
    response = await client.get_balance()
    if not response:
        return None
    return {
        "balance": response.balance,
        "portfolio_value": response.portfolio_value,
    }


def get_balance_sync():
    """Return dict with balance and portfolio_value in cents, or None on failure."""
    return _kalshi_sync(_get_balance, error_msg="Kalshi balance fetch failed")


async def _paginate_raw(client, method_name, result_key, limit=200):
    """Fetch all pages from a paginated Kalshi endpoint using raw responses."""
    all_items = []
    cursor = None
    while True:
        kwargs = {"limit": limit}
        if cursor:
            kwargs["cursor"] = cursor
        method = getattr(client, method_name)
        raw = await method(**kwargs)
        data = json.loads(await raw.read())
        items = data.get(result_key) or []
        if not items:
            break
        all_items.extend(items)
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_items


def get_settlements_sync():
    """Return list of all settlement dicts from Kalshi."""
    return _kalshi_sync(
        lambda c: _paginate_raw(c, "get_settlements_without_preload_content", "settlements"),
        default=[], error_msg="Kalshi settlements fetch failed",
    )


def get_fills_sync():
    """Return list of all fill dicts from Kalshi."""
    return _kalshi_sync(
        lambda c: _paginate_raw(c, "get_fills_without_preload_content", "fills"),
        default=[], error_msg="Kalshi fills fetch failed",
    )


def get_transfers_sync():
    """Fetch deposit/withdrawal history via GET /portfolio/transfers.

    This endpoint isn't exposed in the SDK, so we call it directly.
    """
    async def _fetch(client):
        raw = await client.call_api("GET", KALSHI_API_BASE_URL + "/portfolio/transfers")
        body = await raw.read()
        text = body.decode("utf-8") if isinstance(body, bytes) else str(body)
        # API sometimes returns concatenated responses; find the JSON object
        start = text.find("{")
        if start < 0:
            return []
        data = json.loads(text[start:])
        return data.get("transfers") or []

    return _kalshi_sync(_fetch, default=[], error_msg="Kalshi transfers fetch failed",
                        log_level="warning")


# Kalshi's V2 order endpoint quotes everything from the YES leg: "bid" buys
# YES, "ask" sells YES. There is one book per market, so an ask at p from an
# account holding no Yes IS a No position bought at 1 - p, and the positions
# endpoint reports it as a negative contract count (docs.kalshi.com, Create
# Order V2 and Get Positions).
_V2_SIDE_FOR_BUY = {"yes": "bid", "no": "ask"}


def _v2_yes_price_cents(side, price_cents):
    """The YES-leg price the V2 endpoint quotes buying ``side`` at ``price_cents`` in."""
    return price_cents if side == "yes" else 100 - price_cents


def _v2_order_result(data, side="yes"):
    """Map a Kalshi V2 order response into the internal result dict.

    V2 returns fixed-point strings (count "10.00", price "0.55") and reports
    fills via average_fill_price rather than a taker/maker cost split, so the
    contract cost in cents is derived here.  average_fill_price/average_fee_paid
    are only present when at least one contract filled.

    ``average_fill_price`` is the YES-leg price whichever side was bought, so a
    No fill's cost per contract is its complement. ``yes_price`` stays the YES
    price: it is what the exchange reports and what the fee is computed on.
    """
    fill_count = int(float(data.get("fill_count") or 0))
    avg_price_dollars = float(data.get("average_fill_price") or 0)
    avg_fee_dollars = float(data.get("average_fee_paid") or 0)
    paid_dollars = avg_price_dollars if side == "yes" else 1.0 - avg_price_dollars
    return {
        "order_id": data.get("order_id", ""),
        "fill_count": fill_count,
        "remaining_count": int(float(data.get("remaining_count") or 0)),
        "taker_fill_cost": round(fill_count * paid_dollars * 100) if fill_count else 0,
        "maker_fill_cost": 0,
        # What the exchange ACTUALLY charged, per contract, rather than what the
        # series multiplier says it should have. The two differ: KXMLBHRR is one
        # of the thirteen series below the default multiplier, and a fee
        # recomputed at the default is roughly double what was paid.
        "fee_cents": round(fill_count * avg_fee_dollars * 100),
        "yes_price": round(avg_price_dollars * 100),
    }


@dataclass(frozen=True)
class OrderFill:
    """The part of an order that actually executed.

    **An ``order_id`` means the exchange ACCEPTED the order, not that it
    filled**, and that distinction cost real money before this existed. Every
    order this repo places is immediate-or-cancel, so a response reading
    ``fill_count 0`` with ``remaining_count 0`` is an order that was cancelled
    unfilled -- and it carries an id exactly like one that traded. Two of four
    real prop orders on 2026-08-15 came back that way and were written to
    ``prop_wagers`` as filled positions at their requested size, 670c of
    exposure the account never had.

    Deliberately shaped like :class:`code.book_sizing.BookFill`, which is what
    the caller sized the order against: the pair are the intent and the outcome
    of the same trade, and reading one against the other should not require
    translating between two vocabularies.

    ``quantity`` and ``cost_cents`` are what the exchange executed, never what
    was asked for. A partial fill is a real position of a different size, and
    recording the request instead overstates the book by the unfilled part.
    """

    order_id: str
    quantity: int
    cost_cents: int
    fee_cents: int
    #: Average price actually paid per contract.
    vwap_cents: float
    #: How much of the request went unfilled, for the log.
    requested: int

    @property
    def is_filled(self):
        return self.quantity > 0

    @property
    def status(self):
        """``filled`` or ``partial`` -- the vocabulary the wager tables store."""
        return "filled" if self.quantity >= self.requested else "partial"


def order_fill(result, requested):
    """What ``result`` executed, or ``None`` if the order did not fill at all.

    ``None`` is the "no position was opened" answer, and every caller placing a
    real order must treat it as one: no row, no count, no exposure. Returning a
    zero-quantity object instead would let a caller that forgot to check write a
    position of size zero, which settles and books P/L just as wrongly.
    """
    if not result or not result.get("order_id"):
        return None
    quantity = int(result.get("fill_count") or 0)
    if quantity < 1:
        return None
    cost = int(result.get("taker_fill_cost") or 0) + int(
        result.get("maker_fill_cost") or 0)
    return OrderFill(order_id=result["order_id"], quantity=quantity,
                     cost_cents=cost,
                     fee_cents=int(result.get("fee_cents") or 0),
                     vwap_cents=cost / quantity if quantity else 0.0,
                     requested=int(requested))


#: Kalshi split its matching engine into shards ("Exchange Sharding" in the API
#: docs): 0 is the default, 1 combos, 2 crypto, 3 tennis and baseball. A
#: market's ``exchange_index`` is the authority. An order that omits it is
#: routed to shard 0, and a shard-3 market answers 404 -- which is what every
#: baseball order did on 2026-09-06, the first real order after the split.
#: Collateral is held PER SHARD, so a correctly routed order still needs a
#: balance on that shard (a target allocation or an intra-exchange transfer).
DEFAULT_EXCHANGE_INDEX = 0
_exchange_index_by_ticker = {}


def _api_error(status, data):
    """An ``ApiException`` whose reason carries Kalshi's own code and message.

    Kalshi nests them -- ``{"error": {"code", "message", "details"}}`` -- and a
    reason read from the top level logged a bare ``(404)`` for a mis-routed
    order, which took an afternoon to trace to the shard.
    """
    from kalshi_python_async.exceptions import ApiException
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    parts = [data.get("message") or error.get("message"), error.get("code"),
             error.get("details")]
    reason = " | ".join(str(p) for p in parts if p) or str(data)
    return ApiException(status=status, reason=reason)


async def _market_exchange_index(client, ticker):
    """The shard ``ticker`` trades on, read from the market itself.

    Cached per process -- a market never changes shard. Falls back to the
    default shard with a warning when the market cannot be read, so the order
    still goes out and the exchange, not this client, gives the final answer.
    """
    if ticker in _exchange_index_by_ticker:
        return _exchange_index_by_ticker[ticker]
    raw = await client.call_api("GET", KALSHI_API_BASE_URL + f"/markets/{ticker}")
    data = _read_json_body(await raw.read())
    index = (data.get("market") or {}).get("exchange_index")
    if raw.status >= 400 or index is None:
        logger.warning("Kalshi market %s has no readable exchange_index "
                       "(status %s); routing to shard %d", ticker, raw.status,
                       DEFAULT_EXCHANGE_INDEX)
        return DEFAULT_EXCHANGE_INDEX
    _exchange_index_by_ticker[ticker] = int(index)
    return int(index)


async def _create_order_safe(client, ticker, side, price_cents, count):
    """Submit an immediate-or-cancel buy order via Kalshi's V2 order endpoint.

    ``price_cents`` is what one contract of ``side`` costs; it is converted to
    the YES-leg price the endpoint quotes in. The order is routed to the shard
    the market lives on -- see :data:`DEFAULT_EXCHANGE_INDEX`.

    The legacy POST /portfolio/orders endpoint now returns 410 (deprecated);
    V2 lives at POST /portfolio/events/orders, quotes from the YES leg, and
    expects fixed-point dollar/count strings.  Uses call_api directly to skip
    the SDK's pydantic models, which still encode the deprecated V1 schema.
    """
    v2_side = _V2_SIDE_FOR_BUY.get(side)
    if v2_side is None:
        raise ValueError(
            f"Kalshi V2 orders buy 'yes' or 'no'; got side={side!r}"
        )
    body = {
        "ticker": ticker,
        "side": v2_side,
        "count": str(count),
        "price": f"{_v2_yes_price_cents(side, price_cents) / 100:.2f}",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "exchange_index": await _market_exchange_index(client, ticker),
    }
    raw = await client.call_api(
        "POST", KALSHI_API_BASE_URL + "/portfolio/events/orders", body=body,
    )
    data = json.loads(await raw.read())
    if raw.status >= 400:
        raise _api_error(raw.status, data)
    if not data.get("order_id"):
        return None
    logger.info(f"Kalshi V2 order response for {ticker}: {data}")
    return _v2_order_result(data, side)


def order_collateral_cents(ticker, price_cents, count):
    """What one IOC order locks up on its shard: cost plus the taker fee."""
    return int(count) * int(price_cents) + kalshi_trading_fee_cents(
        price_cents, count, ticker)


async def _shard_balances(client):
    """``{exchange_index: cents}`` from the balance breakdown.

    A pre-sharding response carries only ``balance``; it is read as shard 0 so
    the arithmetic below still holds.
    """
    raw = await client.call_api("GET", KALSHI_API_BASE_URL + "/portfolio/balance")
    data = _read_json_body(await raw.read())
    if raw.status >= 400:
        raise _api_error(raw.status, data)
    balances = {int(row["exchange_index"]): int(round(float(row["balance"]) * 100))
                for row in data.get("balance_breakdown") or []}
    return balances or {DEFAULT_EXCHANGE_INDEX: int(data.get("balance") or 0)}


async def _transfer_between_shards(client, source, destination, cents):
    """Move ``cents`` of event-contract collateral from one shard to another.

    Kalshi prices the body in centicents. Cross-shard transfers run in up to
    three non-atomic steps on the exchange side; a failure part-way leaves the
    money in the primary account, never lost, which is why the caller re-reads
    balances rather than trusting its own arithmetic afterwards.
    """
    body = {"source": "event_contract", "destination": "event_contract",
            "amount": int(cents) * 100,
            "source_exchange_shard": int(source),
            "destination_exchange_shard": int(destination)}
    raw = await client.call_api(
        "POST", KALSHI_API_BASE_URL + "/portfolio/intra_exchange_instance_transfer",
        body=body)
    data = _read_json_body(await raw.read())
    if raw.status >= 400:
        raise _api_error(raw.status, data)
    return data.get("transfer_id")


def shard_transfers(demand, balances):
    """``[(source, destination, cents)]`` that tops each shard up to its demand.

    Pure. ``demand`` is what the real orders about to be placed need per shard,
    ``balances`` what each shard holds. A shard short of its demand draws from
    whichever shard has the largest surplus over its OWN demand, so collateral
    a shard's own orders need is never lent out. When the account as a whole
    cannot cover the plan, the shortest shard is served first and the rest is
    left for the exchange to refuse -- an order the account cannot fund is not
    a position, and the log names the gap.
    """
    surplus = {shard: balances.get(shard, 0) - demand.get(shard, 0)
               for shard in set(balances) | set(demand)}
    plan = []
    for shard in sorted((s for s, v in surplus.items() if v < 0),
                        key=lambda s: surplus[s]):
        while surplus[shard] < 0:
            donors = [s for s, v in surplus.items() if v > 0]
            if not donors:
                break
            donor = max(donors, key=lambda s: surplus[s])
            move = min(-surplus[shard], surplus[donor])
            plan.append((donor, shard, move))
            surplus[donor] -= move
            surplus[shard] += move
    return plan


def fund_shards_for_orders(orders):
    """Move collateral so every shard holds what its real orders are about to spend.

    ``orders`` is an iterable of ``(ticker, cents)`` -- the real order plan, not
    the paper book. Each market's shard is read from the exchange, demand is
    summed per shard, and the shortfalls are transferred from the shard with
    the largest surplus (deposits land on shard 0). Returns the transfers made.
    Nothing moves when every shard already covers its demand.
    """
    orders = [(ticker, int(cents)) for ticker, cents in orders if cents > 0]
    if not orders or not _kalshi_available():
        return []

    async def _fund(client):
        demand = {}
        for ticker, cents in orders:
            shard = await _market_exchange_index(client, ticker)
            demand[shard] = demand.get(shard, 0) + cents
        balances = await _shard_balances(client)
        moved = []
        for source, destination, cents in shard_transfers(demand, balances):
            transfer_id = await _transfer_between_shards(client, source,
                                                         destination, cents)
            logger.info("Kalshi collateral: $%.2f shard %d -> shard %d for "
                        "$%.2f of real orders there (transfer %s)",
                        cents / 100, source, destination,
                        demand[destination] / 100, transfer_id)
            moved.append((source, destination, cents))
        for shard, need in demand.items():
            held = balances.get(shard, 0) + sum(c for _s, d, c in moved if d == shard)
            if held < need:
                logger.warning("Kalshi collateral: shard %d holds $%.2f against "
                               "$%.2f of real orders; the account cannot cover "
                               "the plan", shard, held / 100, need / 100)
        return moved

    return _kalshi_sync(_fund, default=[],
                        error_msg="Kalshi collateral transfer failed") or []


def place_order_sync(ticker, side, price_cents, count):
    """Place an IOC order buying ``count`` of ``side`` at ``price_cents`` each. Returns result dict or None.

    Funds the market's shard for this one order first. Callers with a whole
    slate to place fund it in one pass with :func:`fund_shards_for_orders`;
    this covers the single-order paths (moneyline, combos) and costs a
    balance read when the shard is already funded.
    """
    fund_shards_for_orders([(ticker, order_collateral_cents(ticker, price_cents,
                                                            count))])

    async def _place(client):
        result = await _create_order_safe(client, ticker, side, price_cents, count)
        if not result:
            logger.error(f"Kalshi returned empty response for {ticker}")
            return None
        return result

    return _kalshi_sync(_place, error_msg=f"Kalshi order failed for {ticker}")


def get_portfolio_positions_sync():
    """Return list of open positions from Kalshi."""
    async def _fetch(client):
        raw = await client.get_positions_without_preload_content(limit=200)
        data = json.loads(await raw.read())
        return data.get("market_positions") or data.get("positions") or []

    return _kalshi_sync(_fetch, default=[], error_msg="Failed to fetch Kalshi positions")


def get_open_orders_sync():
    """Return list of resting orders from Kalshi."""
    async def _fetch(client):
        raw = await client.get_orders_without_preload_content(status="resting", limit=200)
        data = json.loads(await raw.read())
        return data.get("orders") or []

    return _kalshi_sync(_fetch, default=[], error_msg="Failed to fetch Kalshi orders")


# ---------------------------------------------------------------------------
# Multivariate "combo"/parlay markets (real multi-leg placement). The rest of
# this client trades single-leg markets only; these wrap Kalshi's multivariate
# event collections against the same trade-api/v2 host + key-pair auth. A combo
# market is created from a set of leg (event_ticker, market_ticker) pairs and
# then trades as an ordinary single YES market, so placement reuses
# place_order_sync. Real placement stays dormant behind ENABLE_REAL_PARLAYS +
# the promotion gate + parlay_combos_available() (see code/parlay.py), so this
# untested-against-live path can never fire unintentionally.
# ---------------------------------------------------------------------------

_MULTIVARIATE_PATH = "/multivariate_event_collections"
_parlay_combos_available_cache = None  # None=unknown, then bool


def _read_json_body(text):
    """Parse a Kalshi raw body, tolerating the occasional concatenated response
    (as in get_transfers_sync)."""
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    start = text.find("{")
    return json.loads(text[start:]) if start >= 0 else {}


async def _list_multivariate_collections(client, status="open"):
    raw = await client.call_api(
        "GET", KALSHI_API_BASE_URL + _MULTIVARIATE_PATH + f"?status={status}",
    )
    if raw.status >= 400:
        return []
    data = _read_json_body(await raw.read())
    return data.get("multivariate_contracts") or []


def list_multivariate_collections_sync(status="open"):
    """Return Kalshi's multivariate (combo/parlay) event collections, or []."""
    return _kalshi_sync(lambda c: _list_multivariate_collections(c, status),
                        default=[], error_msg="Failed to list multivariate collections",
                        log_level="warning")


def parlay_combos_available():
    """True only if Kalshi currently exposes at least one multivariate collection
    (cached per process). The hard availability gate for real parlays — when
    False, every league stays paper-only regardless of the --parlay flag or
    promotion state."""
    global _parlay_combos_available_cache
    if _parlay_combos_available_cache is not None:
        return _parlay_combos_available_cache
    if not _kalshi_available():
        _parlay_combos_available_cache = False
        return False
    _parlay_combos_available_cache = bool(list_multivariate_collections_sync())
    return _parlay_combos_available_cache


def find_parlay_collection_sync(event_tickers):
    """Return a collection_ticker whose associated events cover ALL of the given
    leg event_tickers, or None. The collection is the combo market that can host
    the parlay's legs."""
    wanted = {t for t in event_tickers if t}
    if not wanted:
        return None
    for col in list_multivariate_collections_sync():
        assoc = set(col.get("associated_event_tickers") or [])
        if wanted <= assoc:
            return col.get("collection_ticker")
    return None


async def _create_combo_market(client, collection_ticker, legs):
    """The combo market for ``legs``, created or found (Kalshi is idempotent).

    Each leg carries its own ``side``: the collection endpoint takes ``yes`` or
    ``no`` per selected market and records both in the market's strike
    (verified live 2026-09-06, "Associated Market Sides: no,yes"), so a No rung
    is as expressible in a combo as in a single order. The payload's
    ``exchange_index`` is cached for the order that follows -- collections and
    their markets live on shard 1, not on any leg's shard.
    """
    selected = [
        {"market_ticker": lg["market_ticker"], "event_ticker": lg["event_ticker"],
         "side": lg.get("side", "yes")}
        for lg in legs
    ]
    body = {"selected_markets": selected, "with_market_payload": True}
    raw = await client.call_api(
        "POST", KALSHI_API_BASE_URL + _MULTIVARIATE_PATH + f"/{collection_ticker}", body=body,
    )
    data = _read_json_body(await raw.read())
    if raw.status >= 400:
        raise _api_error(raw.status, data)
    ticker = data.get("market_ticker")
    shard = (data.get("market") or {}).get("exchange_index")
    if ticker and shard is not None:
        _exchange_index_by_ticker[ticker] = int(shard)
    return ticker


def create_combo_market_sync(collection_ticker, legs):
    """Create (idempotent per Kalshi's "hit once before trading") the combo market
    for a set of legs and return its market_ticker, or None. Each leg needs
    market_ticker + event_ticker."""
    return _kalshi_sync(lambda c: _create_combo_market(c, collection_ticker, legs),
                        error_msg="Failed to create Kalshi combo market")


def place_combo_order_sync(collection_ticker, legs, yes_price_cents, count):
    """Create the combo market for `legs`, then place an IOC buy on it. Returns
    the order result dict (with the combo market_ticker attached) or None."""
    combo_ticker = create_combo_market_sync(collection_ticker, legs)
    if not combo_ticker:
        return None
    result = place_order_sync(combo_ticker, "yes", yes_price_cents, count)
    if result:
        result["combo_ticker"] = combo_ticker
    return result


