# Vendored verbatim from better_bot code/venue_kalshi.py by tools/vendor_kalshi.py; sports sections removed. Do not edit.
"""Kalshi as a :class:`code.venue_register.Venue`.

A thin adapter, deliberately. ``code.kalshi_client`` is 1,931 lines with a
1,975-line test file pointed at it, and none of that moves: the seam is this
class, not a relocation. Those tests staying green *unchanged* is what proves
the fee model was not perturbed while the abstraction was built around it.

Everything here delegates. The one piece of real logic is
:meth:`KalshiVenue.simulate`, which reproduces the paper arm's arithmetic
exactly -- see its docstring for why that matters more than it looks.
"""

import logging

from . import kalshi_client as kc
from .book_sizing import (MAX_PRICE_CENTS, MIN_PRICE_CENTS, fill_budget,
                          parse_kalshi_orderbook, validate_against_quote)
from .config import ENABLE_VENUE_KALSHI
from .venue_register import Fill, MarketRef, NO_QUOTE, Quote, TopOfBook

logger = logging.getLogger(__name__)

#: A Kalshi contract settles at one dollar. This is the venue's payout algebra
#: and the only place the literal belongs.
CONTRACT_PAYOUT_CENTS = 100

#: What a half-spread can credibly be, in cents. Kalshi quotes whole cents, so a
#: one-cent book puts the mid at x.5 and 0.5c is the tightest a real two-sided
#: quote can show; a gap past the ceiling is a one-sided or stale row rather than
#: a spread, and taking it at face value would price that market off the slate.
SPREAD_BAND = (0.0, 5.0)

#: Used only when there is neither a quote to read nor a settled record to
#: measure -- a cold start. Deliberately the mean of the 26 fills the frozen
#: constant was provenanced from, so an unmeasured venue behaves as this one did
#: before any of this existed.
DECLARED_HALF_SPREAD_CENTS = 0.52

#: Below this many settled fills the realised mean is noise, not a measurement.
MIN_SPREAD_SAMPLE = 30

_REALISED_HALF_SPREAD = None
_SPREAD_BY_MARKET = {}


def _in_band(value):
    return SPREAD_BAND[0] <= float(value) <= SPREAD_BAND[1]


def reset_spread_cache():
    """Drop the memoised spreads. Tests, and after an attach or settlement pass."""
    global _REALISED_HALF_SPREAD
    _REALISED_HALF_SPREAD = None
    _SPREAD_BY_MARKET.clear()


class KalshiVenue:
    """The CFTC-designated contract market this system has always traded."""

    name = "kalshi"

    # -- activation ---------------------------------------------------------

    def is_available(self):
        """Credentials load, the calibration gate is on, and the flag is set.

        ``_kalshi_available`` already answers the first two; the flag is the
        venue-level switch every venue carries, so "which venues are active" is
        one question with one shape rather than a special case per venue.
        """
        return bool(ENABLE_VENUE_KALSHI) and kc._kalshi_available()

    # -- identity -----------------------------------------------------------

    def market_group_of(self, market_id):
        return kc.series_ticker_of(market_id)


    # -- pricing: pure, no network ------------------------------------------

    def fee_rate_cents(self, price_cents, market_id=None):
        return kc.kalshi_fee_rate_cents(price_cents, market_id)

    def effective_price_cents(self, price_cents, market_id=None):
        return kc.effective_price_cents(price_cents, market_id)

    def order_fee_cents(self, price_cents, quantity, market_id=None):
        return kc.kalshi_trading_fee_cents(price_cents, quantity, market_id)

    def order_cost_cents(self, price_cents, quantity, market_id=None):
        return kc.order_collateral_cents(market_id, price_cents, quantity)

    def payout_cents(self, quantity, market_id=None):
        """A winning position returns a dollar a contract."""
        return int(quantity) * CONTRACT_PAYOUT_CENTS

    def half_spread_cents(self, market_id=None, top=None):
        """What crossing this market costs, per contract, in cents.

        Three tiers, in order, and the order is the point:

        1. **Observed** -- ``ask - mid`` for this market. Live if the caller
           already holds a quote, otherwise from the attach pass's stored pair.
           This is per-market and needs no fill history.
        2. **Realised** -- the mean of ``price - mid`` over settled fills at
           this venue. What a caller without a market gets, because there is no
           book to read.
        3. **Declared** -- a floor, when neither exists.

        Read rather than fetched: the stored pair was written by the attach pass,
        so sizing a slate twice sizes it identically. An observed value outside
        :data:`SPREAD_BAND` is refused and falls to tier 2 -- a one-sided or
        stale cache row can show an arbitrary gap, and trusting it would silently
        price a market out of the slate.
        """
        if top is not None:
            observed = top.half_spread_cents
            if observed is not None and _in_band(observed):
                return observed
        if market_id:
            observed = self._cached_half_spread(market_id)
            if observed is not None:
                if _in_band(observed):
                    return observed
                logger.warning(
                    "%s: cached half-spread %.2fc for %s outside %s -- using the "
                    "realised mean instead", self.name, observed, market_id,
                    SPREAD_BAND)
        return self._realised_half_spread()

    def _cached_half_spread(self, market_id):
        """``ask - mid`` from the attach pass's stored quote, or ``None``.

        Memoised per market: sizing asks this once per candidate, and re-reading
        halfway through a slate could price two candidates on the same card
        against two different snapshots.
        """
        key = (self.name, str(market_id))
        if key in _SPREAD_BY_MARKET:
            return _SPREAD_BY_MARKET[key]
        from .database import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT yes_ask_cents, market_prob FROM venue_market_cache "
                "WHERE venue = ? AND market_id = ?",
                (self.name, str(market_id))).fetchone()
        except Exception as exc:                                # noqa: BLE001
            logger.warning("%s: could not read the quote cache (%s)", self.name, exc)
            return None
        finally:
            conn.close()
        observed = None
        if row and row["yes_ask_cents"] is not None and row["market_prob"] is not None:
            observed = float(row["yes_ask_cents"]) - float(row["market_prob"])
        _SPREAD_BY_MARKET[key] = observed
        return observed

    def _realised_half_spread(self):
        """The mean gap between what was paid and the mid it was sized on.

        Memoised per process: it is a property of a settled record, not of the
        moment it is asked for, and re-reading it mid-slate would let two
        candidates on one card be sized against different constants.
        """
        global _REALISED_HALF_SPREAD
        if _REALISED_HALF_SPREAD is None:
            from .kelly_sizing import measure_half_spread_cents
            try:
                mean, n = measure_half_spread_cents()
            except Exception as exc:                            # noqa: BLE001
                logger.warning("%s: could not measure the realised spread (%s)",
                               self.name, exc)
                mean, n = None, 0
            if mean is None or n < MIN_SPREAD_SAMPLE or not _in_band(mean):
                logger.info("%s: realised spread unusable (n=%s, mean=%s); "
                            "using the declared floor %.2fc",
                            self.name, n, mean, DECLARED_HALF_SPREAD_CENTS)
                mean = DECLARED_HALF_SPREAD_CENTS
            _REALISED_HALF_SPREAD = float(mean)
        return _REALISED_HALF_SPREAD

    def devig(self, yes_ask_cents, no_ask_cents):
        """The market's BELIEF for the yes side, overround removed.

        Proportional normalisation, which is the standard two-sided treatment
        for a central limit order book: with both sides quoted, the asks sum to
        more than 100 by the overround, and each side's share of that sum is its
        implied probability once the overround is divided out.

        A venue operation rather than a general one: this is right for a
        two-sided book, but a parimutuel pool's de-vig is its takeout rate and
        an AMM's is its curve mid. ``None`` when only one side is quoted, since
        a one-sided book carries no overround to remove.
        """
        if yes_ask_cents is None or no_ask_cents is None:
            return None
        total = float(yes_ask_cents) + float(no_ask_cents)
        if total <= 0:
            return None
        return float(yes_ask_cents) / total

    def price_bounds(self):
        """Kalshi quotes whole cents in 1..99; outside that is not a contract."""
        return MIN_PRICE_CENTS, MAX_PRICE_CENTS

    # -- quoting and execution: network -------------------------------------


    def top_of_book(self, market_id, side="yes"):
        market = kc.get_market_sync(market_id)
        if not market:
            return None
        quote = kc.market_quote(market) or {}
        return TopOfBook(
            market=MarketRef(venue=self.name, market_id=market_id,
                             group_id=self.market_group_of(market_id)),
            side=side,
            mid_cents=quote.get("market_prob"),
            ask_cents=kc.market_ask_cents(market, side),
            volume=quote.get("volume"))

    def quote_for_size(self, market_id, side, *, budget_cents, p_win,
                       min_payout_cents, max_quantity=None,
                       quoted_ask_cents=None):
        """Walk the book and price the size actually reachable.

        One call covers fetch, parse, sanity-check and walk, because those are
        one question -- "what would this cost here" -- and splitting them across
        the interface would force every caller to know that this venue answers
        it with a ladder.

        Two refusals are deliberately distinguishable by ``stopped_reason``, and
        the caller must keep treating them differently: ``no_book`` and
        ``stale_book`` mean this venue could not be asked, so the caller falls
        back to top-of-book rather than dropping the position, while anything
        else means it WAS asked and offered nothing worth taking.
        """
        payload = kc.get_orderbook_sync(market_id)
        if payload is None:
            logger.info("no orderbook for %s; caller sizes off top-of-book",
                        market_id)
            return Quote(market=self._ref(market_id), side=side, quantity=0,
                         cost_cents=0.0, payout_cents=0.0,
                         stopped_reason="no_book")
        book = parse_kalshi_orderbook(payload, ticker=market_id)
        if validate_against_quote(book, side, quoted_ask_cents) is False:
            # Inverted sides or a stale snapshot: fall back rather than trust it.
            logger.warning(
                "orderbook for %s implies best ask %sc against quoted %sc -- "
                "ignoring depth", market_id, book.best_ask(side),
                quoted_ask_cents)
            return Quote(market=self._ref(market_id), side=side, quantity=0,
                         cost_cents=0.0, payout_cents=0.0,
                         stopped_reason="stale_book")
        fill = fill_budget(book, side, p_win, budget_cents,
                           min_payout_cents=min_payout_cents,
                           max_contracts=max_quantity)
        if fill.is_fillable:
            logger.info(
                "depth sizing %s: %d contracts across %d level(s), vwap %.2fc "
                "vs best %sc, slippage %+.2fc/contract, cost $%.2f of $%.2f "
                "allocated (%s)", market_id, fill.quantity, fill.levels_taken,
                fill.vwap_cents, fill.best_ask_cents, fill.slippage_cents,
                fill.cost_cents / 100, budget_cents / 100, fill.stopped_reason)
        return self._quote_from_fill(market_id, side, fill)

    def place(self, quote):
        """Submit ``quote`` as an IOC order; ``None`` when nothing filled."""
        if not quote or not quote.is_fillable:
            return None
        market_id = quote.market.market_id
        price = quote.limit_price_cents or quote.best_price_cents
        result = kc.place_order_sync(market_id, quote.side, price, quote.quantity)
        if not result or not result.get("order_id"):
            logger.error("order rejected by %s for %s at %sc",
                         self.name, market_id, price)
            return None
        executed = kc.order_fill(result, quote.quantity)
        if executed is None:
            # Accepted and filled nothing. An IOC that crossed no resting size
            # comes back carrying an order id exactly like one that traded, so
            # this is logged apart from a rejection: they are the same outcome
            # for the caller and very different things to debug.
            logger.warning("%s accepted then filled 0 of %d for %s at %sc",
                           self.name, quote.quantity, market_id, price)
            return None
        return Fill(order_id=executed.order_id, market=quote.market,
                    quantity=executed.quantity,
                    cost_cents=executed.cost_cents,
                    fee_cents=executed.fee_cents,
                    payout_cents=self.payout_cents(executed.quantity,
                                                   market_id),
                    requested=executed.requested)

    def simulate(self, quote):
        """The paper fill for ``quote``, priced exactly as the paper book is.

        **Deliberately the worst level consumed, not the volume-weighted price.**
        The real arm books the VWAP the exchange reported; paper charges
        ``limit_price_cents``, which is the most expensive rung the walk
        touched. That is conservative by construction, and it is load-bearing:
        paper P/L feeds ``league_paper_gate``'s strategy comparison, which
        decides which leagues trade real money. Pricing paper at the VWAP would
        make it cheaper, raise its measured ROI, and move that verdict -- so any
        change here is a change to capital routing and belongs in its own commit
        with its own measurement.
        """
        if not quote or not quote.is_fillable:
            return None
        market_id = quote.market.market_id
        price = quote.limit_price_cents or quote.best_price_cents
        fee = self.order_fee_cents(price, quote.quantity, market_id)
        return Fill(order_id="", market=quote.market, quantity=quote.quantity,
                    cost_cents=int(quote.quantity) * int(price),
                    fee_cents=fee,
                    payout_cents=self.payout_cents(quote.quantity, market_id),
                    requested=quote.quantity)

    def balance_cents(self):
        balance = kc.get_balance_sync()
        return balance.get("balance") if balance else None

    # -- internals ----------------------------------------------------------

    def _ref(self, market_id):
        return MarketRef(venue=self.name, market_id=market_id,
                         group_id=self.market_group_of(market_id))

    def _quote_from_fill(self, market_id, side, fill):
        """Carry a :class:`code.book_sizing.BookFill` into venue vocabulary."""
        if fill is None or not fill.is_fillable:
            return Quote(market=self._ref(market_id), side=side, quantity=0,
                         cost_cents=0.0, payout_cents=0.0,
                         stopped_reason=getattr(fill, "stopped_reason",
                                                NO_QUOTE.stopped_reason))
        return Quote(market=self._ref(market_id), side=side,
                     quantity=fill.quantity,
                     cost_cents=float(fill.cost_cents),
                     payout_cents=float(self.payout_cents(fill.quantity,
                                                          market_id)),
                     fee_cents=float(fill.fee_cents),
                     best_price_cents=fill.best_ask_cents,
                     limit_price_cents=fill.limit_price_cents,
                     stopped_reason=fill.stopped_reason,
                     levels_taken=fill.levels_taken)
