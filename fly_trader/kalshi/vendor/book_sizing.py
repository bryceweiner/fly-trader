# Vendored verbatim from better_bot code/book_sizing.py by tools/vendor_kalshi.py; sports sections removed. Do not edit.
"""Depth-aware position sizing: walk the order book instead of assuming a full
fill at the best ask.

A level is acceptable exactly while it is still +EV net of fee, so the
threshold falls out of break-even. Pure functions over a levels list; the
Kalshi fetch lives in ``kalshi_client``.
"""
import logging
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

from .kalshi_client import effective_price_cents, kalshi_fee_rate_cents
from .venue_register import net_odds_at

logger = logging.getLogger(__name__)

#: Kalshi prices are whole cents in 1..99.
MIN_PRICE_CENTS = 1
MAX_PRICE_CENTS = 99

@dataclass(frozen=True)
class BookLevel:
    """One resting price level."""

    price_cents: int
    quantity: int

    def __post_init__(self):
        if not MIN_PRICE_CENTS <= self.price_cents <= MAX_PRICE_CENTS:
            raise ValueError(
                f"price {self.price_cents} outside Kalshi's 1-99c range")
        if self.quantity < 0:
            raise ValueError(f"negative quantity {self.quantity}")


@dataclass(frozen=True)
class OrderBook:
    """Both sides of a Kalshi book, as resting bids.

    Kalshi publishes two bid ladders and no ask ladder: a No bid at ``q`` is a
    Yes offer at ``100 - q``. :meth:`ask_ladder` is the only supported way to
    ask what a side can be bought at, and :func:`validate_against_quote` catches
    an inverted book against the quoted ask.
    """

    yes_bids: Tuple[BookLevel, ...] = ()
    no_bids: Tuple[BookLevel, ...] = ()
    ticker: str = ""

    def ask_ladder(self, side):
        """Levels at which ``side`` can be BOUGHT, cheapest first.

        Buying Yes consumes No bids and vice versa.
        """
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        opposing = self.no_bids if side == "yes" else self.yes_bids
        levels = [BookLevel(100 - lvl.price_cents, lvl.quantity)
                  for lvl in opposing if lvl.quantity > 0]
        return tuple(sorted(levels, key=lambda l: l.price_cents))

    def best_ask(self, side):
        ladder = self.ask_ladder(side)
        return ladder[0].price_cents if ladder else None

    def depth(self, side):
        return sum(lvl.quantity for lvl in self.ask_ladder(side))

    @property
    def is_empty(self):
        return not self.yes_bids and not self.no_bids


def validate_against_quote(book, side, quoted_ask_cents, tolerance=1):
    """True when the book's best ask agrees with the separately quoted ask, ``None``
    when either is missing. A mismatch means inverted sides or a stale book.
    """
    derived = book.best_ask(side)
    if derived is None or quoted_ask_cents is None:
        return None
    return abs(derived - quoted_ask_cents) <= tolerance


@dataclass(frozen=True)
class BookFill:
    """The result of walking a book: how much to take, and at what real price."""

    quantity: int
    #: Fee-inclusive volume-weighted price per contract.
    vwap_cents: float
    cost_cents: int
    fee_cents: int
    stopped_reason: str
    levels_taken: int = 0
    best_ask_cents: int = None
    #: Worst level price consumed -- the limit to submit an IOC order at, so the
    #: order can reach the depth we sized against without paying beyond it.
    limit_price_cents: int = None
    #: The book's ticker, carried so fee-inclusive figures charge that market's
    #: own series multiplier rather than the exchange default.
    ticker: str = ""

    @property
    def is_fillable(self):
        return self.quantity > 0

    @property
    def slippage_cents(self):
        """Fee-inclusive VWAP minus the fee-inclusive best ask.

        The measured cost of depth, per contract. Logged per wager so the
        slippage figure can be re-measured against fee-adjusted fills rather
        than inherited from a study that predates the fee correction.
        """
        if self.best_ask_cents is None or not self.is_fillable:
            return 0.0
        return self.vwap_cents - effective_price_cents(self.best_ask_cents,
                                                        self.ticker)

    @property
    def expected_profit_cents(self):
        """Profit if the contract settles in our favour."""
        if not self.is_fillable:
            return 0
        return int(self.quantity * (100.0 - self.vwap_cents))


NO_FILL = BookFill(quantity=0, vwap_cents=0.0, cost_cents=0, fee_cents=0,
                   stopped_reason="no_liquidity")


def kelly_fraction_at(p, vwap_cents):
    """Raw Kelly fraction at an already fee-inclusive fill price; zero without
    edge. Distinct from ``kelly_sizing._kelly_bet_fraction``, which adds the
    spread and fee itself.
    """
    if vwap_cents <= 0 or vwap_cents >= 100:
        return 0.0
    b = net_odds_at(vwap_cents)
    return max((p * b - (1.0 - p)) / b, 0.0)


def kelly_contracts(p, vwap_cents, bankroll_cents, kelly_fraction):
    """Fractional-Kelly position size, in contracts, at a given fill price.

    ``vwap_cents`` is already fee-inclusive, so this is Kelly on the true cost
    basis rather than on a quoted price we would not actually pay.
    """
    raw = kelly_fraction_at(p, vwap_cents)
    if raw <= 0:
        return 0.0
    return raw * kelly_fraction * bankroll_cents / vwap_cents


def _contracts_at_level(taken, cost, level_price_eff, level_qty, p,
                        bankroll_cents, kelly_fraction):
    """How many more contracts to take at one level before Kelly is satisfied.
    ``taken + k - kelly_target(k)`` is increasing in ``k``, so a bisection finds
    the crossing.
    """
    def surplus(k):
        vwap = (cost + k * level_price_eff) / (taken + k) if (taken + k) else \
            level_price_eff
        return (taken + k) - kelly_contracts(p, vwap, bankroll_cents,
                                             kelly_fraction)

    if surplus(level_qty) <= 0:
        return level_qty                    # the whole level fits inside Kelly
    if surplus(0) >= 0:
        return 0                            # already at or past the target

    lo, hi = 0, level_qty
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if surplus(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return lo


def fill_budget(book, side, p, budget_cents, *, min_payout_cents,
                max_contracts=None):
    """Spend a pre-decided allocation against real depth.

    Unlike :func:`walk_book`, Kelly has already run upstream. This answers how
    much of the allocation can be spent at a price still +EV, and what it will
    cost. Levels past break-even are refused even with budget left, and a fill
    whose profit is under ``min_payout_cents`` is refused whole.
    """
    ladder = book.ask_ladder(side)
    if not ladder:
        return NO_FILL
    if budget_cents <= 0:
        # Zero allocation is the allocator's answer, not an exchange problem.
        return replace(NO_FILL, stopped_reason="no_allocation",
                       ticker=book.ticker)

    best_ask = ladder[0].price_cents
    taken, cost, fee, levels_taken = 0, 0.0, 0.0, 0
    limit_price = None
    reason = "depth_exhausted"

    for level in ladder:
        price_eff = effective_price_cents(level.price_cents, book.ticker)
        if price_eff >= 100.0 * p:
            reason = "price_exceeds_edge"
            break

        affordable = int((budget_cents - cost) // price_eff)
        if affordable <= 0:
            reason = "budget_spent"
            break

        available = min(level.quantity, affordable)
        if max_contracts is not None:
            available = min(available, max_contracts - taken)
            if available <= 0:
                reason = "max_contracts"
                break

        taken += available
        cost += available * price_eff
        fee += available * kalshi_fee_rate_cents(level.price_cents, book.ticker)
        limit_price = level.price_cents
        levels_taken += 1

        if available < level.quantity:
            reason = "budget_spent" if max_contracts is None else "max_contracts"
            break

    if taken <= 0:
        # Carry the real stop reason: a price the model declines is not an
        # empty market.
        return replace(NO_FILL, stopped_reason=reason, best_ask_cents=best_ask,
                       ticker=book.ticker)

    vwap = cost / taken
    fill = BookFill(
        quantity=int(taken), vwap_cents=vwap, cost_cents=int(round(cost)),
        fee_cents=int(round(fee)), stopped_reason=reason,
        levels_taken=levels_taken, best_ask_cents=best_ask,
        limit_price_cents=limit_price, ticker=book.ticker)

    if fill.expected_profit_cents < min_payout_cents:
        logger.info(
            f"Skipping {book.ticker or side}: fillable size {fill.quantity} at "
            f"{vwap:.2f}c yields ${fill.expected_profit_cents / 100:.2f} profit, "
            f"below the ${min_payout_cents / 100:.2f} minimum")
        return BookFill(quantity=0, vwap_cents=vwap, cost_cents=0, fee_cents=0,
                        stopped_reason="below_min_payout",
                        best_ask_cents=best_ask, ticker=book.ticker)
    return fill


def kelly_allocation_cents(p, price_cents, bankroll_cents, kelly_fraction):
    """What one prop position may spend: fractional Kelly at the fee-inclusive
    best ask, capped by the prop sub-budget's concentration ceiling. The
    tighter bound binds, as in ``prop_betting._position_caps``.
    """
    from .config import PROP_BUDGET_FRACTION, PROP_MAX_POSITION_FRACTION

    raw = kelly_fraction_at(p, price_cents)
    if raw <= 0:
        return 0
    ceiling = PROP_BUDGET_FRACTION * PROP_MAX_POSITION_FRACTION
    return int(min(ceiling, raw * kelly_fraction) * bankroll_cents)


def walk_book(book, side, p, bankroll_cents, *, kelly_fraction,
              min_payout_cents, max_contracts=None):
    """Size a position against real depth, solving the Kelly size and the fill
    price together: ascend the ladder while each level is +EV net of fee and
    stop at whichever binds first -- Kelly, depth, or break-even. Returns
    :data:`NO_FILL` when nothing is fillable at a +EV price or the fill cannot
    clear ``min_payout_cents``.
    """
    ladder = book.ask_ladder(side)
    if not ladder:
        return NO_FILL

    best_ask = ladder[0].price_cents
    taken = 0
    cost = 0.0
    fee = 0.0
    levels_taken = 0
    reason = "depth_exhausted"

    for level in ladder:
        price_eff = effective_price_cents(level.price_cents, book.ticker)
        if price_eff >= 100.0 * p:
            reason = "price_exceeds_edge"
            break

        available = level.quantity
        if max_contracts is not None:
            available = min(available, max_contracts - taken)
            if available <= 0:
                reason = "max_contracts"
                break

        take = _contracts_at_level(taken, cost, price_eff, available, p,
                                   bankroll_cents, kelly_fraction)
        if take <= 0:
            reason = "kelly_satisfied"
            break

        taken += take
        cost += take * price_eff
        # The fee is quadratic in price, so each rung pays its own rate.
        fee += take * kalshi_fee_rate_cents(level.price_cents, book.ticker)
        levels_taken += 1

        if take < available:
            reason = "kelly_satisfied"
            break

    if taken <= 0:
        # Carry the real stop reason: a price the model declines is not an
        # empty market.
        return replace(NO_FILL, stopped_reason=reason, best_ask_cents=best_ask,
                       ticker=book.ticker)

    vwap = cost / taken
    fill = BookFill(
        quantity=int(taken), vwap_cents=vwap, cost_cents=int(round(cost)),
        fee_cents=int(round(fee)), stopped_reason=reason,
        levels_taken=levels_taken, best_ask_cents=best_ask,
        ticker=book.ticker)

    if fill.expected_profit_cents < min_payout_cents:
        logger.info(
            f"Skipping {book.ticker or side}: fillable size {fill.quantity} at "
            f"{vwap:.2f}c yields ${fill.expected_profit_cents / 100:.2f} profit, "
            f"below the ${min_payout_cents / 100:.2f} minimum")
        return BookFill(quantity=0, vwap_cents=vwap, cost_cents=0, fee_cents=0,
                        stopped_reason="below_min_payout",
                        best_ask_cents=best_ask, ticker=book.ticker)
    return fill


def parse_kalshi_orderbook(payload, ticker=""):
    """Build an :class:`OrderBook` from Kalshi's ``orderbook_fp`` payload. Levels
    arrive as ``[price_dollars, count]`` string pairs; ``"0.47"`` is 47c.
    """
    book = (payload or {}).get("orderbook_fp") or (payload or {}).get("orderbook") or {}

    def levels(key_dollars, key_cents):
        """Read one side, converting by which key it came from.

        Not by sniffing the value: 1 could be one cent or one dollar, and
        guessing wrong is a silent 100x. The key name is the authority.
        """
        raw, in_dollars = book.get(key_dollars), True
        if raw is None:
            raw, in_dollars = book.get(key_cents) or [], False

        out = []
        for entry in raw:
            try:
                price, count = entry[0], entry[1]
                cents = int(round(float(price) * 100)) if in_dollars else int(price)
                quantity = int(float(count))
            except (TypeError, ValueError, IndexError):
                logger.warning(f"unparseable orderbook level {entry!r} for {ticker}")
                continue
            if quantity <= 0 or not MIN_PRICE_CENTS <= cents <= MAX_PRICE_CENTS:
                continue
            out.append(BookLevel(cents, quantity))
        return tuple(out)

    return OrderBook(yes_bids=levels("yes_dollars", "yes"),
                     no_bids=levels("no_dollars", "no"), ticker=ticker)
