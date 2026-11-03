# Vendored verbatim from better_bot code/venue_register.py by tools/vendor_kalshi.py; sports sections removed. Do not edit.
"""What a trading venue is, and which ones this process may reach.

One protocol, one registry, and the four value types every caller passes around:
a market's identity, its top of book, what buying a size there would cost, and
what an order actually executed. The Kalshi implementation lives in
``code.venue_kalshi``; routing between venues lives in ``code.venue_routing``.

The reason this exists is that "the fee" and "the payout" are venue facts and
were written as global ones. ``effective_price_cents`` had twenty-five direct
importers and ``quantity * 100`` was spelled out at fifteen sites, so a second
venue could not be added without editing arithmetic in fifteen files. Here they
are asked of the venue that quoted the price.

Named ``venue_register`` rather than ``venue`` on purpose: ``code.venue_geo``
already exists and ``teams.venue_name`` is a stadium. The word is taken.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketRef:
    """One tradeable contract at one venue.

    ``market_id`` is whatever that venue calls a market -- a Kalshi ticker, an
    exchange's market id, an on-chain condition id -- and is opaque above this
    layer. ``group_id`` is what the venue's FEE SCHEDULE keys on (Kalshi's
    series), which is why it is carried separately rather than parsed out of the
    market id by every caller that needs a fee.
    """

    venue: str
    market_id: str
    group_id: Optional[str] = None
    event_id: Optional[str] = None
    title: str = ""


@dataclass(frozen=True)
class TopOfBook:
    """The best quote on one side, and the spread implied by it.

    ``mid_cents`` and ``ask_cents`` are both carried because sizing and
    execution ask different questions of the same quote: sizing is done on the
    mid plus a modelled half-spread, while the order is submitted at the ask.
    """

    market: MarketRef
    side: str
    mid_cents: Optional[float] = None
    ask_cents: Optional[int] = None
    volume: Optional[int] = None

    @property
    def half_spread_cents(self):
        """``ask - mid``, or ``None`` when either is missing.

        The observed cost of crossing, per contract, at this market right now --
        as opposed to a constant measured once across a whole book of fills.
        """
        if self.ask_cents is None or self.mid_cents is None:
            return None
        return float(self.ask_cents) - float(self.mid_cents)


@dataclass(frozen=True)
class Quote:
    """What buying ``quantity`` at one venue would cost, and what it returns.

    This is the primitive the whole abstraction rests on, because it is the only
    shape that fits every venue we might add. A central limit order book answers
    it by walking its ask ladder; a fixed-odds book by reading one price and its
    posted maximum; an automated market maker by walking its own curve, where the
    price genuinely depends on the size asked for; a parimutuel pool cannot
    answer it before the round closes and says so via ``stopped_reason``.

    ``cost_cents`` is ALL-IN for the whole quantity -- fee included -- and
    ``payout_cents`` is what the position returns if it wins. Every existing
    formula in the codebase is expressible in those two numbers, which is what
    makes the venues interchangeable without rewriting the arithmetic.
    """

    market: MarketRef
    side: str
    quantity: int
    cost_cents: float
    payout_cents: float
    fee_cents: float = 0.0
    best_price_cents: Optional[int] = None
    #: Worst level consumed: the limit an IOC order is submitted at, so it can
    #: reach the depth it was sized against without paying beyond it.
    limit_price_cents: Optional[int] = None
    stopped_reason: str = ""
    levels_taken: int = 0

    @property
    def is_fillable(self):
        return self.quantity > 0

    @property
    def effective_price_cents(self):
        """Cost per 100c of payout: the venue-neutral all-in basis.

        This is the bridge that lets the existing formulas stand unchanged. On a
        binary $1 contract it equals price + fee, because payout is
        ``quantity * 100``; on a commission book it is the stake per 100c of
        return once commission is taken. ``p - effective_price/100`` is the edge
        at either.
        """
        if not self.payout_cents:
            return 0.0
        return 100.0 * self.cost_cents / self.payout_cents

    @property
    def net_odds(self):
        """``b`` in Kelly: profit per unit staked if the position wins.

        Written out longhand as ``(100 - p_eff) / p_eff`` at three sites before
        this existed, which is exactly where the literal 100 was wrong for any
        venue that does not settle at a dollar.
        """
        if not self.cost_cents:
            return 0.0
        return (self.payout_cents - self.cost_cents) / self.cost_cents


@dataclass(frozen=True)
class Fill:
    """What an order actually executed -- never what was requested.

    **An order id means the venue ACCEPTED the order, not that it filled.** Every
    order this repo places is immediate-or-cancel, so a response reading zero
    fills carries an id exactly like one that traded; treating that as a position
    booked exposure the account never had. A caller that gets ``None`` from
    :meth:`Venue.place` has no position, and must write no row.
    """

    order_id: str
    market: MarketRef
    quantity: int
    cost_cents: float
    fee_cents: float
    payout_cents: float
    requested: int = 0

    @property
    def is_filled(self):
        return self.quantity >= self.requested > 0

    @property
    def status(self):
        return "filled" if self.is_filled else "partial"

    @property
    def vwap_cents(self):
        return self.cost_cents / self.quantity if self.quantity else 0.0


def net_odds_at(effective_price_cents):
    """``b`` in Kelly at an all-in basis: profit per unit staked on a win.

    **Venue-independent, and that is the finding.** It was written longhand at
    three sites -- ``book_sizing``, ``kelly_sizing`` and ``ev_ledger`` -- and the
    literal 100 in each looks like Kalshi's dollar settlement. It is not. An
    effective price is defined as cost per 100c of PAYOUT, so the 100 is the
    normalisation constant of the basis itself, and the formula is already right
    at a commission book or an AMM. What was duplicated is the expression, not a
    venue assumption; the venue-dependence lives entirely upstream, in how the
    basis was computed.
    """
    basis = float(effective_price_cents)
    if basis <= 0:
        return 0.0
    return (100.0 - basis) / basis


#: A quote that cannot be filled. ``stopped_reason`` says why, because "no
#: depth" and "this venue cannot price a size before the round closes" are
#: different answers and the router treats them differently.
NO_QUOTE = Quote(market=None, side="", quantity=0, cost_cents=0.0,
                 payout_cents=0.0, stopped_reason="no_liquidity")


@runtime_checkable
class Venue(Protocol):
    """A place this system can price and take a position.

    Split into three bands, and the split is load-bearing:

    * **identity** -- what a market is called here, and which of ours it is.
    * **pricing** -- PURE. No network, no lazy fetch, no database. Sizing runs
      through these on every candidate on the slate, and a fetch inside the fee
      arithmetic would make a slate's allocation depend on whether the venue was
      reachable at the moment it was priced. Warm caches explicitly instead.
    * **quoting and execution** -- network. Called once per candidate, not once
      per arithmetic step.
    """

    name: str

    # -- activation ---------------------------------------------------------
    def is_available(self) -> bool: ...

    # -- identity -----------------------------------------------------------
    def market_group_of(self, market_id): ...
    def series_for_league(self, league_id): ...
    def series_for_slug(self, league_slug): ...
    def tradable_leagues(self): ...
    def market_for_event(self, league_slug, home_team, away_team, *,
                         predicted_winner=None, event_date=None,
                         home_abbrev=None, away_abbrev=None): ...

    # -- pricing: pure ------------------------------------------------------
    def fee_rate_cents(self, price_cents, market_id=None): ...
    def effective_price_cents(self, price_cents, market_id=None): ...
    def order_fee_cents(self, price_cents, quantity, market_id=None): ...
    def order_cost_cents(self, price_cents, quantity, market_id=None): ...
    def payout_cents(self, quantity, market_id=None): ...
    def half_spread_cents(self, market_id=None, top=None): ...
    def devig(self, yes_ask_cents, no_ask_cents): ...
    def price_bounds(self): ...

    # -- quoting and execution: network -------------------------------------
    def cached_odds(self, event_id, predicted_winner, home_team, away_team, *,
                    league_slug=None, home_abbrev=None, away_abbrev=None,
                    conn=None): ...
    def top_of_book(self, market_id, side="yes"): ...
    def quote_for_size(self, market_id, side, *, budget_cents, p_win,
                       min_payout_cents, max_quantity=None): ...
    def place(self, quote): ...
    def simulate(self, quote): ...
    def balance_cents(self): ...


# ---------------------------------------------------------------------------
# Registry
#
# Lazy, and mirroring code.gate_register: the module that owns the mechanism
# imports the module that declares the content, not the other way round, so a
# venue implementation can import these types without a cycle.
# ---------------------------------------------------------------------------

#: The venue whose balance anchors Kelly sizing, for every book, whatever venue
#: a position is finally routed to.
#:
#: **Sizing does not read the venue a position lands on, and that is deliberate.**
#: A second exchange is funded ahead of integration -- KYC has to clear before it
#: can hold anything -- and carries only enough to cover what routes to it. Its
#: local balance is a settlement constraint, not a statement of how much capital
#: the book has, and sizing against it would shrink every stake to the float
#: sitting at whichever venue happened to quote best.
#:
#: One bankroll, one solve: the same edge earns the same stake wherever it is
#: finally taken, and adding an exchange moves where a position rests without
#: moving what the optimiser thinks it is playing with.
SIZING_ANCHOR_VENUE = "kalshi"


_REGISTRY = None


def _build_registry():
    from .venue_kalshi import KalshiVenue
    return {v.name: v for v in (KalshiVenue(),)}


def venues():
    """Every venue this build knows how to reach, declared or not."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def venue_for(name):
    """The venue called *name*, or ``KeyError`` naming what is registered.

    A position row carries its venue as a string, so a missing implementation
    surfaces here rather than as an attribute error three frames deeper.
    """
    registry = venues()
    if name not in registry:
        raise KeyError(
            f"no venue named {name!r}; registered: {sorted(registry)}")
    return registry[name]


def active_venues():
    """Registered venues whose credentials load and whose flag is set."""
    return [v for v in venues().values() if v.is_available()]


def default_venue():
    """The single active venue.

    Raises when none is active -- that is a misconfiguration, not an empty
    slate -- and when several are, because choosing between them is
    ``code.venue_routing``'s job and doing it implicitly here would hide the
    choice from the ledger.
    """
    active = active_venues()
    if not active:
        raise RuntimeError(
            "no venue is active: check credentials and ENABLE_VENUE_* flags")
    if len(active) > 1:
        raise RuntimeError(
            "several venues are active (%s); route explicitly via "
            "code.venue_routing" % ", ".join(sorted(v.name for v in active)))
    return active[0]


def sizing_bankroll_cents():
    """The balance every Kelly solve sizes against, or ``None`` when unfunded.

    Always the anchor venue's, never the routed one's -- see
    :data:`SIZING_ANCHOR_VENUE` for why. Callers own the fallback when this is
    ``None``: an unfunded account still has to size a paper slate.
    """
    return venue_for(SIZING_ANCHOR_VENUE).balance_cents()


def reset():
    """Drop the registry. Tests only."""
    global _REGISTRY
    _REGISTRY = None


def register_for_test(venue):
    """Add *venue* to the registry. The only supported way in for a fake.

    Production code must never call this: ``tests/unit/test_venue_isolation.py``
    greps ``code/`` for it, because pytest puts the repo root on the path and a
    test double is therefore importable from production without this guard.
    """
    venues()[venue.name] = venue
    return venue
