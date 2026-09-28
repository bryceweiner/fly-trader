"""The two memecoin markets the one selector and fly trade: Solana (pump.fun graduates, SOL) and Robinhood Chain (Pons
graduates, native ETH).

A ``MarketSpec`` is everything that differs between them in the trading and accounting path: the unit its books keep
their money in, the book names, the rails circuit, the ui_settings keys, and the money settings (capital, the label size
every edge was measured at, the minimum position, the gas reserve, the kill-switch drawdown). Money settings are read from
``config`` at call time, so a test that monkeypatches config sees its value and Solana's are exactly today's settings.

``k`` converts the few fixed-unit thresholds (eligibility, unit-bearing strategy triggers): SOL 1, RH = ETH per SOL at
build time (``config.RH_ETH_PER_SOL``). Shared functions take ``market=SOL`` as their default, so no Solana call site
changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config

SOLANA_CIRCUIT, KALSHI_CIRCUIT, RH_CIRCUIT = 1, 2, 3


@dataclass(frozen=True)
class MarketSpec:
    chain: str                       # 'sol' | 'rh'
    name: str                        # human name
    unit: str                        # 'SOL' | 'ETH'
    circuit_id: int
    selector_book: str
    fly_book: str
    live_book: str
    handover_key: str                # ui_settings key of the fly's live seat on this chain
    fly_status_key: str
    selector_status_key: str
    stream_status_key: str           # ui_settings key whose flushed_through gates this chain's minutes
    program_label: str               # TokenMeta.program_label of this market's pools (the cost model dispatches on it)
    _capital: str
    _label_size: str
    _min_position: str
    _gas_reserve: str
    _kill_drawdown: str
    _k: str | None                   # config name of ETH-per-SOL, None for SOL (k = 1)

    @property
    def books(self) -> tuple[str, str, str]:
        return self.selector_book, self.fly_book, self.live_book

    @property
    def paper_books(self) -> tuple[str, str]:
        return self.selector_book, self.fly_book

    def k(self) -> float:
        return 1.0 if self._k is None else float(getattr(config, self._k))

    def capital(self) -> float:
        return float(getattr(config, self._capital))

    def label_size(self) -> float:
        return float(getattr(config, self._label_size))

    def min_position(self) -> float:
        return float(getattr(config, self._min_position))

    def gas_reserve(self) -> float:
        return float(getattr(config, self._gas_reserve))

    def kill_drawdown(self) -> float:
        return float(getattr(config, self._kill_drawdown))

    def min_resq(self) -> float:
        from .train.decisions import MIN_RESQ_SOL
        return MIN_RESQ_SOL * self.k()

    def min_vol_15m(self) -> float:
        from .train.decisions import MIN_VOL_15M_SOL
        return MIN_VOL_15M_SOL * self.k()


SOL = MarketSpec(chain="sol", name="Solana", unit="SOL", circuit_id=SOLANA_CIRCUIT, selector_book="paper_selector", fly_book="paper_fly",
                 live_book="live", handover_key="handover", fly_status_key="fly_status", selector_status_key="selector_status",
                 stream_status_key="pumpstream_status", program_label="Pump.fun Amm", _capital="CAPITAL_SOL", _label_size="LABEL_SIZE_SOL",
                 _min_position="MIN_POSITION_SOL", _gas_reserve="GAS_RESERVE_SOL", _kill_drawdown="KILL_SWITCH_DRAWDOWN", _k=None)
RH = MarketSpec(chain="rh", name="Robinhood Chain", unit="ETH", circuit_id=RH_CIRCUIT, selector_book="paper_rh_selector", fly_book="paper_rh_fly",
                live_book="live_rh", handover_key="handover_rh", fly_status_key="fly_status_rh", selector_status_key="selector_status_rh",
                stream_status_key="rh_stream_status", program_label="Pons v4", _capital="RH_CAPITAL_ETH", _label_size="RH_LABEL_SIZE_ETH",
                _min_position="RH_MIN_POSITION_ETH", _gas_reserve="RH_GAS_RESERVE_ETH", _kill_drawdown="RH_KILL_SWITCH_DRAWDOWN", _k="RH_ETH_PER_SOL")
MARKETS = {"sol": SOL, "rh": RH}


def for_chain(chain: str | None) -> MarketSpec:
    """The market of a chain id ('sol' | 'rh'); None is Solana (rows and positions from before the second chain)."""
    return MARKETS[chain or "sol"]


def for_book(book: str) -> MarketSpec:
    """The market a book belongs to: an RH book by name, every other (Solana books, replay_* test books) Solana."""
    for m in MARKETS.values():
        if book in m.books:
            return m
    return SOL


def enabled() -> list[MarketSpec]:
    """The markets this install trades: Solana always, RH when ``RH_ENABLED``."""
    return [SOL] + ([RH] if config.RH_ENABLED else [])
