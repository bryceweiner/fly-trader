"""Danger score in [0, 1] — a SENSORY signal injected into the fly's aversive olfactory pathway.

It is never a filter (operator decision). The linear weights below are unsourced defaults; the
encoder standardizes downstream so only their relative ordering matters much.
"""
from __future__ import annotations

import math

from .features import FIDX, STATS_BIT

W_LOW_LIQ = 1.0        # per decade of liquidity below $10k
W_YOUNG = 1.0          # age < 1 h
W_TOP_HOLDERS = 2.0    # × top-holder fraction
W_DEV_BALANCE = 2.0    # × dev balance fraction
W_SUS = 2.0
W_TOKEN2022 = 0.5
W_EXIT_COST = 5.0      # × exit-cost fraction for a max-size position
W_FEW_HOLDERS = 1.0    # holder count < 100
BIAS = -2.0


def _frac(x: float) -> float:
    """Jupiter audit topHoldersPercentage/devBalancePercentage are percents (0-100; ~1/4 of rows are below 1 %), as is
    the corpus proxy (100 × top-10 share); normalise to [0,1]."""
    return 0.0 if x is None else x / 100.0


def danger_score(features: list[float], mask: int, age_hours: float | None) -> float:
    z = BIAS
    liq_usd = math.expm1(features[FIDX["log_liq_usd"]]) if mask & STATS_BIT else None
    if liq_usd is not None:
        z += W_LOW_LIQ * max(0.0, 4.0 - math.log10(liq_usd + 1.0))
    if age_hours is not None and age_hours < 1.0:
        z += W_YOUNG
    z += W_TOP_HOLDERS * _frac(features[FIDX["top_holders_pct"]])
    z += W_DEV_BALANCE * _frac(features[FIDX["dev_balance_pct"]])
    z += W_SUS * features[FIDX["is_sus"]]
    z += W_TOKEN2022 * features[FIDX["token2022"]]
    z += W_EXIT_COST * features[FIDX["exit_cost_0p1"]]
    holders = math.expm1(features[FIDX["log_holders"]]) if mask & STATS_BIT else None
    if holders is not None and holders < 100:
        z += W_FEW_HOLDERS
    return 1.0 / (1.0 + math.exp(-z))
