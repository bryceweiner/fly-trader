"""Exit-cost model: what fraction of a position's marked value is lost by selling it now.

Constant-product pools (PumpSwap, Raydium v4/CPMM, Meteora DAMM): impact = v / (v + R_q) where v is the
position value in quote and R_q the pool's quote reserve (exact for x*y=k selling into the quote side).
Meteora DLMM: impact ≈ LAMBDA_IMPACT_DLMM · v / TVL (VOC dexlp measurement, config.py:213), TVL ≈ 2·R_q.
Fees: Jupiter platform fee (10 bps; 50 bps for tokens younger than 24 h) plus a pool fee
(POOL_FEE_BPS, default 30 bps ≈ PumpSwap 0.25 % + rounding; unsourced beyond the PumpSwap docs).
"""
from __future__ import annotations

from .. import config

POOL_FEE_BPS = 30
JUPITER_FEE_BPS = 10
JUPITER_FEE_BPS_NEW = 50
DLMM_LABELS = ("Meteora DLMM", "meteora", "Meteora")


def jupiter_fee_bps(age_hours: float | None) -> int:
    if age_hours is not None and age_hours < 24:
        return JUPITER_FEE_BPS_NEW
    return JUPITER_FEE_BPS


def impact_fraction(value_sol: float, res_quote_sol: float | None, program_label: str | None = None) -> float:
    if value_sol <= 0:
        return 0.0
    if res_quote_sol is None or res_quote_sol <= 0:
        return 1.0  # unknown liquidity: assume the position cannot be exited
    if program_label and any(lbl.lower() in program_label.lower() for lbl in ("dlmm",)):
        return min(1.0, config.LAMBDA_IMPACT_DLMM * value_sol / (2.0 * res_quote_sol))
    return value_sol / (value_sol + res_quote_sol)


def exit_cost_fraction(value_sol: float, res_quote_sol: float | None, age_hours: float | None,
                       program_label: str | None = None) -> float:
    fee = (jupiter_fee_bps(age_hours) + POOL_FEE_BPS) / 1e4
    return min(1.0, fee + impact_fraction(value_sol, res_quote_sol, program_label))
