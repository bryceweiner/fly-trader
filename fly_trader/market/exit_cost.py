"""Trading costs of a swap, as actually charged (measured 2026-09-15).

- Pool fee. PumpSwap pools created by a pump.fun graduation charge a fee that steps down with the token's market cap
  in SOL (``PUMP_FEE_TIERS``): measured from 157,059 archive trades (June–September 2026, ``poolFeeRate`` against
  ``marketCapQuote``); the table reproduces the charged rate exactly on 99.7 % of them and is never more than one tier
  off. Market cap = price × the token's supply (1 billion for pump.fun tokens, 2 billion in mayhem mode). The live
  stream reports the charged rate itself, which is used when given (``pool_fee``). A token with unknown market cap is
  charged the highest tier.
- Jupiter Ultra platform fee: 10 bps on every PumpSwap route (``feeBps`` of ``/order`` quotes with our key, from a
  58 SOL to a 169,000 SOL market cap; no age dependence).
- Network: the 5,000-lamport signature fee per transaction (Ultra quoted no priority or rent fees).
- Price impact. Constant-product pools (PumpSwap, Raydium v4/CPMM, Meteora DAMM): v / (v + R_q), v the traded value
  and R_q the pool's quote reserve. Meteora DLMM: LAMBDA_IMPACT_DLMM · v / TVL, TVL ≈ 2·R_q.
"""
from __future__ import annotations

import math

import numpy as np

from .. import config

PUMP_SUPPLY = 1e9
PUMP_FEE_TIERS = ((420, 0.0125), (1470, 0.012), (2461, 0.0115), (3440, 0.011), (4430, 0.0105), (9832, 0.01), (14744, 0.0095), (19649, 0.009),
                  (24519, 0.0085), (29616, 0.008), (34892, 0.0075), (39306, 0.007), (44800, 0.0065), (49054, 0.006), (54174, 0.0055), (58935, 0.0053),
                  (63864, 0.005), (69943, 0.0048), (73798, 0.0045), (78617, 0.0043), (83514, 0.004), (88407, 0.0038), (93311, 0.0035), (98178, 0.0033))
PUMP_FEE_FLOOR = 0.003                       # market cap above the last tier
JUPITER_FEE_BPS = 10
TX_FEE_LAMPORTS = 5000
_UPPER = np.array([u for u, _ in PUMP_FEE_TIERS], dtype=float)
_FEE = np.array([f for _, f in PUMP_FEE_TIERS] + [PUMP_FEE_FLOOR])


def pool_fee_rate(mcap_sol):
    """The pump.fun pool fee per swap for a token of this market cap (SOL); scalar or array."""
    if mcap_sol is None:
        return float(_FEE[0])
    m = np.asarray(mcap_sol, dtype=float)
    out = np.where(np.isfinite(m), _FEE[np.searchsorted(_UPPER, np.nan_to_num(m, nan=0.0), side="right")], _FEE[0])
    return float(out) if out.ndim == 0 else out


def impact_fraction(value_sol: float, res_quote_sol: float | None, program_label: str | None = None) -> float:
    if value_sol <= 0:
        return 0.0
    if res_quote_sol is None or res_quote_sol <= 0:
        return 1.0  # unknown liquidity: assume the position cannot be exited
    if program_label and any(lbl.lower() in program_label.lower() for lbl in ("dlmm",)):
        return min(1.0, config.LAMBDA_IMPACT_DLMM * value_sol / (2.0 * res_quote_sol))
    return value_sol / (value_sol + res_quote_sol)


def fee_fraction(value_sol: float, mcap_sol: float | None, pool_fee: float | None = None) -> float:
    """Fees of one swap of ``value_sol`` as a fraction of it: pool fee + Jupiter's platform fee + the network fee."""
    if value_sol <= 0:
        return 0.0
    pf = pool_fee if pool_fee is not None and math.isfinite(pool_fee) and pool_fee > 0 else pool_fee_rate(mcap_sol)
    return pf + JUPITER_FEE_BPS / 1e4 + TX_FEE_LAMPORTS / config.LAMPORTS_PER_SOL / value_sol


def exit_cost_fraction(value_sol: float, res_quote_sol: float | None, mcap_sol: float | None, program_label: str | None = None,
                       pool_fee: float | None = None) -> float:
    """Everything a swap of ``value_sol`` loses, as a fraction of it: fees plus price impact."""
    if value_sol <= 0:
        return 0.0
    return min(1.0, fee_fraction(value_sol, mcap_sol, pool_fee) + impact_fraction(value_sol, res_quote_sol, program_label))
