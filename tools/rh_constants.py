"""The Robinhood Chain book's fixed constants, from today's prices — run once, then the values are pinned in config.

The RH book trades the same way as the Solana book, in ETH: its capital, the size every RH label is priced at and its
minimum position are the Solana values at today's USD value (5 SOL, 0.5 SOL, 0.02 SOL), and K = ETH per SOL converts
the few fixed-unit thresholds (eligibility, unit-bearing strategy triggers). They are fixed on purpose: derived live,
they would move the labels with the market (see config.LABEL_SIZE_SOL).

    .venv/bin/python tools/rh_constants.py          # prints the config lines to paste (or env overrides)

SOL/USD from Jupiter (vault/prices.sol_usd); ETH/USD from a KyberSwap quote of 1 ETH into USDG on Robinhood Chain
(vault/prices.eth_usd).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_trader import config  # noqa: E402
from fly_trader.vault import prices  # noqa: E402


def main() -> None:
    sol, eth = prices.sol_usd()[0], prices.eth_usd()[0]          # each returns (price, fetched_at)
    if not sol or not eth:
        raise SystemExit(f"no price (SOL/USD {sol}, ETH/USD {eth})")
    k = sol / eth                                   # ETH per SOL
    r = lambda v: float(f"{v:.6g}")
    print(f"# {config.utcnow():%Y-%m-%d %H:%M} UTC: SOL/USD {sol:.2f}, ETH/USD {eth:.2f}")
    print(f"RH_ETH_PER_SOL={r(k)}")
    print(f"RH_CAPITAL_ETH={r(5.0 * k)}")
    print(f"RH_LABEL_SIZE_ETH={r(config.LABEL_SIZE_SOL * k)}")
    print(f"RH_MIN_POSITION_ETH={r(config.MIN_POSITION_SOL * k)}")


if __name__ == "__main__":
    main()
