"""Uniswap v4 PoolManager events and concentrated-liquidity math for the Pons pools.

v4 emits ``Swap(id, sender, amount0, amount1, sqrtPriceX96, liquidity, tick, fee)`` with the amounts as the swapper's
balance delta (negative = paid into the pool). A Pons pool's locked graduation position is full range, so between two
swaps the pool behaves as constant product with virtual reserves x = L / √P (currency0) and y = L · √P (currency1), √P =
sqrtPriceX96 / 2^96 in raw units; P = y / x = currency1 per currency0.

The same sqrt-price word sits third in the data of Uniswap v3, v4 and Pancake-Infinity-CL swaps, which is how the base
asset marks (rh/prices.py) read any reference pool.
"""
from __future__ import annotations

from . import abi

Q96 = 2 ** 96
SIGS = {
    "Initialize": "Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)",   # id, currency0, currency1 (indexed)
    "Swap": "Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)",                # id, sender (indexed)
    "ModifyLiquidity": "ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)",          # id, sender (indexed)
}
TOPIC = {k: abi.event_topic(v) for k, v in SIGS.items()}
NAME = {v: k for k, v in TOPIC.items()}
MIN_TICK, MAX_TICK = -887272, 887272


def decode(log: dict) -> dict | None:
    name = NAME.get(log["topics"][0])
    if name is None:
        return None
    t = log["topics"]; d = bytes.fromhex(log["data"][2:])
    base = {"event": name, "pool_id": t[1].lower(), "block": int(log["blockNumber"], 16), "log_index": int(log["logIndex"], 16),
            "tx_hash": log["transactionHash"], "block_hash": log.get("blockHash"), "ts": int(log.get("blockTimestamp") or "0x0", 16) or None}
    if name == "Initialize":
        fee, ts_, hooks, sp, tick = abi.decode(["uint24", "int24", "address", "uint160", "int24"], d)
        return {**base, "currency0": "0x" + t[2][-40:].lower(), "currency1": "0x" + t[3][-40:].lower(), "fee": fee, "tick_spacing": ts_, "hooks": hooks,
                "sqrt_price_x96": sp, "tick": tick}
    if name == "Swap":
        a0, a1, sp, liq, tick, fee = abi.decode(["int128", "int128", "uint160", "uint128", "int24", "uint24"], d)
        return {**base, "sender": "0x" + t[2][-40:].lower(), "amount0": a0, "amount1": a1, "sqrt_price_x96": sp, "liquidity": liq, "tick": tick, "fee": fee}
    if name == "ModifyLiquidity":
        lo, hi, delta, salt = abi.decode(["int24", "int24", "int256", "bytes32"], d)
        return {**base, "sender": "0x" + t[2][-40:].lower(), "tick_lower": lo, "tick_upper": hi, "liquidity_delta": delta, "salt": "0x" + salt.hex()}
    return None


def sqrt_price_word(log: dict) -> int:
    """The sqrtPriceX96 of any v3 / v4 / Pancake-Infinity-CL swap log (the third data word)."""
    d = log["data"][2:]
    return int(d[128:192], 16)


def price1_per_0(sqrt_price_x96: int, dec0: int, dec1: int) -> float:
    """Whole currency1 per whole currency0."""
    p = (sqrt_price_x96 / Q96) ** 2
    return p * 10 ** (dec0 - dec1)


def token_price_in_quote(sqrt_price_x96: int, token_is_0: bool, dec_token: int, dec_quote: int) -> float:
    """Whole quote per whole memecoin."""
    if token_is_0:
        return price1_per_0(sqrt_price_x96, dec_token, dec_quote)
    p = price1_per_0(sqrt_price_x96, dec_quote, dec_token)
    return 1.0 / p if p > 0 else float("nan")


def virtual_reserves(sqrt_price_x96: int, liquidity: int) -> tuple[float, float]:
    """(x raw currency0, y raw currency1) of the constant-product equivalent at this price."""
    s = sqrt_price_x96 / Q96
    return (liquidity / s if s > 0 else float("inf"), liquidity * s)


def quote_reserve(sqrt_price_x96: int, liquidity: int, token_is_0: bool, dec_quote: int) -> float:
    """The quote side's virtual reserve in whole quote units."""
    x, y = virtual_reserves(sqrt_price_x96, liquidity)
    return (y if token_is_0 else x) / 10 ** dec_quote


def is_full_range(tick_lower: int, tick_upper: int, tick_spacing: int) -> bool:
    lo = (MIN_TICK // tick_spacing) * tick_spacing + (tick_spacing if MIN_TICK % tick_spacing else 0)
    hi = (MAX_TICK // tick_spacing) * tick_spacing
    return tick_lower <= lo and tick_upper >= hi
