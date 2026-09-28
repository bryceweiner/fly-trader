"""Direct Uniswap v4 swaps through the UniversalRouter — the fallback when KyberSwap has no route for a Pons pool.

One pool, exact input: ``execute(commands=[V4_SWAP], inputs=[abi.encode(actions, params)], deadline)`` with the v4
actions SWAP_EXACT_IN_SINGLE → SETTLE_ALL → TAKE_ALL. Native ETH input is paid as the call's value; an ERC-20 input is
pulled through Permit2 (the token approves Permit2 exactly, Permit2 approves the router for exactly the amount, expiring
within the hour). minOut comes from the V4Quoter (an eth_call) minus the slippage. A Pons pool's key is (currency0,
currency1, fee 0, tickSpacing 200, the Pons hook); its id is keccak(abi.encode(key)).
"""
from __future__ import annotations

import time

from .. import config
from ..vault.evm import keccak256
from . import abi

ZERO = "0x" + "00" * 20
V4_SWAP = 0x10
SWAP_EXACT_IN_SINGLE, SETTLE_ALL, TAKE_ALL = 0x06, 0x0C, 0x0F
POOL_KEY = "(address,address,uint24,int24,address)"
EXACT_IN_SINGLE = f"({POOL_KEY},bool,uint128,uint128,bytes)"
QUOTE_SINGLE = f"({POOL_KEY},bool,uint128,bytes)"
EXECUTE_SIG = "execute(bytes,bytes[],uint256)"
PERMIT2_APPROVE_SIG = "approve(address,address,uint160,uint48)"
DEADLINE_S = 20 * 60
PERMIT_EXPIRY_S = 3600


def pool_key(currency0: str, currency1: str, fee: int = 0, tick_spacing: int = 200, hooks: str | None = None) -> tuple:
    c0, c1 = currency0.lower(), currency1.lower()
    if int(c0, 16) >= int(c1, 16):
        raise ValueError("currency0 must sort below currency1")
    return (c0, c1, int(fee), int(tick_spacing), (hooks or config.PONS_HOOK).lower())


def pool_id(key: tuple) -> str:
    return "0x" + keccak256(abi.encode([POOL_KEY], [key])).hex()


def swap_calldata(key: tuple, token_in: str, amount_in: int, min_out: int, deadline: int | None = None, hook_data: bytes = b"") -> tuple[bytes, int]:
    """(UniversalRouter calldata, native value) for an exact-input swap of ``amount_in`` of ``token_in`` in pool ``key``."""
    tin = token_in.lower()
    if tin not in (key[0], key[1]):
        raise ValueError("token_in is not in the pool")
    zero_for_one = tin == key[0]
    tout = key[1] if zero_for_one else key[0]
    if not (0 < amount_in < 2 ** 128 and 0 <= min_out < 2 ** 128):
        raise ValueError("amounts out of uint128 range")
    actions = bytes([SWAP_EXACT_IN_SINGLE, SETTLE_ALL, TAKE_ALL])
    params = [abi.encode([EXACT_IN_SINGLE], [(key, zero_for_one, amount_in, min_out, hook_data)]),
              abi.encode(["address", "uint256"], [tin, amount_in]),
              abi.encode(["address", "uint256"], [tout, min_out])]
    inp = abi.encode(["bytes", "bytes[]"], [actions, params])
    dl = deadline if deadline is not None else int(time.time()) + DEADLINE_S
    data = abi.encode_call(EXECUTE_SIG, bytes([V4_SWAP]), [inp], dl)
    return data, (amount_in if tin == ZERO else 0)


def decode_swap(data: bytes) -> dict:
    """Parse our own UniversalRouter calldata back (tests, and a last check before signing)."""
    commands, inputs, deadline = abi.decode_call(EXECUTE_SIG, data)
    if commands != bytes([V4_SWAP]) or len(inputs) != 1:
        raise ValueError("not a single V4_SWAP")
    actions, params = abi.decode(["bytes", "bytes[]"], inputs[0])
    if actions != bytes([SWAP_EXACT_IN_SINGLE, SETTLE_ALL, TAKE_ALL]):
        raise ValueError("unexpected v4 actions")
    (key, zfo, amt, mn, hook_data), = abi.decode([EXACT_IN_SINGLE], params[0])
    settle = abi.decode(["address", "uint256"], params[1]); take = abi.decode(["address", "uint256"], params[2])
    return {"key": key, "zero_for_one": zfo, "amount_in": amt, "min_out": mn, "settle": settle, "take": take, "deadline": deadline}


def quote_calldata(key: tuple, token_in: str, amount_in: int, hook_data: bytes = b"") -> bytes:
    zfo = token_in.lower() == key[0]
    return abi.encode_call(f"quoteExactInputSingle({QUOTE_SINGLE})", (key, zfo, amount_in, hook_data))


def quote(rpc, key: tuple, token_in: str, amount_in: int) -> tuple[int, int]:
    """(amountOut, gasEstimate) from the V4Quoter."""
    out = rpc.eth_call(config.V4_QUOTER, "0x" + quote_calldata(key, token_in, amount_in).hex())
    a, g = abi.decode(["uint256", "uint256"], bytes.fromhex(out[2:]))
    return a, g


def permit2_approve_calldata(token: str, amount: int, now_s: int | None = None) -> bytes:
    if not 0 < amount < 2 ** 160:
        raise ValueError("Permit2 amounts are uint160 and must be exact")
    exp = (int(time.time()) if now_s is None else now_s) + PERMIT_EXPIRY_S
    return abi.encode_call(PERMIT2_APPROVE_SIG, token.lower(), config.UNIVERSAL_ROUTER.lower(), amount, exp)
