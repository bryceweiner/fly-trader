"""One swap's transaction plan: KyberSwap first (atomic, multi-hop — a USDG- or stock-quoted Pons coin is bought as
ETH→base→meme in one transaction), the UniversalRouter v4 single-pool swap when Kyber has no route.

A plan is what the executor (rh/exec.py) signs, in order: the approvals the route needs (exact amounts), then the swap.
Before the swap is signed the executor simulates it (eth_call as the wallet) — a plan is never trusted blind.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from . import kyber as K, univ4 as U


class NoRoute(RuntimeError):
    pass


@dataclass
class Call:
    to: str
    data: bytes
    value: int = 0
    kind: str = "swap"                          # rh_txs.kind: 'approve' | 'swap'


@dataclass
class Plan:
    route: str                                  # 'atomic_kyber' | 'atomic_ur'
    token_in: str
    token_out: str
    amount_in: int
    expected_out: int
    min_out: int
    swap: Call
    approvals: list = field(default_factory=list)      # [(token, spender, exact amount)] made before the swap (skipped when covered)
    pre_calls: list = field(default_factory=list)      # other calls before the swap (Permit2 → router allowance)
    gas_hint: int | None = None
    quote: dict | None = None
    build: dict | None = None


def _is_native(t: str) -> bool:
    return t.lower() in (K.NATIVE, U.ZERO)


def plan_swap(account: str, token_in: str, token_out: str, amount_in: int, slippage_bps: float, *, kyber: K.Kyber | None = None,
              rpc=None, pool: dict | None = None) -> Plan:
    """``token_in``/``token_out``: an ERC-20 address, or native ETH (Kyber's 0xeeee… or the zero address). ``pool``: the
    Pons pool (currency0, currency1, fee, tick_spacing, hooks) for the fallback."""
    kin = K.NATIVE if _is_native(token_in) else token_in.lower(); kout = K.NATIVE if _is_native(token_out) else token_out.lower()
    kerr = None
    try:
        ky = kyber or K.Kyber()
        route = ky.get_route(kin, kout, amount_in)
        built = ky.build_route(route, account, slippage_bps)
        exp = int(built["amountOut"]); mn = K.decode_desc(built["data"])["minReturnAmount"]
        approvals = [] if kin == K.NATIVE else [(kin, config.KYBER_ROUTER.lower(), amount_in)]
        return Plan("atomic_kyber", kin, kout, amount_in, exp, mn, Call(config.KYBER_ROUTER.lower(), bytes.fromhex(built["data"][2:]), int(built.get("transactionValue") or 0)),
                    approvals=approvals, gas_hint=int(route["routeSummary"].get("gas") or 0) or None, quote=route["routeSummary"],
                    build={k: built.get(k) for k in ("amountIn", "amountOut", "transactionValue", "routerAddress")})
    except K.KyberError as e:
        kerr = str(e)
    if pool is None or rpc is None:
        raise NoRoute(f"KyberSwap: {kerr}; no v4 pool for the fallback")
    key = U.pool_key(pool["currency0"], pool["currency1"], pool.get("fee", 0), pool.get("tick_spacing", 200), pool.get("hooks"))
    tin = U.ZERO if _is_native(token_in) else token_in.lower(); tout = U.ZERO if _is_native(token_out) else token_out.lower()
    if {tin, tout} != {key[0], key[1]}:
        raise NoRoute(f"KyberSwap: {kerr}; the v4 fallback swaps within one pool only ({tin}→{tout} is not its pair)")
    exp, gas = U.quote(rpc, key, tin, amount_in)
    mn = K.min_received(exp, slippage_bps)
    data, value = U.swap_calldata(key, tin, amount_in, mn)
    approvals, pre = [], []
    if tin != U.ZERO:
        approvals = [(tin, config.PERMIT2.lower(), amount_in)]
        pre = [Call(config.PERMIT2.lower(), U.permit2_approve_calldata(tin, amount_in), 0, "approve")]
    return Plan("atomic_ur", tin, tout, amount_in, exp, mn, Call(config.UNIVERSAL_ROUTER.lower(), data, value), approvals=approvals, pre_calls=pre,
                gas_hint=gas, quote={"kyber_error": kerr, "v4_quote": str(exp)})
