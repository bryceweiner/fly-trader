"""Swap plans: KyberSwap first (with the exact approval an ERC-20 input needs), the UniversalRouter v4 fallback when Kyber
has no route (quoter floor, Permit2 for an ERC-20 input), refusal outside the pool's pair; the v4 calldata round-trips and
a Pons pool id is keccak(abi.encode(PoolKey)) — checked against a real registered pool.

Set RH_LIVE_TESTS=1 to also simulate a UniversalRouter buy against Robinhood Chain mainnet state (read-only eth_calls)."""
import os

import pytest

from fly_trader import config
from fly_trader.rh import kyber as K, router as R, univ4 as U

MEME = "0x9ed412f8fadfe63ba0686e06a2b32cb4cd614140"          # a Pons graduate (ETH-quoted), registered as the id below
MEME_ID = "0x9863bd8d000d618e9fd3a56c8626afcf461848d4b4e4d569f05433dc5d14d1c9"
ME = "0x" + "a1" * 20


def test_pool_id_matches_the_registered_pons_pool():
    assert U.pool_id(U.pool_key(U.ZERO, MEME)) == MEME_ID
    with pytest.raises(ValueError):
        U.pool_key(MEME, U.ZERO)                                    # currencies must be sorted


def test_v4_calldata_round_trips():
    key = U.pool_key(U.ZERO, MEME)
    data, value = U.swap_calldata(key, U.ZERO, 10 ** 14, 123, deadline=1_900_000_000)
    d = U.decode_swap(data)
    assert value == 10 ** 14 and d["zero_for_one"] is True and d["amount_in"] == 10 ** 14 and d["min_out"] == 123
    assert d["settle"] == [U.ZERO, 10 ** 14] and d["take"] == [MEME, 123] and d["deadline"] == 1_900_000_000
    data2, value2 = U.swap_calldata(key, MEME, 5 * 10 ** 20, 7)
    assert value2 == 0 and U.decode_swap(data2)["zero_for_one"] is False
    with pytest.raises(ValueError):
        U.swap_calldata(key, "0x" + "77" * 20, 1, 0)


class NoKyber:
    def get_route(self, *a):
        raise K.KyberError("KyberSwap: route not found")


class FakeRpc:
    def eth_call(self, to, data, block="latest"):
        assert to == config.V4_QUOTER
        from fly_trader.rh import abi
        return "0x" + abi.encode(["uint256", "uint256"], [10_000, 85_000]).hex()


def test_fallback_to_v4_with_permit2_for_an_erc20_input():
    pool = {"currency0": U.ZERO, "currency1": MEME}
    p = R.plan_swap(ME, K.NATIVE, MEME, 10 ** 14, 300, kyber=NoKyber(), rpc=FakeRpc(), pool=pool)
    assert p.route == "atomic_ur" and p.swap.to == config.UNIVERSAL_ROUTER.lower() and p.swap.value == 10 ** 14
    assert p.min_out == 9_700 and p.approvals == [] and p.pre_calls == []
    s = R.plan_swap(ME, MEME, K.NATIVE, 5 * 10 ** 20, 300, kyber=NoKyber(), rpc=FakeRpc(), pool=pool)
    assert s.swap.value == 0 and s.approvals == [(MEME, config.PERMIT2.lower(), 5 * 10 ** 20)] and s.pre_calls[0].to == config.PERMIT2.lower()
    with pytest.raises(R.NoRoute, match="one pool"):
        R.plan_swap(ME, K.NATIVE, "0x" + "55" * 20, 10 ** 14, 300, kyber=NoKyber(), rpc=FakeRpc(), pool=pool)
    with pytest.raises(R.NoRoute, match="no v4 pool"):
        R.plan_swap(ME, K.NATIVE, MEME, 10 ** 14, 300, kyber=NoKyber())


def test_kyber_first_with_its_exact_approval(monkeypatch):
    desc = (MEME, K.NATIVE, [], [], [], [], ME, 5 * 10 ** 20, 990, 0, b"")
    from fly_trader.rh import abi
    data = "0x" + abi.encode_call(K.SWAP_SIG, ("0x" + "e1" * 20, "0x" + "e1" * 20, b"", desc, b"")).hex()

    class Ky:
        def get_route(self, a, b, n):
            return {"routeSummary": {"tokenIn": a, "tokenOut": b, "amountIn": str(n), "amountOut": "1000", "gas": "356167"}, "routerAddress": config.KYBER_ROUTER}

        def build_route(self, route, account, slip):
            return {"amountIn": route["routeSummary"]["amountIn"], "amountOut": "1000", "data": data, "transactionValue": "0", "routerAddress": config.KYBER_ROUTER}
    p = R.plan_swap(ME, MEME, K.NATIVE, 5 * 10 ** 20, 100, kyber=Ky())
    assert p.route == "atomic_kyber" and p.approvals == [(MEME, config.KYBER_ROUTER.lower(), 5 * 10 ** 20)] and p.min_out == 990 and p.gas_hint == 356167


@pytest.mark.skipif(not os.environ.get("RH_LIVE_TESTS"), reason="RH_LIVE_TESTS=1 simulates against Robinhood Chain mainnet")
def test_live_ur_buy_simulates_on_mainnet():
    from fly_trader.rh.rpc import RhRpc
    rpc = RhRpc(); key = U.pool_key(U.ZERO, MEME)
    out, _ = U.quote(rpc, key, U.ZERO, 10 ** 14)
    data, value = U.swap_calldata(key, U.ZERO, 10 ** 14, out * 95 // 100)
    rpc.call("eth_call", [{"from": config.RH_WETH, "to": config.UNIVERSAL_ROUTER, "data": "0x" + data.hex(), "value": hex(value)}, "latest"])
