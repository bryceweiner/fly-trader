"""KyberSwap builds are refused unless they swap exactly what was quoted, pay this wallet, carry no fee, keep the slippage
floor, stay within the drift of the quote, go to the allowlisted router and send the right ETH value."""
import pytest

from fly_trader import config
from fly_trader.rh import abi, kyber as K

ME = "0x" + "a1" * 20
MEME = "0x" + "c3" * 20
DESC_T = "(address,address,address[],uint256[],address[],uint256[],address,uint256,uint256,uint256,bytes)"


def _calldata(simple=False, src=K.NATIVE, dst=MEME, receiver=ME, amount=10 ** 16, min_ret=990, fee=False):
    desc = (src, dst, ["0x" + "e1" * 20], [amount], (["0x" + "fe" * 20] if fee else []), ([5] if fee else []), receiver, amount, min_ret, 0, b"")
    if simple:
        return "0x" + abi.encode_call(K.SWAP_SIMPLE_SIG, "0x" + "e1" * 20, desc, b"\x01", b"").hex()
    return "0x" + abi.encode_call(K.SWAP_SIG, ("0x" + "e1" * 20, "0x" + "e1" * 20, b"\x02", desc, b"")).hex()


def _route(src=K.NATIVE, dst=MEME, amount=10 ** 16, out=1000):
    return {"routeSummary": {"tokenIn": src, "tokenOut": dst, "amountIn": str(amount), "amountOut": str(out)}, "routerAddress": config.KYBER_ROUTER}


def _built(route, data, out=1000, value=None):
    amt = int(route["routeSummary"]["amountIn"])
    return {"amountIn": str(amt), "amountOut": str(out), "data": data, "routerAddress": config.KYBER_ROUTER,
            "transactionValue": str(amt if value is None and route["routeSummary"]["tokenIn"] == K.NATIVE else (value or 0))}


@pytest.mark.parametrize("simple", [False, True])
def test_a_faithful_build_passes(simple):
    r = _route(); K.check_build(_built(r, _calldata(simple=simple)), r, ME, 100)          # min 990 = 1000 × (1 − 1 %)
    r2 = _route(src=MEME, dst=K.NATIVE)
    K.check_build(_built(r2, _calldata(simple=simple, src=MEME, dst=K.NATIVE)), r2, ME, 100)


@pytest.mark.parametrize("why,kw,bkw", [
    ("pays another address", {"receiver": "0x" + "66" * 20}, {}),
    ("pays a fee", {"fee": True}, {}),
    ("slippage limit", {"min_ret": 900}, {}),
    ("other tokens", {"dst": "0x" + "d4" * 20}, {}),
    ("different amount", {"amount": 10 ** 16 + 1}, {}),
    ("much less than quoted", {"min_ret": 980}, {"out": 990}),
    ("ETH value", {}, {"value": 1}),
])
def test_tampered_builds_are_refused(why, kw, bkw):
    r = _route()
    b = _built(r, _calldata(**kw), **{"out": bkw.get("out", 1000)})
    if "value" in bkw:
        b["transactionValue"] = str(int(b["transactionValue"]) + bkw["value"])
    with pytest.raises(K.KyberError, match=why):
        K.check_build(b, r, ME, 100)


def test_an_erc20_input_must_send_no_eth():
    r = _route(src=MEME, dst=K.NATIVE)
    b = _built(r, _calldata(src=MEME, dst=K.NATIVE)); b["transactionValue"] = "1"
    with pytest.raises(K.KyberError, match="ETH value"):
        K.check_build(b, r, ME, 100)


def test_router_allowlist_and_unknown_calldata():
    with pytest.raises(K.KyberError, match="allowlisted"):
        K._allowed_router("0x" + "99" * 20)
    r = _route(); b = _built(r, "0xdeadbeef" + "00" * 64)
    with pytest.raises(K.KyberError, match="neither swap"):
        K.check_build(b, r, ME, 100)


def test_slippage_clamps():
    assert K.clamp_slippage(1) == 20 and K.clamp_slippage(5000) == 1000 and K.clamp_slippage("x") == 20
    assert K.min_received(10_000, 100) == 9_900
