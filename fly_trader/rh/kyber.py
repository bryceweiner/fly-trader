"""KyberSwap aggregator on Robinhood Chain for the bot: quote (routes) and calldata (route/build), with every check the
website's client makes (web/src/lib/kyber.ts) before anything is signed:

- the router is the allowlisted MetaAggregationRouterV2 (``config.KYBER_ROUTER``), in the quote and in the build;
- the quote answers the swap asked for (tokens, amount);
- the built calldata (``swap`` 0xe21fd0e9 / ``swapSimpleMode`` 0x8af033fb, decoded here) pays this wallet, swaps exactly
  the quoted tokens and amount, lists no fee, and carries a minReturnAmount of at least the slippage floor;
- the build delivers at most ``BUILD_DRIFT_BPS`` less than the quote;
- the bot's own addition: the transaction's native value equals the amount when the input is native ETH, and is 0 otherwise.
"""
from __future__ import annotations

import time

import httpx

from .. import config
from ..db.apilog import record_api_call
from . import abi

NATIVE = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
MIN_SLIPPAGE_BPS, MAX_SLIPPAGE_BPS = 20, 1000
BUILD_DRIFT_BPS = 50
DEADLINE_S = 20 * 60
_DESC = "(address,address,address[],uint256[],address[],uint256[],address,uint256,uint256,uint256,bytes)"
SWAP_SIG = f"swap((address,address,bytes,{_DESC},bytes))"
SWAP_SIMPLE_SIG = f"swapSimpleMode(address,{_DESC},bytes,bytes)"


class KyberError(RuntimeError):
    pass


def _same(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.lower() == b.lower()


def _big(v) -> int | None:
    return int(v) if isinstance(v, str) and v.isdigit() else (v if isinstance(v, int) and v >= 0 else None)


def clamp_slippage(bps: float) -> int:
    try:
        return int(min(MAX_SLIPPAGE_BPS, max(MIN_SLIPPAGE_BPS, round(float(bps)))))
    except (TypeError, ValueError):
        return MIN_SLIPPAGE_BPS


def min_received(amount_out: int, slippage_bps: float) -> int:
    return amount_out * (10_000 - clamp_slippage(slippage_bps)) // 10_000


def _allowed_router(addr: str | None) -> str:
    if not _same(addr, config.KYBER_ROUTER):
        raise KyberError(f"refusing: KyberSwap returned router {addr or '(none)'}, not the allowlisted {config.KYBER_ROUTER}")
    return addr.lower()


class Kyber:
    def __init__(self, base: str | None = None, client: httpx.Client | None = None):
        self.base = base or config.KYBER_API
        self.http = client or httpx.Client(timeout=15.0, headers={"x-client-id": config.KYBER_CLIENT_ID})

    def _req(self, method: str, path: str, **kw) -> dict:
        t0 = time.monotonic(); status, ok, err = None, False, None
        try:
            r = self.http.request(method, f"{self.base}{path}", **kw)
            status = r.status_code
            body = r.json() if r.content else None
            if r.status_code != 200 or not body or body.get("code") != 0 or not body.get("data"):
                err = (body or {}).get("message") or f"HTTP {r.status_code}"
                raise KyberError(f"KyberSwap: {err}")
            ok = True
            return body["data"]
        except httpx.HTTPError as e:
            err = type(e).__name__
            raise KyberError("KyberSwap could not be reached") from None
        finally:
            record_api_call("kyber", path.split("?")[0], method, status, int((time.monotonic() - t0) * 1000), ok, err)

    def get_route(self, token_in: str, token_out: str, amount_in: int, included_sources: str | None = None) -> dict:
        """{routeSummary, routerAddress} for exactly this swap, or KyberError (incl. 'no route'). ``included_sources``: only
        these venues (a comma list of KyberSwap source ids)."""
        params = {"tokenIn": token_in, "tokenOut": token_out, "amountIn": str(int(amount_in)), "gasInclude": "true"}
        if included_sources:
            params["includedSources"] = included_sources
        d = self._req("GET", "/routes", params=params)
        router = _allowed_router(d.get("routerAddress"))
        rs = d.get("routeSummary") or {}
        if not _same(rs.get("tokenIn"), token_in) or not _same(rs.get("tokenOut"), token_out) or _big(rs.get("amountIn")) != int(amount_in) \
                or _big(rs.get("amountOut")) is None:
            raise KyberError("refusing the quote: KyberSwap answered for a different swap than the one asked for")
        return {"routeSummary": rs, "routerAddress": router}

    def build_route(self, route: dict, account: str, slippage_bps: float, now_s: int | None = None) -> dict:
        now_s = int(time.time()) if now_s is None else now_s
        d = self._req("POST", "/route/build", json={"routeSummary": route["routeSummary"], "sender": account, "recipient": account,
                                                    "slippageTolerance": clamp_slippage(slippage_bps), "deadline": now_s + DEADLINE_S,
                                                    "source": config.KYBER_CLIENT_ID})
        built = {**d, "routerAddress": _allowed_router(d.get("routerAddress"))}
        check_build(built, route, account, slippage_bps)
        return built


def decode_desc(data: str) -> dict:
    """The SwapDescriptionV2 of a built call, as a dict."""
    cd = bytes.fromhex(data[2:] if data.startswith("0x") else data)
    if cd[:4] == abi.selector(SWAP_SIG):
        desc = abi.decode_call(SWAP_SIG, cd)[0][3]
    elif cd[:4] == abi.selector(SWAP_SIMPLE_SIG):
        desc = abi.decode_call(SWAP_SIMPLE_SIG, cd)[1]
    else:
        raise KyberError("refusing the swap: calldata is neither swap nor swapSimpleMode")
    keys = ("srcToken", "dstToken", "srcReceivers", "srcAmounts", "feeReceivers", "feeAmounts", "dstReceiver", "amount", "minReturnAmount", "flags", "permit")
    return dict(zip(keys, desc))


def check_build(built: dict, route: dict, account: str, slippage_bps: float) -> None:
    rs = route["routeSummary"]
    refuse = lambda why: (_ for _ in ()).throw(KyberError(f"refusing the swap: KyberSwap built {why}"))
    amount_in, quoted_out, built_out = _big(rs.get("amountIn")), _big(rs.get("amountOut")), _big(built.get("amountOut"))
    if amount_in is None or quoted_out is None or built_out is None:
        refuse("a transaction without readable amounts")
    if _big(built.get("amountIn")) != amount_in:
        refuse("a transaction for a different amount")
    if built_out * 10_000 < quoted_out * (10_000 - BUILD_DRIFT_BPS):
        refuse("a transaction that delivers much less than quoted")
    try:
        desc = decode_desc(built["data"])
    except KyberError:
        raise
    except Exception:
        refuse("calldata the bot cannot verify")
    if not _same(desc["dstReceiver"], account):
        refuse("a transaction that pays another address")
    if not _same(desc["srcToken"], rs.get("tokenIn")) or not _same(desc["dstToken"], rs.get("tokenOut")):
        refuse("a transaction for other tokens")
    if desc["amount"] != amount_in:
        refuse("a transaction for a different amount")
    if desc["feeReceivers"] or any(desc["feeAmounts"]):
        refuse("a transaction that pays a fee")
    if desc["minReturnAmount"] < min_received(built_out, slippage_bps):
        refuse("a transaction without the slippage limit asked for")
    value = _big(str(built.get("transactionValue") or "0"))
    if value != (amount_in if _same(rs.get("tokenIn"), NATIVE) else 0):
        refuse("a transaction whose ETH value does not match the swap")
