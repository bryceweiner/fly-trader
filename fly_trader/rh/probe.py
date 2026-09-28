"""``fly-trader rh-probe``: a read-only check of everything the RH trading path relies on, before any key is used.

1. the RPC answers the expected chain id and every configured contract carries code;
2. for the most recent Pons graduations (per quote class): does KyberSwap route a buy and a sell, what gas does it
   quote, and does the UniversalRouter v4 fallback quote and simulate a buy (eth_call from a contract holding ETH);
3. the gas units measured feed ``RH_GAS_RESERVE_ETH`` and the cost model's per-swap transaction fee.
"""
from __future__ import annotations

import statistics
import time

from .. import config
from . import abi, kyber as K, univ4 as U
from .rpc import RhRpc

CONTRACTS = ("PONS_FACTORY", "PONS_ROUTER", "PONS_HOOK", "PONS_LOCKER", "PONS_GRAD_EXECUTOR", "V4_POOL_MANAGER", "V4_QUOTER", "V4_STATE_VIEW",
             "UNIVERSAL_ROUTER", "PERMIT2", "V3_FACTORY", "V3_QUOTER_V2", "RH_WETH", "RH_USDG", "KYBER_ROUTER")
POOL_REGISTERED = "PoolRegistered(bytes32,address,address,address)"


def recent_graduations(rpc: RhRpc, blocks: int = 400_000, step: int = 50_000) -> list[dict]:
    head = rpc.block_number(); t = abi.event_topic(POOL_REGISTERED); out = []; b = head - blocks
    while b < head:
        try:
            logs = rpc.get_logs(config.PONS_HOOK, b, min(head, b + step), [t]); b += step + 1
        except Exception:
            time.sleep(2.0); continue
        for lg in logs:
            meme, quote, creator = abi.decode(["address", "address", "address"], bytes.fromhex(lg["data"][2:]))
            out.append({"pool_id": lg["topics"][1], "meme": meme, "quote": quote, "creator": creator, "block": int(lg["blockNumber"], 16)})
    return out


def run(sample: int = 12, amount_wei: int = 10 ** 14) -> dict:
    rpc = RhRpc(); rep = {"chain_id": rpc.chain(), "expected": config.RH_EXPECTED_CHAIN_ID, "contracts": {}, "pools": []}
    for name in CONTRACTS:
        a = getattr(config, name); rep["contracts"][name] = {"address": a, "code_bytes": max(0, len(rpc.code(a)) // 2 - 1)}
    grads = recent_graduations(rpc)[-sample:]
    ky = K.Kyber(); gas_k = []
    for g in grads:
        row = {"meme": g["meme"], "quote": "ETH" if int(g["quote"], 16) == 0 else g["quote"]}
        try:
            r = ky.get_route(K.NATIVE, g["meme"], amount_wei); row["kyber_buy_gas"] = int(r["routeSummary"].get("gas") or 0); gas_k.append(row["kyber_buy_gas"])
            out = int(r["routeSummary"]["amountOut"])
            s = ky.get_route(g["meme"], K.NATIVE, out); row["kyber_round_trip"] = int(s["routeSummary"]["amountOut"]) / amount_wei
        except K.KyberError as e:
            row["kyber"] = str(e)[:80]
        if row["quote"] == "ETH":
            try:
                key = U.pool_key(U.ZERO, g["meme"]); q, gq = U.quote(rpc, key, U.ZERO, amount_wei)
                data, value = U.swap_calldata(key, U.ZERO, amount_wei, q * 95 // 100)
                rpc.call("eth_call", [{"from": config.RH_WETH, "to": config.UNIVERSAL_ROUTER, "data": "0x" + data.hex(), "value": hex(value)}, "latest"])
                row["ur_buy"] = "simulates"; row["ur_quote_gas"] = gq
            except Exception as e:
                row["ur_buy"] = str(e)[:80]
        rep["pools"].append(row); time.sleep(0.3)
    base = rpc.base_fee()
    rep["base_fee_gwei"] = base / 1e9
    if gas_k:
        med = statistics.median(gas_k)
        rep["kyber_gas_median"] = med
        rep["swap_fee_eth_at_base"] = med * base / 1e18
        rep["suggested_gas_reserve_eth"] = round(max(config.RH_GAS_RESERVE_ETH, 200 * 2 * med * base / 1e18), 6)
    trips = [p["kyber_round_trip"] for p in rep["pools"] if "kyber_round_trip" in p]
    if trips:
        rep["round_trip_median"] = statistics.median(trips)
    return rep
