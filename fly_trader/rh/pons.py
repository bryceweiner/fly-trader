"""Pons V2 events (Robinhood Chain), pinned from chain data on 2026-09-28: topic0 = keccak of each signature below matched
the logs the factory, launch router, curves, meme hook and launch locker emit (the explorer's verified sources sit behind a
bot wall). Argument roles were read from real transactions (a launch: mint Transfer 0 → curve; the router's buy for the
deployer; the curve's buy/sell with a 1 % fee word).

Emitters: TokenLaunched / LaunchSwept / PoolGraduated / GraduationTokensPermanentlyLocked — the V2 factory; Launched — the
launch router (the deployer's own first buy); CurveBuy / CurveSell — each launch's own curve contract; PoolRegistered /
HookFeeCollected — the meme hook; TokenSupplyLocked / PositionLocked — the launch locker.
"""
from __future__ import annotations

from . import abi

ZERO = "0x" + "00" * 20
SIGS = {
    "TokenLaunched": "TokenLaunched(address,address,address,address,uint256,uint256)",      # token, curve, deployer (indexed); pairToken, ?, graduationThreshold
    "Launched": "Launched(address,address,address,address,uint256,uint256)",                # token, curve, deployer (indexed); recipient, quoteIn, tokensOut
    "CurveBuy": "CurveBuy(address,address,uint256,uint256,uint256,uint256)",                # buyer, recipient (indexed); quoteIn, tokensOut, fee, tax
    "CurveSell": "CurveSell(address,address,uint256,uint256,uint256,uint256)",              # seller, recipient (indexed); tokensIn, quoteOut, fee, tax
    "LaunchSwept": "LaunchSwept(address,uint256,uint256)",                                  # token (indexed); quoteOut, tokenOut
    "PoolGraduated": "PoolGraduated(address,uint256,uint256,uint256)",                      # token (indexed); positionId, tokenAmount, pairTokenAmount
    "GraduationTokensPermanentlyLocked": "GraduationTokensPermanentlyLocked(address,uint256)",
    "PoolRegistered": "PoolRegistered(bytes32,address,address,address)",                    # poolId (indexed); memecoin, quoteToken, creator
    "HookFeeCollected": "HookFeeCollected(bytes32,address,uint256,uint256)",                # poolId (indexed); currency, feeAmount, taxAmount
    "TokenSupplyLocked": "TokenSupplyLocked(address,uint256)",
    "PositionLocked": "PositionLocked(address,uint256)",
}
TOPIC = {k: abi.event_topic(v) for k, v in SIGS.items()}
NAME = {v: k for k, v in TOPIC.items()}
FACTORY_EVENTS = ("TokenLaunched", "LaunchSwept", "PoolGraduated", "GraduationTokensPermanentlyLocked")
CURVE_EVENTS = ("CurveBuy", "CurveSell")
HOOK_EVENTS = ("PoolRegistered", "HookFeeCollected")


def _addr_topic(t: str) -> str:
    return "0x" + t[-40:].lower()


def _words(data: str, types: list[str]) -> list:
    return abi.decode(types, bytes.fromhex(data[2:] if data.startswith("0x") else data))


def decode(log: dict) -> dict | None:
    """One Pons log → a dict with ``event`` and its fields (addresses lowercase), or None for an event we do not use."""
    name = NAME.get(log["topics"][0])
    if name is None:
        return None
    t, d = log["topics"], log["data"]
    base = {"event": name, "emitter": log["address"].lower(), "block": int(log["blockNumber"], 16), "log_index": int(log["logIndex"], 16),
            "tx_hash": log["transactionHash"], "ts": int(log.get("blockTimestamp") or "0x0", 16) or None}
    if name == "TokenLaunched":
        pair, x, thr = _words(d, ["address", "uint256", "uint256"])
        return {**base, "token": _addr_topic(t[1]), "curve": _addr_topic(t[2]), "deployer": _addr_topic(t[3]), "pair_token": pair, "word1": x, "threshold_raw": thr}
    if name == "Launched":
        rcpt, qin, tout = _words(d, ["address", "uint256", "uint256"])
        return {**base, "token": _addr_topic(t[1]), "curve": _addr_topic(t[2]), "deployer": _addr_topic(t[3]), "recipient": rcpt, "quote_in_raw": qin, "tokens_out_raw": tout}
    if name in CURVE_EVENTS:
        a, b, fee, tax = _words(d, ["uint256"] * 4)
        side = 1 if name == "CurveBuy" else -1
        return {**base, "side": side, "trader": _addr_topic(t[1]), "recipient": _addr_topic(t[2]),
                "quote_raw": a if side > 0 else b, "tokens_raw": b if side > 0 else a, "fee_raw": fee, "tax_raw": tax}
    if name == "LaunchSwept":
        q, tok = _words(d, ["uint256", "uint256"])
        return {**base, "token": _addr_topic(t[1]), "quote_out_raw": q, "token_out_raw": tok}
    if name == "PoolGraduated":
        pos, tok, q = _words(d, ["uint256", "uint256", "uint256"])
        return {**base, "token": _addr_topic(t[1]), "position_id": pos, "token_amount_raw": tok, "pair_amount_raw": q}
    if name in ("GraduationTokensPermanentlyLocked", "TokenSupplyLocked"):
        return {**base, "token": _addr_topic(t[1]), "amount_raw": _words(d, ["uint256"])[0]}
    if name == "PoolRegistered":
        meme, quote, creator = _words(d, ["address", "address", "address"])
        return {**base, "pool_id": t[1].lower(), "token": meme, "quote_asset": quote, "creator": creator}
    if name == "HookFeeCollected":
        cur, fee, tax = _words(d, ["address", "uint256", "uint256"])
        return {**base, "pool_id": t[1].lower(), "currency": cur, "fee_raw": fee, "tax_raw": tax}
    if name == "PositionLocked":
        return {**base, "owner": _addr_topic(t[1]), "token_id": int(t[2], 16)}
    return None
