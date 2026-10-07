"""Each Pons quote asset's value in ETH per minute (``rh_base_prices``): the base leg's mark, and the conversion that puts
every RH row in ETH.

Reference path: the AMM pools KyberSwap routes 1 unit of the asset into WETH through, restricted to venues whose swap
logs carry the pool price (discovered once, stored as ``rh_assets.ref_path``): on 2026-09-28 USDG → WETH and a stock →
USDG → WETH resolve to Uniswap v3 pools; the unrestricted best route may be an order book, which leaves no log price. Every one of those venues puts the pool's sqrtPriceX96
third in its swap event's data, so a hop's price at a minute is the last swap's sqrt price in (or before) that minute;
the asset's ETH value is the product over its hops. A minute with no swap on some hop carries the last price forward and
is marked ``stale`` once that price is older than ``STALE_S``.
"""
from __future__ import annotations

import json
import logging

from .. import config
from ..db.connection import transaction
from . import abi, kyber as K, scan, v4
from .rpc import logs_rpc

log = logging.getLogger(__name__)

STALE_S = 600
ETH_ASSET = "0x" + "00" * 20
# plain v3-style pools first (address-based, the standard Swap event), then the singleton CL venues (bytes32 pool ids)
REF_SOURCES = ("uniswapv3,slipstream", "uniswapv3,uniswap-v4,slipstream,pancake-infinity-cl,orvex-cl-feemanager")


def _is_id(pool: str) -> bool:
    return len(pool) == 66


def discover(conn, rpc, asset: str, decimals: int, kyber: K.Kyber | None = None) -> list[dict]:
    """The hops from ``asset`` to WETH (``[{pool, emitter, token0, token1, dec0, dec1, token_in}]``), stored on rh_assets."""
    ky = kyber or K.Kyber()
    paths = []
    for sources in REF_SOURCES:                                   # AMM venues only: an order book leaves no price in the logs
        try:
            paths = ky.get_route(asset, config.RH_WETH, 10 ** decimals, included_sources=sources)["routeSummary"].get("route") or []
        except K.KyberError:
            paths = []
        if paths and paths[0]:
            break
    if not paths or not paths[0]:
        raise K.KyberError(f"no AMM route for {asset}")
    hops = []
    for h in paths[0]:                                             # the first split carries the path; splits share hops
        tin, tout, pool = h["tokenIn"].lower(), h["tokenOut"].lower(), h["pool"].lower()
        t0, t1 = sorted([tin, tout], key=lambda a: int(a, 16))
        if _is_id(pool):                                           # a singleton manager's pool id: find the manager from its logs
            head = rpc.block_number(); emitter = None; lo = head - 200_000
            while emitter is None and lo > head - 2_000_000:
                got = rpc.get_logs_multi(None, lo, lo + 20_000, [None, pool]); scan.pause()
                emitter = next((g["address"].lower() for g in got if len(g["data"]) >= 2 + 64 * 3), None); lo -= 20_000
            if emitter is None:
                raise K.KyberError(f"no swaps seen for reference pool {pool}")
        else:
            emitter = pool
        dec = {}
        for t in (t0, t1):
            dec[t] = 18 if int(t, 16) == 0 else abi.decode(["uint8"], bytes.fromhex(rpc.eth_call(t, "0x" + abi.selector("decimals()").hex())[2:]))[0]
        hops.append({"pool": pool, "emitter": emitter, "token0": t0, "token1": t1, "dec0": dec[t0], "dec1": dec[t1], "token_in": tin})
    conn.execute("UPDATE rh_assets SET ref_path = %s, ref_pool = %s, ref_kind = %s, updated_at = now() WHERE asset = %s",
                 (json.dumps(hops), hops[0]["pool"], paths[0][0].get("poolType"), asset.lower()))
    return hops


def hop_price(h: dict, sqrt_price_x96: int) -> float:
    """Whole token_out per whole token_in on this hop."""
    p10 = v4.price1_per_0(sqrt_price_x96, h["dec0"], h["dec1"])     # token1 per token0
    return p10 if h["token_in"] == h["token0"] else (1.0 / p10 if p10 > 0 else float("nan"))


SWAP_V3 = abi.event_topic("Swap(address,address,int256,int256,uint160,uint128,int24)")      # Uniswap v3 and Slipstream pools


def _logs(rpc, addresses, lo: int, hi: int, topics) -> list[dict]:
    from .index import _get_logs
    return _get_logs(rpc, addresses, lo, hi, topics)


def _swap_logs(rpc, h: dict, lo: int, hi: int) -> list[dict]:
    if _is_id(h["pool"]):
        return _logs(rpc, [h["emitter"]], lo, hi, [None, h["pool"]])
    return _logs(rpc, [h["emitter"]], lo, hi, [SWAP_V3])


def fetch_pool_logs(rpc, pools: dict, lo: int, hi: int) -> dict:
    """Swap logs of every reference pool in ``pools`` (``_hop_key`` → hop) over lo..hi, in as few queries as the venues
    allow: one for all the standalone pool contracts (one address list, one Swap topic) and one per singleton manager
    (its pool ids as the topic-1 list). 2026-10-07: 69 pools queried one by one made a live pass take 1–3.5 min."""
    out = {k: [] for k in pools}
    plain = sorted({h["emitter"].lower() for h in pools.values() if not _is_id(h["pool"])})
    if plain:
        for g in _logs(rpc, plain, lo, hi, [SWAP_V3]):
            k = (g["address"].lower(), g["address"].lower())
            if k in out:
                out[k].append(g)
    managers: dict[str, list[str]] = {}
    for h in pools.values():
        if _is_id(h["pool"]):
            managers.setdefault(h["emitter"].lower(), []).append(h["pool"].lower())
    for em, ids in sorted(managers.items()):
        for g in _logs(rpc, [em], lo, hi, [None, sorted(set(ids))]):
            tp = g.get("topics") or []
            k = (em, tp[1].lower()) if len(tp) > 1 else None
            if k in out:
                out[k].append(g)
    return out


def _hop_key(h: dict) -> tuple[str, str]:
    return h["emitter"].lower(), h["pool"].lower()


def scan_asset(conn, rpc, asset: str, hops: list[dict], lo: int, hi: int, state: dict, times, logs_by_hop: dict | None = None) -> int:
    """Minute closes of ``asset`` over blocks lo..hi → rh_base_prices. ``state`` carries each hop's last (t, sqrtP) across
    ranges; ``times(blocks) -> {block: ts}`` times the swaps. ``logs_by_hop``: swap logs already fetched per hop
    (``_hop_key``), shared by every asset whose path uses that pool. Returns minutes written."""
    per_hop = []
    for i, h in enumerate(hops):
        got = logs_by_hop[_hop_key(h)] if logs_by_hop is not None else _swap_logs(rpc, h, lo, hi)
        logs = [g for g in got if len(g["data"]) >= 2 + 64 * 3 and lo <= int(g["blockNumber"], 16) <= hi]
        ts = times(sorted({int(g["blockNumber"], 16) for g in logs})) if logs else {}
        seq = sorted((ts[int(g["blockNumber"], 16)], int(g["logIndex"], 16), v4.sqrt_price_word(g)) for g in logs)
        per_hop.append(seq)
    minutes = sorted({int(t // 60) * 60 for seq in per_hop for t, _, _ in seq})
    rows = []
    idx = [0] * len(hops)
    for m in minutes:
        end = m + 60; stale = False; price = 1.0; n_obs = 0
        for i, h in enumerate(hops):
            seq = per_hop[i]
            while idx[i] < len(seq) and seq[idx[i]][0] < end:
                t, _, sp = seq[idx[i]]; state[str(i)] = [t, sp]; idx[i] += 1; n_obs += 1
            last = state.get(str(i))
            if last is None:
                price = None; break
            if end - last[0] > STALE_S:
                stale = True
            price *= hop_price(h, int(last[1]))
        if price is not None and price > 0:
            rows.append((asset, m, price, n_obs, stale))
    if rows:
        conn.cursor().executemany("INSERT INTO rh_base_prices (asset, ts, price_eth, n_obs, stale) VALUES (%s, to_timestamp(%s), %s, %s, %s) "
                                  "ON CONFLICT (asset, ts) DO UPDATE SET price_eth = EXCLUDED.price_eth, n_obs = EXCLUDED.n_obs, stale = EXCLUDED.stale", rows)
    return len(rows)


def base_eth(conn, asset: str | None, ts) -> tuple[float | None, bool]:
    """(ETH per whole unit of ``asset`` at minute ``ts``, stale). ETH itself is 1. The last mark at or before the minute;
    stale when older than ``STALE_S`` or flagged stale."""
    if asset is None or int(asset, 16) == 0:
        return 1.0, False
    r = conn.execute("SELECT ts, price_eth, stale FROM rh_base_prices WHERE asset = %s AND ts <= %s ORDER BY ts DESC LIMIT 1", (asset.lower(), ts)).fetchone()
    if r is None:
        return None, True
    age = (ts - r["ts"]).total_seconds() if hasattr(ts, "tzinfo") else 0.0
    return float(r["price_eth"]), bool(r["stale"]) or age > STALE_S


def run_once(max_ranges: int = 1) -> dict:
    """Advance every traded quote asset's price cursor (``price:<asset>``) toward the index cursor, all in lockstep: each
    round takes the range after the laggard's cursor, fetches every distinct reference pool's swaps in it once (the
    stock paths share USDG → WETH), times them once, and moves every asset whose cursor is inside the range."""
    from . import blocktime
    rpc = logs_rpc(); out: dict = {}
    with transaction() as conn:
        top = conn.execute("SELECT block, detail FROM rh_scan WHERE name = 'rh'").fetchone()
        rows = conn.execute("SELECT DISTINCT a.asset, a.decimals, a.ref_path FROM rh_assets a JOIN rh_pools p ON p.quote_asset = a.asset "
                            "WHERE p.is_pons AND a.class <> 'eth'").fetchall()
    if not top:
        return out
    through = int(top["block"])
    start = int(((top["detail"] or {}).get("start_block")) or config.RH_START_BLOCK or through)      # where the index began
    assets = []
    for a in rows:
        try:
            with transaction() as conn:
                hops = a["ref_path"] if a["ref_path"] else discover(conn, rpc, a["asset"], int(a["decimals"]))
                if isinstance(hops, str):
                    hops = json.loads(hops)
                # marks are needed from the first graduation quoted in the asset (a day of history before it for the carry-forward)
                fg = conn.execute("SELECT min(grad_block) AS b FROM rh_tokens WHERE quote_asset = %s AND grad_block IS NOT NULL", (a["asset"],)).fetchone()["b"]
                first = max(start, int(fg) - 800_000) if fg is not None else through
                at, rng = scan.cursor(conn, f"price:{a['asset']}", first - 1)
                r = conn.execute("SELECT detail FROM rh_scan WHERE name = %s", (f"price:{a['asset']}",)).fetchone()
                state = ((r["detail"] or {}) if r else {}).get("state", {})
            assets.append({"asset": a["asset"], "hops": hops, "at": at, "rng": rng, "state": state})
        except Exception as e:
            log.warning("rh prices %s: %s", a["asset"], str(e)[:200])
    for _ in range(max_ranges):
        active = [a for a in assets if a["at"] < through]
        if not active:
            break
        lo = min(a["at"] for a in active) + 1
        rng = min(a["rng"] for a in active if a["at"] + 1 == lo)
        hi = min(through, lo + rng - 1)
        group = [a for a in active if a["at"] < hi]
        try:
            pools = {}
            for a in group:
                for h in a["hops"]:
                    pools.setdefault(_hop_key(h), h)
            logs = fetch_pool_logs(rpc, pools, lo, hi)
            ts = {int(g["blockNumber"], 16): int(g["blockTimestamp"], 16) for got in logs.values() for g in got
                  if int(g.get("blockTimestamp") or "0x0", 16)}                    # HyperSync logs carry their block's time
            blocks = sorted({int(g["blockNumber"], 16) for got in logs.values() for g in got} - set(ts))
            if blocks:
                ts.update(blocktime.times(rpc, lo, hi, blocks))
            with transaction() as conn:
                for a in group:
                    n = scan_asset(conn, rpc, a["asset"], a["hops"], a["at"] + 1, hi, a["state"], lambda bs: {b: ts[b] for b in bs}, logs)
                    a["at"] = hi; a["rng"] = scan.grow(rng)
                    scan.set_cursor(conn, f"price:{a['asset']}", hi, a["rng"], {"state": a["state"]})
                    out[a["asset"]] = out.get(a["asset"], 0) + n
        except Exception as e:
            log.warning("rh prices %s..%s: %s", lo, hi, str(e)[:200])
            for a in group:
                a["rng"] = scan.shrink(a["rng"])
            break
    return out
