"""The Robinhood Chain indexer: Pons launches, curve trades and graduations, and the swaps of every graduated Pons pool,
read from our own RPC (one cursor ``rh``; backfill from Pons V2's deployment, then live near the head).

Per block range, in order:
1. the factory / launch router / meme hook logs (one request): launches, dev buys, graduations, pool registrations, and
   the hook's fee on every Pons swap;
2. those events applied in (block, log index) order → rh_tokens, rh_pools, rh_insiders; a graduation also pulls that
   launch's curve trades from its own curve contract (launch → graduation, address-filtered): only the ~1 % of launches
   that graduate are ever traded, and their curve phase is all the features need (curve facts, time to graduate);
3. the v4 PoolManager's Initialize / Swap / ModifyLiquidity logs for the Pons pools known by then (topic1 = pool id,
   200 ids per request), each swap matched to its hook fee by (tx, pool) in log order;
4. each swap's trader from the memecoin's own Transfer logs in that transaction (address-filtered, one request per 200
   tokens): the wallet whose token balance rose on a buy (fell on a sell), not the pool manager or a router — the swap
   event itself names the router (measured 2026-09-28: per-swap tx lookups made a 55-day backfill take ~43 days);
5. rh_swaps rows priced in the pool's quote (price_q, virtual quote reserve resq_q, fee_frac); the ETH conversion of
   non-ETH quotes happens per minute in rh/minutes.py with the base asset's mark.

Everything a range writes is committed with its cursor, so a crash replays at most one range (inserts are idempotent).
"""
from __future__ import annotations

import logging

from .. import config
from ..db.connection import transaction
from . import abi, blocktime, pons, scan, v4
from .rpc import RhRpc, logs_rpc

log = logging.getLogger(__name__)

CURSOR = "rh"
IDS_PER_REQUEST = 200
TOKENS_PER_REQUEST = 200
TX_BATCH = 100
TRANSFER = abi.event_topic("Transfer(address,address,uint256)")
PONS_DECIMALS = 18                          # every Pons launch mints 1e9 × 1e18 raw
STABLES = ("USDG", "USDC", "USDT", "PYUSD", "USDE")


# ---------------------------------------------------------------- quote assets
def ensure_asset(conn, rpc, asset: str) -> dict:
    """The rh_assets row of a quote asset (decimals, symbol, class), created on first sight."""
    asset = asset.lower()
    r = conn.execute("SELECT asset, symbol, class, decimals FROM rh_assets WHERE asset = %s", (asset,)).fetchone()
    if r:
        return dict(r)
    if int(asset, 16) == 0:
        row = {"asset": asset, "symbol": "ETH", "class": "eth", "decimals": 18}
    else:
        dec = abi.decode(["uint8"], bytes.fromhex(rpc.eth_call(asset, "0x" + abi.selector("decimals()").hex())[2:]))[0]
        try:
            sym = abi.decode(["string"], bytes.fromhex(rpc.eth_call(asset, "0x" + abi.selector("symbol()").hex())[2:]))[0]
        except Exception:
            sym = asset[:10]
        s = sym.upper()
        cls = "stable" if s in STABLES else "btc" if "BTC" in s else "eth" if s in ("WETH", "ETH") else "stock"
        row = {"asset": asset, "symbol": sym, "class": cls, "decimals": int(dec)}
    conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%(asset)s, %(symbol)s, %(class)s, %(decimals)s) ON CONFLICT (asset) DO NOTHING", row)
    return row


def find_deploy_block(rpc, address: str) -> int:
    """The first block at which ``address`` carries code (binary search over getCode)."""
    lo, hi = 0, rpc.block_number()
    while lo < hi:
        mid = (lo + hi) // 2
        if len(rpc.code(address, hex(mid))) > 2:
            hi = mid
        else:
            lo = mid + 1
        scan.pause()
    return lo


# ---------------------------------------------------------------- one range
def _get_logs(rpc, addresses, lo, hi, topics):
    """eth_getLogs, splitting the range in halves while the RPC's 10,000-logs-per-query cap is hit."""
    try:
        out = rpc.get_logs_multi(addresses, lo, hi, topics)
    except Exception as e:
        if ("exceeds limit" in str(e) or "too many" in str(e).lower()) and hi > lo:
            mid = (lo + hi) // 2
            return _get_logs(rpc, addresses, lo, mid, topics) + _get_logs(rpc, addresses, mid + 1, hi, topics)
        raise
    scan.pause()
    return out


def _order(logs: list[dict]) -> list[dict]:
    return sorted(logs, key=lambda x: (int(x["blockNumber"], 16), int(x["logIndex"], 16)))


def _not_trader() -> set[str]:
    return {a.lower() for a in (config.V4_POOL_MANAGER, config.UNIVERSAL_ROUTER, config.KYBER_ROUTER, config.PERMIT2, config.PONS_HOOK,
                                config.PONS_ROUTER, "0x" + "00" * 20)}


def traders_from_transfers(transfers: list[dict], swaps: list[dict], tokens: dict[str, str]) -> dict[tuple, str]:
    """(tx, pool) → the swapping wallet: the address with the largest net memecoin gain (buy) or loss (sell) in that tx,
    excluding the pool manager, routers and the hook. ``tokens``: pool id → memecoin."""
    skip = _not_trader(); net: dict[tuple, dict[str, int]] = {}
    for t in transfers:
        tok = t["address"].lower(); tx = t["transactionHash"]
        a, b = "0x" + t["topics"][1][-40:].lower(), "0x" + t["topics"][2][-40:].lower(); v = int(t["data"], 16)
        d = net.setdefault((tx, tok), {})
        d[a] = d.get(a, 0) - v; d[b] = d.get(b, 0) + v
    out = {}
    for s in swaps:
        tok = tokens[s["pool_id"]]; d = net.get((s["tx_hash"], tok)) or {}
        cands = [(v, a) for a, v in d.items() if a not in skip and a != tok]
        if not cands:
            continue
        v, a = (max(cands) if s["side"] > 0 else min(cands))
        if (s["side"] > 0 and v > 0) or (s["side"] < 0 and v < 0):
            out[(s["tx_hash"], s["pool_id"])] = a
    return out


def fill_timestamps(rpc, events: list[dict], lo: int, hi: int) -> None:
    """The RPC's ``blockTimestamp`` on logs is usually 0x0 (measured 2026-09-28): every event that lands in a minute bar
    gets its time from rh/blocktime (exact minute, interpolated seconds), never one lookup per block."""
    need = [e for e in events if not e.get("ts")]
    if not need:
        return
    t = blocktime.times(rpc, lo, hi, sorted({e["block"] for e in need}))
    for e in need:
        e["ts"] = t[e["block"]]


def index_curve(conn, rpc, token: str, until_block: int) -> int:
    """A graduating launch's curve trades (its own curve contract, launch → graduation). Idempotent; returns rows written."""
    r = conn.execute("SELECT curve, launch_block, curve_indexed FROM rh_tokens WHERE token = %s", (token,)).fetchone()
    if not r or not r["curve"] or r["launch_block"] is None or r["curve_indexed"]:
        return 0
    router = config.PONS_ROUTER.lower(); n = 0; lo = int(r["launch_block"]); step = scan.MAX_RANGE
    while lo <= until_block:
        hi = min(until_block, lo + step - 1)
        logs = _get_logs(rpc, [r["curve"]], lo, hi, [[pons.TOPIC["CurveBuy"], pons.TOPIC["CurveSell"]]])
        evs = [pons.decode(x) for x in _order(logs)]; evs = [e for e in evs if e]
        fill_timestamps(rpc, evs, lo, hi)
        for e in evs:
            trader = e["recipient"] if e["trader"] == router else e["trader"]
            conn.execute("INSERT INTO rh_curve_trades (block, log_index, tx_hash, ts, token, side, trader, recipient, quote_raw, tokens_raw, fee_raw, tax_raw) "
                         "VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                         (e["block"], e["log_index"], e["tx_hash"], e["ts"], token, e["side"], trader, e["recipient"], e["quote_raw"], e["tokens_raw"], e["fee_raw"], e["tax_raw"]))
            n += 1
        lo = hi + 1
    conn.execute("UPDATE rh_tokens SET curve_indexed = true WHERE token = %s", (token,))
    return n


def apply_pons(conn, rpc, events: list[dict]) -> dict:
    """Pons events of one range, in order. Returns counts."""
    n = {"launches": 0, "curve_trades": 0, "graduations": 0, "pools": 0}
    for e in events:
        ev = e["event"]
        if ev == "TokenLaunched":
            a = ensure_asset(conn, rpc, e["pair_token"])
            conn.execute("INSERT INTO rh_tokens (token, launch_block, launch_ts, launch_tx, creator, curve, quote_asset, quote_class, decimals, supply_raw) "
                         "VALUES (%s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (token) DO NOTHING",
                         (e["token"], e["block"], e["ts"], e["tx_hash"], e["deployer"], e["curve"], a["asset"], a["class"], PONS_DECIMALS, 10 ** 27))
            conn.execute("INSERT INTO rh_assets (asset, class, decimals, graduation_threshold_raw) VALUES (%s, %s, %s, %s) "
                         "ON CONFLICT (asset) DO UPDATE SET graduation_threshold_raw = EXCLUDED.graduation_threshold_raw", (a["asset"], a["class"], a["decimals"], e["threshold_raw"]))
            conn.execute("INSERT INTO rh_insiders (mint, wallet, kind) VALUES (%s, %s, 'dev') ON CONFLICT DO NOTHING", (e["token"], e["deployer"]))
            n["launches"] += 1
        elif ev == "Launched":
            conn.execute("UPDATE rh_tokens SET dev_quote_in_raw = COALESCE(dev_quote_in_raw, 0) + %s, dev_tokens_raw = COALESCE(dev_tokens_raw, 0) + %s WHERE token = %s",
                         (e["quote_in_raw"], e["tokens_out_raw"], e["token"]))
            conn.execute("INSERT INTO rh_insiders (mint, wallet, kind) VALUES (%s, %s, 'dev') ON CONFLICT DO NOTHING", (e["token"], e["recipient"]))
        elif ev == "PoolGraduated":
            conn.execute("UPDATE rh_tokens SET graduated_at = to_timestamp(%s), grad_block = %s, grad_position_id = %s, grad_token_raw = %s, grad_quote_raw = %s, "
                         "status = 'graduated', updated_at = now() WHERE token = %s",
                         (e["ts"], e["block"], e["position_id"], e["token_amount_raw"], e["pair_amount_raw"], e["token"]))
            n["graduations"] += 1
        elif ev == "PoolRegistered":
            q = e["quote_asset"]; tok = e["token"]
            c0, c1 = sorted([q, tok], key=lambda a: int(a, 16))
            conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, fee, tick_spacing, hooks, quote_asset, init_block, init_ts, is_pons) "
                         "VALUES (%s, %s, %s, %s, %s, 0, 200, %s, %s, %s, to_timestamp(%s), true) ON CONFLICT (pool_id) DO UPDATE SET is_pons = true, token = EXCLUDED.token",
                         (e["pool_id"], tok, c0, c1, c0 == tok, config.PONS_HOOK.lower(), q, e["block"], e["ts"]))
            conn.execute("UPDATE rh_tokens SET pool_id = %s, status = 'graduated', graduated_at = COALESCE(graduated_at, to_timestamp(%s)) WHERE token = %s",
                         (e["pool_id"], e["ts"], tok))
            n["pools"] += 1
            n["curve_trades"] += index_curve(conn, rpc, tok, e["block"])
    return n


def apply_v4(conn, rpc, logs: list[dict], fees: list[dict]) -> dict:
    """v4 logs of Pons pools (+ their hook fees) → rh_swaps / rh_liquidity / rh_pools state."""
    pools = {r["pool_id"]: dict(r) for r in conn.execute(
        "SELECT p.pool_id, p.token, p.token_is_0, p.quote_asset, a.decimals AS qdec FROM rh_pools p JOIN rh_assets a ON a.asset = p.quote_asset WHERE p.is_pons").fetchall()}
    fee_q: dict[tuple, list] = {}
    for f in fees:
        fee_q.setdefault((f["tx_hash"], f["pool_id"]), []).append(f)
    evs = [v4.decode(x) for x in _order(logs)]
    evs = [e for e in evs if e is not None]
    if evs:
        fill_timestamps(rpc, evs, min(e["block"] for e in evs), max(e["block"] for e in evs))
    swaps = [e for e in evs if e and e["event"] == "Swap" and e["pool_id"] in pools]
    for s_ in swaps:
        ti0 = bool(pools[s_["pool_id"]]["token_is_0"]); s_["side"] = 1 if (s_["amount0"] if ti0 else s_["amount1"]) > 0 else -1
    toks = sorted({pools[s_["pool_id"]]["token"] for s_ in swaps})
    transfers = []
    if swaps:
        lo_b, hi_b = min(s_["block"] for s_ in swaps), max(s_["block"] for s_ in swaps)
        for i in range(0, len(toks), TOKENS_PER_REQUEST):
            transfers += _get_logs(rpc, toks[i:i + TOKENS_PER_REQUEST], lo_b, hi_b, [TRANSFER])
    senders = traders_from_transfers(transfers, swaps, {k: v["token"] for k, v in pools.items()})
    n = {"swaps": 0, "liquidity": 0}
    last_state: dict[str, dict] = {}
    for e in evs:
        if e is None or e["pool_id"] not in pools:
            continue
        p = pools[e["pool_id"]]
        if e["event"] == "Initialize":
            conn.execute("UPDATE rh_pools SET sqrt_price_x96 = %s, tick = %s, fee = %s, tick_spacing = %s, hooks = %s, updated_block = %s WHERE pool_id = %s",
                         (e["sqrt_price_x96"], e["tick"], e["fee"], e["tick_spacing"], e["hooks"], e["block"], e["pool_id"]))
        elif e["event"] == "ModifyLiquidity":
            conn.execute("INSERT INTO rh_liquidity (block, log_index, tx_hash, ts, pool_id, sender, tick_lower, tick_upper, liquidity_delta, salt) "
                         "VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                         (e["block"], e["log_index"], e["tx_hash"], e["ts"], e["pool_id"], e["sender"], e["tick_lower"], e["tick_upper"], e["liquidity_delta"], e["salt"]))
            n["liquidity"] += 1
        elif e["event"] == "Swap":
            ti0 = bool(p["token_is_0"]); qdec = int(p["qdec"])
            tok_amt, q_amt = (e["amount0"], e["amount1"]) if ti0 else (e["amount1"], e["amount0"])
            side = 1 if tok_amt > 0 else -1                                   # the swapper received tokens: a buy
            fq = fee_q.get((e["tx_hash"], e["pool_id"])) or []
            fee = fq.pop(0) if fq else None
            price_q = v4.token_price_in_quote(e["sqrt_price_x96"], ti0, PONS_DECIMALS, qdec)
            fee_frac = None
            if fee:
                net = abs(tok_amt) if fee["currency"] == p["token"] else abs(q_amt)          # the fee is taken from the incoming currency
                taken = fee["fee_raw"] + fee["tax_raw"]
                fee_frac = taken / (net + taken) if net + taken > 0 else None
            resq_q = v4.quote_reserve(e["sqrt_price_x96"], e["liquidity"], ti0, qdec)
            conn.execute(
                "INSERT INTO rh_swaps (block, log_index, tx_hash, block_hash, ts, pool_id, token, trader, side, token_raw, quote_raw, hook_fee_raw, hook_tax_raw, "
                "sqrt_price_x96, liquidity, tick, price_q, resq_q, fee_frac) VALUES (%s,%s,%s,%s,to_timestamp(%s),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (e["block"], e["log_index"], e["tx_hash"], e["block_hash"], e["ts"], e["pool_id"], p["token"], senders.get((e["tx_hash"], e["pool_id"])), side, abs(tok_amt),
                 abs(q_amt), fee["fee_raw"] if fee else None, fee["tax_raw"] if fee else None, e["sqrt_price_x96"], e["liquidity"], e["tick"], price_q, resq_q, fee_frac))
            last_state[e["pool_id"]] = e; n["swaps"] += 1
    for pid, e in last_state.items():
        conn.execute("UPDATE rh_pools SET sqrt_price_x96 = %s, liquidity = %s, tick = %s, updated_block = %s WHERE pool_id = %s",
                     (e["sqrt_price_x96"], e["liquidity"], e["tick"], e["block"], pid))
    return n


def step(conn, rpc, lo: int, hi: int) -> dict:
    """Index blocks lo..hi (inclusive) into ``conn`` (the caller commits with the cursor)."""
    pons_topics = [[pons.TOPIC[k] for k in (*pons.FACTORY_EVENTS, "Launched", *pons.HOOK_EVENTS)]]
    main = _get_logs(rpc, [config.PONS_FACTORY, config.PONS_ROUTER, config.PONS_HOOK], lo, hi, pons_topics)
    evs = [pons.decode(x) for x in _order(main)]
    evs = [e for e in evs if e is not None]
    fill_timestamps(rpc, [e for e in evs if e["event"] != "HookFeeCollected"], lo, hi)      # fees are matched to swaps, never timed
    fees = [e for e in evs if e["event"] == "HookFeeCollected"]
    n = apply_pons(conn, rpc, [e for e in evs if e["event"] != "HookFeeCollected"])
    ids = [r["pool_id"] for r in conn.execute("SELECT pool_id FROM rh_pools WHERE is_pons ORDER BY pool_id").fetchall()]
    v4_logs = []
    topics0 = [v4.TOPIC["Initialize"], v4.TOPIC["Swap"], v4.TOPIC["ModifyLiquidity"]]
    for i in range(0, len(ids), IDS_PER_REQUEST):
        v4_logs += _get_logs(rpc, [config.V4_POOL_MANAGER], lo, hi, [topics0, ids[i:i + IDS_PER_REQUEST]])
    n.update(apply_v4(conn, rpc, v4_logs, fees))
    n["hook_fees"] = len(fees)
    return n


def start_block(rpc) -> int:
    return config.RH_START_BLOCK or find_deploy_block(rpc, config.PONS_FACTORY)


def run_once(rpc: RhRpc | None = None, *, head_margin: int | None = None, max_ranges: int = 1) -> dict:
    """Advance the cursor by up to ``max_ranges`` ranges, each ending at most ``head − head_margin`` (RH_CONFIRMATIONS).
    Returns {through, head, ranges, counts}."""
    rpc = rpc or logs_rpc()
    margin = config.RH_CONFIRMATIONS if head_margin is None else head_margin
    head = rpc.block_number() - margin
    total: dict = {}; done = 0
    with transaction() as conn:
        at, rng = scan.cursor(conn, CURSOR, -1)
    if at < 0:
        sb = start_block(rpc)
        with transaction() as conn:
            scan.set_cursor(conn, CURSOR, sb - 1, rng, {"start_block": sb}); at = sb - 1
    for _ in range(max_ranges):
        r = scan.next_range(at, rng, head)
        if r is None:
            break
        lo, hi = r
        try:
            with transaction() as conn:
                c = step(conn, rpc, lo, hi)
                rng = scan.grow(rng); at = hi
                scan.set_cursor(conn, CURSOR, at, rng)
            for k, v in c.items():
                total[k] = total.get(k, 0) + v
            done += 1
        except Exception as e:
            rng = scan.shrink(rng) if not scan.is_rate_limit(e) else rng
            with transaction() as conn:
                scan.set_cursor(conn, CURSOR, at, rng)
            log.warning("rh index %d..%d: %s (range now %d)", lo, hi, str(e)[:200], rng)
            import time
            time.sleep(scan.BACKOFF_S)
            break
    return {"through": at, "head": head, "ranges": done, "counts": total}
