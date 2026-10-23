"""Robinhood Chain's other Uniswap v4 pools (2026-10-23): a Pons coin's own pools outside its Pons pool, and plain v4
memecoins, traded like the Pons pools once they qualify.

Measured 2026-10-23: 1.07 million v4 pools had been initialized on the chain, 700,000 of them without a hook. In one
hour 1,480 pools traded (40,960 swaps, 3,375 of them in Pons pools); 85 hookless pools pairing a coin with ETH, WETH or
USDG cleared the bot's floors (17,051 swaps), and 64 hooked ones did (4,810).

Every PoolManager log already reaches the stream each live pass (rh/prefetch.py), and the index step dropped those
outside the Pons pools (rh/index.step). Nothing below runs in front of a minute's publish:

1. ``note`` (rh/stream.one_pass, once the pass's minutes are written): the pass's Initialize and Swap logs are kept.
2. ``slot`` (rh/stream.v4_slot: mid-minute, once the newest minute is written, while no live order waits):
   - the kept Initialize logs go to ``rh_v4_keys``, every pool's key (pools initialized before the stream kept them come
     from ``fly-trader rh-v4-backfill``); the kept swaps to each pool's activity (``_ACT``, 15-minute buckets);
   - a pool whose last half hour cleared the bot's floors (reserve and 15-minute volume, in ETH) is classified from its
     key (``classify``) and registered in ``rh_pools`` (kind 'pons_other' or 'v4', not yet ``indexed``), and a plain
     coin is queued in ``rh_v4_tokens``;
   - ``RH_V4_CHECKS_PER_SLOT`` plain coins have their contract checked (rh/token_check.py), the busiest first;
   - a qualifying pool goes live: its coin's rh_meta row (rh/meta.write_v4) and ``indexed`` with ``live_from`` (the
     next block the stream indexes) in one commit, so the next pass indexes its swaps with the Pons pools'.
3. ``backfill_main`` (``fly-trader rh-v4-backfill``, its own process): every Initialize since block 0 into
   ``rh_v4_keys``, then each live pool's swaps before ``live_from`` with its coin's minutes over them, and a plain
   coin's first-hour outcomes once its history is complete.

Only the slot's contract checks reach the network: the RPC, each read once with a short timeout and none after the
slot's end (``SlotRpc``). A rate limit or the slot's end leaves the rest for the next minute.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..vault.evm import EvmRpc, EvmRpcError
from . import token_check, v4
from .rpc import RhRpc, _rate_limited

log = logging.getLogger(__name__)
ZERO = "0x" + "00" * 20
DYNAMIC_FEE = 0x800000          # a v4 pool's fee field with this flag: the LP fee is set per swap (the Swap event's fee)
BUCKET_S = 900                  # a pool's activity in 15-minute buckets, the volume floor's window
MIN_SWAPS = 5                   # swaps over the current and the previous bucket before a pool is looked at
RECHECK_S = 900.0               # a pool below the floors, or without a known key, is looked at again after this long
FORGET_S = 86400.0              # a pool that is no coin's pool (a hook, two quote assets): after this long
KEEP_SWAPS = 200_000            # swap logs kept between slots (~5 h at 2026-10-23's rate); older ones are dropped
SLOT_TIMEOUT_S = 5.0            # one RPC read of a slot
MAX_ATTEMPTS = 5                # contract checks of a coin before it is given up ('error')
HISTORY_CHUNK = 100_000         # backfill_main: blocks per history step (~3 h of chain)
KEYS_STEP = 2_000_000           # backfill_main: blocks per key-import step
_STASH: dict = {"inits": [], "swaps": []}
_ACT: dict[str, list] = {}      # pool id -> [bucket, n, |amount0|, |amount1|, n before, |amount0| before, |amount1| before, sqrtPriceX96, tick, last ts]
_KNOWN: set[str] = set()        # pools found in rh_pools (Pons pools, registered pools)
_SEEN: dict[str, float] = {}    # pools looked at and not registered -> when to look again
_RPC: dict = {"rpc": None}


class SlotOver(RuntimeError):
    """The slot's time is up, or the RPC refused it a read: the rest waits for the next minute."""


class SlotRpc(RhRpc):
    """The RPC for a slot's contract checks: each read once (no rate-limit waits), with a short timeout, and none after
    ``deadline`` (epoch s); a rate limit or a timeout ends the slot's reads (``SlotOver``)."""

    def __init__(self, url: str):
        super().__init__(url, service="rh_v4", timeout=SLOT_TIMEOUT_S)
        self.deadline = 0.0

    def _guard(self, what: str, fn):
        if time.time() > self.deadline:
            raise SlotOver(f"the slot ended before {what}")
        try:
            return fn()
        except EvmRpcError as e:
            if _rate_limited(e) or "Timeout" in str(e):
                raise SlotOver(f"{what}: {str(e)[:120]}") from None
            raise

    def call(self, method: str, params: list | None = None):
        return self._guard(method, lambda: EvmRpc.call(self, method, params))

    def batch(self, calls: list[tuple[str, list]]) -> list:
        return self._guard("a batch", lambda: self._batch(calls))


def slot_rpc(deadline: float) -> SlotRpc:
    r = _RPC["rpc"]
    if r is None:
        r = _RPC["rpc"] = SlotRpc(config.RH_RPC_URL_LOGS or config.RH_RPC_URL)
    r.deadline = deadline
    return r


# ---------------------------------------------------------------- what qualifies
def _weth() -> str:
    return config.RH_WETH.lower()


def quote_ok_v4(asset: str) -> bool:
    """A plain memecoin's quote: ETH, WETH or one of ``RH_V4_QUOTES``."""
    a = asset.lower()
    return a in (ZERO, _weth()) or a in config.RH_V4_QUOTES


def hook_ok(hooks: str | None) -> bool:
    h = (hooks or ZERO).lower()
    return h == ZERO or h in config.RH_V4_HOOKS


def classify(key: dict, assets: dict[str, dict], pons: dict[str, str]) -> dict | None:
    """{"token", "quote", "kind"} for a pool key, or None (a Pons pool, a pair of two quote assets or of two unknown
    tokens, a quote with no ETH mark, a hook that is not allowed). ``assets``: rh_assets by address (``priced``: class
    'eth' or a reference path); ``pons``: the Pons launches among the pair -> status. A Pons coin's other pool counts once
    the coin graduated; a Pons coin is never a plain v4 coin."""
    c0, c1 = key["currency0"].lower(), key["currency1"].lower(); hooks = (key.get("hooks") or ZERO).lower()
    if hooks == config.PONS_HOOK.lower() or not hook_ok(hooks):
        return None
    for tok, q in ((c0, c1), (c1, c0)):
        if pons.get(tok) == "graduated" and q not in pons and (q in (ZERO, _weth()) or (assets.get(q) or {}).get("priced")):
            return {"token": tok, "quote": q, "kind": "pons_other"}
    if c0 in pons or c1 in pons:
        return None
    q0, q1 = quote_ok_v4(c0), quote_ok_v4(c1)
    if q0 == q1:
        return None
    q, tok = (c0, c1) if q0 else (c1, c0)
    if tok in assets or tok == _weth():
        return None                                              # a quote asset (stable, stock, BTC) is not a memecoin
    return {"token": tok, "quote": q, "kind": "v4"}


def floors() -> tuple[float, float]:
    """(reserve, 15-minute volume) in ETH: the RH book's entry floors (markets.RH, train/decisions)."""
    from ..train.decisions import MIN_RESQ_SOL, MIN_VOL_15M_SOL
    return MIN_RESQ_SOL * config.RH_ETH_PER_SOL, MIN_VOL_15M_SOL * config.RH_ETH_PER_SOL


def activity(a: list, bucket: int) -> tuple[int, int, int]:
    """(swaps, max |amount0|, max |amount1|) of a pool's buckets within the last half hour (``_ACT`` row)."""
    cur = a[0] >= bucket - 1; prev = a[0] - 1 >= bucket - 1
    n = (a[1] if cur else 0) + (a[4] if prev else 0)
    return n, max(a[2] if cur else 0, a[5] if prev else 0), max(a[3] if cur else 0, a[6] if prev else 0)


def clears(a: list, bucket: int, token_is_0: bool, qdec: int, base: float) -> tuple[bool, float, float]:
    """(clears the floors, reserve ETH, best 15-minute volume ETH) of a pool from its activity and its last swap."""
    _n, v0, v1 = activity(a, bucket)
    if not a[7] or not a[8]:
        return False, 0.0, 0.0                                   # no price or no liquidity after its last swap
    resq = v4.quote_reserve(int(a[7]), int(a[8]), token_is_0, qdec) * base
    vol = (v1 if token_is_0 else v0) / 10 ** qdec * base
    min_resq, min_vol = floors()
    return resq >= min_resq and vol >= min_vol, resq, vol


# ---------------------------------------------------------------- the stream's pass and the slot
def note(logs: list[dict]) -> None:
    """Keep a pass's PoolManager Initialize and Swap logs for the next slot (rh/stream.one_pass, after its minutes)."""
    pm = config.V4_POOL_MANAGER.lower(); init, swap = v4.TOPIC["Initialize"], v4.TOPIC["Swap"]
    for g in logs:
        t = g.get("topics") or ()
        if not t or (g.get("address") or "").lower() != pm:
            continue
        t0 = t[0].lower()
        if t0 == swap:
            _STASH["swaps"].append(g)
        elif t0 == init:
            _STASH["inits"].append(g)
    if len(_STASH["swaps"]) > KEEP_SWAPS:
        del _STASH["swaps"][:len(_STASH["swaps"]) - KEEP_SWAPS]


def tally(swaps: list[dict], now: float) -> int:
    """Kept swap logs into each pool's activity (the current 15-minute bucket and the one before); idle pools dropped."""
    n = 0
    for g in swaps:
        e = v4.decode(g)
        if not e or e["event"] != "Swap":
            continue
        ts = e["ts"] or now; b = int(ts // BUCKET_S); a = _ACT.get(e["pool_id"])
        if a is None:
            a = _ACT[e["pool_id"]] = [b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]
        if b > a[0]:
            a[4:7] = a[1:4] if b == a[0] + 1 else [0, 0, 0]
            a[0], a[1], a[2], a[3] = b, 0, 0, 0
        elif b < a[0]:
            continue
        a[1] += 1; a[2] += abs(int(e["amount0"])); a[3] += abs(int(e["amount1"]))
        a[7], a[8], a[9], a[10] = e["sqrt_price_x96"], e["liquidity"], e["tick"], ts
        n += 1
    cut = int(now // BUCKET_S) - 1
    for pid in [p for p, a in _ACT.items() if a[0] < cut]:
        del _ACT[pid]
    return n


def save_keys(conn, rpc, inits: list[dict]) -> int:
    """Decoded Initialize events -> ``rh_v4_keys`` (all but the Pons hook's pools: those are indexed as Pons pools)."""
    pons_hook = config.PONS_HOOK.lower()
    evs = [e for e in inits if (e.get("hooks") or ZERO).lower() != pons_hook]
    if not evs:
        return 0
    if any(not e.get("ts") for e in evs):                       # logs the RPC served while HyperSync was down carry no time
        from . import index
        index.fill_timestamps(rpc, evs, min(e["block"] for e in evs), max(e["block"] for e in evs))
    conn.cursor().executemany(
        "INSERT INTO rh_v4_keys (pool_id, currency0, currency1, fee, tick_spacing, hooks, init_block, init_ts, init_tx, sqrt_price_x96, tick) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), %s, %s, %s) ON CONFLICT (pool_id) DO NOTHING",
        [(e["pool_id"], e["currency0"].lower(), e["currency1"].lower(), e.get("fee"), e.get("tick_spacing"),
          None if (e.get("hooks") or ZERO).lower() == ZERO else e["hooks"].lower(), e["block"], e.get("ts"), e.get("tx_hash"),
          e.get("sqrt_price_x96"), e.get("tick")) for e in evs])
    return len(evs)


def _assets(conn) -> dict[str, dict]:
    return {r["asset"]: {"class": r["class"], "decimals": r["decimals"], "priced": r["class"] == "eth" or bool(r["has_path"])}
            for r in conn.execute("SELECT asset, class, decimals, ref_path IS NOT NULL AS has_path FROM rh_assets").fetchall()}


def discover(conn, now: float) -> dict:
    """Register the active pools that cleared the floors over the last half hour (``classify`` on their keys)."""
    from . import prices
    b = int(now // BUCKET_S); out = {"looked": 0, "registered": 0, "no_key": 0, "below_floors": 0, "not_coin_pools": 0}
    for pid in [p for p, when in _SEEN.items() if when <= now]:
        del _SEEN[pid]
    todo = [pid for pid, a in _ACT.items() if pid not in _KNOWN and pid not in _SEEN and activity(a, b)[0] >= MIN_SWAPS]
    if not todo:
        return out
    known = {r["pool_id"] for r in conn.execute("SELECT pool_id FROM rh_pools WHERE pool_id = ANY(%s)", (todo,)).fetchall()}
    _KNOWN.update(known)
    todo = [p for p in todo if p not in known]
    keys = {r["pool_id"]: dict(r) for r in conn.execute("SELECT * FROM rh_v4_keys WHERE pool_id = ANY(%s)", (todo,)).fetchall()}
    side = sorted({k["currency0"] for k in keys.values()} | {k["currency1"] for k in keys.values()})
    pons = {r["token"]: r["status"] for r in conn.execute("SELECT token, status FROM rh_tokens WHERE token = ANY(%s)", (side,)).fetchall()} if side else {}
    assets = _assets(conn); at = datetime.fromtimestamp(now, timezone.utc)
    for pid in todo:
        out["looked"] += 1; k = keys.get(pid)
        if k is None:
            _SEEN[pid] = now + RECHECK_S; out["no_key"] += 1; continue                  # initialized before the stream kept keys
        c = classify(k, assets, pons)
        if c is None:
            _SEEN[pid] = now + FORGET_S; out["not_coin_pools"] += 1; continue
        if c["quote"] == _weth() and c["quote"] not in assets:
            conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%s, 'WETH', 'eth', 18) ON CONFLICT (asset) DO NOTHING", (c["quote"],))
            assets = _assets(conn)
        qa = assets.get(c["quote"]) or ({"decimals": 18} if c["quote"] == ZERO else None)
        base = prices.base_eth(conn, c["quote"], at)[0] if qa else None
        if qa is None or base is None:
            _SEEN[pid] = now + RECHECK_S; out["below_floors"] += 1; continue
        ti0 = k["currency0"].lower() == c["token"]
        ok, resq, vol = clears(_ACT[pid], b, ti0, int(qa["decimals"]), float(base))
        if not ok:
            _SEEN[pid] = now + RECHECK_S; out["below_floors"] += 1; continue
        register(conn, k, c, activity(_ACT[pid], b)[0])
        _KNOWN.add(pid); out["registered"] += 1
        log.info("rh v4: %s pool %s of %s registered (reserve %.2f ETH, 15-min volume %.2f ETH)", c["kind"], pid[:10], c["token"][:10], resq, vol)
    return out


def register(conn, key: dict, c: dict, swaps: int) -> None:
    """A qualifying pool into ``rh_pools`` (not yet indexed); a plain coin into ``rh_v4_tokens`` (pending its check),
    dated by the first pool of it ever initialized."""
    tok = c["token"]; fee = int(key.get("fee") or 0); a = _ACT.get(key["pool_id"]) or [None] * 11
    if c["kind"] == "v4":
        first = conn.execute("SELECT pool_id, init_block, init_ts FROM rh_v4_keys WHERE currency0 = %(t)s OR currency1 = %(t)s "
                             "ORDER BY init_block LIMIT 1", {"t": tok}).fetchone() or key
        conn.execute("INSERT INTO rh_v4_tokens (token, first_pool, first_block, first_ts, quote_asset) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (token) DO NOTHING",
                     (tok, first["pool_id"], first["init_block"], first["init_ts"], c["quote"]))
    conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, fee, tick_spacing, hooks, quote_asset, init_block, init_ts, is_pons, "
                 "kind, indexed, lp_fee, init_tx, sqrt_price_x96, tick, swaps_seen, last_swap_at) "
                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,false,%s,false,%s,%s,%s,%s,%s,to_timestamp(%s)) ON CONFLICT (pool_id) DO NOTHING",
                 (key["pool_id"], tok, key["currency0"].lower(), key["currency1"].lower(), key["currency0"].lower() == tok, fee, key.get("tick_spacing"),
                  (key.get("hooks") or ZERO).lower(), c["quote"], key["init_block"], key["init_ts"], c["kind"],
                  None if fee & DYNAMIC_FEE else fee / 1e6, key.get("init_tx"), a[7] or key.get("sqrt_price_x96"), a[9] if a[9] is not None else key.get("tick"),
                  swaps, a[10]))


def refresh_activity(conn, now: float) -> None:
    """Registered pools not yet live: their last half hour's swaps (``run_checks`` takes the busiest coins first)."""
    b = int(now // BUCKET_S)
    vals = [(pid, activity(a, b)[0], a[10]) for pid, a in _ACT.items() if pid in _KNOWN]
    if vals:
        conn.cursor().executemany("UPDATE rh_pools SET swaps_seen = %s, last_swap_at = to_timestamp(%s) WHERE pool_id = %s AND NOT is_pons AND NOT indexed",
                                  [(n, ts, pid) for pid, n, ts in vals])


def run_checks(conn, rpc, limit: int | None = None) -> dict:
    """The contract check (rh/token_check.check) of the busiest pending coins; the creator is the sender of the coin's
    first pool's initialization. A ``SlotOver`` ends the checks without counting an attempt."""
    limit = config.RH_V4_CHECKS_PER_SLOT if limit is None else limit
    todo = conn.execute("""SELECT v.token, (SELECT k.init_tx FROM rh_v4_keys k WHERE k.pool_id = v.first_pool) AS init_tx,
                                  (SELECT coalesce(sum(p.swaps_seen), 0) FROM rh_pools p WHERE p.token = v.token) AS swaps
                           FROM rh_v4_tokens v WHERE v.status = 'pending' AND v.attempts < %s ORDER BY swaps DESC, v.first_ts DESC LIMIT %s""",
                        (MAX_ATTEMPTS, limit)).fetchall()
    out = {"ok": 0, "refused": 0, "error": 0}
    for t in todo:
        try:
            r = token_check.check(rpc, t["token"])
            tx = rpc.tx_by_hash(t["init_tx"]) if t["init_tx"] else None
            creator = ((tx or {}).get("from") or "").lower() or None
        except SlotOver:
            raise
        except Exception as e:
            conn.execute("UPDATE rh_v4_tokens SET attempts = attempts + 1, reason = %s, status = CASE WHEN attempts + 1 >= %s THEN 'error' ELSE status END, "
                         "checked_at = now() WHERE token = %s", (str(e)[:300], MAX_ATTEMPTS, t["token"]))
            out["error"] += 1; continue
        st = "ok" if r["ok"] else "refused"
        conn.execute("UPDATE rh_v4_tokens SET status = %s, reason = %s, detail = %s, symbol = %s, name = %s, decimals = %s, supply_raw = %s, creator = %s, "
                     "attempts = attempts + 1, checked_at = now() WHERE token = %s",
                     (st, r["reason"], json.dumps(r, default=str), r["symbol"], r["name"], r["decimals"], r["supply_raw"], creator, t["token"]))
        out[st] += 1
        log.info("rh v4 coin %s (%s): %s, %s", t["token"], r["symbol"], st, r["reason"])
    return out


def go_live(conn, cursor: int) -> int:
    """Each registered pool that qualifies now (a graduated Pons coin's, or an accepted plain coin's) goes live: its
    coin's rh_meta row and ``indexed`` with ``live_from`` = the next block the stream indexes, in this commit. Runs in
    the stream's thread between passes, so the cursor cannot move under it."""
    from . import meta
    rows = conn.execute("""SELECT p.pool_id, p.token, p.kind FROM rh_pools p LEFT JOIN rh_v4_tokens v ON v.token = p.token
                           WHERE NOT p.is_pons AND NOT p.indexed AND p.kind IN ('pons_other', 'v4')
                             AND ((p.kind = 'pons_other' AND EXISTS (SELECT 1 FROM rh_meta m WHERE m.mint = p.token)) OR (p.kind = 'v4' AND v.status = 'ok'))
                           ORDER BY p.swaps_seen DESC""").fetchall()
    n = 0
    for r in rows:
        if r["kind"] == "v4" and not meta.write_v4(conn, r["token"]):
            continue
        conn.execute("UPDATE rh_pools SET indexed = true, live_from = %s WHERE pool_id = %s", (cursor + 1, r["pool_id"]))
        n += 1
        log.info("rh v4: %s pool %s of %s live from block %d", r["kind"], r["pool_id"][:10], r["token"][:10], cursor + 1)
    return n


def slot(deadline: float, now: float | None = None) -> dict:
    """One minute's work (module docstring, step 2), in the stream's thread between passes. What it did, for the
    stream's status."""
    t = time.time() if now is None else now; t0 = time.monotonic()
    out: dict = {"at": datetime.fromtimestamp(t, timezone.utc).isoformat()}
    inits, swaps = _STASH["inits"], _STASH["swaps"]; _STASH["inits"], _STASH["swaps"] = [], []
    out["swaps"] = tally(swaps, t)
    rpc = slot_rpc(deadline)
    try:
        with transaction() as conn:
            out["keys"] = save_keys(conn, rpc, [e for e in (v4.decode(g) for g in inits) if e])
            out.update(discover(conn, t))
            refresh_activity(conn, t)
    except SlotOver as e:
        _STASH["inits"][:0] = inits; out["stopped"] = str(e)[:200]
    except Exception as e:
        _STASH["inits"][:0] = inits; out["error"] = str(e)[:200]; log.exception("rh v4 slot: discovery failed")
    try:
        with transaction() as conn:
            out["checks"] = run_checks(conn, rpc)
    except SlotOver as e:
        out["stopped"] = str(e)[:200]
    except Exception as e:
        out["error"] = str(e)[:200]; log.exception("rh v4 slot: contract checks failed")
    try:
        with transaction() as conn:
            cur = conn.execute("SELECT block FROM rh_scan WHERE name = 'rh'").fetchone()
            out["live"] = go_live(conn, int(cur["block"])) if cur else 0
            out["v4"] = status(conn)
    except Exception as e:
        out["error"] = str(e)[:200]; log.exception("rh v4 slot: going live failed")
    out["active_pools"] = len(_ACT); out["took_s"] = round(time.monotonic() - t0, 2)
    return out


def status(conn) -> dict:
    """Counts for the console and the logs."""
    r = conn.execute("""SELECT count(*) FILTER (WHERE kind = 'pons_other') AS pons_other, count(*) FILTER (WHERE kind = 'v4') AS v4,
                               count(*) FILTER (WHERE indexed) AS live FROM rh_pools WHERE NOT is_pons""").fetchone()
    t = {x["status"]: x["n"] for x in conn.execute("SELECT status, count(*) AS n FROM rh_v4_tokens GROUP BY 1").fetchall()}
    return {**dict(r), "coins": t}


# ---------------------------------------------------------------- fly-trader rh-v4-backfill (its own process)
def import_keys(rpc, stop: threading.Event) -> int:
    """Every Initialize from block 0 to the stream's cursor into ``rh_v4_keys`` (resumable: cursor ``rh_v4_inits``)."""
    from . import index
    n = 0
    with transaction() as conn:
        r = conn.execute("SELECT block FROM rh_scan WHERE name = 'rh_v4_inits'").fetchone()
        top = conn.execute("SELECT block FROM rh_scan WHERE name = %s", (index.CURSOR,)).fetchone()
    frm = int(r["block"]) + 1 if r else 0; to = int(top["block"]) if top else 0
    while frm <= to and not stop.is_set():
        hi = min(to, frm + KEYS_STEP - 1)
        logs = index._get_logs(rpc, [config.V4_POOL_MANAGER], frm, hi, [[v4.TOPIC["Initialize"]]])
        with transaction() as conn:
            k = save_keys(conn, rpc, [e for e in (v4.decode(g) for g in logs) if e])
            conn.execute("INSERT INTO rh_scan (name, block) VALUES ('rh_v4_inits', %s) ON CONFLICT (name) DO UPDATE SET block = EXCLUDED.block, updated_at = now()", (hi,))
        n += k; log.info("rh v4 keys: blocks %d..%d, %d pools", frm, hi, k)
        frm = hi + 1
    return n


def backfill_pool(conn, rpc, p: dict, hi: int) -> dict:
    """One history step of a live pool: its logs from ``p["lo"]`` to ``hi`` (< its ``live_from``), its coin's minutes
    over them recomputed (each day with that day's skill table, never past the minutes the stream has written), and
    ``ready_through = hi``."""
    from . import index, minutes, wallet_skill
    lo = int(p["lo"])
    logs = index._get_logs(rpc, [config.V4_POOL_MANAGER], lo, hi, [index.V4_TOPICS0, [p["pool_id"]]])
    n = index.apply_v4(conn, rpc, logs, [], pool_ids=[p["pool_id"]])
    evs = [e for e in (v4.decode(g) for g in logs) if e and e.get("ts")]
    if evs:
        r = conn.execute("SELECT detail FROM rh_scan WHERE name = 'rh_minutes'").fetchone()
        through = int(((r["detail"] or {}) if r else {}).get("through") or 0)
        t0 = int(min(e["ts"] for e in evs) // 60) * 60; t1 = min(through, int(max(e["ts"] for e in evs) // 60) * 60 + 60)
        d = t0 - t0 % 86400
        while d < t1:
            a, z = max(t0, d), min(t1, d + 86400)
            if z > a:
                minutes.write(conn, minutes.aggregate(conn, a, z, wallet_skill.table_for(datetime.fromtimestamp(d, timezone.utc).date()), tokens=[p["token"]]))
            d += 86400
    conn.execute("UPDATE rh_pools SET ready_through = %s WHERE pool_id = %s", (hi, p["pool_id"]))
    return {"logs": len(logs), **n}


def backfill_history(rpc, stop: threading.Event) -> dict:
    """Each live pool's swaps before its ``live_from`` (busiest first), then a plain coin's first-hour outcomes once its
    pools' histories are complete."""
    from . import meta
    out = {"pools": 0, "steps": 0, "outcomes": 0}
    with transaction() as conn:
        pools = [dict(r) for r in conn.execute(
            """SELECT pool_id, token, kind, live_from, coalesce(ready_through, init_block - 1) + 1 AS lo FROM rh_pools
               WHERE NOT is_pons AND indexed AND live_from IS NOT NULL AND coalesce(ready_through, init_block - 1) < live_from - 1
               ORDER BY swaps_seen DESC""").fetchall()]
    for p in pools:
        while not stop.is_set() and int(p["lo"]) <= int(p["live_from"]) - 1:
            hi = min(int(p["live_from"]) - 1, int(p["lo"]) + HISTORY_CHUNK - 1)
            with transaction() as conn:
                r = backfill_pool(conn, rpc, p, hi)
            out["steps"] += 1; p["lo"] = hi + 1
            log.info("rh v4 history %s (%s): through block %d, %s", p["pool_id"][:10], p["kind"], hi, r)
        out["pools"] += 1
        if stop.is_set():
            break
    with transaction() as conn:                                  # plain coins whose every live pool has its history
        due = conn.execute("""SELECT m.mint, m.graduated_at FROM rh_meta m WHERE m.source = 'v4' AND m.own_dd60 IS NULL
                              AND NOT EXISTS (SELECT 1 FROM rh_pools p WHERE p.token = m.mint AND p.indexed AND NOT p.is_pons
                                              AND coalesce(p.ready_through, p.init_block - 1) < p.live_from - 1)""").fetchall()
        for r in due:
            o = meta.outcomes(conn, r["mint"], r["graduated_at"])
            if o is not None:
                conn.execute("UPDATE rh_meta SET own_dd60 = %s, own_max60 = %s, updated_at = now() WHERE mint = %s", (o[0], o[1], r["mint"])); out["outcomes"] += 1
    return out


def backfill_main(stop_event: threading.Event | None = None, keys: bool = True, history: bool = True) -> dict:
    """``fly-trader rh-v4-backfill``: the key import, then the live pools' histories. Run it apart from the console, under
    nice; it paces HyperSync at ``RH_V4_BACKFILL_RPM`` of its own, so the console keeps its share of the plan."""
    from ..logging_setup import setup
    from .rpc import logs_rpc
    setup("rh_v4_backfill"); stop = stop_event or threading.Event()
    config.RH_HYPERSYNC_RPM = config.RH_V4_BACKFILL_RPM
    rpc = logs_rpc(); out: dict = {}
    if keys:
        out["keys"] = import_keys(rpc, stop)
    if history and not stop.is_set():
        out.update(backfill_history(rpc, stop))
    log.info("rh v4 backfill done: %s", out)
    return out
