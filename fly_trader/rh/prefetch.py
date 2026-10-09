"""One log query per live pass of the RH stream (rh/stream.py), and the RPC behind HyperSync when HyperSync fails.

Until 2026-10-09 a live pass asked HyperSync 8-12 times, one query after another (Pons events, PoolManager events, hook
fees, the swappers' Transfers, every reference-pool venue, twice over for two price ranges, each paced and each on a fresh
connection for the price scan): ~21 s a pass, so RH decisions landed 45-70 s after their minute.

``fetch`` reads every log the pass's index step and price scan will ask for over the pass's block range at once: from
HyperSync as one multi-selection query (capped at its archive height, every block's time included), or, while HyperSync
is failing, from the RPC as one eth_getLogs over every selection's addresses (any topic, at most ``rpc_span`` blocks).
``Prefetch`` wraps the stream's rpc for the pass: ``get_logs_multi`` answers from memory a request inside the fetched
range that a fetched selection covers (applying the request's own filter, so a broad fetch answers a narrower request
exactly) and passes anything else on — to HyperSync, or to the RPC while HyperSync is down. ``block`` answers the times
the fetch saw. A request the selections do not cover is only slower, never wrong.

HyperSync is asked ``LIVE_TRIES`` times in a pass; when it still fails the RPC serves the logs for ``COOLDOWN_S`` before
HyperSync is tried again, so the RH books keep trading through a HyperSync outage instead of holding until it returns.
"""
from __future__ import annotations

import json
import logging
import time

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from . import hypersync as H
from .rpc import RhRpc

log = logging.getLogger(__name__)

LIVE_TRIES = 3               # HyperSync attempts in a live pass before the RPC takes the pass over
COOLDOWN_S = 60.0            # after HyperSync failed a pass: this long on the RPC before HyperSync is asked again
RPC_SPAN = 400               # blocks per pass from the RPC (its 10,000-log cap; ~800 PoolManager logs in 400 blocks, 2026-10-09)
RPC_SPAN_MIN = 25
_STATE: dict = {"down_until": 0.0, "down_since": None, "last_error": None, "rpc_span": RPC_SPAN}


def hypersync_down() -> bool:
    """HyperSync failed a pass less than ``COOLDOWN_S`` ago: the RPC serves the live logs."""
    return time.monotonic() < _STATE["down_until"]


def status() -> dict:
    return {"hypersync_down_since": _STATE["down_since"], "last_error": _STATE["last_error"], "rpc_span": _STATE["rpc_span"]} if _STATE["down_since"] else {}


def _failed(e: Exception) -> None:
    _STATE["down_until"] = time.monotonic() + COOLDOWN_S; _STATE["last_error"] = str(e)[:200]
    if _STATE["down_since"] is None:
        _STATE["down_since"] = time.time()
        log.warning("HyperSync failed a live pass (%s): the RPC serves the RH logs for now", str(e)[:200])
        record_event("warning", "rh_stream", "HyperSync is failing: the public RPC serves the RH logs", {"error": str(e)[:300]})


def _recovered() -> None:
    if _STATE["down_since"] is not None:
        down = time.time() - _STATE["down_since"]
        log.info("HyperSync is back after %.0f s", down)
        record_event("info", "rh_stream", "HyperSync is back: it serves the RH logs again", {"down_s": round(down)})
    _STATE.update(down_until=0.0, down_since=None, last_error=None)


# ---------------------------------------------------------------- coverage and matching (eth_getLogs semantics)
def _sets(topics) -> list:
    """Topic positions as sets (None: any), from eth_getLogs' or HyperSync's form."""
    out = []
    for t in topics or []:
        if t is None or (isinstance(t, (list, tuple)) and not t):
            out.append(None)
        else:
            out.append({x.lower() for x in (t if isinstance(t, (list, tuple)) else [t])})
    return out


def covered(addresses, topics, sels: list[dict]) -> bool:
    """Whether some selection's logs include every log the request (``addresses``, ``topics``) can match."""
    if not addresses:
        return False                                            # an any-address request is never answered from memory
    want_a = {a.lower() for a in addresses}; want_t = _sets(topics)
    for s in sels:
        if not want_a <= {a.lower() for a in (s.get("address") or [])}:
            continue
        ok = True
        for i, have in enumerate(_sets(s.get("topics"))):
            if have is None:
                continue
            want = want_t[i] if i < len(want_t) else None
            if want is None or not want <= have:
                ok = False; break
        if ok:
            return True
    return False


def matches(g: dict, want_a: set, want_t: list) -> bool:
    if g["address"].lower() not in want_a:
        return False
    tp = g.get("topics") or []
    for i, want in enumerate(want_t):
        if want is not None and (i >= len(tp) or tp[i].lower() not in want):
            return False
    return True


# ---------------------------------------------------------------- what a live pass reads
def selections(conn) -> list[dict]:
    """The index step's and the price scan's reads (index.step, prices.fetch_pool_logs) as HyperSync selections."""
    from . import index, prices
    sels = [H.selection([config.PONS_FACTORY, config.PONS_ROUTER, config.PONS_HOOK], index.PONS_TOPICS),
            H.selection([config.V4_POOL_MANAGER], [index.V4_TOPICS0]),
            H.selection([config.PONS_HOOK], [[index.FEE_TOPIC]])]
    hops: dict = {}
    for r in conn.execute("SELECT DISTINCT a.asset, a.ref_path::text AS ref_path FROM rh_assets a JOIN rh_pools p ON p.quote_asset = a.asset "
                          "WHERE p.is_pons AND a.class <> 'eth' AND a.ref_path IS NOT NULL").fetchall():
        for h in json.loads(r["ref_path"]):
            hops.setdefault(prices._hop_key(h), h)
    plain = sorted({h["emitter"].lower() for h in hops.values() if not prices._is_id(h["pool"])})
    if plain:
        sels.append(H.selection(plain, [[prices.SWAP_V3]]))
    managers: dict[str, set] = {}
    for h in hops.values():
        if prices._is_id(h["pool"]):
            managers.setdefault(h["emitter"].lower(), set()).add(h["pool"].lower())
    for em, ids in sorted(managers.items()):
        sels.append(H.selection([em], [None, sorted(ids)]))
    return sels


class Prefetch:
    """The stream's rpc for one pass, answering the fetched range's log reads from memory (module docstring)."""

    def __init__(self, rpc, lo: int, hi: int, logs: list[dict], times: dict[int, int], sels: list[dict], source: str):
        self._rpc, self.lo, self.hi, self.logs, self.times, self.sels, self.source = rpc, lo, hi, logs, times, sels, source
        self.served = 0; self.passed = 0; self.served_last = False

    def __getattr__(self, name):
        return getattr(self._rpc, name)

    def block(self, tag, full: bool = False):
        if isinstance(tag, int) and not full and tag in self.times:
            return {"number": hex(tag), "timestamp": hex(self.times[tag])}
        return self._rpc.block(tag, full)

    def get_logs_multi(self, addresses, from_block: int, to_block: int, topics=None) -> list[dict]:
        if self.lo <= from_block and to_block <= self.hi and covered(addresses, topics, self.sels):
            want_a = {a.lower() for a in addresses}; want_t = _sets(topics)
            self.served += 1; self.served_last = True
            return [g for g in self.logs if from_block <= int(g["blockNumber"], 16) <= to_block and matches(g, want_a, want_t)]
        self.passed += 1; self.served_last = False
        if self.source == "rpc":                                 # HyperSync is down: the RPC answers what memory cannot
            return RhRpc.get_logs_multi(self._rpc, addresses, from_block, to_block, topics)
        return self._rpc.get_logs_multi(addresses, from_block, to_block, topics)


def _too_big(e: Exception) -> bool:
    msg = str(e).lower()
    return "exceeds limit" in msg or "too many" in msg or "timed out" in msg or "timeout" in msg or "response size" in msg


def fetch(rpc, lo: int, hi: int) -> Prefetch:
    """The pass's logs over lo..hi: one HyperSync query (to its archive height) unless HyperSync is failing, else one
    RPC read of at most ``rpc_span`` blocks (halved while the RPC refuses it). The returned ``hi`` is what was covered."""
    with transaction() as conn:
        sels = selections(conn)
    hs = getattr(rpc, "hs", None)
    if hs is not None and not hypersync_down():
        try:
            logs, through, times = hs.get_logs_sel(sels, lo, hi, upto=True, tries=LIVE_TRIES, all_blocks=True)
            _recovered()
            return Prefetch(rpc, lo, through, logs, times, sels, "hypersync")
        except Exception as e:                                  # whatever HyperSync's failure, the pass goes on without it
            _failed(e)
    addrs = sorted({a for s in sels for a in (s.get("address") or [])})
    while True:
        top = min(hi, lo + _STATE["rpc_span"] - 1)
        try:
            logs = RhRpc.get_logs_multi(rpc, addrs, lo, top, None)
        except Exception as e:
            if _too_big(e) and _STATE["rpc_span"] > RPC_SPAN_MIN:
                _STATE["rpc_span"] = max(RPC_SPAN_MIN, _STATE["rpc_span"] // 2); continue
            raise
        _STATE["rpc_span"] = min(RPC_SPAN, _STATE["rpc_span"] * 2)
        logs.sort(key=lambda g: (int(g["blockNumber"], 16), int(g["logIndex"], 16)))
        return Prefetch(rpc, lo, top, logs, {}, [{"address": addrs, "topics": []}], "rpc")
