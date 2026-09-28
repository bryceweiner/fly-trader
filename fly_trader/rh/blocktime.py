"""Block → time on Robinhood Chain without one lookup per block.

The RPC's ``blockTimestamp`` on logs reads 0x0 (2026-09-28) and the chain makes 10–20 blocks a second, so timing every
swap's block cost thousands of lookups per chain-minute. Minute bars only need each event's exact minute: block times never
decrease, so sample one block in every ``SAMPLE`` (batched), and where two neighbouring samples fall in different minutes
find each minute's first block by binary search (all open searches advance together, one batch per round). An event's
minute is then exact; its seconds are interpolated between the samples around it.
"""
from __future__ import annotations

import bisect

from . import scan

SAMPLE = 256
BATCH = 100


def _fetch(rpc, blocks: list[int], cache: dict[int, int]) -> None:
    need = [b for b in sorted(set(blocks)) if b not in cache]
    for i in range(0, len(need), BATCH):
        chunk = need[i:i + BATCH]
        res = rpc.batch([("eth_getBlockByNumber", [hex(b), False]) for b in chunk]); scan.pause()
        for b, r in zip(chunk, res):
            if not isinstance(r, dict) or not r.get("timestamp"):
                raise RuntimeError(f"no block {b} from the RPC")
            cache[b] = int(r["timestamp"], 16)


def times(rpc, lo: int, hi: int, wanted: list[int], sample: int = SAMPLE) -> dict[int, float]:
    """Seconds since epoch for each block in ``wanted`` (all within lo..hi): exact minute, interpolated seconds."""
    if not wanted:
        return {}
    cache: dict[int, int] = {}
    grid = list(range(lo, hi + 1, sample))
    if grid[-1] != hi:
        grid.append(hi)
    _fetch(rpc, grid, cache)
    # the first block of every minute that starts inside the range
    searches = []                                           # [lo_block, hi_block, minute_start_s]
    for a, b in zip(grid, grid[1:]):
        ma, mb = cache[a] // 60, cache[b] // 60
        for m in range(ma + 1, mb + 1):
            searches.append([a, b, m * 60])                  # ts(a) < m*60 <= ts(b)
    while any(s[1] - s[0] > 1 for s in searches):
        mids = [(s[0] + s[1]) // 2 for s in searches if s[1] - s[0] > 1]
        _fetch(rpc, mids, cache)
        for s in searches:
            if s[1] - s[0] > 1:
                mid = (s[0] + s[1]) // 2
                if cache[mid] >= s[2]:
                    s[1] = mid
                else:
                    s[0] = mid
    starts = sorted((s[1], s[2]) for s in searches)          # (first block of the minute, minute start)
    start_blocks = [b for b, _ in starts]
    known = sorted(cache.items()); kb = [b for b, _ in known]
    out = {}
    for blk in wanted:
        if blk in cache:
            out[blk] = float(cache[blk]); continue
        i = bisect.bisect_right(start_blocks, blk) - 1
        # exact minute: the last minute whose first block is at or before blk (else the minute of the range start)
        minute = starts[i][1] if i >= 0 else (cache[lo] // 60) * 60
        j = bisect.bisect_right(kb, blk)
        (b0, t0), (b1, t1) = known[max(0, j - 1)], known[min(len(known) - 1, j)]
        t = t0 + (t1 - t0) * ((blk - b0) / (b1 - b0)) if b1 != b0 else float(t0)
        out[blk] = min(max(t, float(minute)), minute + 59.999)
    return out
