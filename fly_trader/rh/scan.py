"""A cursor over block ranges for the RH indexer: the vault indexer's adaptive-range pattern (vault/rh_index.run_once)
with the public RPC's limits measured on 2026-09-28 built in — a 20,000-block eth_getLogs range sustains ~12,000 blocks/s
at one request per second; wider ranges or faster requests return 429.

``advance(conn, name, head, step_fn)`` runs ``step_fn(from_block, to_block)`` on the next range after the cursor, commits
the cursor with the work, grows the range after a success and halves it after a failure (never below ``MIN_RANGE``).
"""
from __future__ import annotations

import logging
import time

from .. import config

log = logging.getLogger(__name__)

MIN_RANGE, MAX_RANGE, START_RANGE = 250, 20_000, 5_000
MAX_RANGE_FAST = 50_000          # HyperSync answers whole ranges in one paginated query; ~0.04 busy days of logs held at once
PAUSE_S = 0.8                    # between RPC requests: the public endpoint answers 429 above ~1 request/s
BACKOFF_S = 3.0


def fast() -> bool:
    """A log source without the public endpoint's ~1 request/s limit (a dedicated RPC or HyperSync)."""
    return bool(config.RH_RPC_URL_LOGS or config.env_str("ENVIO_API_TOKEN"))


def max_range() -> int:
    return MAX_RANGE_FAST if config.env_str("ENVIO_API_TOKEN") else MAX_RANGE


def pause() -> None:
    time.sleep(PAUSE_S if not fast() else 0.05)


def cursor(conn, name: str, default: int) -> tuple[int, int]:
    r = conn.execute("SELECT block, range_blocks FROM rh_scan WHERE name = %s", (name,)).fetchone()
    if r is None:
        conn.execute("INSERT INTO rh_scan (name, block, range_blocks) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING", (name, default, START_RANGE))
        return default, START_RANGE
    return int(r["block"]), int(r["range_blocks"])


def set_cursor(conn, name: str, block: int, rng: int, detail: dict | None = None) -> None:
    import json
    conn.execute("UPDATE rh_scan SET block = %s, range_blocks = %s, detail = COALESCE(%s::jsonb, detail), updated_at = now() WHERE name = %s",
                 (block, rng, json.dumps(detail) if detail else None, name))


def next_range(done_through: int, rng: int, head: int) -> tuple[int, int] | None:
    lo = done_through + 1
    if lo > head:
        return None
    return lo, min(head, lo + rng - 1)


def grow(rng: int) -> int:
    return min(max_range(), int(rng * 1.5) + 1)


def shrink(rng: int) -> int:
    return max(MIN_RANGE, rng // 2)


def is_rate_limit(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "rate" in s.lower() or "Too Many" in s
