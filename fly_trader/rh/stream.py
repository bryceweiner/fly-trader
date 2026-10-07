"""The ``rh_stream`` worker: Robinhood Chain from the Pons V2 deployment to the head, then live.

Each pass: the index cursor advances (rh/index.py; many ranges while backfilling, one near the head), the quote assets'
marks follow it (rh/prices.py), the complete minutes are written (rh/minutes.py), and the graduated launches' meta
(rh/meta.py) is refreshed. ``rh_stream_status.flushed_through`` gates the engine's RH minutes, as pumpstream_status gates
Solana's. Caught up, it polls every ``RH_STREAM_POLL_S``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..logging_setup import setup
from . import index, minutes, prices
from .rpc import logs_rpc

log = logging.getLogger(__name__)
BACKFILL_RANGES = 20            # ranges per pass while far behind (then a status write)
LIVE_LAG_BLOCKS = 5_000         # closer than this to the head counts as live


def _status(**kv) -> None:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
        cur = (r["value"] if r and isinstance(r["value"], dict) else json.loads(r["value"])) if r else {}
        cur.update(kv); cur["updated_at"] = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('rh_stream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps(cur, default=str),))


def one_pass(rpc, skill=None) -> dict:
    tm: dict[str, float] = {}; t = time.monotonic()

    def lap(step: str) -> None:                             # seconds per step, in the status: where a slow pass spends its time
        nonlocal t
        now = time.monotonic(); tm[step] = round(now - t, 2); t = now
    head = rpc.block_number()
    with transaction() as conn:
        r = conn.execute("SELECT block FROM rh_scan WHERE name = 'rh'").fetchone()
    behind = head - (int(r["block"]) if r else 0)
    lap("head")
    ix = index.run_once(rpc, max_ranges=BACKFILL_RANGES if behind > LIVE_LAG_BLOCKS else 1); lap("index")
    px = prices.run_once(max_ranges=BACKFILL_RANGES if behind > LIVE_LAG_BLOCKS else 2); lap("prices")
    mn = minutes.run_once(rpc, skill); lap("minutes")
    from . import corpus, meta
    mt = meta.run_once(); lap("meta")
    cp = corpus.run_once(max_days=1) if behind <= LIVE_LAG_BLOCKS * 20 else {}          # the training corpus follows once near the head
    lap("corpus")
    out = {"head": head, "index_through": ix.get("through"), "behind_blocks": head - (ix.get("through") or 0), "index": ix.get("counts"), "prices": px,
           "minutes": mn, "meta": mt, "corpus": cp, "mode": "backfill" if behind > LIVE_LAG_BLOCKS else "live", "timings_s": tm}
    _status(**{k: v for k, v in out.items() if k != "minutes"}, minutes_through=mn.get("through"))
    return out


def main(stop_event: threading.Event | None = None) -> None:
    setup("rh_stream")
    if not config.RH_ENABLED:
        log.info("RH_ENABLED is off: rh_stream idles"); return
    from ..db import schema
    schema.apply_schema()
    rpc = logs_rpc(); stop = stop_event or threading.Event()
    log.info("rh_stream: %s", config.RH_RPC_URL_LOGS and "dedicated log RPC" or "public RPC")
    while not stop.is_set():
        try:
            skill = None
            try:
                from . import wallet_skill
                skill = wallet_skill.load_today()
            except ImportError:
                pass
            out = one_pass(rpc, skill)
            if out["mode"] == "backfill":
                log.info("rh backfill: through %s, %s blocks behind, %s", out["index_through"], out["behind_blocks"], out["index"])
                continue
        except Exception as e:
            log.exception("rh_stream pass failed: %s", str(e)[:200]); _status(error=str(e)[:300])
            stop.wait(10.0); continue
        stop.wait(config.RH_STREAM_POLL_S)


def status() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
        scans = [dict(x) for x in conn.execute("SELECT name, block, range_blocks, updated_at FROM rh_scan ORDER BY name").fetchall()]
        counts = {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in ("rh_tokens", "rh_pools", "rh_swaps", "rh_curve_trades", "rh_minutes", "rh_base_prices")}
    return {"status": (r["value"] if r else None), "cursors": scans, "rows": counts}


if __name__ == "__main__":
    main()
    time.sleep(0)
