"""The ``rh_stream`` worker: Robinhood Chain from the Pons V2 deployment to the head, then live.

Each pass: the index cursor advances (rh/index.py; many ranges while backfilling, one near the head), the quote assets'
marks follow it (rh/prices.py), the complete minutes are written (rh/minutes.py), and the graduated launches' meta
(rh/meta.py) is refreshed. ``rh_stream_status.flushed_through`` gates the engine's RH minutes, as pumpstream_status gates
Solana's. Near the head a pass reads its logs in one query (rh/prefetch.py: HyperSync, or the RPC while HyperSync fails)
and the next pass starts as soon as the next minute can be written (``_wait_s``), at most ``RH_STREAM_POLL_S`` later.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..logging_setup import setup
from . import index, minutes, prefetch, prices, scan
from .rpc import logs_rpc

log = logging.getLogger(__name__)
BACKFILL_RANGES = 20            # ranges per pass while far behind (then a status write)
LIVE_LAG_BLOCKS = 5_000         # closer than this to the head counts as live
RPC_CATCHUP_S = 3.0             # between catch-up passes while HyperSync is down (each reads ~2-3 times from the public RPC)
SLOW_EVERY_S = 60.0             # meta and corpus refresh with each written minute, else at least this often
_SLOW = {"at": 0.0}
_CORPUS: dict = {"proc": None}  # the child building a due corpus day (rh/corpus.run_locked)


def _status(**kv) -> None:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
        cur = (r["value"] if r and isinstance(r["value"], dict) else json.loads(r["value"])) if r else {}
        cur.update(kv); cur["updated_at"] = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('rh_stream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps(cur, default=str),))


def _head(rpc) -> int:
    """The chain head: the header endpoint's (RH_RPC_URL_BLOCKS: the freshest head measured, 2026-10-09, and no ~20
    requests/min limit like the official RPC's), else the RPC's, else HyperSync's archive height — a pass every few
    seconds must not stop because one of them is down."""
    err = None
    for src in (getattr(rpc, "blocks", None), rpc):
        if src is None:
            continue
        try:
            return src.block_number()
        except Exception as e:
            err = e
    hs = getattr(rpc, "hs", None)
    if hs is not None and hs.height:
        return int(hs.height)
    raise err


def one_pass(rpc, skill=None) -> dict:
    tm: dict[str, float] = {}; t = time.monotonic()

    def lap(step: str) -> None:                             # seconds per step, in the status: where a slow pass spends its time
        nonlocal t
        now = time.monotonic(); tm[step] = round(now - t, 2); t = now
    head = _head(rpc)
    with transaction() as conn:
        r = conn.execute("SELECT block, range_blocks FROM rh_scan WHERE name = %s", (index.CURSOR,)).fetchone()
    at = int(r["block"]) if r else -1
    behind = head - (at if r else 0)
    lap("head")
    src = rpc; pre = None
    if r is not None and at >= 0 and (behind <= LIVE_LAG_BLOCKS or prefetch.hypersync_down()):
        rng = scan.next_range(at, int(r["range_blocks"] or scan.START_RANGE), head - config.RH_CONFIRMATIONS)
        if rng is not None:                                     # one query for everything the pass reads (rh/prefetch.py)
            pre = src = prefetch.fetch(rpc, *rng); lap("logs")
            ix = index.run_once(src, max_ranges=1, head=pre.hi)
        else:
            ix = {"through": at, "head": head, "ranges": 0, "counts": {}}
        lap("index")
        px = prices.run_once(max_ranges=2, rpc=src); lap("prices")
    else:
        ix = index.run_once(rpc, max_ranges=BACKFILL_RANGES if behind > LIVE_LAG_BLOCKS else 1); lap("index")
        px = prices.run_once(max_ranges=BACKFILL_RANGES if behind > LIVE_LAG_BLOCKS else 2, rpc=rpc); lap("prices")
    from . import wallet_skill
    wallet_skill.daily(); lap("skill_tables")                  # today's and tomorrow's tables, before their minutes are written
    mn = minutes.run_once(src, wallet_skill.load_today() if skill is None else skill); lap("minutes")
    from . import meta
    mt, cp = {}, {}
    if mn.get("minutes") or time.monotonic() - _SLOW["at"] >= SLOW_EVERY_S:   # after a written minute: passes in between stay short
        _SLOW["at"] = time.monotonic()
        mt = meta.run_once(); lap("meta")
        cp = corpus_apart() if behind <= LIVE_LAG_BLOCKS * 20 else {}               # the training corpus follows once near the head
        lap("corpus")
    out = {"head": head, "index_through": ix.get("through"), "behind_blocks": head - (ix.get("through") or 0), "index": ix.get("counts"), "prices": px,
           "minutes": mn, "meta": mt, "corpus": cp, "mode": "backfill" if behind > LIVE_LAG_BLOCKS else "live", "timings_s": tm,
           "logs_from": pre.source if pre is not None else ("hypersync" if hasattr(rpc, "hs") else "rpc"),
           "memory_reads": {"served": pre.served, "passed": pre.passed} if pre is not None else None, "hypersync": prefetch.status() or None}
    _status(**{k: v for k, v in out.items() if k != "minutes"}, minutes_through=mn.get("through"))
    return out


def main(stop_event: threading.Event | None = None) -> None:
    setup("rh_stream")
    if not config.RH_ENABLED:
        log.info("RH_ENABLED is off: rh_stream idles"); return
    from ..db import schema
    schema.apply_schema()
    rpc = logs_rpc(); stop = stop_event or threading.Event()
    log.info("rh_stream: logs from %s; the head from %s", "HyperSync" if config.env_str("ENVIO_API_TOKEN") else "the RPC",
             "the dedicated RPC (RH_RPC_URL_LOGS)" if config.RH_RPC_URL_LOGS else "HyperSync / the public RPC")
    while not stop.is_set():
        try:
            out = one_pass(rpc)                                     # the pass writes and loads the day's skill table itself
            if out["mode"] == "backfill":
                log.info("rh backfill: through %s, %s blocks behind, %s", out["index_through"], out["behind_blocks"], out["index"])
                if out.get("logs_from") == "rpc" and prefetch.hypersync_down():
                    stop.wait(RPC_CATCHUP_S)                          # catching up on the public RPC: under its rate limit
                continue
        except Exception as e:
            log.exception("rh_stream pass failed: %s", str(e)[:200]); _status(error=str(e)[:300])
            stop.wait(10.0); continue
        stop.wait(_wait_s((out.get("minutes") or {}).get("through")))
    _stop_corpus()


def corpus_apart() -> dict:
    """A due corpus day is built by a child process at the lowest CPU priority (``fly-trader rh-build-corpus``), never in
    this thread (rh/corpus.py). One child at a time; this only looks at the files and starts it."""
    from . import corpus
    p = _CORPUS["proc"]
    if p is not None and p.poll() is None:
        return {"building": p.pid}
    if p is not None:
        _CORPUS["proc"] = None
        if p.returncode:
            log.warning("rh corpus child %d exited %d (its log: rh_corpus.log)", p.pid, p.returncode)
    if not corpus.due():
        return {}
    nice = ["/usr/bin/nice", "-n", "19"] if os.path.exists("/usr/bin/nice") else []      # not preexec_fn: unsafe in a threaded process
    p = _CORPUS["proc"] = subprocess.Popen([*nice, sys.executable, "-m", "fly_trader", "rh-build-corpus"], stdout=subprocess.DEVNULL)
    log.info("rh corpus: a day is due, built by child process %d at the lowest CPU priority", p.pid)
    return {"started": p.pid}


def _stop_corpus() -> None:
    p = _CORPUS["proc"]
    if p is not None and p.poll() is None:
        p.terminate()                                            # every file is written atomically: an ended child loses only its day
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def _wait_s(written_through: int | None, now: float | None = None) -> float:
    """Until the next pass: the moment the last ended minute can be written (its end + ``minutes.GRACE_S``) if it is not
    yet, else the next minute's, but never longer than ``RH_STREAM_POLL_S`` (``written_through``: the end of the last
    minute written, epoch s)."""
    now = time.time() if now is None else now
    end = int(now // 60) * 60
    due = (end if (written_through or 0) < end else end + 60) + minutes.GRACE_S + 0.5
    return max(0.05, min(config.RH_STREAM_POLL_S, due - now))


def status() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
        scans = [dict(x) for x in conn.execute("SELECT name, block, range_blocks, updated_at FROM rh_scan ORDER BY name").fetchall()]
        counts = {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in ("rh_tokens", "rh_pools", "rh_swaps", "rh_curve_trades", "rh_minutes", "rh_base_prices")}
    return {"status": (r["value"] if r else None), "cursors": scans, "rows": counts}


if __name__ == "__main__":
    main()
    time.sleep(0)
