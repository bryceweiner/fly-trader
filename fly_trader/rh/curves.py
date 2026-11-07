"""Graduates' curve back-fills, off the live stream. When a Pons coin graduates, its curve trades (launch → graduation, its
own curve contract) give its launch facts (rh/meta.py: creator holding, bundle, holder concentration). A coin that sat weeks
on its curve is hundreds of ``scan.MAX_RANGE`` chunks; run inside the live index step (until 2026-11-05) one such graduation
held every RH minute -- and the fly -- for 40+ minutes at a time. Here a thread of the rh_stream worker takes the oldest
graduate not yet back-filled, one chunk per transaction (progress in ``rh_tokens.curve_through``, so a restart resumes),
paced so the live passes keep HyperSync's quota. rh/meta.py builds a graduate's row only once its back-fill is done; a new
graduate cannot trade for its first ``MIN_AGE_H`` hours anyway."""
from __future__ import annotations

import logging
import threading

from ..db.connection import transaction
from . import index
from .rpc import logs_rpc

log = logging.getLogger(__name__)
PACE_S = 2.0                     # between chunks
IDLE_S = 30.0                    # when no graduate waits
ERROR_S = 30.0


def next_token(conn) -> str | None:
    r = conn.execute("""SELECT token FROM rh_tokens WHERE NOT curve_indexed AND curve IS NOT NULL AND launch_block IS NOT NULL
                        AND COALESCE(curve_until, grad_block) IS NOT NULL ORDER BY graduated_at NULLS LAST, token LIMIT 1""").fetchone()
    return r["token"] if r else None


def run(stop: threading.Event, rpc=None) -> None:
    rpc = rpc or logs_rpc(); rows = 0; chunks = 0
    while not stop.is_set():
        try:
            with transaction() as conn:
                tok = next_token(conn)
            if tok is None:
                stop.wait(IDLE_S); continue
            with transaction() as conn:
                n, done = index.index_curve_chunk(conn, rpc, tok)
            rows += n; chunks += 1
            if done:
                log.info("rh curve back-fill: %s done (%d chunks, %d trades)", tok, chunks, rows); rows = chunks = 0
        except Exception as e:
            log.warning("rh curve back-fill: %s", str(e)[:200]); stop.wait(ERROR_S); continue
        stop.wait(PACE_S)


def start(stop: threading.Event) -> threading.Thread:
    t = threading.Thread(target=run, args=(stop,), name="rh_curves", daemon=True)
    t.start()
    return t
