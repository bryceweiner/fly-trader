"""Jupiter token stats (the `discover` worker).

Every STATS_REFRESH_S, token_stats via /search batches (100 mints per call) for every token of the selector's universe
(traded on the stream in the last hour, pool above its gate). Jupiter serves only current values, so this is how a
point-in-time history accrues for future model inputs.
"""
from __future__ import annotations

import json
import logging
import signal
import time

from .. import config
from ..chain.jupiter_tokens import JupiterTokens
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup

log = logging.getLogger(__name__)


def insert_stats(conn, tok: dict) -> None:
    audit = tok.get("audit") or {}
    stats = {k: tok.get(k) for k in ("stats5m", "stats1h", "stats6h", "stats24h")}
    conn.execute(
        """INSERT INTO token_stats (mint, organic_score, organic_label, holder_count, liquidity_usd, usd_price, mcap,
              top_holders_pct, dev_balance_pct, dev_mints, is_sus, mint_auth_disabled, freeze_auth_disabled, is_verified, stats)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (tok["id"], tok.get("organicScore"), tok.get("organicScoreLabel"), tok.get("holderCount"), tok.get("liquidity"),
         tok.get("usdPrice"), tok.get("mcap"), audit.get("topHoldersPercentage"), audit.get("devBalancePercentage"),
         audit.get("devMints"), audit.get("isSus"), audit.get("mintAuthorityDisabled"), audit.get("freezeAuthorityDisabled"),
         tok.get("isVerified"), json.dumps(stats, default=str)),
    )


def refresh_stats(conn, client: JupiterTokens) -> dict:
    """token_stats for the selector's universe: every pump.fun PumpSwap token the stream saw trade in the last hour with a
    pool above the selector's gate, so a point-in-time history accrues for the tokens actually traded."""
    from ..train.decisions import MIN_RESQ_SOL   # lazy: keeps the training stack out of the worker's import
    uni = conn.execute("SELECT DISTINCT mint FROM pump_minutes WHERE ts > now() - interval '1 hour' AND resq_sol >= %s",
                       (MIN_RESQ_SOL,)).fetchall()
    mints = sorted(r["mint"] for r in uni)
    toks = [t for t in client.search_many(mints) if t.get("id")] if mints else []
    for tok in toks:
        insert_stats(conn, tok)
    return {"universe": len(mints), "refreshed": len(toks)}


def run_once() -> None:
    setup("discover")
    with transaction() as conn:
        print(json.dumps(refresh_stats(conn, JupiterTokens()), indent=1))


def run_forever(stop_event=None) -> None:
    """Loop until stop_event is set (console thread) or SIGTERM/SIGINT (CLI, main thread only)."""
    import threading
    setup("discover")
    client = JupiterTokens(); stopped = {"v": False}
    if stop_event is None and threading.current_thread() is threading.main_thread():
        def _stop(*_):
            stopped["v"] = True
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    record_event("info", "discover", "token stats worker started", {"refresh_s": config.STATS_REFRESH_S})
    while not stopped["v"] and not (stop_event is not None and stop_event.is_set()):
        t0 = time.monotonic()
        try:
            with transaction() as conn:
                log.info("stats refresh %s", refresh_stats(conn, client))
        except Exception as e:
            log.exception("token stats refresh error")
            record_event("error", "discover", f"stats refresh error: {type(e).__name__}: {e}")
        # sleep the remainder of the interval, waking early on stop
        remaining = config.STATS_REFRESH_S - (time.monotonic() - t0)
        while remaining > 0 and not stopped["v"] and not (stop_event is not None and stop_event.is_set()):
            time.sleep(min(1.0, remaining))
            remaining -= 1.0
    record_event("info", "discover", "token stats worker stopped")
