"""Jupiter token stats (the `discover` worker) and the legacy watch-list helpers.

Worker: every STATS_REFRESH_S, token_stats via /search batches (100 mints per call) for every token of the selector's
universe (traded on the stream in the last hour, pool above its gate). Jupiter serves only current values, so this
is how a point-in-time history accrues for future model inputs.

The watch-list helpers below (graduation gate, tokens/watch_pools upserts, category polling for ``--probe``) fed the
legacy brain modes and are no longer run by the worker.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timedelta, timezone

from .. import config
from ..chain.jupiter_tokens import JupiterTokens
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup

log = logging.getLogger(__name__)

PRE_STALE_HOURS = 48.0
CATEGORIES_FAST = [("toptrending", "5m"), ("toptraded", "5m"), ("toporganicscore", "5m")]
CATEGORIES_SLOW = [("toptrending", "1h"), ("toptraded", "1h"), ("toporganicscore", "1h"), ("toptraded", "24h")]
PUMP_AMM_LABEL = "Pump.fun Amm"


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def launchpad_ok(launchpad: str | None) -> bool:
    if "*" in config.LAUNCHPADS:
        return launchpad is not None
    return launchpad is not None and launchpad.lower() in {x.lower() for x in config.LAUNCHPADS}


def evaluate(tok: dict) -> tuple[str, str]:
    """Return (watch_status, reason)."""
    audit = tok.get("audit") or {}
    lp = tok.get("launchpad")
    graduated = bool(tok.get("graduatedAt"))
    if not launchpad_ok(lp):
        return "excluded", f"launchpad={lp}"
    if not graduated:
        return "pre", "not graduated"
    ma, fa = audit.get("mintAuthorityDisabled"), audit.get("freezeAuthorityDisabled")
    if ma is False:
        return "excluded", "mint authority not disabled"
    if fa is False:
        return "excluded", "freeze authority not disabled"
    if ma is not True or fa is not True:
        return "unknown", "audit missing from payload"      # transient payload variance: no status change
    if not tok.get("graduatedPool"):
        return "unknown", "graduated without graduatedPool"  # payload variance too: keep the stored status
    return "watch", "graduated, authorities disabled"


def dedupe(toks: list[dict]) -> list[dict]:
    """First payload per mint: /recent and the category lists overlap (one token_stats row per poll)."""
    seen: set[str] = set()
    out = []
    for t in toks:
        if t.get("id") and t["id"] not in seen:
            seen.add(t["id"])
            out.append(t)
    return out


def upsert_token(conn, tok: dict, status: str) -> None:
    audit = tok.get("audit") or {}
    fp = tok.get("firstPool") or {}
    conn.execute(
        """INSERT INTO tokens (mint, symbol, name, decimals, token_program, launchpad, partner_config,
              graduated_pool, graduated_at, first_pool_id, first_pool_created_at, mint_auth_disabled,
              freeze_auth_disabled, first_seen, last_seen, watch_status, raw)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now(),%s,%s)
           ON CONFLICT (mint) DO UPDATE SET symbol=EXCLUDED.symbol, name=EXCLUDED.name,
              decimals=EXCLUDED.decimals, token_program=EXCLUDED.token_program, launchpad=EXCLUDED.launchpad,
              partner_config=EXCLUDED.partner_config, graduated_pool=EXCLUDED.graduated_pool,
              graduated_at=EXCLUDED.graduated_at, first_pool_id=EXCLUDED.first_pool_id,
              first_pool_created_at=EXCLUDED.first_pool_created_at, mint_auth_disabled=EXCLUDED.mint_auth_disabled,
              freeze_auth_disabled=EXCLUDED.freeze_auth_disabled, last_seen=now(),
              watch_status=CASE WHEN EXCLUDED.watch_status='unknown' THEN tokens.watch_status
                                WHEN tokens.watch_status='watch' AND EXCLUDED.watch_status='pre' THEN tokens.watch_status
                                ELSE EXCLUDED.watch_status END,
              raw=EXCLUDED.raw""",
        (tok["id"], tok.get("symbol"), tok.get("name"), tok.get("decimals"), tok.get("tokenProgram"),
         tok.get("launchpad"), tok.get("partnerConfig"), tok.get("graduatedPool"), _ts(tok.get("graduatedAt")),
         fp.get("id"), _ts(fp.get("createdAt")), audit.get("mintAuthorityDisabled"), audit.get("freezeAuthorityDisabled"),
         status, json.dumps(tok, default=str)),
    )


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


def ensure_watch_pool(conn, tok: dict) -> bool:
    """Create the watch_pools row for a 'watch' token (evaluate() guarantees graduatedPool). Returns True if newly added."""
    mint, pool, source = tok["id"], tok["graduatedPool"], "jupiter.graduatedPool"
    label = PUMP_AMM_LABEL if (tok.get("launchpad") or "").lower() == "pump.fun" else None
    row = conn.execute("SELECT pool, active, source, reason FROM watch_pools WHERE pool = %s", (pool,)).fetchone()
    if row:
        # re-arm only a deactivation the token has now disproved (authorities disabled again, the only reason
        # process_tokens deactivates for); pools retired by maintain_pools (watch window elapsed) stay retired so the two do not flap
        if not row["active"] and row["source"] != "smoke" and "authority" in (row["reason"] or "").lower():
            conn.execute("UPDATE watch_pools SET active = true, deactivated_at = NULL, reason = NULL WHERE pool = %s", (pool,))
        return False
    conn.execute(
        """INSERT INTO watch_pools (pool, mint, quote_mint, program_label, base_decimals, source, active, tradable)
           VALUES (%s,%s,%s,%s,%s,%s,true,true) ON CONFLICT (pool) DO NOTHING""",
        (pool, mint, config.WSOL_MINT if label == PUMP_AMM_LABEL else None, label, tok.get("decimals"), source),
    )
    return True


def process_tokens(conn, toks: list[dict], with_stats: bool) -> dict:
    counts = {"seen": 0, "watch": 0, "pre": 0, "excluded": 0, "unknown": 0, "new_pools": 0, "deactivated": 0}
    for tok in toks:
        if not tok.get("id"):
            continue
        counts["seen"] += 1
        status, reason = evaluate(tok)
        counts[status] += 1
        upsert_token(conn, tok, status)
        if with_stats:
            insert_stats(conn, tok)
        if status == "watch":
            if ensure_watch_pool(conn, tok):
                counts["new_pools"] += 1
                record_event("info", "discover", "new watch pool", {"mint": tok["id"], "symbol": tok.get("symbol"),
                                                                      "pool": tok.get("graduatedPool"), "launchpad": tok.get("launchpad")})
        elif status == "excluded" and tok.get("graduatedAt") and "authority" in reason:
            # only a token that regained an authority is deactivated; a payload missing graduatedPool/audit is not a reason
            n = conn.execute(
                "UPDATE watch_pools SET active=false, deactivated_at=now(), reason=%s WHERE mint=%s AND active",
                (reason, tok["id"]),
            ).rowcount
            counts["deactivated"] += n
    return counts


def maintain_pools(conn) -> int:
    """Deactivate pools past the watch window with no recent swaps and no open position.

    Only runs once the capture tape has at least 24 h of history; before that "no swaps in 24 h" is
    vacuously true and would deactivate every pool."""
    cov = conn.execute(
        "SELECT min(ts) AS first_ts, max(ts) AS last_ts FROM swap_tape WHERE ts > now() - interval '30 hours'"
    ).fetchone()
    if not cov or cov["first_ts"] is None or cov["last_ts"] is None:
        return 0
    if (cov["last_ts"] - cov["first_ts"]).total_seconds() < 24 * 3600:
        return 0
    n = conn.execute(
        """UPDATE watch_pools wp SET active=false, deactivated_at=now(), reason='watch window elapsed'
           FROM tokens t WHERE wp.mint = t.mint AND wp.active AND wp.source <> 'smoke'
             AND COALESCE(t.graduated_at, wp.added_at) < now() - (%s * interval '1 day')
             AND NOT EXISTS (SELECT 1 FROM swap_tape s WHERE s.pool = wp.pool AND s.ts > now() - interval '24 hours')
             AND NOT EXISTS (SELECT 1 FROM positions p WHERE p.mint = wp.mint AND p.status = 'open')""",
        (config.WATCH_DAYS_AFTER_GRADUATION,),
    ).rowcount
    return n


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


def probe() -> None:
    setup("discover", to_file=False)
    client = JupiterTokens()
    toks = client.recent()
    for cat, iv in CATEGORIES_FAST + CATEGORIES_SLOW:
        toks += client.category(cat, iv)
    toks = dedupe(toks)
    from collections import Counter
    lps = Counter(t.get("launchpad") for t in toks)
    grad = [t for t in toks if t.get("graduatedAt")]
    print(f"tokens={len(toks)} graduated={len(grad)} with_pool={sum(1 for t in grad if t.get('graduatedPool'))}")
    print("launchpads:", dict(lps))
    print("graduated by launchpad:", dict(Counter(t.get("launchpad") for t in grad)))
    ok = [t for t in grad if evaluate(t)[0] == "watch"]
    print(f"pass gate with LAUNCHPADS={config.LAUNCHPADS}: {len(ok)}")
    print("token programs among gate-passing:", dict(Counter(t.get("tokenProgram") for t in ok)))


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
    record_event("info", "discover", "discover worker stopped")
