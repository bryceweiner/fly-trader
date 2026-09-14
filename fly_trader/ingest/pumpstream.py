"""PumpAPI live stream → per-minute rich candles (Supervisor thread ``pumpstream``).

``wss://stream.pumpapi.io`` is a free, keyless firehose (~400 events/s, one connection per IP) whose events are
identical to the replay archive. This worker keeps only what the selector needs: PumpSwap (``pump-amm``,
SOL-quoted) buys and sells of pump.fun-origin mints, aggregated per (mint, minute) into the same fields
``train/mature.py`` derives from the archive, written to ``pump_minutes`` as each minute closes; and the
lifecycle events (``create`` on the curve, ``migrate`` to PumpSwap) into ``pump_events``, from which
``corpus_meta`` rows are upserted immediately at graduation (creation time, creator, dev buy, reserve at
migration, and the creator's point-in-time history from the table). Rows older than ``PUMP_MINUTES_KEEP_DAYS``
are archived to ``data/corpus/stream/<day>.parquet`` before removal.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import ssl
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import certifi
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup

log = logging.getLogger(__name__)
WSOL = "So11111111111111111111111111111111111111112"


class Minute:
    __slots__ = ("open", "high", "low", "close", "buy", "sell", "nb", "ns", "traders", "resq", "pool_id")

    def __init__(self):
        self.open = None; self.high = -math.inf; self.low = math.inf; self.close = None; self.buy = 0.0; self.sell = 0.0
        self.nb = 0; self.ns = 0; self.traders = set(); self.resq = None; self.pool_id = None


class Aggregator:
    def __init__(self):
        self.minutes: dict[int, dict[str, Minute]] = defaultdict(dict)   # minute start (s) → mint → Minute
        self.creates: dict[str, dict] = {}                                 # mint → create facts (24 h)
        self.pending_creates: list[tuple[str, dict]] = []                          # creates not yet written to pump_events
        self.flushed_through: datetime | None = None                                # start of the newest minute written
        self.stats = {"events": 0, "trades": 0, "flushed_minutes": 0, "flushed_rows": 0, "creates": 0, "migrates": 0, "reconnects": 0, "started_at": datetime.now(timezone.utc).isoformat()}

    def ingest(self, e: dict) -> None:
        self.stats["events"] += 1
        a = e.get("action"); pool = e.get("pool"); mint = e.get("mint"); ts = e.get("timestamp")
        if ts is None or not mint:
            return
        if a in ("buy", "sell"):
            if pool != "pump-amm" or e.get("quoteMint") != WSOL or not str(mint).endswith("pump"):
                return
            price = e.get("price")
            if not price or price <= 0:
                return
            m = int(ts // 60000) * 60; row = self.minutes[m].get(mint)
            if row is None:
                row = self.minutes[m][mint] = Minute()
            legs = e.get("breakdown") or [{"action": a, "trader": e.get("txSigner"), "quoteAmount": e.get("quoteAmount")}]
            p = float(price)
            row.open = p if row.open is None else row.open; row.high = max(row.high, p); row.low = min(row.low, p); row.close = p
            for b in legs:
                sol = float(b.get("quoteAmount") or 0.0)
                if b.get("action") == "buy":
                    row.buy += sol; row.nb += 1
                else:
                    row.sell += sol; row.ns += 1
                if b.get("trader"):
                    row.traders.add(b["trader"])
            q = e.get("quoteInPool")
            if q is not None:
                row.resq = float(q)
            row.pool_id = e.get("poolId") or row.pool_id
            self.stats["trades"] += 1
        elif a == "create" and pool == "pump":
            c = {"ts": ts, "creator": e.get("txSigner"), "dev_sol": e.get("quoteAmount"), "dev_tokens": e.get("initialBuy"), "supply": e.get("supply"),
                 "mayhem": e.get("mayhemMode"), "name": e.get("name"), "symbol": e.get("symbol"), "uri": e.get("uri"), "sig": e.get("signature")}
            self.creates[mint] = c; self.pending_creates.append((mint, c))
            self.stats["creates"] += 1
            if len(self.creates) > 200_000:
                cutoff = ts - 86_400_000
                self.creates = {k: v for k, v in self.creates.items() if v["ts"] >= cutoff}
        elif a == "migrate" and pool == "pump-amm":
            self.stats["migrates"] += 1
            self._graduation(e)

    def write_creates(self) -> int:
        """Persist creates as they arrive (keyed by signature), so graduations after a restart still find their creation facts."""
        rows, self.pending_creates = self.pending_creates, []
        if not rows:
            return 0
        with transaction() as conn:
            conn.cursor().executemany("INSERT INTO pump_events (sig, ts, action, pool, mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri) VALUES (%s,%s,'create','pump',%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sig) DO NOTHING",
                                      [(c["sig"], datetime.fromtimestamp(c["ts"] / 1000, timezone.utc), m, c["creator"], c["dev_sol"], c["dev_tokens"], c["supply"], c["mayhem"], c["name"], c["symbol"], c["uri"]) for m, c in rows if c.get("sig")])
        return len(rows)

    def _graduation(self, e: dict) -> None:
        mint = e["mint"]; g = datetime.fromtimestamp(e["timestamp"] / 1000, timezone.utc); c = self.creates.get(mint)
        try:
            with transaction() as conn:
                if c is None:                                       # created before this process started: look it up
                    r = conn.execute("SELECT sig, ts, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri FROM pump_events WHERE mint = %s AND action = 'create' ORDER BY ts LIMIT 1", (mint,)).fetchone()
                    if r:
                        c = {"ts": r["ts"].timestamp() * 1000, "creator": r["signer"], "dev_sol": r["dev_sol"], "dev_tokens": r["dev_tokens"], "supply": r["supply"], "mayhem": r["mayhem"], "name": r["name"], "symbol": r["symbol"], "uri": r["uri"], "sig": r["sig"]}
                conn.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, pool_id, signer, quote_in_pool) VALUES (%s,%s,'migrate','pump-amm',%s,%s,%s,%s) ON CONFLICT (sig) DO NOTHING",
                             (e.get("signature"), g, mint, e.get("poolId"), e.get("txSigner"), e.get("quoteInPool")))
                if c:
                    conn.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri) VALUES (%s,%s,'create','pump',%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sig) DO NOTHING",
                                 (c["sig"], datetime.fromtimestamp(c["ts"] / 1000, timezone.utc), mint, c["creator"], c["dev_sol"], c["dev_tokens"], c["supply"], c["mayhem"], c["name"], c["symbol"], c["uri"]))
                hist = None; launches = None
                if c and c.get("creator"):
                    ct = datetime.fromtimestamp(c["ts"] / 1000, timezone.utc)
                    launches = conn.execute("SELECT count(*) AS n FROM (SELECT mint FROM pump_events WHERE signer = %s AND action = 'create' AND ts < %s UNION SELECT mint FROM corpus_meta WHERE creator = %s AND create_ts < %s) u",
                                            (c["creator"], ct, c["creator"], ct)).fetchone()["n"]      # prior creates, graduated or not (the training definition)
                    hist = conn.execute("""SELECT count(*) AS launches, count(*) FILTER (WHERE own_alive6h IS NOT NULL) AS known,
                                                  avg(CASE WHEN own_dd60 <= -0.9 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_alive6h IS NOT NULL) AS rug,
                                                  avg(CASE WHEN own_max60 >= 1.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_alive6h IS NOT NULL) AS moon
                                           FROM corpus_meta WHERE creator = %s AND create_ts < %s""", (c["creator"], datetime.fromtimestamp(c["ts"] / 1000, timezone.utc))).fetchone()
                conn.execute("""INSERT INTO corpus_meta (mint, create_ts, creator, dev_sol, dev_tokens, dev_share, supply, mayhem, uri, name, symbol, ttg_min, rq0, pool_id,
                                                         prior_launches, prior_grads, prior_known, prior_rug_share, prior_moon_share)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint) DO UPDATE SET
                                  create_ts = COALESCE(corpus_meta.create_ts, EXCLUDED.create_ts), creator = COALESCE(corpus_meta.creator, EXCLUDED.creator),
                                  dev_sol = COALESCE(corpus_meta.dev_sol, EXCLUDED.dev_sol), dev_tokens = COALESCE(corpus_meta.dev_tokens, EXCLUDED.dev_tokens),
                                  dev_share = COALESCE(corpus_meta.dev_share, EXCLUDED.dev_share), supply = COALESCE(corpus_meta.supply, EXCLUDED.supply),
                                  mayhem = COALESCE(corpus_meta.mayhem, EXCLUDED.mayhem), uri = COALESCE(corpus_meta.uri, EXCLUDED.uri), ttg_min = COALESCE(corpus_meta.ttg_min, EXCLUDED.ttg_min),
                                  rq0 = COALESCE(corpus_meta.rq0, EXCLUDED.rq0), pool_id = COALESCE(corpus_meta.pool_id, EXCLUDED.pool_id),
                                  prior_launches = COALESCE(corpus_meta.prior_launches, EXCLUDED.prior_launches), prior_grads = COALESCE(corpus_meta.prior_grads, EXCLUDED.prior_grads),
                                  prior_known = COALESCE(corpus_meta.prior_known, EXCLUDED.prior_known), prior_rug_share = COALESCE(corpus_meta.prior_rug_share, EXCLUDED.prior_rug_share),
                                  prior_moon_share = COALESCE(corpus_meta.prior_moon_share, EXCLUDED.prior_moon_share), updated_at = now()""",
                             (mint, datetime.fromtimestamp(c["ts"] / 1000, timezone.utc) if c else None, c["creator"] if c else None, c["dev_sol"] if c else None,
                              c["dev_tokens"] if c else None, (float(c["dev_tokens"]) / float(c["supply"])) if c and c.get("dev_tokens") and c.get("supply") else None,
                              c["supply"] if c else None, c["mayhem"] if c else None, c["uri"] if c else None, c["name"] if c else None, c["symbol"] if c else None,
                              ((e["timestamp"] - c["ts"]) / 60000.0) if c else None, e.get("quoteInPool"), e.get("poolId"),
                              int(launches) if launches is not None else None, int(hist["launches"]) if hist else None, int(hist["known"]) if hist else None,
                              float(hist["rug"]) if hist and hist["rug"] is not None else None, float(hist["moon"]) if hist and hist["moon"] is not None else None))
                conn.execute("""INSERT INTO corpus_tokens (mint, graduated_at, grad_slot, source, status) VALUES (%s,%s,%s,'stream','pending') ON CONFLICT (mint) DO NOTHING""",
                             (mint, g, int(e.get("block") or 0)))
        except Exception:
            log.exception("graduation upsert failed for %s", mint)

    def flush(self, before_minute: int) -> int:
        """Write every closed minute strictly older than ``before_minute`` (seconds)."""
        done = [m for m in self.minutes if m < before_minute]
        if not done:
            return 0
        rows = []
        for m in sorted(done):
            ts = datetime.fromtimestamp(m, timezone.utc)
            for mint, r in self.minutes.pop(m).items():
                if r.close is None:
                    continue
                rows.append((mint, ts, r.pool_id, r.open, r.high, r.low, r.close, r.buy, r.sell, r.nb, r.ns, len(r.traders), r.resq))
        if rows:
            with transaction() as conn:
                conn.cursor().executemany("INSERT INTO pump_minutes (mint, ts, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol) "
                                          "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint, ts) DO UPDATE SET close = EXCLUDED.close, high = greatest(pump_minutes.high, EXCLUDED.high), "
                                          "low = least(pump_minutes.low, EXCLUDED.low), buy_sol = pump_minutes.buy_sol + EXCLUDED.buy_sol, sell_sol = pump_minutes.sell_sol + EXCLUDED.sell_sol, "
                                          "n_buys = pump_minutes.n_buys + EXCLUDED.n_buys, n_sells = pump_minutes.n_sells + EXCLUDED.n_sells, n_traders = greatest(pump_minutes.n_traders, EXCLUDED.n_traders), resq_sol = EXCLUDED.resq_sol", rows)
        self.stats["flushed_minutes"] += len(done); self.stats["flushed_rows"] += len(rows)
        self.flushed_through = datetime.fromtimestamp(max(done), timezone.utc)
        return len(rows)


def archive_old(keep_days: int) -> int:
    """Archive minute rows older than ``keep_days`` to Parquet, then remove them from the table."""
    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=keep_days)
    with transaction() as conn:
        days = [r["d"] for r in conn.execute("SELECT DISTINCT date_trunc('day', ts) AS d FROM pump_minutes WHERE ts < %s", (cutoff,)).fetchall()]
    n = 0
    for d in days:
        with transaction() as conn:
            rows = conn.execute("SELECT * FROM pump_minutes WHERE ts >= %s AND ts < %s ORDER BY mint, ts", (d, d + timedelta(days=1))).fetchall()
            if rows:
                out = config.CORPUS_DIR / "stream"; out.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist([dict(r) for r in rows]), out / f"{d:%Y-%m-%d}.parquet", compression="zstd")
                conn.execute("DELETE FROM pump_minutes WHERE ts >= %s AND ts < %s", (d, d + timedelta(days=1)))
                n += len(rows)
    return n


def _status(agg: Aggregator, extra: dict) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('pumpstream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps({**agg.stats, **extra, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str),))
    except Exception:
        log.debug("pumpstream status write failed", exc_info=True)


async def _run(stop: threading.Event | None, agg: Aggregator) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where()); backoff = 1.0; last_status = 0.0; last_flush = 0.0; last_archive = 0.0; last_msg = 0.0; rate_n = 0; rate_t = time.time()
    while not (stop is not None and stop.is_set()):
        try:
            async with websockets.connect(config.PUMPSTREAM_URL, ssl=ctx, max_size=8_000_000, ping_interval=20, ping_timeout=20) as ws:
                log.info("pumpstream connected"); backoff = 1.0
                while not (stop is not None and stop.is_set()):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        log.warning("pumpstream: no events for 30 s; reconnecting"); break
                    last_msg = time.time(); rate_n += 1
                    try:
                        e = orjson.loads(msg)
                    except orjson.JSONDecodeError:
                        continue
                    if isinstance(e, dict):
                        agg.ingest(e)
                    now = time.time()
                    if now - last_flush >= 1.0:
                        cur = int(now // 60) * 60
                        agg.flush(cur if now % 60 >= 2 else cur - 60)   # the minute that just closed is written ~2 s after its end
                        last_flush = now
                    if now - last_status >= 5.0:
                        try:
                            agg.write_creates()
                        except Exception:
                            log.exception("create rows failed")
                        _status(agg, {"events_per_s": rate_n / max(now - rate_t, 1e-9), "pending_minutes": len(agg.minutes), "lag_s": now - last_msg, "connected": True,
                                      "flushed_through": agg.flushed_through.isoformat() if agg.flushed_through else None}); rate_n = 0; rate_t = now; last_status = now
                    if now - last_archive >= 3600:
                        n = archive_old(config.PUMP_MINUTES_KEEP_DAYS); last_archive = now
                        if n:
                            log.info("archived %d old minute rows", n)
        except Exception as e:
            agg.stats["reconnects"] += 1; log.warning("pumpstream: %s; reconnecting in %.0fs", e, backoff)
            _status(agg, {"connected": False, "last_error": str(e)[:200]})
            await asyncio.sleep(backoff); backoff = min(60.0, backoff * 2)
    agg.flush(int(time.time() // 60) * 60 + 60)


def main(stop_event: threading.Event | None = None) -> None:
    setup("pumpstream"); agg = Aggregator()
    record_event("info", "pumpstream", "pumpstream started", {"url": config.PUMPSTREAM_URL})
    try:
        asyncio.run(_run(stop_event, agg))
    finally:
        _status(agg, {"connected": False, "stage": "stopped"})
        record_event("info", "pumpstream", "pumpstream stopped", {k: agg.stats.get(k) for k in ("events", "trades", "flushed_rows", "creates", "migrates", "reconnects")})
