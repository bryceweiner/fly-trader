"""PumpAPI historical replay → hourly Parquet (Supervisor thread ``replay``).

pumpapi.io publishes every event its decoded data stream emitted, hour by hour, since 2026-04-18:
``https://replay.pumpapi.io/YYYY/MM/DD/HH.jsonl.zst`` (~400 MB zstd, ~2 GB JSONL, ~1.7 M events; free, no key).
Each line is one decoded transaction: buys/sells with the trader, SOL and token amounts, price, and the pool's real
reserves; ``create`` with the creator wallet and metadata URI; ``migrate`` (graduation); ``createPool``.

This worker downloads ``REPLAY_PARALLEL`` hours at a time (measured ~2–4 MB/s per stream), parses each in one
thread and writes, per hour, ``data/corpus/replay/<date>/<HH>_trades.parquet`` (pump.fun bonding-curve and
PumpSwap buys/sells, one row per trader leg; signatures and per-mint constants are left out to keep it ~70 MB/hour, the
lifecycle table carries creator, pool id, metadata URI and the reserves at migration) and ``<HH>_events.parquet`` (every non-trade lifecycle event across
all pools, small). Hours are processed newest first so the current regime lands first, and hours that close while the backfill runs
are fetched ahead of it, so the archive stays within ~2 hours of live; the registry is ``replay_hours``. The compressed download is a temporary file and is removed after a successful parse — the
public archive remains the source of truth. Downstream: ``train/replay_assemble.py`` turns days into per-token
candle/trade files for the feature builder.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from .corpus_pull import _sleep

log = logging.getLogger(__name__)
TRADE_POOLS = {"pump", "pump-amm"}
SKIP_ACTIONS = {"buy", "sell", "transfer", "add", "remove", "claimCreatorFees"}
TRADE_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("slot", pa.int64()), ("pool", pa.dictionary(pa.int8(), pa.string())), ("mint", pa.string()),
                          ("trader", pa.string()), ("side", pa.int8()), ("sol", pa.float64()), ("tokens", pa.float64()), ("price", pa.float64()),
                          ("quote_in_pool", pa.float64()), ("tokens_in_pool", pa.float64()), ("vquote", pa.float32()), ("n_legs", pa.int8()),
                          ("quote_mint", pa.dictionary(pa.int16(), pa.string())), ("pool_id", pa.dictionary(pa.int32(), pa.string()))])
EVENT_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("slot", pa.int64()), ("sig", pa.string()), ("action", pa.string()), ("pool", pa.string()),
                          ("mint", pa.string()), ("pool_id", pa.string()), ("signer", pa.string()), ("creator_fee_addr", pa.string()), ("name", pa.string()),
                          ("symbol", pa.string()), ("uri", pa.string()), ("supply", pa.float64()), ("decimals", pa.int16()), ("token_program", pa.string()),
                          ("initial_buy", pa.float64()), ("quote_amount", pa.float64()), ("quote_in_pool", pa.float64()), ("tokens_in_pool", pa.float64()),
                          ("mayhem", pa.bool_()), ("pool_created_by", pa.string()), ("mint_authority", pa.string()), ("freeze_authority", pa.string())])


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def parse_hour(path: Path) -> tuple[pa.Table, pa.Table, int]:
    """Stream-decompress one hour file into (trades, events, n_events)."""
    T = {name: [] for name in TRADE_SCHEMA.names}; E = {name: [] for name in EVENT_SCHEMA.names}; n = 0
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.BufferedReader(reader, buffer_size=8 << 20):
            if not line.strip():
                continue
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            n += 1
            a = e.get("action"); pool = e.get("pool")
            ts = e.get("timestamp")
            if ts is None:
                continue
            tsdt = datetime.fromtimestamp(int(ts) / 1000, timezone.utc)
            if a in ("buy", "sell"):
                if pool not in TRADE_POOLS:
                    continue
                legs = e.get("breakdown") or [{"action": a, "trader": e.get("txSigner"), "tokenAmount": e.get("tokenAmount"), "quoteAmount": e.get("quoteAmount")}]
                vq = e.get("vQuoteInBondingCurve") if pool == "pump" else e.get("virtualQuoteInPool")
                for b in legs:
                    T["ts"].append(tsdt); T["slot"].append(int(e.get("block") or 0)); T["pool"].append(pool); T["mint"].append(e.get("mint"))
                    T["trader"].append(b.get("trader") or e.get("txSigner")); T["side"].append(1 if b.get("action") == "buy" else -1)
                    T["sol"].append(_f(b.get("quoteAmount")) or 0.0); T["tokens"].append(_f(b.get("tokenAmount")) or 0.0); T["price"].append(_f(e.get("price")))
                    T["quote_in_pool"].append(_f(e.get("quoteInPool"))); T["tokens_in_pool"].append(_f(e.get("tokensInPool"))); T["vquote"].append(_f(vq))
                    T["n_legs"].append(len(legs)); T["quote_mint"].append(e.get("quoteMint")); T["pool_id"].append(e.get("poolId"))
            elif a not in SKIP_ACTIONS:
                E["ts"].append(tsdt); E["slot"].append(int(e.get("block") or 0)); E["sig"].append(e.get("signature")); E["action"].append(a); E["pool"].append(pool)
                E["mint"].append(e.get("mint")); E["pool_id"].append(e.get("poolId")); E["signer"].append(e.get("txSigner")); E["creator_fee_addr"].append(e.get("creatorFeeAddress"))
                E["name"].append(e.get("name")); E["symbol"].append(e.get("symbol")); E["uri"].append(e.get("uri")); E["supply"].append(_f(e.get("supply")))
                E["decimals"].append(int(e["decimals"]) if e.get("decimals") is not None else None); E["token_program"].append(e.get("tokenProgram"))
                E["initial_buy"].append(_f(e.get("initialBuy"))); E["quote_amount"].append(_f(e.get("quoteAmount"))); E["quote_in_pool"].append(_f(e.get("quoteInPool")))
                E["tokens_in_pool"].append(_f(e.get("tokensInPool"))); E["mayhem"].append(bool(e.get("mayhemMode")) if e.get("mayhemMode") is not None else None)
                E["pool_created_by"].append(e.get("poolCreatedBy")); E["mint_authority"].append(e.get("mintAuthority")); E["freeze_authority"].append(e.get("freezeAuthority"))
    return pa.table(T, schema=TRADE_SCHEMA), pa.table(E, schema=EVENT_SCHEMA), n


def plan_hours(now: datetime | None = None) -> list[datetime]:
    now = now or datetime.now(timezone.utc)   # callers pass the same ``now`` they seed the re-plan bound with
    start = datetime.fromisoformat(config.REPLAY_START).replace(tzinfo=timezone.utc)
    last = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    with transaction() as conn:
        done = {r["hour"] for r in conn.execute("SELECT hour FROM replay_hours WHERE status = 'done'").fetchall()}
    hours = []; h = last
    while h >= start:
        if h not in done:
            hours.append(h)
        h -= timedelta(hours=1)
    return hours     # newest first


def _download(client: httpx.Client, hour: datetime, tmp: Path, stop: threading.Event | None) -> int:
    url = f"{config.REPLAY_URL}/{hour:%Y/%m/%d/%H}.jsonl.zst"
    part = tmp.with_suffix(".part"); size = 0
    with client.stream("GET", url) as r:
        if r.status_code == 404:
            return -1
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                if stop is not None and stop.is_set():
                    raise InterruptedError
                f.write(chunk); size += len(chunk)
    part.rename(tmp)
    return size


class _Status:
    def __init__(self, total: int):
        self.s = {"hours_total": total, "hours_done": 0, "trades": 0, "events": 0, "bytes": 0, "started_at": datetime.now(timezone.utc).isoformat(), "errors": 0}
        self.lock = threading.Lock(); self.last = 0.0; self.t0 = time.time()

    def update(self, force=False, **kv):
        with self.lock:
            self.s.update(kv); el = time.time() - self.t0
            self.s["mb_s"] = self.s["bytes"] / 1e6 / el if el > 1 else 0.0
            self.s["hours_per_h"] = self.s["hours_done"] / el * 3600 if el > 1 else 0.0
            remaining = self.s["hours_total"] - self.s["hours_done"]
            self.s["eta_h"] = remaining / self.s["hours_per_h"] if self.s["hours_per_h"] else None
            self.s["updated_at"] = datetime.now(timezone.utc).isoformat()
            if not force and time.time() - self.last < 3.0:
                return
            self.last = time.time(); payload = dict(self.s)
        try:
            with transaction() as conn:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES ('replay_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (json.dumps(payload, default=str),))
        except Exception:
            log.debug("replay status write failed", exc_info=True)


def main(stop_event: threading.Event | None = None) -> None:
    setup("replay"); stop = stop_event
    tmpdir = config.REPLAY_DIR / "_tmp"; tmpdir.mkdir(parents=True, exist_ok=True)
    for stale in tmpdir.glob("*"):
        stale.unlink()
    now0 = datetime.now(timezone.utc); hours = plan_hours(now0); status = _Status(len(hours))
    record_event("info", "replay", "replay ingest started", {"hours_pending": len(hours), "parallel": config.REPLAY_PARALLEL})
    log.info("replay ingest: %d hours pending (newest first)", len(hours))
    todo: queue.Queue = queue.Queue(); ready: queue.Queue = queue.Queue(maxsize=config.REPLAY_PARALLEL + 2)
    for h in hours:
        todo.put(h)
    client = httpx.Client(headers={"User-Agent": "fly-trader replay ingest"}, timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)

    newest_planned = [(now0 - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)]   # the plan's upper bound (same clock)
    fresh: queue.Queue = queue.Queue()      # hours that closed after the plan was made; served before the backfill
    retried: dict[datetime, float] = {}

    def replan():
        last = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
        h = newest_planned[0] + timedelta(hours=1)
        while h <= last:
            fresh.put(h); h += timedelta(hours=1)
        if last > newest_planned[0]:
            newest_planned[0] = last
        # an hour fetched before the archive published it (404) or that failed is retried every 30 min for 12 h
        try:
            with transaction() as conn:
                again = [r["hour"] for r in conn.execute("SELECT hour FROM replay_hours WHERE status IN ('missing','error') AND hour > now() - interval '12 hours'").fetchall()]
        except Exception:
            again = []
        for h in again:
            if time.time() - retried.get(h, 0.0) >= 1800:
                retried[h] = time.time(); fresh.put(h)

    def downloader():
        while not (stop is not None and stop.is_set()):
            try:
                h = fresh.get_nowait()
            except queue.Empty:
                try:
                    h = todo.get_nowait()
                except queue.Empty:
                    return
            tmp = tmpdir / f"{h:%Y%m%d%H}.zst"
            for attempt in range(4):
                try:
                    t0 = time.time(); size = _download(client, h, tmp, stop)
                    while not (stop is not None and stop.is_set()):
                        try:
                            ready.put((h, tmp, size, time.time() - t0), timeout=5.0); break
                        except queue.Full:
                            continue
                    break
                except InterruptedError:
                    return
                except Exception as e:
                    log.warning("download %s failed (%d): %s", h, attempt, e); time.sleep(10 * (attempt + 1))
            else:
                while not (stop is not None and stop.is_set()):
                    try:
                        ready.put((h, None, 0, 0.0), timeout=5.0); break
                    except queue.Full:
                        continue

    from ..train import replay_assemble
    assembler = threading.Thread(target=replay_assemble.assemble_loop, args=(stop,), name="replay-assemble", daemon=True); assembler.start()
    threads = [threading.Thread(target=downloader, name=f"replay-dl-{i}", daemon=True) for i in range(config.REPLAY_PARALLEL)]
    for t in threads:
        t.start()
    try:
        pending = len(hours)
        while not (stop is not None and stop.is_set()):
            try:
                h, tmp, size, dl_s = ready.get(timeout=5.0)
            except queue.Empty:
                replan()
                if not any(t.is_alive() for t in threads) and ready.empty():
                    if fresh.empty():
                        _sleep(300, stop); replan()          # caught up: poll for the next closed hour
                        if fresh.empty():
                            continue
                    threads = [threading.Thread(target=downloader, name=f"replay-dl-{i}", daemon=True) for i in range(config.REPLAY_PARALLEL)]
                    for t in threads:
                        t.start()
                continue
            pending -= 1
            replan()
            if not fresh.empty():
                pending += fresh.qsize()      # keep the parser loop alive for the hours just queued
            if tmp is None or size < 0:
                with transaction() as conn:
                    conn.execute("INSERT INTO replay_hours (hour, status, last_error) VALUES (%s, %s, %s) ON CONFLICT (hour) DO UPDATE SET status = EXCLUDED.status, last_error = EXCLUDED.last_error, done_at = now()",
                                 (h, "missing" if size < 0 else "error", None if size < 0 else "download failed"))
                status.update(errors=status.s["errors"] + (0 if size < 0 else 1)); continue
            t0 = time.time()
            try:
                trades, events, n = parse_hour(tmp)
                day = config.REPLAY_DIR / f"{h:%Y-%m-%d}"; day.mkdir(parents=True, exist_ok=True)
                pq.write_table(trades, day / f"{h:%H}_trades.parquet", compression="zstd"); pq.write_table(events, day / f"{h:%H}_events.parquet", compression="zstd")
                tmp.unlink()
                with transaction() as conn:
                    conn.execute("INSERT INTO replay_hours (hour, status, trades, events, bytes, took_s) VALUES (%s,'done',%s,%s,%s,%s) "
                                 "ON CONFLICT (hour) DO UPDATE SET status = 'done', trades = EXCLUDED.trades, events = EXCLUDED.events, bytes = EXCLUDED.bytes, took_s = EXCLUDED.took_s, last_error = NULL, done_at = now()",
                                 (h, trades.num_rows, events.num_rows, size, dl_s + time.time() - t0))
                status.update(hours_done=status.s["hours_done"] + 1, trades=status.s["trades"] + trades.num_rows, events=status.s["events"] + events.num_rows,
                              bytes=status.s["bytes"] + size, last_hour=h.isoformat(), last_parse_s=round(time.time() - t0, 1), last_dl_s=round(dl_s, 1))
                log.info("hour %s: %d events -> %d trade rows, %d lifecycle rows (%.0f MB, dl %.0fs, parse %.1fs)", h.isoformat(), n, trades.num_rows, events.num_rows, size / 1e6, dl_s, time.time() - t0)
            except Exception as e:
                log.exception("parse %s failed", h)
                with transaction() as conn:
                    conn.execute("INSERT INTO replay_hours (hour, status, last_error) VALUES (%s,'error',%s) ON CONFLICT (hour) DO UPDATE SET status = 'error', last_error = EXCLUDED.last_error, done_at = now()", (h, str(e)[:300]))
                status.update(errors=status.s["errors"] + 1)
    finally:
        if stop is not None:
            stop.set()
        status.update(stage="stopped", force=True)
        record_event("info", "replay", "replay ingest stopped", {k: status.s.get(k) for k in ("hours_done", "hours_total", "trades", "events", "bytes", "errors")})
        log.info("replay ingest stopped")
