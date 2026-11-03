"""PumpAPI historical replay → hourly Parquet (Supervisor thread ``replay``).

pumpapi.io publishes every event its decoded data stream emitted, hour by hour, since 2026-04-18:
``https://replay.pumpapi.io/YYYY/MM/DD/HH.jsonl.zst`` (~400 MB zstd, ~2 GB JSONL, ~1.7 M events; free, no key).
Each line is one decoded transaction: buys/sells with the trader, SOL and token amounts, price, and the pool's real
reserves; ``create`` with the creator wallet and metadata URI; ``migrate`` (graduation); ``createPool``.

This worker downloads ``REPLAY_PARALLEL`` hours at a time (measured ~2–4 MB/s per stream), parses each in one
thread and writes, per hour, ``data/corpus/replay/<date>/<HH>_trades.parquet`` (pump.fun bonding-curve and
PumpSwap buys/sells in any quote, and since 2026-10-23 every other venue's SOL-quoted buys/sells: Meteora DLMM, DAMM,
Raydium, Orca, other launchpads' curves; one row per trader leg; signatures and per-mint constants are left out to keep it
~90 MB/hour, the lifecycle table carries creator, pool id, metadata URI and the reserves at migration) and
``<HH>_events.parquet`` (every non-trade lifecycle event across all pools, small; the quote mint since 2026-10-23). Both are
written atomically; the trades file is stamped with what it keeps (``TRADES_VERSION``), and ``refetch_main`` downloads
the hours stamped otherwise again in its own process (``fly-trader refetch-archive``). Hours are processed newest first so the current regime lands first, and hours that close while the backfill runs
are fetched ahead of it, so the archive stays within ~2 hours of live; the registry is ``replay_hours``. The compressed download is a temporary file and is removed after a successful parse — the
public archive remains the source of truth. Downstream: ``train/replay_assemble.py`` turns days into per-token
candle/trade files for the feature builder.
"""
from __future__ import annotations

import io
import json
import logging
import os
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
from .corpus_pull import _db_retry, _sleep

log = logging.getLogger(__name__)
HOUR_SQL = ("INSERT INTO replay_hours (hour, status, last_error) VALUES (%s, %s, %s) "
            "ON CONFLICT (hour) DO UPDATE SET status = EXCLUDED.status, last_error = EXCLUDED.last_error, done_at = now()")
TRADE_POOLS = {"pump", "pump-amm"}          # kept in any quote; every other venue's legs are kept when SOL-quoted
WSOL = "So11111111111111111111111111111111111111112"
TRADES_VERSION = "pumpfun+sol-1"             # the trades file's ``fly_pools`` stamp: what its legs cover (refetch_main)
SKIP_ACTIONS = {"buy", "sell", "transfer", "add", "remove", "claimCreatorFees"}
TRADE_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("slot", pa.int64()), ("pool", pa.dictionary(pa.int8(), pa.string())), ("mint", pa.string()),
                          ("trader", pa.string()), ("side", pa.int8()), ("sol", pa.float64()), ("tokens", pa.float64()), ("price", pa.float64()),
                          ("quote_in_pool", pa.float64()), ("tokens_in_pool", pa.float64()), ("vquote", pa.float32()), ("n_legs", pa.int8()),
                          ("quote_mint", pa.dictionary(pa.int16(), pa.string())), ("pool_id", pa.dictionary(pa.int32(), pa.string()))])
EVENT_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("slot", pa.int64()), ("sig", pa.string()), ("action", pa.string()), ("pool", pa.string()),
                          ("mint", pa.string()), ("pool_id", pa.string()), ("signer", pa.string()), ("creator_fee_addr", pa.string()), ("name", pa.string()),
                          ("symbol", pa.string()), ("uri", pa.string()), ("supply", pa.float64()), ("decimals", pa.int16()), ("token_program", pa.string()),
                          ("initial_buy", pa.float64()), ("quote_amount", pa.float64()), ("quote_in_pool", pa.float64()), ("tokens_in_pool", pa.float64()),
                          ("mayhem", pa.bool_()), ("pool_created_by", pa.string()), ("mint_authority", pa.string()), ("freeze_authority", pa.string()),
                          ("quote_mint", pa.string())])


def _f(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _q(d: dict, new: str, old: str):
    """A quote-side field under its current name, else under the archive's pre-2026-05-21 SOL-only name."""
    v = d.get(new)
    return v if v is not None else d.get(old)


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
            # the archive before 2026-05-21 names fields txType / solAmount / solInPool / vSolInBondingCurve; later: action / quote*
            a = e.get("action") or e.get("txType"); pool = e.get("pool")
            ts = e.get("timestamp")
            if ts is None:
                continue
            tsdt = datetime.fromtimestamp(int(ts) / 1000, timezone.utc)
            if a in ("buy", "sell"):
                if pool not in TRADE_POOLS and not (pool and e.get("quoteMint") == WSOL):
                    continue
                legs = e.get("breakdown") or [{"action": a, "trader": e.get("txSigner"), "tokenAmount": e.get("tokenAmount"), "quoteAmount": _q(e, "quoteAmount", "solAmount")}]
                vq = _q(e, "vQuoteInBondingCurve", "vSolInBondingCurve") if pool == "pump" else _q(e, "virtualQuoteInPool", "virtualSolInPool")
                for b in legs:
                    T["ts"].append(tsdt); T["slot"].append(int(e.get("block") or 0)); T["pool"].append(pool); T["mint"].append(e.get("mint"))
                    T["trader"].append(b.get("trader") or e.get("txSigner")); T["side"].append(1 if (b.get("action") or b.get("txType")) == "buy" else -1)
                    T["sol"].append(_f(_q(b, "quoteAmount", "solAmount")) or 0.0); T["tokens"].append(_f(b.get("tokenAmount")) or 0.0); T["price"].append(_f(e.get("price")))
                    T["quote_in_pool"].append(_f(_q(e, "quoteInPool", "solInPool"))); T["tokens_in_pool"].append(_f(e.get("tokensInPool"))); T["vquote"].append(_f(vq))
                    T["n_legs"].append(len(legs)); T["quote_mint"].append(e.get("quoteMint")); T["pool_id"].append(e.get("poolId"))
            elif a not in SKIP_ACTIONS:
                E["ts"].append(tsdt); E["slot"].append(int(e.get("block") or 0)); E["sig"].append(e.get("signature")); E["action"].append(a); E["pool"].append(pool)
                E["mint"].append(e.get("mint")); E["pool_id"].append(e.get("poolId")); E["signer"].append(e.get("txSigner")); E["creator_fee_addr"].append(e.get("creatorFeeAddress"))
                E["name"].append(e.get("name")); E["symbol"].append(e.get("symbol")); E["uri"].append(e.get("uri")); E["supply"].append(_f(e.get("supply")))
                E["decimals"].append(int(e["decimals"]) if e.get("decimals") is not None else None); E["token_program"].append(e.get("tokenProgram"))
                E["initial_buy"].append(_f(e.get("initialBuy"))); E["quote_amount"].append(_f(_q(e, "quoteAmount", "solAmount"))); E["quote_in_pool"].append(_f(_q(e, "quoteInPool", "solInPool")))
                E["tokens_in_pool"].append(_f(e.get("tokensInPool"))); E["mayhem"].append(bool(e.get("mayhemMode")) if e.get("mayhemMode") is not None else None)
                E["pool_created_by"].append(e.get("poolCreatedBy")); E["mint_authority"].append(e.get("mintAuthority")); E["freeze_authority"].append(e.get("freezeAuthority"))
                E["quote_mint"].append(e.get("quoteMint"))
    return pa.table(T, schema=TRADE_SCHEMA), pa.table(E, schema=EVENT_SCHEMA), n


def write_hour(day: Path, h: datetime, trades: pa.Table, events: pa.Table) -> None:
    """The hour's two files, each written atomically (a reader never meets half a file, and a rewrite by ``refetch_main``
    replaces it whole), the trades file stamped with what its legs cover (``TRADES_VERSION``)."""
    day.mkdir(parents=True, exist_ok=True)
    trades = trades.replace_schema_metadata({**(trades.schema.metadata or {}), b"fly_pools": TRADES_VERSION.encode()})
    for t, name in ((trades, f"{h:%H}_trades.parquet"), (events, f"{h:%H}_events.parquet")):
        tmp = day / f"{name}.{os.getpid()}.{threading.get_ident()}.tmp"
        pq.write_table(t, tmp, compression="zstd"); tmp.replace(day / name)


def trades_version(path) -> str | None:
    """The ``fly_pools`` stamp of an hour's trades file (None: written before 2026-10-23, PumpSwap and the curve only)."""
    v = (pq.read_schema(path).metadata or {}).get(b"fly_pools")
    return v.decode() if v else None


def refetch_missing_pool_id() -> int:
    """Hours whose trades file predates the ``pool_id`` column (written 2026-08-20..09-13 by an older parser) are marked
    ``refetch``: ``plan_hours`` downloads them again and ``parse_hour`` writes the column, so the archive applies the live
    stream's blocked-pool filter and dominant-pool choice (train/mature.py). Returns the number of hours marked."""
    hours = []
    for f in sorted(config.REPLAY_DIR.glob("*/*_trades.parquet")):
        if "pool_id" not in pq.read_schema(f).names:
            d = datetime.fromisoformat(f.parent.name)
            hours.append(datetime(d.year, d.month, d.day, int(f.name[:2]), tzinfo=timezone.utc))
    if hours:
        with transaction() as conn:
            conn.cursor().executemany(HOUR_SQL, [(h, "refetch", "trades file lacks pool_id") for h in hours])
    return len(hours)


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


def _put_hour(sql: str, params: tuple) -> None:
    with transaction() as conn:
        conn.execute(sql, params)


def _kept_downloads(tmpdir: Path, planned: set[datetime]) -> list[tuple[datetime, Path, int, float]]:
    """Complete hour files left by a stopped worker, for hours still planned: parsed instead of downloaded again.
    Partial files are removed (the archive ignores Range requests, so a download cannot resume), as are complete
    files for hours no longer planned."""
    out = []
    for f in sorted(tmpdir.glob("*")):
        if f.suffix == ".zst":
            try:
                h = datetime.strptime(f.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                h = None
            if h in planned:
                out.append((h, f, f.stat().st_size, 0.0)); continue
        f.unlink()
    return out


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
    def __init__(self, total: int, key: str = "replay_status"):
        self.key = key
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
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (self.key, json.dumps(payload, default=str)))
        except Exception:
            log.debug("replay status write failed", exc_info=True)


def main(stop_event: threading.Event | None = None) -> None:
    setup("replay"); stop = stop_event
    tmpdir = config.REPLAY_DIR / "_tmp"; tmpdir.mkdir(parents=True, exist_ok=True)
    now0 = datetime.now(timezone.utc); hours = plan_hours(now0); status = _Status(len(hours))
    preloaded = _kept_downloads(tmpdir, set(hours))                 # finished before a restart: parse, don't download again
    if preloaded:
        kept = {p[0] for p in preloaded}; hours = [h for h in hours if h not in kept]
        log.info("replay ingest: %d complete hour file(s) kept from before the restart", len(preloaded))
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

    if config.REPLAY_ASSEMBLE:                   # REPLAY_ASSEMBLE=0: a download-only run (e.g. refetch-pool-ids) leaves the builds alone
        from ..train import replay_assemble
        assembler = threading.Thread(target=replay_assemble.assemble_loop, args=(stop,), name="replay-assemble", daemon=True); assembler.start()
    threads = [threading.Thread(target=downloader, name=f"replay-dl-{i}", daemon=True) for i in range(config.REPLAY_PARALLEL)]
    for t in threads:
        t.start()
    try:
        while not (stop is not None and stop.is_set()):
            try:
                h, tmp, size, dl_s = preloaded.pop(0) if preloaded else ready.get(timeout=5.0)
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
            replan()
            if tmp is None or size < 0:
                _db_retry(stop, "replay_hours write", _put_hour, HOUR_SQL, (h, "missing" if size < 0 else "error", None if size < 0 else "download failed"))
                status.update(errors=status.s["errors"] + (0 if size < 0 else 1)); continue
            t0 = time.time()
            try:
                trades, events, n = parse_hour(tmp)
                write_hour(config.REPLAY_DIR / f"{h:%Y-%m-%d}", h, trades, events)
                tmp.unlink()
                _db_retry(stop, "replay_hours write", _put_hour,
                          "INSERT INTO replay_hours (hour, status, trades, events, bytes, took_s) VALUES (%s,'done',%s,%s,%s,%s) "
                          "ON CONFLICT (hour) DO UPDATE SET status = 'done', trades = EXCLUDED.trades, events = EXCLUDED.events, bytes = EXCLUDED.bytes, took_s = EXCLUDED.took_s, last_error = NULL, done_at = now()",
                          (h, trades.num_rows, events.num_rows, size, dl_s + time.time() - t0))
                status.update(hours_done=status.s["hours_done"] + 1, trades=status.s["trades"] + trades.num_rows, events=status.s["events"] + events.num_rows,
                              bytes=status.s["bytes"] + size, last_hour=h.isoformat(), last_parse_s=round(time.time() - t0, 1), last_dl_s=round(dl_s, 1))
                log.info("hour %s: %d events -> %d trade rows, %d lifecycle rows (%.0f MB, dl %.0fs, parse %.1fs)", h.isoformat(), n, trades.num_rows, events.num_rows, size / 1e6, dl_s, time.time() - t0)
            except Exception as e:
                log.exception("parse %s failed", h)
                tmp.unlink(missing_ok=True)      # the ~400 MB download; replan() refetches the hour
                _db_retry(stop, "replay_hours write", _put_hour, HOUR_SQL, (h, "error", str(e)[:300]))
                status.update(errors=status.s["errors"] + 1)
    finally:
        if stop is not None:
            stop.set()
        status.update(stage="stopped", force=True)
        record_event("info", "replay", "replay ingest stopped", {k: status.s.get(k) for k in ("hours_done", "hours_total", "trades", "events", "bytes", "errors")})
        log.info("replay ingest stopped")


# ---------------------------------------------------------------- downloading the archive again (fly-trader refetch-archive)
def refetch_plan(now: datetime | None = None) -> list[datetime]:
    """Hours whose trades file is stamped otherwise than ``TRADES_VERSION`` (written before the files kept every venue's SOL
    legs), newest first, leaving the newest ``REFETCH_SKIP_RECENT_H`` hours to the console's replay worker."""
    now = now or datetime.now(timezone.utc)
    cut = now - timedelta(hours=config.REFETCH_SKIP_RECENT_H)
    out = []
    for f in config.REPLAY_DIR.glob("*/*_trades.parquet"):
        try:
            d = datetime.fromisoformat(f.parent.name)
            h = datetime(d.year, d.month, d.day, int(f.name[:2]), tzinfo=timezone.utc)
        except ValueError:
            continue
        if h < cut and trades_version(f) != TRADES_VERSION:
            out.append(h)
    return sorted(out, reverse=True)


def live_ok() -> tuple[bool, str]:
    """The live stream keeps up: its status is fresh and its newest event is at most ``REFETCH_MAX_LAG_S`` behind the
    status write. A refetch download competes with it for this machine's connection, so none starts while it lags."""
    try:
        with transaction() as conn:
            r = conn.execute("SELECT value, updated_at FROM ui_settings WHERE key = 'pumpstream_status'").fetchone()
    except Exception as e:
        return False, f"status unreadable ({e})"
    if not r:
        return False, "no stream status"
    v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")
    upd = r["updated_at"]; age = (datetime.now(timezone.utc) - upd).total_seconds()
    if age > 60:
        return False, f"stream status {age:.0f} s old"
    wm = v.get("event_watermark")
    if not wm:
        return False, "no stream watermark"
    lag = (upd - datetime.fromisoformat(wm)).total_seconds()
    return (lag <= config.REFETCH_MAX_LAG_S, f"stream {lag:.0f} s behind")


def refetch_main(stop_event: threading.Event | None = None) -> dict:
    """Download again every hour ``refetch_plan`` lists, ``REFETCH_PARALLEL`` at a time, in this process (run it apart from
    the console, at low priority: ``nice -n 19 fly-trader refetch-archive``). Each hour is parsed and written as the replay
    worker would (``write_hour``); a download starts only while the live stream keeps up (``live_ok``). When a day's hours
    are all rewritten its aggregate is stale (train/mature.agg_stale: inputs newer) and a build round is tried here, never
    waiting behind another process's round. Status: ``ui_settings['replay_refetch_status']``."""
    setup("replay_refetch"); stop = stop_event or threading.Event()
    hours = refetch_plan(); status = _Status(len(hours), "replay_refetch_status"); status.update(stage="refetching", force=True)
    by_day: dict = {}
    for h in hours:
        by_day.setdefault(h.date(), set()).add(h)
    log.info("archive refetch: %d hours on %d days to download again (%d at a time)", len(hours), len(by_day), config.REFETCH_PARALLEL)
    tmpdir = config.REPLAY_DIR / "_refetch_tmp"; tmpdir.mkdir(parents=True, exist_ok=True)
    for f in tmpdir.glob("*"):
        f.unlink()
    todo: queue.Queue = queue.Queue(); ready: queue.Queue = queue.Queue(maxsize=config.REFETCH_PARALLEL + 1)
    for h in hours:
        todo.put(h)
    client = httpx.Client(headers={"User-Agent": "fly-trader replay refetch"}, timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)
    paused = {"why": None}

    def downloader():
        while not stop.is_set():
            ok, why = live_ok()
            if not ok:
                if paused["why"] is None:
                    log.info("refetch paused: %s", why)
                paused["why"] = why; status.update(paused=why); stop.wait(30.0); continue
            if paused["why"] is not None:
                log.info("refetch resumed"); paused["why"] = None; status.update(paused=None)
            try:
                h = todo.get_nowait()
            except queue.Empty:
                return
            tmp = tmpdir / f"{h:%Y%m%d%H}.zst"
            for attempt in range(4):
                try:
                    t0 = time.time(); size = _download(client, h, tmp, stop)
                    ready.put((h, tmp, size, time.time() - t0)); break
                except InterruptedError:
                    return
                except Exception as e:
                    log.warning("refetch download %s failed (%d): %s", h, attempt, e); stop.wait(10 * (attempt + 1))
            else:
                ready.put((h, None, 0, 0.0))

    threads = [threading.Thread(target=downloader, name=f"replay_refetch-dl-{i}", daemon=True) for i in range(config.REFETCH_PARALLEL)]
    for t in threads:
        t.start()
    from ..train import mature
    rebuild: dict = {"thread": None}
    done = failed = 0
    try:
        while not stop.is_set() and (any(t.is_alive() for t in threads) or not ready.empty()):
            try:
                h, tmp, size, dl_s = ready.get(timeout=5.0)
            except queue.Empty:
                continue
            if tmp is None or size < 0:
                failed += 1; status.update(errors=failed)
                log.warning("refetch %s: %s; the hour keeps its old file", h, "not in the archive" if size < 0 else "download failed")
            else:
                t0 = time.time()
                try:
                    trades, events, n = parse_hour(tmp)
                    write_hour(config.REPLAY_DIR / f"{h:%Y-%m-%d}", h, trades, events)
                    done += 1
                    status.update(hours_done=done, bytes=status.s["bytes"] + size, last_hour=h.isoformat(), last_parse_s=round(time.time() - t0, 1), last_dl_s=round(dl_s, 1))
                    log.info("refetched %s: %d events -> %d trade rows (%.0f MB, dl %.0fs, parse %.1fs)", h.isoformat(), n, trades.num_rows, size / 1e6, dl_s, time.time() - t0)
                except Exception:
                    failed += 1; status.update(errors=failed); log.exception("refetch parse %s failed; the hour keeps its old file", h)
                finally:
                    tmp.unlink(missing_ok=True)
            day = by_day.get(h.date())
            if day is not None:
                day.discard(h)
                if not day:                                  # the day is rewritten: rebuild it here (one round at a time)
                    del by_day[h.date()]
                    if rebuild["thread"] is None or not rebuild["thread"].is_alive():
                        def _round(d=h.date()):
                            try:
                                n = mature.try_loop_once()
                                log.info("refetch: build round after %s: %s", d, "another process holds the build" if n is None else f"{n} day steps")
                            except Exception:
                                log.exception("refetch: build round after %s failed (the console's round rebuilds it)", d)
                        rebuild["thread"] = threading.Thread(target=_round, name="replay_refetch-build", daemon=True); rebuild["thread"].start()
    finally:
        stop.set()
        if rebuild["thread"] is not None:
            rebuild["thread"].join()                     # its day's files are written atomically either way
        status.update(stage="stopped", force=True)
        log.info("archive refetch stopped: %d hours rewritten, %d failed, %d planned", done, failed, len(hours))
    return {"planned": len(hours), "done": done, "failed": failed}
