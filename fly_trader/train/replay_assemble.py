"""Replay days → per-token corpus files (thread inside the ``replay`` worker; also ``fly-trader assemble-replay``).

A day D is assembled once every hour of D and the first 12 hours of D+1 are in ``replay_hours``. For each
pump.fun graduation on D (``migrate`` on pool ``pump-amm`` in the lifecycle rows) it writes the same files the
swap-api puller writes, so ``train/corpus_features.py`` needs no changes:

* ``data/corpus/trades/<mint>.parquet``: every curve and PumpSwap trade leg in [graduation − 30 min, + CORPUS_TRADES_H]
  (wallet, side, SOL, tokens, price) — for every token, not a sample;
* ``data/corpus/candles/<mint>.parquet``: 1-minute OHLCV in SOL for the first CORPUS_CANDLE_1M_H hours and 5-minute
  candles for the rest of the 36 h scanned, plus ``resq_sol`` (the pool's real quote reserve at the last trade of the
  candle), which the feature builder prefers over its sqrt(price) approximation when present;
* a ``corpus_tokens`` row (source 'replay', status 'done') with the graduation time and slot from the migrate event.

DuckDB scans the day's hourly trade files with the graduated-mint list pushed down; a day is ~2–3 GB on disk.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..ingest.corpus_pull import CANDLE_SCHEMA, TRADE_SCHEMA

log = logging.getLogger(__name__)
CANDLE_SCHEMA_R = CANDLE_SCHEMA.append(pa.field("resq_sol", pa.float64()))


def _hours_done() -> set[datetime]:
    with transaction() as conn:
        return {r["hour"] for r in conn.execute("SELECT hour FROM replay_hours WHERE status IN ('done', 'missing')").fetchall()}


def assemblable_days() -> list[date]:
    """Days whose 24 hours and the next day's first 12 hours are ingested, newest first, not yet assembled."""
    done = _hours_done()
    if not done:
        return []
    with transaction() as conn:
        did = {r["day"] for r in conn.execute("SELECT day FROM replay_days").fetchall()}
    days = sorted({h.astimezone(timezone.utc).date() for h in done}, reverse=True); out = []
    for d in days:
        if d in did:
            continue
        need = [datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc) for h in range(24)]
        nxt = d + timedelta(days=1)
        need += [datetime(nxt.year, nxt.month, nxt.day, h, tzinfo=timezone.utc) for h in range(12)]
        if all(h in done for h in need):
            out.append(d)
    return out


def _files(d: date, kind: str, hours: range) -> list[str]:
    day = config.REPLAY_DIR / d.isoformat()
    return [str(day / f"{h:02d}_{kind}.parquet") for h in hours if (day / f"{h:02d}_{kind}.parquet").exists()]


def assemble_day(d: date) -> tuple[int, int]:
    t0 = time.time(); nxt = d + timedelta(days=1)
    con = duckdb.connect()
    ev_files = _files(d, "events", range(24))
    grads = con.execute("SELECT mint, min(ts) AS g, arg_min(slot, ts) AS slot, arg_min(quote_in_pool, ts) AS rq0 FROM read_parquet(?) "
                        "WHERE action = 'migrate' AND pool = 'pump-amm' AND mint IS NOT NULL GROUP BY mint", [ev_files]).fetchall()
    if not grads:
        return 0, 0
    mints = [g[0] for g in grads]
    tr_files = _files(d, "trades", range(24)) + _files(nxt, "trades", range(12))
    con.execute("CREATE TEMP TABLE grads AS SELECT * FROM (VALUES " + ",".join("(?, ?)" for _ in grads) + ") t(mint, g)",
                [v for g in grads for v in (g[0], g[1])])
    # every trade leg of a graduated mint within [g − 30 min, g + 36 h]
    con.execute("CREATE TEMP TABLE tr AS SELECT t.mint, t.ts, t.slot, t.pool, t.trader, t.side, t.sol, t.tokens, t.price, t.quote_in_pool, "
                "epoch_ms(t.ts) - epoch_ms(g.g) AS rel_ms FROM read_parquet(?) t JOIN grads g USING (mint) "
                "WHERE t.ts >= g.g - INTERVAL 30 MINUTE AND t.ts < g.g + INTERVAL 36 HOUR AND t.price > 0 "
                "AND (t.pool = 'pump' OR t.quote_in_pool BETWEEN 0.001 AND 100000)", [tr_files])
    # one price scale per mint: drop legs more than 50x away from the mint's median AMM price (secondary pools in other quotes)
    con.execute("CREATE TEMP TABLE pm AS SELECT mint, median(price) AS pmed FROM tr WHERE pool = 'pump-amm' GROUP BY mint")
    con.execute("DELETE FROM tr WHERE pool = 'pump-amm' AND mint IN (SELECT mint FROM pm) AND price NOT BETWEEN "
                "(SELECT pmed FROM pm WHERE pm.mint = tr.mint) / 50 AND (SELECT pmed FROM pm WHERE pm.mint = tr.mint) * 50")
    (cdir := config.CORPUS_DIR / "candles").mkdir(parents=True, exist_ok=True); (tdir := config.CORPUS_DIR / "trades").mkdir(parents=True, exist_ok=True)
    win1 = int(config.CORPUS_CANDLE_1M_H * 3_600_000); trades_ms = int(config.CORPUS_TRADES_H * 3_600_000)
    # per-mint trade rows for the first hours
    tr_tab = con.execute("SELECT mint, ts, CAST(slot AS VARCHAR) AS slot_index, trader AS wallet, side, "
                         "CASE WHEN pool = 'pump-amm' THEN 'pump_amm' ELSE 'pump' END AS program, price AS price_sol, sol, tokens "
                         "FROM tr WHERE rel_ms < ? ORDER BY mint, ts", [trades_ms]).fetch_arrow_table()
    # candles: 1-minute inside win1, 5-minute after
    cd_tab = con.execute("""
        SELECT mint, bucket AS ts, interval, first(price ORDER BY ts, slot) AS open, max(price) AS high, min(price) AS low,
               last(price ORDER BY ts, slot) AS close, sum(sol) AS volume_sol, last(quote_in_pool ORDER BY ts, slot) AS resq_sol
        FROM (SELECT *, CASE WHEN rel_ms < ? THEN '1m' ELSE '5m' END AS interval,
                        CASE WHEN rel_ms < ? THEN time_bucket(INTERVAL 1 MINUTE, ts) ELSE time_bucket(INTERVAL 5 MINUTE, ts) END AS bucket
              FROM tr WHERE pool = 'pump-amm' OR rel_ms < 0)
        GROUP BY mint, bucket, interval ORDER BY mint, bucket""", [win1, win1]).fetch_arrow_table()
    n_written = 0; rows = []
    tr_by = tr_tab.to_pandas().groupby("mint") if tr_tab.num_rows else None
    cd_by = cd_tab.to_pandas().groupby("mint") if cd_tab.num_rows else None
    for mint, g, slot, rq0 in grads:
        g_dt = g if g.tzinfo else g.replace(tzinfo=timezone.utc)
        cpath, tpath = cdir / f"{mint}.parquet", tdir / f"{mint}.parquet"
        upd = {"life_h": None, "candles_1m": 0, "candles_5m": 0, "trades": 0}
        if cd_by is not None and mint in cd_by.groups:
            c = cd_by.get_group(mint).drop(columns=["mint"])
            upd["candles_1m"] = int((c["interval"] == "1m").sum()); upd["candles_5m"] = int((c["interval"] == "5m").sum())
            upd["life_h"] = max(0.0, (c["ts"].max().timestamp() - g_dt.timestamp()) / 3600.0)
            if not cpath.exists():
                pq.write_table(pa.Table.from_pandas(c, schema=CANDLE_SCHEMA_R, preserve_index=False), cpath, compression="zstd")
        if tr_by is not None and mint in tr_by.groups:
            t = tr_by.get_group(mint).drop(columns=["mint"]); t.insert(2, "tx", None)
            upd["trades"] = len(t)
            if not tpath.exists():
                pq.write_table(pa.Table.from_pandas(t, schema=TRADE_SCHEMA, preserve_index=False), tpath, compression="zstd")
        rows.append((mint, g_dt, int(slot or 0), "replay", "done" if cpath.exists() else "empty", upd["life_h"], upd["candles_1m"], upd["candles_5m"], upd["trades"],
                     g_dt + timedelta(milliseconds=trades_ms) if upd["trades"] else None, str(cpath) if cpath.exists() else None, str(tpath) if tpath.exists() else None))
        n_written += 1
    with transaction() as conn:
        conn.cursor().executemany(
            "INSERT INTO corpus_tokens (mint, graduated_at, grad_slot, source, status, life_h, candles_1m, candles_5m, trades, trades_through, candle_path, trade_path) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint) DO UPDATE SET graduated_at = EXCLUDED.graduated_at, grad_slot = EXCLUDED.grad_slot, "
            "source = EXCLUDED.source, status = EXCLUDED.status, life_h = EXCLUDED.life_h, candles_1m = EXCLUDED.candles_1m, candles_5m = EXCLUDED.candles_5m, "
            "trades = EXCLUDED.trades, trades_through = EXCLUDED.trades_through, candle_path = EXCLUDED.candle_path, trade_path = EXCLUDED.trade_path, updated_at = now()", rows)
        conn.execute("INSERT INTO replay_days (day, graduations, tokens_written, took_s) VALUES (%s,%s,%s,%s) ON CONFLICT (day) DO UPDATE SET graduations = EXCLUDED.graduations, "
                     "tokens_written = EXCLUDED.tokens_written, took_s = EXCLUDED.took_s, assembled_at = now()", (d, len(grads), n_written, time.time() - t0))
    con.close()
    log.info("assembled %s: %d graduations, %d tokens written, %d trade rows, %d candle rows in %.0fs", d, len(grads), n_written, tr_tab.num_rows, cd_tab.num_rows, time.time() - t0)
    return len(grads), n_written


_last_meta = [0.0]


def assemble_loop(stop_event: threading.Event | None = None, idle_s: float = 60.0) -> None:
    while not (stop_event is not None and stop_event.is_set()):
        try:
            days = assemblable_days()
        except Exception:
            log.exception("assemble: planning failed"); days = []
        for d in days:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                assemble_day(d)
            except Exception:
                log.exception("assemble %s failed", d)
        if days or time.time() - _last_meta[0] > 1800:
            try:
                from . import corpus_meta
                corpus_meta.rebuild(); _last_meta[0] = time.time()
            except Exception:
                log.exception("corpus_meta rebuild failed; mature build waits for the next round")
            else:
                try:
                    from . import mature
                    mature.loop_once()
                except Exception:
                    log.exception("mature universe build failed")
        end = time.time() + idle_s
        while time.time() < end and not (stop_event is not None and stop_event.is_set()):
            time.sleep(1.0)


def main() -> None:
    from ..logging_setup import setup
    setup("replay")
    days = assemblable_days()
    print(f"{len(days)} days ready")
    for d in days:
        g, n = assemble_day(d); print(f"{d}: {g} graduations, {n} tokens")
