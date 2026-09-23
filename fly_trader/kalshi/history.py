"""The Kalshi corpus (worker ``kalshi_history``, CLI ``kalshi-history``): every settled market since ``KALSHI_HISTORY_START``
with its 1-minute candles and public trades.

1. Seed (once, idempotent): the open dataset at ``KALSHI_DATASET_DIR`` (jon-becker/prediction-market-analysis:
   ``data/kalshi/markets/*.parquet`` and ``data/kalshi/trades/*.parquet``) → ``kalshi_markets`` rows (source 'dataset')
   and ``trades/<ticker>.parquet`` for every settled yes/no market that closed on or after the start.
2. Refresh from the exchange, resumable and paced (``config.KALSHI_RPS``): settled markets from the live tier
   (``GET /markets?status=settled``) and the archive (``GET /historical/markets``), events (category, mutual exclusivity)
   and series (category, frequency, fee multiplier) on first sight.
3. Candles: for every settled market without a candle file, 1-minute candles over the last ``CANDLE_DAYS`` days before
   close (the strategies only enter inside 10 days; the features look back 7) in ≤ ``CHUNK_MIN``-minute requests, on the
   series path with the historical path as the 404 fallback; trades for markets the dataset did not cover.
Registry: ``kalshi_corpus`` (status pending → done | empty | error); progress in ``ui_settings['kalshi_history_status']``.
"""
from __future__ import annotations

import glob
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from . import data as D
from .client import KalshiApiError, KalshiRest

log = logging.getLogger(__name__)
CANDLE_DAYS = 17.0
CHUNK_MIN = 5000
SEED_KEY = "kalshi_dataset_seed"
WALK_KEY = "kalshi_refresh_walk"          # where the exchange walk is (tier, page cursor) and through when it last completed
RECOVER_DAYS = 3                          # a completed walk is re-covered this far back next round: a market settles after it closes
STATUS_KEY = "kalshi_history_status"
IDLE_S = 300.0


def _start() -> datetime:
    return datetime.fromisoformat(config.KALSHI_HISTORY_START).replace(tzinfo=timezone.utc)


def _status(**kv) -> None:
    try:
        with transaction() as conn:
            r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (STATUS_KEY,)).fetchone()
            cur = (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")) if r else {}
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (STATUS_KEY, json.dumps({**cur, **kv, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str)))
    except Exception:
        log.debug("history status write failed", exc_info=True)


def _sleep(s: float, stop: threading.Event | None) -> None:
    end = time.time() + s
    while time.time() < end and not (stop is not None and stop.is_set()):
        time.sleep(min(1.0, end - time.time()))


# ---------------------------------------------------------------- 1. dataset seed
def seed_from_dataset(stop: threading.Event | None = None) -> dict:
    """Idempotent: skips when ``ui_settings[kalshi_dataset_seed]`` records the same dataset files."""
    root = Path(config.KALSHI_DATASET_DIR).expanduser() if config.KALSHI_DATASET_DIR else None
    if root is None or not (root / "data" / "kalshi" / "markets").exists():
        return {"skipped": "no dataset at KALSHI_DATASET_DIR"}
    import duckdb
    mfiles = sorted(glob.glob(str(root / "data" / "kalshi" / "markets" / "*.parquet"))); tfiles = sorted(glob.glob(str(root / "data" / "kalshi" / "trades" / "*.parquet")))
    stamp = {"markets": [Path(f).name for f in mfiles], "trades": [Path(f).name for f in tfiles], "start": config.KALSHI_HISTORY_START}
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (SEED_KEY,)).fetchone()
    if r and (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")).get("stamp") == stamp:
        return {"skipped": "already seeded"}
    t0 = time.time(); con = duckdb.connect()
    cols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [mfiles]).description]
    want = ["ticker", "event_ticker", "market_type", "title", "yes_sub_title", "no_sub_title", "status", "result", "open_time", "close_time", "volume", "open_interest"]
    sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in want)
    # the most traded markets of each close day: volume ≥ KALSHI_MIN_MARKET_VOLUME, at most KALSHI_MARKETS_PER_DAY per day (6.8 M settled
    # yes/no markets since 2025-01-01 are mostly hourly ladders that never traded; ~1.3 s of API per market bounds the corpus)
    rows = con.execute(f"""SELECT {sel} FROM read_parquet(?, union_by_name = true)
                           WHERE result IN ('yes', 'no') AND close_time >= ? AND COALESCE(volume, 0) >= ?
                           QUALIFY row_number() OVER (PARTITION BY CAST(close_time AS DATE) ORDER BY volume DESC, ticker) <= ?
                           ORDER BY close_time DESC""", [mfiles, _start().replace(tzinfo=None), float(config.KALSHI_MIN_MARKET_VOLUME), int(config.KALSHI_MARKETS_PER_DAY)]).fetchall()
    markets = [dict(zip(want, r)) for r in rows]
    log.info("dataset seed: %d settled yes/no markets since %s (volume ≥ %g, ≤ %d per day)", len(markets), config.KALSHI_HISTORY_START, config.KALSHI_MIN_MARKET_VOLUME, config.KALSHI_MARKETS_PER_DAY)
    n = 0
    with transaction() as conn:
        for i in range(0, len(markets), 5000):
            chunk = markets[i:i + 5000]
            D.upsert_markets(conn, [{**m, "settlement_ts": m.get("close_time")} for m in chunk], "dataset")
            conn.cursor().executemany("INSERT INTO kalshi_corpus (ticker, status, settled_ts, result, open_time, close_time, seeded_from, volume) VALUES (%s,'pending',%s,%s,%s,%s,'dataset',%s) "
                                      "ON CONFLICT (ticker) DO UPDATE SET result = COALESCE(kalshi_corpus.result, EXCLUDED.result), close_time = COALESCE(kalshi_corpus.close_time, EXCLUDED.close_time), "
                                      "volume = COALESCE(EXCLUDED.volume, kalshi_corpus.volume)",
                                      [(m["ticker"], D.ts(str(m["close_time"])), m["result"], D.ts(str(m["open_time"])) if m.get("open_time") else None, D.ts(str(m["close_time"])),
                                        float(m["volume"]) if m.get("volume") is not None else None) for m in chunk])
            n += len(chunk)
    # trades: one file per market, written from the dataset's trade table restricted to the seeded tickers
    n_tr = 0
    if tfiles and markets:
        con.execute("CREATE TEMP TABLE seeded AS SELECT * FROM (VALUES " + ",".join("(?)" for _ in markets) + ") t(ticker)", [m["ticker"] for m in markets])
        tcols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [tfiles]).description]
        side = "taker_side" if "taker_side" in tcols else "NULL"
        con.execute(f"""CREATE TEMP TABLE tr AS SELECT t.ticker, t.created_time AS ts, CAST(t.yes_price AS INTEGER) AS yes_price, CAST(t.count AS DOUBLE) AS count,
                        {side} AS taker_side FROM read_parquet(?, union_by_name = true) t JOIN seeded s USING (ticker)""", [tfiles])
        tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM tr").fetchall()]
        done: list[tuple[str, str, int]] = []
        for i, tk in enumerate(tickers):
            if stop is not None and stop.is_set():
                break
            tab = con.execute("SELECT ts, yes_price, count, taker_side FROM tr WHERE ticker = ? ORDER BY ts", [tk]).fetch_arrow_table()
            rows_t = [{"ts": r["ts"].replace(tzinfo=timezone.utc) if r["ts"].tzinfo is None else r["ts"], "yes_price": int(r["yes_price"]), "count": float(r["count"]),
                       "taker_side": r["taker_side"] or "", "is_block": False} for r in tab.to_pylist()]
            path = D.TRADES_DIR / f"{tk}.parquet"
            if not path.exists():
                D.write_parquet(rows_t, D.TRADE_SCHEMA, path)
            done.append((str(path), len(rows_t), tk)); n_tr += 1
            if len(done) >= 2000 or i == len(tickers) - 1:
                with transaction() as conn:
                    conn.cursor().executemany("UPDATE kalshi_corpus SET trade_path = %s, trades = %s, updated_at = now() WHERE ticker = %s", done)
                done = []
                _status(stage="seeding trades", seed_trades=n_tr, seed_trades_total=len(tickers))
    con.close()
    if not (stop is not None and stop.is_set()):
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (SEED_KEY, json.dumps({"stamp": stamp, "markets": n, "trades": n_tr, "at": datetime.now(timezone.utc).isoformat()})))
    log.info("dataset seed: %d markets, %d trade files in %.0fs", n, n_tr, time.time() - t0)
    return {"markets": n, "trades": n_tr}


# ---------------------------------------------------------------- 2. exchange refresh
def _known_events(conn) -> set[str]:
    return {r["event_ticker"] for r in conn.execute("SELECT event_ticker FROM kalshi_events").fetchall()}


def _known_series(conn) -> set[str]:
    return {r["ticker"] for r in conn.execute("SELECT ticker FROM kalshi_series").fetchall()}


def ensure_catalogue(rest: KalshiRest, conn, event_tickers: set[str], known_events: set[str], known_series: set[str]) -> None:
    """Events and series first seen: one GET each (category, mutual exclusivity, frequency, fee multiplier)."""
    for et in sorted(event_tickers - known_events):
        try:
            e = (rest.event(et, with_nested_markets=False) or {}).get("event") or {}
        except KalshiApiError as ex:
            log.warning("event %s: %s", et, ex); e = {"event_ticker": et}
        D.upsert_event(conn, e); known_events.add(et)
        st = e.get("series_ticker") or D.series_ticker_of(et)
        if st and st not in known_series:
            try:
                D.upsert_series(conn, rest.series(st))
            except KalshiApiError as ex:
                log.warning("series %s: %s", st, ex); D.upsert_series(conn, {"ticker": st})
            known_series.add(st)


def market_volume(m: dict) -> float | None:
    v = m.get("volume_fp") if m.get("volume_fp") is not None else m.get("volume")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def walk_state() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (WALK_KEY,)).fetchone()
    return (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")) if r else {}


def _save_walk(conn, st: dict) -> None:
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                 (WALK_KEY, json.dumps(st, default=str)))


def refresh_markets(rest: KalshiRest, stop: threading.Event | None = None, max_pages: int = 10_000) -> int:
    """Settled yes/no markets with volume ≥ KALSHI_MIN_MARKET_VOLUME closing after the dataset's newest day (or the start),
    from the live tier then the archive (both page newest first). The walk is resumable: after every page its tier and page
    cursor are saved with that page's rows (``ui_settings[WALK_KEY]``), so a timeout, a crash or a restart continues from the
    next page instead of walking the newest months again. A completed walk records when it began; later rounds walk only the
    live tier for markets closing since then (less ``RECOVER_DAYS``), and the archive is never walked twice."""
    start = _start(); n = 0
    with transaction() as conn:
        known_e, known_s = _known_events(conn), _known_series(conn)
        r = conn.execute("SELECT max(close_time) AS t FROM kalshi_corpus WHERE seeded_from = 'dataset'").fetchone()
    if r and r["t"] and r["t"] > start:
        start = r["t"]                                         # the archive covers the days before; the API fills from there on
    min_vol = float(config.KALSHI_MIN_MARKET_VOLUME)
    st = walk_state(); now = datetime.now(timezone.utc)
    if st.get("complete_through"):
        lower = max(start, datetime.fromisoformat(st["complete_through"]) - timedelta(days=RECOVER_DAYS)); tiers = ["live"]
    else:
        lower = start; tiers = ["live", "historical"]
    tier = st.get("tier") if st.get("tier") in tiers else None; cursor = st.get("cursor") if tier else None
    if tier is None:                                           # a fresh walk: from the newest settlements down to ``lower``
        tier = tiers[0]; cursor = None; st = {**st, "walk_started": now.isoformat(), "tier": tier, "cursor": None, "lower": lower.isoformat()}
        with transaction() as conn:
            _save_walk(conn, st)
    if st.get("lower"):
        lower = max(lower, datetime.fromisoformat(st["lower"]))
    for tier in tiers[tiers.index(tier):]:
        gen = (rest.markets(status="settled", mve_filter="exclude", min_close_ts=int(lower.timestamp()), cursor=cursor, with_cursor=True) if tier == "live"
               else rest.historical_markets(mve_filter="exclude", cursor=cursor, with_cursor=True))
        for k, (page, nxt) in enumerate(gen):
            if stop is not None and stop.is_set() or k >= max_pages:
                return n
            settled = [m for m in page if (m.get("result") in ("yes", "no")) and (D.ts(m.get("close_time")) or lower) >= lower and (market_volume(m) or 0.0) >= min_vol]
            closes = [D.ts(m.get("close_time")) for m in page if D.ts(m.get("close_time"))]
            done_tier = not nxt or (closes and max(closes) < lower)
            with transaction() as conn:
                D.upsert_markets(conn, settled, tier)
                conn.cursor().executemany("INSERT INTO kalshi_corpus (ticker, status, settled_ts, result, open_time, close_time, seeded_from, volume) VALUES (%s,'pending',%s,%s,%s,%s,%s,%s) "
                                          "ON CONFLICT (ticker) DO UPDATE SET settled_ts = COALESCE(EXCLUDED.settled_ts, kalshi_corpus.settled_ts), result = COALESCE(EXCLUDED.result, kalshi_corpus.result), "
                                          "open_time = COALESCE(EXCLUDED.open_time, kalshi_corpus.open_time), close_time = COALESCE(EXCLUDED.close_time, kalshi_corpus.close_time), "
                                          "volume = COALESCE(EXCLUDED.volume, kalshi_corpus.volume)",
                                          [(m["ticker"], D.ts(m.get("settlement_ts") or m.get("settled_time")), m.get("result"), D.ts(m.get("open_time")), D.ts(m.get("close_time")), tier, market_volume(m)) for m in settled])
                ensure_catalogue(rest, conn, {m["event_ticker"] for m in settled if m.get("event_ticker")}, known_e, known_s)
                nxt_tier = tiers[tiers.index(tier) + 1] if done_tier and tiers.index(tier) + 1 < len(tiers) else (None if done_tier else tier)
                st = {**st, "tier": nxt_tier, "cursor": None if done_tier else nxt, "oldest_close": min(closes).isoformat() if closes else st.get("oldest_close")}
                if done_tier and nxt_tier is None:             # every tier walked: settlements since this walk began are the next round's work
                    st = {**st, "complete_through": st.get("walk_started") or now.isoformat(), "walk_started": None, "lower": None}
                _save_walk(conn, st)                           # the page's rows and the place after them commit together
            n += len(settled)
            _status(stage=f"refreshing {tier} markets", refreshed=n, tier_page=k, oldest_close=st.get("oldest_close"))
            if done_tier:
                break
        cursor = None
    return n


def prune_pending() -> int:
    """Pending corpus rows beyond the corpus's bounds go to 'skipped': volume below KALSHI_MIN_MARKET_VOLUME (or unknown), or
    not among the KALSHI_MARKETS_PER_DAY most traded of their close day. Volume unknown on the row is read from the market's raw JSON."""
    with transaction() as conn:
        conn.execute("""UPDATE kalshi_corpus c SET volume = COALESCE(NULLIF(m.raw->>'volume_fp', '')::float, NULLIF(m.raw->>'volume', '')::float)
                        FROM kalshi_markets m WHERE m.ticker = c.ticker AND c.volume IS NULL AND m.raw IS NOT NULL""")
        a = conn.execute("UPDATE kalshi_corpus SET status = 'skipped', last_error = 'volume below the corpus minimum', updated_at = now() "
                         "WHERE status = 'pending' AND COALESCE(volume, 0) < %s", (float(config.KALSHI_MIN_MARKET_VOLUME),)).rowcount
        b = conn.execute("""UPDATE kalshi_corpus c SET status = 'skipped', last_error = 'beyond the corpus''s markets per day', updated_at = now()
                            FROM (SELECT ticker, row_number() OVER (PARTITION BY date_trunc('day', close_time) ORDER BY volume DESC NULLS LAST, ticker) AS rk
                                  FROM kalshi_corpus WHERE status IN ('pending', 'done', 'built') AND close_time IS NOT NULL) r
                            WHERE r.ticker = c.ticker AND c.status = 'pending' AND r.rk > %s""", (int(config.KALSHI_MARKETS_PER_DAY),)).rowcount
    return int(a or 0) + int(b or 0)


# ---------------------------------------------------------------- 3. candles and trades
def _series_for(conn, ticker: str) -> str:
    r = conn.execute("SELECT e.series_ticker FROM kalshi_markets m LEFT JOIN kalshi_events e USING (event_ticker) WHERE m.ticker = %s", (ticker,)).fetchone()
    return (r["series_ticker"] if r and r["series_ticker"] else None) or D.series_ticker_of(ticker)


def pull_candles(rest: KalshiRest, ticker: str, series: str, open_time: datetime | None, close_time: datetime, settled: datetime | None) -> list[dict]:
    end = settled or close_time
    lo = max(open_time or (close_time - timedelta(days=CANDLE_DAYS)), close_time - timedelta(days=CANDLE_DAYS))
    t = int(lo.timestamp()) // 60 * 60; t_end = int(end.timestamp()) // 60 * 60 + 60
    out: list[dict] = []
    while t < t_end:
        hi = min(t + CHUNK_MIN * 60, t_end)
        out.extend(D.candle_rows(rest.candlesticks(series, ticker, t, hi, 1)))
        t = hi
    seen = set(); uniq = []
    for c in out:
        if c["end_ts"] not in seen:
            seen.add(c["end_ts"]); uniq.append(c)
    return sorted(uniq, key=lambda c: c["end_ts"])


def pull_trades(rest: KalshiRest, ticker: str, close_time: datetime, settled: datetime | None) -> list[dict]:
    lo = int((close_time - timedelta(days=CANDLE_DAYS)).timestamp()); hi = int((settled or close_time).timestamp()) + 60
    cutoff = None
    try:
        cutoff = D.ts((rest.historical_cutoff() or {}).get("trades_created_ts"))
    except KalshiApiError:
        pass
    historical = bool(cutoff and close_time < cutoff)
    rows: list[dict] = []
    for page in rest.trades(ticker, min_ts=lo, max_ts=hi, historical=historical):
        rows.extend(D.trade_rows(page))
    return sorted(rows, key=lambda r: r["ts"])


def fill(rest: KalshiRest, stop: threading.Event | None = None, limit: int = 200) -> int:
    """Candles (and trades where missing) for pending corpus rows, newest first; returns markets attempted."""
    with transaction() as conn:
        todo = conn.execute("SELECT c.ticker, c.open_time, c.close_time, c.settled_ts, c.candle_path, c.trade_path FROM kalshi_corpus c "
                            "WHERE c.status = 'pending' AND c.close_time IS NOT NULL ORDER BY c.close_time DESC LIMIT %s", (limit,)).fetchall()
        series = {r["ticker"]: _series_for(conn, r["ticker"]) for r in todo}
    n = 0
    for r in todo:
        if stop is not None and stop.is_set():
            break
        tk = r["ticker"]; upd = {"status": "done", "last_error": None}
        try:
            if not r["candle_path"]:
                cs = pull_candles(rest, tk, series[tk], r["open_time"], r["close_time"], r["settled_ts"])
                if cs:
                    p = D.CANDLES_DIR / f"{tk}.parquet"; D.write_parquet(cs, D.CANDLE_SCHEMA, p); upd.update(candle_path=str(p), candles_1m=len(cs))
                else:
                    upd["status"] = "empty"
            if not r["trade_path"]:
                tr = pull_trades(rest, tk, r["close_time"], r["settled_ts"])
                p = D.TRADES_DIR / f"{tk}.parquet"; D.write_parquet(tr, D.TRADE_SCHEMA, p); upd.update(trade_path=str(p), trades=len(tr))
        except KalshiApiError as e:
            upd = {"status": "error" if e.status not in (404,) else "empty", "last_error": str(e)[:300]}
        except Exception as e:
            log.exception("candles %s failed", tk); upd = {"status": "error", "last_error": f"{type(e).__name__}: {e}"[:300]}
        with transaction() as conn:
            conn.execute("UPDATE kalshi_corpus SET " + ", ".join(f"{k} = %s" for k in upd) + ", updated_at = now() WHERE ticker = %s", (*upd.values(), tk))
        n += 1
    return n


def counts() -> dict:
    with transaction() as conn:
        rows = conn.execute("SELECT status, count(*) AS n FROM kalshi_corpus GROUP BY status").fetchall()
        first = conn.execute("SELECT min(close_time) AS a, max(close_time) AS b FROM kalshi_corpus WHERE status = 'done'").fetchone()
    return {**{r["status"]: int(r["n"]) for r in rows}, "first_done": first["a"], "last_done": first["b"]}


def main(stop_event: threading.Event | None = None) -> None:
    setup("kalshi_history"); stop = stop_event
    record_event("info", "kalshi_history", "kalshi history started", {"start": config.KALSHI_HISTORY_START, "dataset": config.KALSHI_DATASET_DIR})
    rest = KalshiRest(subaccount=0)
    try:
        while not (stop is not None and stop.is_set()):
            try:
                _status(stage="seeding from the dataset"); seeded = seed_from_dataset(stop)
                _status(stage="refreshing markets", seed=seeded); n_new = refresh_markets(rest, stop)
                skipped = prune_pending()
                c = counts(); _status(stage="filling candles", refreshed=n_new, skipped_now=skipped, **{k: v for k, v in c.items()})
                t0 = time.time(); done = 0
                while not (stop is not None and stop.is_set()):
                    k = fill(rest, stop)
                    if not k:
                        break
                    done += k; c = counts()
                    rate = done / max(time.time() - t0, 1e-9)
                    _status(stage="filling candles", filled_this_run=done, markets_per_h=rate * 3600, eta_h=(c.get("pending", 0) / rate / 3600) if rate else None, **c)
                _status(stage="idle", **counts())
            except Exception as e:
                log.exception("kalshi history round failed"); record_event("error", "kalshi_history", f"round failed: {type(e).__name__}: {e}")
                _status(stage="error", last_error=str(e)[:200])
            _sleep(IDLE_S, stop)
    finally:
        rest.close(); _status(stage="stopped")
        record_event("info", "kalshi_history", "kalshi history stopped", counts())
