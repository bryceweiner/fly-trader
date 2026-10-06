"""The Kalshi corpus (worker ``kalshi_history``, CLI ``kalshi-history``): every settled market since ``KALSHI_HISTORY_START``
with its 1-minute candles and public trades.

Which markets: an **event sample** that never looks at a market's outcome or its later trading. Every settled single
market Kalshi listed goes into ``kalshi_universe`` (the dataset's, then the exchange walk's); per (Kalshi category of the
series, close day) whole events are taken in md5(event ticker) order until the day holds ``KALSHI_MARKETS_PER_DAY``
markets (``select_corpus``) — a fixed rate per category could not follow Kalshi's listings, which grew several-fold in 2026.
Until 2026-10-02 the corpus kept the markets with the most lifetime volume, which selects on the future: a cheap strike is
traded heavily mostly when the price runs to it, so kept 1–5c crypto-ladder sides returned +94.5 % per dollar against
−78 % for the rest, and the selector learned the selection (snapshots 151–153). Point-in-time liquidity is the decision
rows' eligibility gate (spread, 24 h volume, open interest at that minute), never the corpus's membership.

1. Seed (idempotent per rule): the open dataset at ``KALSHI_DATASET_DIR`` (jon-becker/prediction-market-analysis:
   ``data/kalshi/markets/*.parquet`` and ``data/kalshi/trades/*.parquet``) → ``kalshi_markets`` rows (source 'dataset')
   and ``trades/<ticker>.parquet`` for every settled yes/no market of a sampled event that closed on or after the start.
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
from concurrent.futures import ThreadPoolExecutor
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
SAMPLE_RULE = "event-md5-per-category-day-v1"
EXCLUDED_CATEGORIES = ("exotics",)            # Kalshi's multivariate combos: not single markets, not our universe
NOT_SAMPLED = "not in the event sample"


def event_u(event_ticker: str | None) -> float:
    """A uniform number in [0, 1) fixed by the event ticker alone: independent of the market's outcome and trading."""
    import hashlib
    return int(hashlib.md5(str(event_ticker or "").encode()).hexdigest()[:12], 16) / float(16 ** 12)


def series_categories(conn) -> dict[str, str]:
    return {r["ticker"]: (r["category"] or "unknown").strip().lower() for r in conn.execute("SELECT ticker, category FROM kalshi_series").fetchall()}


def refresh_series(rest: KalshiRest) -> int:
    """Every series with its category (one call): the sample's categories come from here."""
    try:
        ss = rest.series_list()
    except KalshiApiError as e:
        log.warning("series list: %s", e); return 0
    with transaction() as conn:
        for x in ss:
            D.upsert_series(conn, x)
    return len(ss)


def seed_from_dataset(stop: threading.Event | None = None) -> dict:
    """The dataset's settled single markets since the start, all of them, into ``kalshi_universe`` (source 'dataset').
    Idempotent: skips when ``ui_settings[kalshi_dataset_seed]`` records the same files and rule."""
    root = Path(config.KALSHI_DATASET_DIR).expanduser() if config.KALSHI_DATASET_DIR else None
    if root is None or not (root / "data" / "kalshi" / "markets").exists():
        return {"skipped": "no dataset at KALSHI_DATASET_DIR"}
    import duckdb
    mfiles = sorted(glob.glob(str(root / "data" / "kalshi" / "markets" / "*.parquet")))
    stamp = {"markets": [Path(f).name for f in mfiles], "start": config.KALSHI_HISTORY_START, "rule": SAMPLE_RULE}
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (SEED_KEY,)).fetchone()
    if r and (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")).get("stamp") == stamp:
        return {"skipped": "already seeded"}
    t0 = time.time(); con = duckdb.connect()
    cols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [mfiles]).description]
    title = "title" if "title" in cols else "NULL"
    rows = con.execute(f"""SELECT ticker, event_ticker, split_part(event_ticker, '-', 1) AS series_ticker, {title} AS title, open_time, close_time, result,
                                  CAST(volume AS DOUBLE) AS volume
                           FROM read_parquet(?, union_by_name = true) WHERE result IN ('yes', 'no') AND close_time >= ? AND event_ticker IS NOT NULL""",
                       [mfiles, _start().replace(tzinfo=None)]).fetchall()
    con.close()
    n = 0
    with transaction() as conn:
        conn.execute("CREATE TEMP TABLE u_in (LIKE kalshi_universe INCLUDING DEFAULTS) ON COMMIT DROP")
        with conn.cursor().copy("COPY u_in (ticker, event_ticker, series_ticker, title, open_time, close_time, settled_ts, result, volume, source) FROM STDIN") as cp:
            for tk, et, st, ti, ot, ct, res, vol in rows:
                ot = ot.replace(tzinfo=timezone.utc) if ot is not None and ot.tzinfo is None else ot; ct = ct.replace(tzinfo=timezone.utc) if ct.tzinfo is None else ct
                cp.write_row((tk, et, st, ti, ot, ct, ct, res, vol, "dataset")); n += 1
        conn.execute("""INSERT INTO kalshi_universe (ticker, event_ticker, series_ticker, title, open_time, close_time, settled_ts, result, volume, source)
                        SELECT ticker, event_ticker, series_ticker, title, open_time, close_time, settled_ts, result, volume, source FROM u_in
                        ON CONFLICT (ticker) DO NOTHING""")
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (SEED_KEY, json.dumps({"stamp": stamp, "markets": n, "at": datetime.now(timezone.utc).isoformat()})))
    log.info("dataset universe: %d settled single markets since %s in %.0fs", n, config.KALSHI_HISTORY_START, time.time() - t0)
    return {"universe": n}


def seed_dataset_trades(stop: threading.Event | None = None) -> int:
    """Trade files from the dataset for selected corpus markets that have none yet (the dataset covers its own months)."""
    root = Path(config.KALSHI_DATASET_DIR).expanduser() if config.KALSHI_DATASET_DIR else None
    tfiles = sorted(glob.glob(str(root / "data" / "kalshi" / "trades" / "*.parquet"))) if root else []
    if not tfiles:
        return 0
    with transaction() as conn:
        todo = [r["ticker"] for r in conn.execute("""SELECT k.ticker FROM kalshi_corpus k JOIN kalshi_universe u USING (ticker)
                                                     WHERE u.source = 'dataset' AND k.trade_path IS NULL AND k.status IN ('pending', 'done')""").fetchall()]
    if not todo:
        return 0
    import duckdb
    import pandas as pd
    con = duckdb.connect(); con.register("want", pd.DataFrame({"ticker": todo}))
    tcols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [tfiles]).description]
    side = "taker_side" if "taker_side" in tcols else "NULL"
    con.execute(f"""CREATE TEMP TABLE tr AS SELECT t.ticker, t.created_time AS ts, CAST(t.yes_price AS INTEGER) AS yes_price, CAST(t.count AS DOUBLE) AS count,
                    {side} AS taker_side FROM read_parquet(?, union_by_name = true) t JOIN want USING (ticker)""", [tfiles])
    have = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM tr").fetchall()]; done: list = []; n = 0
    for i, tk in enumerate(have):
        if stop is not None and stop.is_set():
            break
        tab = con.execute("SELECT ts, yes_price, count, taker_side FROM tr WHERE ticker = ? ORDER BY ts", [tk]).fetch_arrow_table()
        rows_t = [{"ts": r["ts"].replace(tzinfo=timezone.utc) if r["ts"].tzinfo is None else r["ts"], "yes_price": int(r["yes_price"]), "count": float(r["count"]),
                   "taker_side": r["taker_side"] or "", "is_block": False} for r in tab.to_pylist()]
        path = D.TRADES_DIR / f"{tk}.parquet"
        if not path.exists():
            D.write_parquet(rows_t, D.TRADE_SCHEMA, path)
        done.append((str(path), len(rows_t), tk)); n += 1
        if len(done) >= 2000 or i == len(have) - 1:
            with transaction() as conn:
                conn.cursor().executemany("UPDATE kalshi_corpus SET trade_path = %s, trades = %s, updated_at = now() WHERE ticker = %s", done)
            done = []; _status(stage="seeding dataset trades", seed_trades=n, seed_trades_total=len(have))
    if not (stop is not None and stop.is_set()):                  # markets the dataset has no trades for: an empty file, not an API call
        missing = sorted(set(todo) - set(have)); empty: list = []
        for tk in missing:
            path = D.TRADES_DIR / f"{tk}.parquet"
            if not path.exists():
                D.write_parquet([], D.TRADE_SCHEMA, path)
            empty.append((str(path), 0, tk))
        with transaction() as conn:
            for i in range(0, len(empty), 5000):
                conn.cursor().executemany("UPDATE kalshi_corpus SET trade_path = %s, trades = %s, updated_at = now() WHERE ticker = %s", empty[i:i + 5000])
    con.close()
    return n

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
    """Every settled single market closing after the dataset's newest day (or the start) into ``kalshi_universe``, from the
    live tier then the archive (both page newest first); which of them the corpus takes is ``select_corpus``'s decision. The walk is resumable: after every page its tier and page
    cursor are saved with that page's rows (``ui_settings[WALK_KEY]``), so a timeout, a crash or a restart continues from the
    next page instead of walking the newest months again. A completed walk records when it began; later rounds walk only the
    live tier for markets closing since then (less ``RECOVER_DAYS``), and the archive is never walked twice."""
    start = _start(); n = 0
    with transaction() as conn:
        r = conn.execute("SELECT max(close_time) AS t FROM kalshi_universe WHERE source = 'dataset'").fetchone()
    if r and r["t"] and r["t"] > start:
        start = r["t"]                                         # the dataset covers the days before; the API fills from there on
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
            settled = [m for m in page if (m.get("result") in ("yes", "no")) and (D.ts(m.get("close_time")) or lower) >= lower and m.get("event_ticker")]
            closes = [D.ts(m.get("close_time")) for m in page if D.ts(m.get("close_time"))]
            done_tier = not nxt or (closes and max(closes) < lower)
            with transaction() as conn:
                conn.cursor().executemany(
                    "INSERT INTO kalshi_universe (ticker, event_ticker, series_ticker, title, open_time, close_time, settled_ts, result, volume, source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ticker) DO UPDATE SET result = COALESCE(EXCLUDED.result, kalshi_universe.result), "
                    "settled_ts = COALESCE(EXCLUDED.settled_ts, kalshi_universe.settled_ts), volume = COALESCE(EXCLUDED.volume, kalshi_universe.volume), updated_at = now()",
                    [(m["ticker"], m["event_ticker"], D.series_ticker_of(m["event_ticker"]), m.get("title"), D.ts(m.get("open_time")), D.ts(m.get("close_time")),
                      D.ts(m.get("settlement_ts") or m.get("settled_time")), m.get("result"), market_volume(m), tier) for m in settled])
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


def catalogue_missing(rest: KalshiRest, stop: threading.Event | None = None, threads: int | None = None) -> int:
    """Events (and their series) of corpus markets that have none in the catalogue — the dataset seed never fetched them, so
    those markets built with category 'other', fee multiplier 1, not mutually exclusive and no siblings. Fetched
    ``threads`` at a time under the client's one rate bucket; returns events catalogued."""
    with transaction() as conn:
        todo = [r["event_ticker"] for r in conn.execute("""SELECT DISTINCT m.event_ticker FROM kalshi_corpus k JOIN kalshi_markets m USING (ticker)
                                                           LEFT JOIN kalshi_events e ON e.event_ticker = m.event_ticker
                                                           WHERE k.status IN ('pending', 'done', 'built') AND m.event_ticker IS NOT NULL AND e.event_ticker IS NULL""").fetchall()]
        known_s = _known_series(conn)
    if not todo:
        return 0
    lock = threading.Lock(); done = {"n": 0}

    def one(et: str) -> None:
        if stop is not None and stop.is_set():
            return
        try:
            e = (rest.event(et, with_nested_markets=False) or {}).get("event") or {"event_ticker": et}
        except KalshiApiError as ex:
            if ex.status is not None and (ex.status == 429 or ex.status >= 500):
                return                                          # retried next round
            e = {"event_ticker": et}
        st = e.get("series_ticker") or D.series_ticker_of(et); series = None
        with lock:
            need = bool(st) and st not in known_s
            if need:
                known_s.add(st)
        if need:
            try:
                series = rest.series(st)
            except KalshiApiError:
                series = {"ticker": st}
        with transaction() as conn:
            D.upsert_event(conn, e)
            if series is not None:
                D.upsert_series(conn, series)
        with lock:
            done["n"] += 1
            if done["n"] % 500 == 0:
                _status(stage="cataloguing seeded markets", catalogued=done["n"], catalogue_total=len(todo))

    with ThreadPoolExecutor(max_workers=max(1, int(threads or config.KALSHI_FILL_THREADS)), thread_name_prefix="kalshi-cat") as pool:
        list(pool.map(one, todo))
    log.info("catalogued %d of %d missing events", done["n"], len(todo))
    return done["n"]


def ensure_series(rest: KalshiRest, stop: threading.Event | None = None, threads: int | None = None) -> int:
    """Series of corpus markets missing from the catalogue (the dataset's markets carry no category; their series does)."""
    with transaction() as conn:
        todo = [r["st"] for r in conn.execute("""SELECT DISTINCT COALESCE(e.series_ticker, split_part(COALESCE(m.event_ticker, m.ticker), '-', 1)) AS st
                                                 FROM kalshi_corpus k JOIN kalshi_markets m USING (ticker) LEFT JOIN kalshi_events e USING (event_ticker)
                                                 WHERE k.status <> 'empty'""").fetchall() if r["st"]]
        known = _known_series(conn)
    todo = [t for t in todo if t not in known]

    def one(st: str) -> None:
        if stop is not None and stop.is_set():
            return
        try:
            s_ = rest.series(st)
        except KalshiApiError as ex:
            if ex.status is not None and (ex.status == 429 or ex.status >= 500):
                return
            s_ = {"ticker": st}
        with transaction() as conn:
            D.upsert_series(conn, s_ if s_.get("ticker") else {**s_, "ticker": st})
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, int(threads or config.KALSHI_FILL_THREADS)), thread_name_prefix="kalshi-series") as pool:
            list(pool.map(one, todo))
        log.info("catalogued %d series of corpus markets", len(todo))
    return len(todo)


COMPLETE_AFTER_DAYS = 3        # a close day is selected once every market of it has settled (settlement follows close)


def select_events(df, target: int) -> set:
    """Whole events per (category, close day) in md5(event ticker) order until the day holds ``target`` markets (the event
    that crosses the budget is kept whole). ``df``: ticker, event_ticker, cat, day. Blind to volume and outcome."""
    import pandas as pd
    df = df[~df["cat"].isin(EXCLUDED_CATEGORIES)]
    if not len(df):
        return set()
    ev = df.groupby("event_ticker").agg(n=("ticker", "size"), day=("day", "min"), cat=("cat", "first")).reset_index()
    ev["u"] = [event_u(e) for e in ev["event_ticker"]]
    ev = ev.sort_values(["cat", "day", "u"]); before = ev.groupby(["cat", "day"])["n"].cumsum() - ev["n"]
    keep = set(ev.loc[before < target, "event_ticker"])
    return set(df.loc[df["event_ticker"].isin(keep), "ticker"])


def select_corpus() -> dict:
    """The corpus from the universe: ``select_events`` over every complete close day, marked in ``kalshi_universe.selected``;
    newly selected markets enter ``kalshi_markets`` and ``kalshi_corpus`` (pending), and the corpus follows the selection
    (``apply_sample``). Deterministic: the same universe always gives the same corpus."""
    import pandas as pd
    t0 = time.time(); cutoff = datetime.now(timezone.utc) - timedelta(days=COMPLETE_AFTER_DAYS)
    with transaction() as conn:
        cats = series_categories(conn)
        df = pd.DataFrame([dict(r) for r in conn.execute("SELECT ticker, event_ticker, series_ticker, close_time FROM kalshi_universe WHERE close_time < %s", (cutoff,)).fetchall()])
    if not len(df):
        return {"selected": 0}
    df["cat"] = [cats.get(x, "unknown") for x in df["series_ticker"]]
    df["day"] = pd.to_datetime(df["close_time"], utc=True).dt.floor("D")
    sel = select_events(df[["ticker", "event_ticker", "cat", "day"]], int(config.KALSHI_MARKETS_PER_DAY))
    sel_l = sorted(sel)
    with transaction() as conn:
        conn.execute("CREATE TEMP TABLE sel (ticker text PRIMARY KEY) ON COMMIT DROP")
        with conn.cursor().copy("COPY sel (ticker) FROM STDIN") as cp:
            for tk in sel_l:
                cp.write_row((tk,))
        conn.execute("UPDATE kalshi_universe u SET selected = true, updated_at = now() FROM sel WHERE sel.ticker = u.ticker AND NOT u.selected")
        conn.execute("UPDATE kalshi_universe u SET selected = false, updated_at = now() WHERE u.selected AND u.close_time < %s "
                     "AND NOT EXISTS (SELECT 1 FROM sel WHERE sel.ticker = u.ticker)", (cutoff,))
        new = conn.execute("""SELECT u.* FROM kalshi_universe u JOIN sel USING (ticker) LEFT JOIN kalshi_corpus k USING (ticker) WHERE k.ticker IS NULL""").fetchall()
        mk = [{"ticker": r["ticker"], "event_ticker": r["event_ticker"], "title": r["title"], "status": "settled", "open_time": r["open_time"], "close_time": r["close_time"],
               "settlement_ts": r["settled_ts"], "result": r["result"]} for r in new]
        for i in range(0, len(mk), 5000):
            D.upsert_markets(conn, mk[i:i + 5000], "universe")
        conn.cursor().executemany("INSERT INTO kalshi_corpus (ticker, status, settled_ts, result, open_time, close_time, seeded_from, volume) VALUES (%s,'pending',%s,%s,%s,%s,%s,%s) "
                                  "ON CONFLICT (ticker) DO NOTHING",
                                  [(r["ticker"], r["settled_ts"], r["result"], r["open_time"], r["close_time"], r["source"], r["volume"]) for r in new])
    changed = apply_sample()
    log.info("corpus selection: %d markets of %d in complete days (%d new to the corpus, %d corpus rows changed) in %.0fs",
             len(sel), len(df), len(new), changed, time.time() - t0)
    return {"selected": len(sel), "universe": int(len(df)), "new": len(new), "changed": changed}


def apply_sample() -> int:
    """Corpus rows follow the selection: a market not selected is 'skipped' (its files stay on disk, it is never built or
    trained on); a selected market an older rule skipped comes back ('done' when its candles are already on disk, else
    'pending'). Markets of days not yet selectable (the last COMPLETE_AFTER_DAYS) are left as they are. Returns rows changed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=COMPLETE_AFTER_DAYS)
    with transaction() as conn:
        a = conn.execute("""UPDATE kalshi_corpus k SET status = 'skipped', last_error = %s, updated_at = now()
                            WHERE k.status IN ('pending', 'done', 'built') AND k.close_time < %s
                              AND NOT EXISTS (SELECT 1 FROM kalshi_universe u WHERE u.ticker = k.ticker AND u.selected)""", (NOT_SAMPLED, cutoff)).rowcount
        b = conn.execute("""UPDATE kalshi_corpus k SET status = CASE WHEN k.candle_path IS NOT NULL THEN 'done' ELSE 'pending' END, last_error = NULL, updated_at = now()
                            FROM kalshi_universe u WHERE u.ticker = k.ticker AND u.selected AND k.status = 'skipped'""").rowcount
    log.info("event sample: %d corpus rows left, %d came back", a, b)
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


_CUTOFF: dict = {"at": 0.0, "value": None}
_CUTOFF_LOCK = threading.Lock()
CUTOFF_TTL_S = 3600.0


def trades_cutoff(rest: KalshiRest) -> datetime | None:
    """The exchange's historical cutoff for trades, fetched at most hourly (it moves daily; every market used to ask for it)."""
    with _CUTOFF_LOCK:
        if time.time() - _CUTOFF["at"] < CUTOFF_TTL_S:
            return _CUTOFF["value"]
        try:
            _CUTOFF["value"] = D.ts((rest.historical_cutoff() or {}).get("trades_created_ts")); _CUTOFF["at"] = time.time()
        except KalshiApiError:
            pass
        return _CUTOFF["value"]


def pull_trades(rest: KalshiRest, ticker: str, close_time: datetime, settled: datetime | None) -> list[dict]:
    lo = int((close_time - timedelta(days=CANDLE_DAYS)).timestamp()); hi = int((settled or close_time).timestamp()) + 60
    cutoff = trades_cutoff(rest)
    historical = bool(cutoff and close_time < cutoff)
    rows: list[dict] = []
    for page in rest.trades(ticker, min_ts=lo, max_ts=hi, historical=historical):
        rows.extend(D.trade_rows(page))
    return sorted(rows, key=lambda r: r["ts"])


TRANSIENT_WAIT_S = 60.0


def fill(rest: KalshiRest, stop: threading.Event | None = None, limit: int | None = None, threads: int | None = None) -> int:
    """Candles (and trades where missing) for pending corpus rows, newest first, ``threads`` markets at a time; every call
    still passes the client's one rate bucket (KALSHI_RPS), so the threads only overlap the calls' latency. A market whose
    fetch failed on the network stays pending (retried next batch); a batch that failed entirely waits ``TRANSIENT_WAIT_S``.
    Returns the markets finished (done, empty or error)."""
    threads = max(1, int(threads or config.KALSHI_FILL_THREADS)); limit = limit or max(200, threads * 25)
    with transaction() as conn:
        todo = conn.execute("SELECT c.ticker, c.open_time, c.close_time, c.settled_ts, c.candle_path, c.trade_path FROM kalshi_corpus c "
                            "WHERE c.status = 'pending' AND c.close_time IS NOT NULL ORDER BY c.close_time DESC LIMIT %s", (limit,)).fetchall()
        series = {r["ticker"]: _series_for(conn, r["ticker"]) for r in todo}
    if not todo:
        return 0
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="kalshi-fill") as pool:
        results = list(pool.map(lambda r: _fill_one(rest, r, series[r["ticker"]], stop), todo))
    n = sum(1 for x in results if x == "finished")
    if n == 0 and any(x == "transient" for x in results):
        log.warning("kalshi fill: every market of the batch failed on the network; waiting %.0f s", TRANSIENT_WAIT_S)
        _sleep(TRANSIENT_WAIT_S, stop)
    return n if n else (0 if not any(x == "transient" for x in results) else -1)


def _fill_one(rest: KalshiRest, r: dict, series: str, stop: threading.Event | None) -> str:
    """One market's candles and trades; 'finished' (row updated), 'transient' (left pending) or 'stopped'."""
    if stop is not None and stop.is_set():
        return "stopped"
    tk = r["ticker"]; upd = {"status": "done", "last_error": None}
    try:
        if not r["candle_path"]:
            cs = pull_candles(rest, tk, series, r["open_time"], r["close_time"], r["settled_ts"])
            if cs:
                p = D.CANDLES_DIR / f"{tk}.parquet"; D.write_parquet(cs, D.CANDLE_SCHEMA, p); upd.update(candle_path=str(p), candles_1m=len(cs))
            else:
                upd["status"] = "empty"
        if not r["trade_path"]:
            tr = pull_trades(rest, tk, r["close_time"], r["settled_ts"])
            p = D.TRADES_DIR / f"{tk}.parquet"; D.write_parquet(tr, D.TRADE_SCHEMA, p); upd.update(trade_path=str(p), trades=len(tr))
    except KalshiApiError as e:
        if e.status is not None and (e.status == 429 or e.status >= 500):         # the exchange is busy or down: retry later
            _mark_transient(tk, str(e)); return "transient"
        upd = {"status": "error" if e.status not in (404,) else "empty", "last_error": str(e)[:300]}
    except (OSError, TimeoutError, ConnectionError) as e:                           # the network: the market stays pending
        _mark_transient(tk, f"{type(e).__name__}: {e}"); return "transient"
    except Exception as e:
        if type(e).__module__.startswith(("aiohttp", "asyncio", "concurrent")) or "Timeout" in type(e).__name__:
            _mark_transient(tk, f"{type(e).__name__}: {e}"); return "transient"
        log.exception("candles %s failed", tk); upd = {"status": "error", "last_error": f"{type(e).__name__}: {e}"[:300]}
    with transaction() as conn:
        conn.execute("UPDATE kalshi_corpus SET " + ", ".join(f"{k} = %s" for k in upd) + ", updated_at = now() WHERE ticker = %s", (*upd.values(), tk))
    return "finished"


def _mark_transient(tk: str, err: str) -> None:
    try:
        with transaction() as conn:
            conn.execute("UPDATE kalshi_corpus SET last_error = %s, updated_at = now() WHERE ticker = %s", (f"retrying: {err}"[:300], tk))
    except Exception:
        log.warning("could not note the transient failure of %s", tk)


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
                _status(stage="seeding the universe from the dataset"); refresh_series(rest); seeded = seed_from_dataset(stop)
                _status(stage="refreshing markets", seed=seeded); n_new = refresh_markets(rest, stop)
                refresh_series(rest); sel = select_corpus(); skipped = sel.get("changed")      # the outcome-blind event sample,
                seed_dataset_trades(stop); ensure_series(rest, stop); catalogue_missing(rest, stop)  # then only the selected markets' files and events
                c = counts(); _status(stage="filling candles", refreshed=n_new, skipped_now=skipped, **{k: v for k, v in c.items()})
                t0 = time.time(); done = 0
                while not (stop is not None and stop.is_set()):
                    k = fill(rest, stop)
                    if k == 0:
                        break
                    if k < 0:                                   # the whole batch failed on the network: fill() already waited
                        continue
                    done += k; c = counts()
                    rate = done / max(time.time() - t0, 1e-9)
                    _status(stage="filling candles", filled_this_run=done, markets_per_h=rate * 3600, eta_h=(c.get("pending", 0) / rate / 3600) if rate else None, **c)
                if not (stop is not None and stop.is_set()):
                    from . import mature                # filled markets become feature rows (the training corpus)
                    _status(stage="building features", **counts()); built = mature.loop_once()
                    ok, why = mature.build_complete(); _status(stage="idle", built_this_round=built, build_complete=ok, build_note=why)
                _status(stage="idle", **counts())
            except Exception as e:
                log.exception("kalshi history round failed"); record_event("error", "kalshi_history", f"round failed: {type(e).__name__}: {e}")
                _status(stage="error", last_error=str(e)[:200])
            _sleep(IDLE_S, stop)
    finally:
        rest.close(); _status(stage="stopped")
        record_event("info", "kalshi_history", "kalshi history stopped", counts())
