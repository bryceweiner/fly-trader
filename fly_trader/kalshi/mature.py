"""The Kalshi corpus as feature rows (``kalshi-build``; also run by the history worker after each fill round).

Per event, every settled market's candles and trades are merged into minute bars and stepped through together (the
siblings' quotes feed ``EventState``), and every ``STRIDE_MIN`` minutes inside the entry window (the last
``KALSHI_MAX_DAYS_TO_CLOSE`` days before close, at least ``KALSHI_MIN_MINUTES_TO_CLOSE`` before it) a row per side is
written with the ``K_COLS`` vector, the settlement (``result``), the times the labels need and ``fut_min_ask`` — the
lowest ask the side saw over the rest of the market's life, which is what decides whether a resting order would have
filled (kalshi/decisions.py). Output: ``data/kalshi/features/<day>/part-<batch>.parquet`` (day = the row's UTC date),
version-stamped like train/mature.py; markets go ``done → built`` in ``kalshi_corpus``; ``kalshi_days`` counts rows.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..train.corpus_features import _epoch_s
from . import data as D
from .features import CATEGORIES, K_COLS, KALSHI_FEATURE_VERSION, Bar, EventState, MarketMeta, MarketState, warm_fees

log = logging.getLogger(__name__)
STRIDE_MIN = 5
META_COLS = ["ticker", "side", "ts", "event_ticker", "category", "close_ts", "settled_ts", "result", "fut_min_ask", "fut_min_ask_24h"]
SCHEMA = pa.schema([("ticker", pa.string()), ("side", pa.string()), ("ts", pa.timestamp("ms", tz="UTC")), ("event_ticker", pa.string()), ("category", pa.string()),
                    ("close_ts", pa.float64()), ("settled_ts", pa.float64()), ("result", pa.string()), ("fut_min_ask", pa.float32()), ("fut_min_ask_24h", pa.float32())]
                   + [(c, pa.float32()) for c in K_COLS])
RECURRING = ("hourly", "daily", "15min", "minute", "4h", "weekly")


def part_version(path) -> int:
    md = pq.read_schema(path).metadata or {}
    try:
        return int(md.get(b"fly_version", b"0"))
    except ValueError:
        return 0


def part_current(path) -> bool:
    p = Path(path)
    return p.exists() and part_version(p) == KALSHI_FEATURE_VERSION


def build_complete() -> tuple[bool, str]:
    with transaction() as conn:
        c = {r["status"]: int(r["n"]) for r in conn.execute("SELECT status, count(*) AS n FROM kalshi_corpus GROUP BY status").fetchall()}
    if not c:
        return False, "no Kalshi corpus yet (kalshi-history)"
    if c.get("pending"):
        return False, f"{c['pending']} market(s) still to fetch"
    if c.get("done"):
        return False, f"{c['done']} market(s) still to build"
    stale = [f for f in D.FEATURES_DIR.glob("*/part-*.parquet") if not part_current(f)]
    if stale:
        return False, f"{len(stale)} feature part(s) from another version (rebuild)"
    return True, f"{c.get('built', 0)} markets built"


def load_meta(conn, tickers: list[str]) -> dict[str, MarketMeta]:
    rows = conn.execute("""SELECT m.ticker, m.event_ticker, m.open_time, m.close_time, m.floor_strike, m.cap_strike, e.series_ticker, e.category AS ecat,
                                  e.mutually_exclusive, s.category AS scat, s.frequency, s.fee_type, s.fee_multiplier
                           FROM kalshi_markets m LEFT JOIN kalshi_events e USING (event_ticker) LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker
                           WHERE m.ticker = ANY(%s)""", (tickers,)).fetchall()
    out = {}
    for r in rows:
        freq = (r["frequency"] or "").lower()
        strike = r["floor_strike"] if r["floor_strike"] is not None else r["cap_strike"]
        out[r["ticker"]] = MarketMeta(ticker=r["ticker"], event_ticker=r["event_ticker"], series_ticker=r["series_ticker"] or D.series_ticker_of(r["ticker"]),
                                      category=D.category_key(r["scat"] or r["ecat"]), is_recurring=1.0 if any(k in freq for k in RECURRING) else 0.0,
                                      open_ts=r["open_time"].timestamp() if r["open_time"] else None, close_ts=r["close_time"].timestamp() if r["close_time"] else None,
                                      mutually_exclusive=1.0 if r["mutually_exclusive"] else 0.0,
                                      fee_multiplier=float(r["fee_multiplier"]) if r["fee_multiplier"] is not None else 1.0,
                                      maker_fee=1.0 if (r["fee_type"] or "").endswith("maker_fees") else 0.0, strike=float(strike) if strike is not None else None)
    return out


def bars_from_files(candle_path: str | None, trade_path: str | None) -> list[Bar]:
    """Minute bars from a market's candle file, with the taker flow of its trades folded in by minute."""
    if not candle_path or not Path(candle_path).exists():
        return []
    c = pq.read_table(candle_path).to_pandas()
    flow: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])          # taker_yes, taker_no, n, max, block
    if trade_path and Path(trade_path).exists():
        t = pq.read_table(trade_path).to_pandas()
        if len(t):
            ends = (np.floor(_epoch_s(t["ts"]) / 60.0) * 60 + 60).astype(np.int64)
            for e, side, cnt, blk in zip(ends, t["taker_side"].to_numpy(), t["count"].to_numpy(), t["is_block"].to_numpy()):
                f = flow[int(e)]
                if side == "yes":
                    f[0] += float(cnt)
                else:
                    f[1] += float(cnt)
                f[2] += 1; f[3] = max(f[3], float(cnt)); f[4] += float(cnt) if blk else 0.0
    out = []
    for r in c.itertuples(index=False):
        fl = flow.get(int(r.end_ts), [0.0, 0.0, 0.0, 0.0, 0.0])
        out.append(Bar(float(r.end_ts), float(r.yes_bid_close) if r.yes_bid_close == r.yes_bid_close and 0 < r.yes_bid_close < 100 else float("nan"),
                       float(r.yes_ask_close) if r.yes_ask_close == r.yes_ask_close and 0 < r.yes_ask_close < 100 else float("nan"),
                       float(r.price_close) if r.price_close == r.price_close else float("nan"), 0.0, 0.0, float(r.volume or 0.0), float(r.open_interest or 0.0),
                       fl[0], fl[1], fl[2], fl[3], fl[4]))
    return out


def _side_ask_lows(c_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(end_ts, lowest YES ask of the minute, lowest NO ask of the minute) from a candle file."""
    c = pq.read_table(c_path, columns=["end_ts", "yes_ask_low", "yes_bid_high"]).to_pandas()
    ya = c["yes_ask_low"].to_numpy(dtype=float); yb = c["yes_bid_high"].to_numpy(dtype=float)
    ya = np.where(np.isfinite(ya) & (ya > 0) & (ya < 100), ya, np.nan); nb = np.where(np.isfinite(yb) & (yb > 0) & (yb < 100), 100.0 - yb, np.nan)
    return c["end_ts"].to_numpy(dtype=np.int64), ya, nb


def _suffix_min(a: np.ndarray) -> np.ndarray:
    """min over positions > i (exclusive), NaN-aware; NaN where nothing follows."""
    x = np.where(np.isfinite(a), a, np.inf)
    s = np.minimum.accumulate(x[::-1])[::-1]
    out = np.full(len(a), np.nan); out[:-1] = s[1:]
    return np.where(np.isfinite(out), out, np.nan)


def _window_min(ts: np.ndarray, a: np.ndarray, span_s: float) -> np.ndarray:
    """min over positions j > i with ts[j] <= ts[i] + span (a rolling forward window), NaN-aware."""
    out = np.full(len(a), np.nan); x = np.where(np.isfinite(a), a, np.inf)
    j_end = np.searchsorted(ts, ts + span_s, side="right")
    for i in range(len(a)):
        seg = x[i + 1:j_end[i]]
        if len(seg):
            m = seg.min(); out[i] = m if np.isfinite(m) else np.nan
    return out


def build_event(event_ticker: str, markets: list[dict], metas: dict[str, MarketMeta]) -> list[dict]:
    """Rows of every market of one event, stepped minute by minute together."""
    bars = {m["ticker"]: bars_from_files(m["candle_path"], m["trade_path"]) for m in markets}
    lows = {m["ticker"]: _side_ask_lows(m["candle_path"]) for m in markets if m["candle_path"]}
    fut = {}
    for tk, (ts, ya, nb) in lows.items():
        # a resting order can only fill while the market trades: nothing after close − the maker's quiet margin counts
        cut = (metas[tk].close_ts or float("inf")) - config.KALSHI_MAKER_QUIET_MIN * 60.0
        live = ts <= cut; ya = np.where(live, ya, np.nan); nb = np.where(live, nb, np.nan)
        fut[tk] = (ts, {"yes": (_suffix_min(ya), _window_min(ts, ya, 86400.0)), "no": (_suffix_min(nb), _window_min(ts, nb, 86400.0))})
    states = {tk: MarketState(tk) for tk in bars}; ev = EventState()
    pos = {tk: 0 for tk in bars}
    minutes = sorted({b.t_end for bs in bars.values() for b in bs})
    max_s = config.KALSHI_MAX_DAYS_TO_CLOSE * 86400.0; min_s = config.KALSHI_MIN_MINUTES_TO_CLOSE * 60.0
    out = []
    for t_end in minutes:
        for tk, bs in bars.items():
            i = pos[tk]
            if i < len(bs) and bs[i].t_end == t_end:
                states[tk].append(bs[i]); ev.update(tk, bs[i], metas[tk].strike); pos[tk] = i + 1
        if int(t_end // 60) % STRIDE_MIN:
            continue
        for m in markets:
            tk = m["ticker"]; meta = metas[tk]; st = states[tk]
            if st.last is None or st.last.t_end != t_end or meta.close_ts is None:
                continue
            to_close = meta.close_ts - t_end
            if not (min_s <= to_close <= max_s):
                continue
            ts_arr, fl = fut[tk]
            j = int(np.searchsorted(ts_arr, int(t_end)))
            for side in ("yes", "no"):
                x = st.features(t_end, meta, side, ev)
                fmin, fmin24 = fl[side]
                row = {"ticker": tk, "side": side, "ts": datetime.fromtimestamp(t_end - 60.0, timezone.utc), "event_ticker": event_ticker, "category": meta.category,
                       "close_ts": meta.close_ts, "settled_ts": (m["settled_ts"].timestamp() if m["settled_ts"] else meta.close_ts), "result": m["result"],
                       "fut_min_ask": float(fmin[j]) if j < len(fmin) and np.isfinite(fmin[j]) else float("nan"),
                       "fut_min_ask_24h": float(fmin24[j]) if j < len(fmin24) and np.isfinite(fmin24[j]) else float("nan")}
                row.update(zip(K_COLS, x))
                out.append(row)
    return out


def _events_todo(conn, limit: int) -> dict[str, list[dict]]:
    rows = conn.execute("""SELECT c.ticker, c.candle_path, c.trade_path, c.settled_ts, c.result, m.event_ticker FROM kalshi_corpus c JOIN kalshi_markets m USING (ticker)
                           WHERE c.status = 'done' AND c.candle_path IS NOT NULL AND c.result IN ('yes', 'no') ORDER BY c.close_time DESC LIMIT %s""", (limit,)).fetchall()
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["event_ticker"] or r["ticker"]].append(dict(r))
    if not by:
        return {}
    # an event's markets are built together: pull in its other done markets too
    evs = list(by)
    more = conn.execute("""SELECT c.ticker, c.candle_path, c.trade_path, c.settled_ts, c.result, m.event_ticker FROM kalshi_corpus c JOIN kalshi_markets m USING (ticker)
                           WHERE c.status = 'done' AND c.candle_path IS NOT NULL AND c.result IN ('yes', 'no') AND m.event_ticker = ANY(%s)""", (evs,)).fetchall()
    for r in more:
        if r["ticker"] not in {x["ticker"] for x in by[r["event_ticker"]]}:
            by[r["event_ticker"]].append(dict(r))
    return by


def build_batch(limit: int = 300) -> int:
    """Build the features of up to ``limit`` done markets (whole events); returns markets built."""
    t0 = time.time()
    with transaction() as conn:
        warm_fees(conn); by = _events_todo(conn, limit)
        metas = load_meta(conn, [m["ticker"] for ms in by.values() for m in ms])
    if not by:
        return 0
    rows_by_day: dict[str, list[dict]] = defaultdict(list); built = []
    for ev, ms in by.items():
        ms = [m for m in ms if m["ticker"] in metas]
        try:
            for row in build_event(ev, ms, metas):
                rows_by_day[row["ts"].date().isoformat()].append(row)
            built.extend(m["ticker"] for m in ms)
        except Exception:
            log.exception("build %s failed", ev)
            with transaction() as conn:
                conn.cursor().executemany("UPDATE kalshi_corpus SET status = 'error', last_error = 'feature build failed' WHERE ticker = %s", [(m["ticker"],) for m in ms])
    batch = int(time.time() * 1000); n_rows = 0
    for day, rows in rows_by_day.items():
        path = D.FEATURES_DIR / day / f"part-{batch}.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
        tab = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata({b"fly_version": str(KALSHI_FEATURE_VERSION).encode(), b"stride_min": str(STRIDE_MIN).encode()})
        tmp = path.with_name(path.name + ".tmp"); pq.write_table(tab, tmp, compression="zstd"); tmp.replace(path); n_rows += len(rows)
    with transaction() as conn:
        conn.cursor().executemany("UPDATE kalshi_corpus SET status = 'built', updated_at = now() WHERE ticker = %s", [(t,) for t in built])
        conn.cursor().executemany("INSERT INTO kalshi_days (day, markets, rows, took_s) VALUES (%s,%s,%s,%s) ON CONFLICT (day) DO UPDATE SET markets = kalshi_days.markets + EXCLUDED.markets, "
                                  "rows = kalshi_days.rows + EXCLUDED.rows, took_s = EXCLUDED.took_s, built_at = now()",
                                  [(datetime.fromisoformat(d).date(), len({r["ticker"] for r in rows}), len(rows), time.time() - t0) for d, rows in rows_by_day.items()])
    log.info("kalshi features: %d markets in %d events -> %d rows over %d days (%.0fs)", len(built), len(by), n_rows, len(rows_by_day), time.time() - t0)
    return len(built)


def retire_stale() -> int:
    """Parts from another feature version are removed and their markets rebuilt."""
    stale = [f for f in D.FEATURES_DIR.glob("*/part-*.parquet") if not part_current(f)]
    for f in stale:
        f.unlink()
    if stale:
        with transaction() as conn:
            conn.execute("UPDATE kalshi_corpus SET status = 'done' WHERE status = 'built'")
            conn.execute("DELETE FROM kalshi_days")
    return len(stale)


def loop_once(limit: int = 300) -> int:
    n = 0
    retire_stale()
    while True:
        k = build_batch(limit)
        if not k:
            break
        n += k
    return n


def main() -> None:
    from ..logging_setup import setup
    setup("kalshi_history")
    print(f"built {loop_once()} markets; complete: {build_complete()}")
