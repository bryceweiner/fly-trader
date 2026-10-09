"""The Kalshi corpus as feature rows (``kalshi-build``; also run by the history worker after each fill round).

Per event, every settled market's candles and trades are merged into minute bars and stepped through together (the
siblings' quotes feed ``EventState``), and every ``STRIDE_MIN`` minutes inside the entry window (the last
``KALSHI_MAX_DAYS_TO_CLOSE`` days before close, at least ``KALSHI_MIN_MINUTES_TO_CLOSE`` before it) a row per side is
written with the ``K_COLS`` vector, the settlement (``result``), the times the labels need, and what decides whether the
maker arm's bid would have filled (kalshi/decisions.maker_filled) over the order's life — the minutes after the decision
up to ``KALSHI_MAKER_TTL_H`` later, ending ``KALSHI_MAKER_QUIET_MIN`` before the close known at decision time (the arm's
expiry): ``fill_ask_low``, the lowest ask the side showed, and ``fill_sold_low``, the lowest price a taker sold the side
at (from the trades). The order's window is stamped on every part (``MAKER_WINDOW``): another window is another corpus.
Output: ``data/kalshi/features/<day>/part-<batch>.parquet`` (day = the row's UTC date),
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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..train.corpus_features import _epoch_s
from . import data as D
from .features import CATEGORIES, K_COLS, KALSHI_FEATURE_VERSION, Bar, EventState, MarketMeta, MarketState, warm_fees

log = logging.getLogger(__name__)
STRIDE_MIN = 5
META_COLS = ["ticker", "side", "ts", "event_ticker", "category", "close_ts", "end_ts", "settled_ts", "result", "fill_ask_low", "fill_sold_low"]
SCHEMA = pa.schema([("ticker", pa.string()), ("side", pa.string()), ("ts", pa.timestamp("ms", tz="UTC")), ("event_ticker", pa.string()), ("category", pa.string()),
                    ("close_ts", pa.float64()), ("end_ts", pa.float64()), ("settled_ts", pa.float64()), ("result", pa.string()), ("fill_ask_low", pa.float32()), ("fill_sold_low", pa.float32())]
                   + [(c, pa.float32()) for c in K_COLS])
MAKER_WINDOW = f"{config.KALSHI_MAKER_TTL_H:g}h-quiet{config.KALSHI_MAKER_QUIET_MIN:g}m"     # the maker order's life the labels assume
RECURRING = ("hourly", "daily", "15min", "minute", "4h", "weekly")


def part_version(path) -> int:
    md = pq.read_schema(path).metadata or {}
    try:
        return int(md.get(b"fly_version", b"0"))
    except ValueError:
        return 0


def part_current(path) -> bool:
    p = Path(path)
    return p.exists() and part_version(p) == KALSHI_FEATURE_VERSION and (pq.read_schema(p).metadata or {}).get(b"maker_window") == MAKER_WINDOW.encode()


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


def ticker_strike(ticker: str | None) -> float | None:
    """A ladder market's strike from its ticker's last segment: ``T81299.99`` (threshold) or ``B7650`` (bucket); else None."""
    import re
    m = re.fullmatch(r"[TB](-?\d+(?:\.\d+)?)", str(ticker or "").rsplit("-", 1)[-1])
    return float(m.group(1)) if m else None


def time_anchor(close_time, expected_expiration_time, early_close_condition) -> float | None:
    """The close a trader could know at decision time (epoch s), or None when it cannot be told.

    A market with an early-close condition ("after a winner is declared", "if the price criterion is met", "if the event
    occurs") may close before its listed close when its event happens; the close recorded after settlement is then the
    moment of the event — information from the future, and for a threshold market the outcome itself. Its anchor is the
    scheduled ``expected_expiration_time``, set at listing. A market without one closes on its listed schedule, which is
    known. Markets whose details were never fetched (no expected expiration recorded) give None: no rows, rather than a
    leaky anchor. Until 2026-10-08 every feature, window and trigger used the realized close; the selector learned to buy
    favorites in a match's last 17 minutes, which live it can never know (snapshot 188)."""
    if expected_expiration_time is None:
        return None
    if early_close_condition:
        return expected_expiration_time.timestamp()
    return close_time.timestamp() if close_time is not None else None


def load_meta(conn, tickers: list[str]) -> dict[str, MarketMeta]:
    rows = conn.execute("""SELECT m.ticker, m.event_ticker, m.open_time, m.close_time, m.expected_expiration_time, m.early_close_condition,
                                  m.floor_strike, m.cap_strike, e.series_ticker, e.category AS ecat,
                                  e.mutually_exclusive, s.category AS scat, s.frequency, s.fee_type, s.fee_multiplier
                           FROM kalshi_markets m LEFT JOIN kalshi_events e USING (event_ticker) LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker
                           WHERE m.ticker = ANY(%s)""", (tickers,)).fetchall()
    out = {}
    for r in rows:
        freq = (r["frequency"] or "").lower()
        strike = r["floor_strike"] if r["floor_strike"] is not None else r["cap_strike"]
        if strike is None:
            strike = ticker_strike(r["ticker"])                     # the catalogue lacks it (dataset or universe rows): the ticker carries it
        out[r["ticker"]] = MarketMeta(ticker=r["ticker"], event_ticker=r["event_ticker"], series_ticker=r["series_ticker"] or D.series_ticker_of(r["ticker"]),
                                      category=D.category_key(r["scat"] or r["ecat"]), is_recurring=1.0 if any(k in freq for k in RECURRING) else 0.0,
                                      open_ts=r["open_time"].timestamp() if r["open_time"] else None,
                                      close_ts=time_anchor(r["close_time"], r["expected_expiration_time"], r["early_close_condition"]),
                                      end_ts=r["close_time"].timestamp() if r["close_time"] else None,
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


def _side_sold_lows(t_path: str | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side, (minute end, lowest price a taker sold the side at in that minute) from a trade file: a taker who bought
    NO sold YES at the trade's YES price, one who bought YES sold NO at 100 − it."""
    out = {s: (np.zeros(0, np.int64), np.zeros(0)) for s in ("yes", "no")}
    if not t_path or not Path(t_path).exists():
        return out
    t = pq.read_table(t_path, columns=["ts", "yes_price", "taker_side"]).to_pandas()
    if not len(t):
        return out
    ends = (np.floor(_epoch_s(t["ts"]) / 60.0) * 60 + 60).astype(np.int64); yp = t["yes_price"].to_numpy(dtype=float); who = t["taker_side"].to_numpy()
    for side, seller, price in (("yes", "no", yp), ("no", "yes", 100.0 - yp)):
        m = (who == seller) & (yp > 0) & (yp < 100)
        if m.any():
            s = pd.Series(price[m]).groupby(ends[m]).min()
            out[side] = (s.index.to_numpy(np.int64), s.to_numpy(dtype=float))
    return out


def _window_lows(ends: np.ndarray, vals: np.ndarray, after: np.ndarray, until: np.ndarray) -> np.ndarray:
    """Per query i, the lowest of ``vals`` over the minutes whose end lies in (after[i], until[i]]; NaN where there is none.
    ``ends`` ascending; a sparse table answers every query in O(1)."""
    out = np.full(len(after), np.nan)
    if not len(ends) or not len(after):
        return out
    tab = [np.where(np.isfinite(vals), vals, np.inf)]
    while (1 << len(tab)) <= len(ends):
        h = 1 << (len(tab) - 1); tab.append(np.minimum(tab[-1][:-h], tab[-1][h:]))
    lo = np.searchsorted(ends, after, side="right"); hi = np.searchsorted(ends, until, side="right"); n = hi - lo; ok = n > 0
    k = np.zeros(len(after), np.int64); k[ok] = np.floor(np.log2(n[ok])).astype(np.int64)
    k[ok] -= np.left_shift(1, k[ok]) > n[ok]                                      # float log2 rounding up at a power of two
    for kk in np.unique(k[ok]):
        m = ok & (k == kk); v = np.minimum(tab[kk][lo[m]], tab[kk][hi[m] - (1 << int(kk))])
        out[m] = np.where(np.isfinite(v), v, np.nan)
    return out


def fill_lows(candle_path: str, trade_path: str | None, anchor_close: float | None) -> tuple[np.ndarray, dict]:
    """(candle minute ends, side -> (fill_ask_low, fill_sold_low) per candle minute) for a maker bid posted at the end of
    each candle minute and resting for the order's life: until ``KALSHI_MAKER_TTL_H`` later, ending ``KALSHI_MAKER_QUIET_MIN``
    before the close known at decision time (kalshi/maker.plan's expiry; minutes after an early close have no data)."""
    ends, ya, nb = _side_ask_lows(candle_path); sold = _side_sold_lows(trade_path)
    o = np.argsort(ends, kind="stable"); ends, ya, nb = ends[o], ya[o], nb[o]
    cut = (anchor_close if anchor_close is not None else math.inf) - config.KALSHI_MAKER_QUIET_MIN * 60.0
    until = np.minimum(ends + config.KALSHI_MAKER_TTL_H * 3600.0, cut)
    return ends, {side: (_window_lows(ends, asks, ends, until), _window_lows(*sold[side], ends, until)) for side, asks in (("yes", ya), ("no", nb))}


def build_event(event_ticker: str, markets: list[dict], metas: dict[str, MarketMeta]) -> list[dict]:
    """Rows of every market of one event, stepped minute by minute together."""
    bars = {m["ticker"]: bars_from_files(m["candle_path"], m["trade_path"]) for m in markets}
    fills = {m["ticker"]: fill_lows(m["candle_path"], m["trade_path"], metas[m["ticker"]].close_ts) for m in markets if m["candle_path"]}
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
            ends, fl = fills[tk]
            j = int(np.searchsorted(ends, int(t_end)))
            for side in ("yes", "no"):
                x = st.features(t_end, meta, side, ev)
                f_ask, f_sold = fl[side]
                row = {"ticker": tk, "side": side, "ts": datetime.fromtimestamp(t_end - 60.0, timezone.utc), "event_ticker": event_ticker, "category": meta.category,
                       "close_ts": meta.close_ts, "end_ts": meta.end_ts, "settled_ts": (m["settled_ts"].timestamp() if m["settled_ts"] else meta.end_ts), "result": m["result"],
                       "fill_ask_low": float(f_ask[j]) if j < len(f_ask) else float("nan"), "fill_sold_low": float(f_sold[j]) if j < len(f_sold) else float("nan")}
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
        tab = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata({b"fly_version": str(KALSHI_FEATURE_VERSION).encode(), b"stride_min": str(STRIDE_MIN).encode(),
                                                                         b"maker_window": MAKER_WINDOW.encode()})
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
