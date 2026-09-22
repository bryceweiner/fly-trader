"""The Kalshi corpus on disk and the catalogue rows both the history puller and the live stream write.

Layout (``config.KALSHI_DIR``):
- ``candles/<ticker>.parquet`` — 1-minute candles (``CANDLE_SCHEMA``: period end, YES bid/ask OHLC in cents, trade price
  OHLC when traded, contracts traded in the period, open interest at its end);
- ``trades/<ticker>.parquet`` — public trades (``TRADE_SCHEMA``: time, YES price in cents, contracts, the taker's side);
- ``features/<day>/part.parquet`` — the daily feature rows (kalshi/mature.py).
Rows of ``kalshi_markets`` / ``kalshi_events`` / ``kalshi_series`` are upserted from the exchange's JSON by
``upsert_market`` and friends; prices are kept in cents, contracts as floats, times as UTC timestamps.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from .. import config

CANDLES_DIR = config.KALSHI_DIR / "candles"
TRADES_DIR = config.KALSHI_DIR / "trades"
FEATURES_DIR = config.KALSHI_DIR / "features"
CANDLE_SCHEMA = pa.schema([("end_ts", pa.int64()), ("yes_bid_open", pa.float32()), ("yes_bid_high", pa.float32()), ("yes_bid_low", pa.float32()),
                           ("yes_bid_close", pa.float32()), ("yes_ask_open", pa.float32()), ("yes_ask_high", pa.float32()), ("yes_ask_low", pa.float32()),
                           ("yes_ask_close", pa.float32()), ("price_open", pa.float32()), ("price_high", pa.float32()), ("price_low", pa.float32()),
                           ("price_close", pa.float32()), ("price_mean", pa.float32()), ("volume", pa.float64()), ("open_interest", pa.float64())])
TRADE_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("yes_price", pa.int16()), ("count", pa.float64()), ("taker_side", pa.string()), ("is_block", pa.bool_())])
# discovery categories the exchange uses, folded into the fixed set the features one-hot (kalshi/features.CATEGORIES)
CATEGORY_KEYS = (("politic", "politics"), ("election", "politics"), ("sport", "sports"), ("crypto", "crypto"), ("bitcoin", "crypto"), ("econom", "economics"),
                 ("financ", "economics"), ("weather", "weather"), ("climate", "weather"), ("entertain", "entertainment"), ("culture", "entertainment"),
                 ("science", "science"), ("tech", "science"), ("compan", "companies"), ("world", "world"), ("health", "health"))


def category_key(raw: str | None) -> str:
    s = (raw or "").lower()
    for needle, key in CATEGORY_KEYS:
        if needle in s:
            return key
    return "other"


def ts(v) -> datetime | None:
    """An ISO-8601 string (or epoch seconds) from the exchange as an aware UTC datetime."""
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), timezone.utc)
    s = str(v).replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _f(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def price_cents(row: dict, key: str) -> float | None:
    """``<key>_dollars`` (fixed point) or ``<key>`` (cents) of a market/ticker row, as cents."""
    d = _f(row.get(f"{key}_dollars"))
    if d is not None:
        return d * 100.0
    return _f(row.get(key))


def series_ticker_of(ticker: str | None) -> str | None:
    return str(ticker).split("-")[0] if ticker else None


MARKET_COLS = ("ticker", "event_ticker", "market_type", "title", "yes_sub_title", "no_sub_title", "status", "open_time", "close_time",
               "expected_expiration_time", "latest_expiration_time", "settlement_ts", "result", "settlement_value", "strike_type", "floor_strike",
               "cap_strike", "exchange_index", "price_level_structure", "is_provisional", "can_close_early", "created_time", "source", "raw")


def market_row(m: dict, source: str) -> dict:
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "market_type": m.get("market_type"), "title": m.get("title"),
            "yes_sub_title": m.get("yes_sub_title"), "no_sub_title": m.get("no_sub_title"), "status": m.get("status"), "open_time": ts(m.get("open_time")),
            "close_time": ts(m.get("close_time")), "expected_expiration_time": ts(m.get("expected_expiration_time")),
            "latest_expiration_time": ts(m.get("latest_expiration_time")), "settlement_ts": ts(m.get("settlement_ts") or m.get("settled_time")),
            "result": m.get("result") or None, "settlement_value": _f(m.get("settlement_value_dollars")) if m.get("settlement_value_dollars") is not None else _f(m.get("settlement_value")),
            "strike_type": m.get("strike_type"), "floor_strike": _f(m.get("floor_strike")), "cap_strike": _f(m.get("cap_strike")),
            "exchange_index": m.get("exchange_index"), "price_level_structure": m.get("price_level_structure"), "is_provisional": m.get("is_provisional"),
            "can_close_early": m.get("can_close_early"), "created_time": ts(m.get("created_time")), "source": source,
            "raw": json.dumps({k: v for k, v in m.items() if k not in ("rules_primary", "rules_secondary")}, default=str)}


def upsert_markets(conn, markets: list[dict], source: str) -> int:
    rows = [market_row(m, source) for m in markets if m.get("ticker")]
    if not rows:
        return 0
    cols = MARKET_COLS
    sets = ", ".join(f"{c} = COALESCE(EXCLUDED.{c}, kalshi_markets.{c})" for c in cols if c not in ("ticker", "source", "raw"))
    conn.cursor().executemany(
        f"INSERT INTO kalshi_markets ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) ON CONFLICT (ticker) DO UPDATE SET {sets}, "
        "source = EXCLUDED.source, raw = EXCLUDED.raw, updated_at = now()", [tuple(r[c] for c in cols) for r in rows])
    return len(rows)


def upsert_event(conn, e: dict) -> None:
    if not e.get("event_ticker"):
        return
    conn.execute("INSERT INTO kalshi_events (event_ticker, series_ticker, title, sub_title, category, mutually_exclusive, strike_date, strike_period, "
                 "collateral_return_type, exchange_index, raw) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_ticker) DO UPDATE SET "
                 "series_ticker = COALESCE(EXCLUDED.series_ticker, kalshi_events.series_ticker), title = COALESCE(EXCLUDED.title, kalshi_events.title), "
                 "sub_title = COALESCE(EXCLUDED.sub_title, kalshi_events.sub_title), category = COALESCE(EXCLUDED.category, kalshi_events.category), "
                 "mutually_exclusive = COALESCE(EXCLUDED.mutually_exclusive, kalshi_events.mutually_exclusive), strike_date = COALESCE(EXCLUDED.strike_date, kalshi_events.strike_date), "
                 "strike_period = COALESCE(EXCLUDED.strike_period, kalshi_events.strike_period), collateral_return_type = COALESCE(EXCLUDED.collateral_return_type, kalshi_events.collateral_return_type), "
                 "exchange_index = COALESCE(EXCLUDED.exchange_index, kalshi_events.exchange_index), raw = EXCLUDED.raw, updated_at = now()",
                 (e["event_ticker"], e.get("series_ticker") or series_ticker_of(e["event_ticker"]), e.get("title"), e.get("sub_title"), e.get("category"),
                  e.get("mutually_exclusive"), ts(e.get("strike_date")), e.get("strike_period"), e.get("collateral_return_type"), e.get("exchange_index"),
                  json.dumps({k: v for k, v in e.items() if k != "markets"}, default=str)))


def upsert_series(conn, s: dict) -> None:
    if not s.get("ticker"):
        return
    conn.execute("INSERT INTO kalshi_series (ticker, title, category, categories, tags, frequency, fee_type, fee_multiplier, settlement_sources) "
                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ticker) DO UPDATE SET title = EXCLUDED.title, category = EXCLUDED.category, "
                 "categories = EXCLUDED.categories, tags = EXCLUDED.tags, frequency = EXCLUDED.frequency, fee_type = EXCLUDED.fee_type, "
                 "fee_multiplier = EXCLUDED.fee_multiplier, settlement_sources = EXCLUDED.settlement_sources, updated_at = now()",
                 (s["ticker"], s.get("title"), s.get("category"), list(s.get("categories") or []), list(s.get("tags") or []), s.get("frequency"),
                  s.get("fee_type"), _f(s.get("fee_multiplier")), json.dumps(s.get("settlement_sources") or [], default=str)))


def candle_rows(candles: list[dict]) -> list[dict]:
    """Exchange candlesticks → ``CANDLE_SCHEMA`` rows (cents; missing trade prices stay NaN)."""
    out = []
    for c in candles:
        yb, ya, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}

        def q(d, k):
            v = _f(d.get(f"{k}_dollars"))
            return v * 100.0 if v is not None else (_f(d.get(k)) if d.get(k) is not None else float("nan"))
        out.append({"end_ts": int(c.get("end_period_ts") or 0),
                    "yes_bid_open": q(yb, "open"), "yes_bid_high": q(yb, "high"), "yes_bid_low": q(yb, "low"), "yes_bid_close": q(yb, "close"),
                    "yes_ask_open": q(ya, "open"), "yes_ask_high": q(ya, "high"), "yes_ask_low": q(ya, "low"), "yes_ask_close": q(ya, "close"),
                    "price_open": q(pr, "open"), "price_high": q(pr, "high"), "price_low": q(pr, "low"), "price_close": q(pr, "close"), "price_mean": q(pr, "mean"),
                    "volume": _f(c.get("volume_fp") or c.get("volume")) or 0.0, "open_interest": _f(c.get("open_interest_fp") or c.get("open_interest")) or 0.0})
    return out


def trade_rows(trades: list[dict]) -> list[dict]:
    out = []
    for t in trades:
        p = price_cents(t, "yes_price")
        if p is None:
            continue
        out.append({"ts": ts(t.get("created_time")), "yes_price": int(round(p)), "count": _f(t.get("count_fp") or t.get("count")) or 0.0,
                    "taker_side": t.get("taker_outcome_side") or t.get("taker_side") or "", "is_block": bool(t.get("is_block_trade"))})
    return [r for r in out if r["ts"] is not None]


def write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd"); tmp.replace(path)
