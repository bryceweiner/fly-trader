"""The Kalshi live feed — the ``kalshi_stream`` worker: one authenticated websocket, exchange-wide ``ticker``, ``trade``
and ``market_lifecycle_v2`` plus the subaccount's ``fill`` and ``user_orders`` channels, aggregated per (market, UTC
minute) into ``kalshi_minutes`` — the fields the corpus builder derives from the candle and trade archive
(kalshi/mature.bars_from_files): the minute's closing YES bid/ask and sizes, its lowest ask and highest bid, the last
price, cumulative volume and open interest, and the taker flow by side. A minute row exists only for minutes with a
message, as a candle exists only for minutes with activity; the engine carries quotes forward from its own state.

Minutes are written once complete (an event stamped ≥ 2 s after the minute's end has arrived, or 15 s of silence);
``kalshi_stream_status.flushed_through`` publishes the newest complete minute for the engine. Every ticker message also
refreshes ``kalshi_quotes`` (the live top of book the maker arm and the vendored venue read). Lifecycle: ``determined`` /
``settled`` write the result to ``kalshi_markets`` (positions settle and the fly's tags resolve on it); ``created`` /
``activated`` / ``close_date_updated`` queue the market for a REST fetch, so its event and series (category, fee
multiplier) are in the catalogue before the engine scores it. The open-market catalogue is refreshed from REST at start
and hourly. Combo (``KXMVE…``) tickers are not markets of ours and are dropped at the door.
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
from datetime import datetime, timezone

import certifi
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from . import data as D

log = logging.getLogger(__name__)
STATUS_KEY = "kalshi_stream_status"
FLUSH_GRACE_S, FLUSH_FALLBACK_S = 2.0, 15.0
QUOTE_FLUSH_S, STATUS_S, CATALOGUE_S, LIFECYCLE_S, ARCHIVE_S = 5.0, 5.0, 3600.0, 30.0, 3600.0
PUBLIC_CHANNELS = ("ticker", "trade", "market_lifecycle_v2")
PRIVATE_CHANNELS = ("fill", "user_orders")
COMBO_PREFIX = "KXMVE"
SETTLE_EVENTS = ("determined", "settled")
CATALOGUE_EVENTS = ("created", "activated", "close_date_updated", "reactivated")


def _cents(v) -> float:
    """A ``*_dollars`` string ('0.4600') as cents (46.0); NaN when absent."""
    try:
        return float(v) * 100.0 if v is not None and v != "" else math.nan
    except (TypeError, ValueError):
        return math.nan


def _num(v, default=0.0) -> float:
    try:
        return float(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


class KMinute:
    __slots__ = ("yes_bid", "yes_ask", "last", "bid_size", "ask_size", "volume", "oi", "dollar_volume", "ask_low", "bid_high", "taker_yes", "taker_no",
                 "n_trades", "max_trade", "block")

    def __init__(self):
        self.yes_bid = self.yes_ask = self.last = math.nan; self.bid_size = self.ask_size = 0.0; self.volume = self.oi = self.dollar_volume = math.nan
        self.ask_low = math.inf; self.bid_high = -math.inf; self.taker_yes = self.taker_no = 0.0; self.n_trades = 0; self.max_trade = 0.0; self.block = 0.0


class Aggregator:
    def __init__(self):
        self.minutes: dict[int, dict[str, KMinute]] = defaultdict(dict)        # minute start → ticker → KMinute
        self.quotes: dict[str, tuple] = {}; self.dirty: set[str] = set()       # ticker → (ts, yes_bid, yes_ask, bid_size, ask_size, last)
        self.max_event_s = 0.0; self.last_event_wall = 0.0
        self.flushed_through: datetime | None = None
        self.settled: list[tuple] = []                                            # (ticker, result, settled_ts, event_type)
        self.catalogue_todo: set[str] = set()                                     # tickers to fetch from REST
        self.fills: list[tuple] = []; self.order_updates: list[tuple] = []
        self.stats = {"events": 0, "tickers": 0, "trades": 0, "lifecycle": 0, "fills": 0, "orders": 0, "dropped_late": 0, "dropped_combo": 0, "flushed_minutes": 0,
                      "flushed_rows": 0, "reconnects": 0, "settled": 0, "catalogued": 0, "started_at": datetime.now(timezone.utc).isoformat()}

    # ---- ingest ----
    def _minute(self, ts_s: float, ticker: str) -> KMinute | None:
        m = int(ts_s // 60) * 60
        if self.flushed_through is not None and m <= self.flushed_through.timestamp():
            self.stats["dropped_late"] += 1
            return None
        row = self.minutes[m].get(ticker)
        if row is None:
            row = self.minutes[m][ticker] = KMinute()
        return row

    def _seen(self, ts_s: float) -> None:
        self.max_event_s = max(self.max_event_s, ts_s); self.last_event_wall = time.time()

    def ingest(self, e: dict) -> None:
        self.stats["events"] += 1
        t = e.get("type"); msg = e.get("msg") or {}
        if t == "ticker":
            tk = msg.get("market_ticker")
            if not tk or tk.startswith(COMBO_PREFIX):
                self.stats["dropped_combo"] += bool(tk); return
            ts_s = _num(msg.get("ts_ms"), 0.0) / 1000.0 or _num(msg.get("ts"), 0.0)
            if ts_s <= 0:
                return
            self._seen(ts_s); self.stats["tickers"] += 1
            yb, ya, px = _cents(msg.get("yes_bid_dollars")), _cents(msg.get("yes_ask_dollars")), _cents(msg.get("price_dollars"))
            if "yes_bid_dollars" not in msg and msg.get("yes_bid") is not None:                    # the older cents fields
                yb, ya, px = _num(msg.get("yes_bid"), math.nan), _num(msg.get("yes_ask"), math.nan), _num(msg.get("price"), math.nan)
            bs, as_ = _num(msg.get("yes_bid_size_fp")), _num(msg.get("yes_ask_size_fp"))
            row = self._minute(ts_s, tk)
            if row is not None:
                row.yes_bid, row.yes_ask, row.bid_size, row.ask_size = yb, ya, bs, as_
                if math.isfinite(px) and px > 0:
                    row.last = px
                row.volume = _num(msg.get("volume_fp"), row.volume); row.oi = _num(msg.get("open_interest_fp"), row.oi)
                row.dollar_volume = _num(msg.get("dollar_volume"), row.dollar_volume)
                if math.isfinite(ya):
                    row.ask_low = min(row.ask_low, ya)
                if math.isfinite(yb):
                    row.bid_high = max(row.bid_high, yb)
            self.quotes[tk] = (ts_s, yb, ya, bs, as_, px); self.dirty.add(tk)
        elif t == "trade":
            tk = msg.get("market_ticker")
            if not tk or tk.startswith(COMBO_PREFIX):
                return
            ts_s = _num(msg.get("ts_ms"), 0.0) / 1000.0 or _num(msg.get("ts"), 0.0)
            if ts_s <= 0:
                return
            self._seen(ts_s); self.stats["trades"] += 1
            px = _cents(msg.get("yes_price_dollars")) if "yes_price_dollars" in msg else _num(msg.get("yes_price"), math.nan)
            cnt = _num(msg.get("count_fp")) if "count_fp" in msg else _num(msg.get("count"))
            row = self._minute(ts_s, tk)
            if row is None:
                return
            if msg.get("taker_side") == "yes":
                row.taker_yes += cnt
            else:
                row.taker_no += cnt
            row.n_trades += 1; row.max_trade = max(row.max_trade, cnt)
            if msg.get("is_block_trade"):
                row.block += cnt
            if math.isfinite(px) and px > 0:
                row.last = px
        elif t == "market_lifecycle_v2":
            self.stats["lifecycle"] += 1
            tk = msg.get("market_ticker"); ev = msg.get("event_type") or ""
            if not tk or tk.startswith(COMBO_PREFIX):
                return
            if ev in SETTLE_EVENTS or msg.get("result"):
                self.settled.append((tk, msg.get("result"), _num(msg.get("settled_ts") or msg.get("determination_ts") or msg.get("ts"), 0.0) or time.time(), ev or "settled",
                                     _num(msg.get("close_ts"), 0.0) or None))
            elif ev in CATALOGUE_EVENTS or not ev:
                self.catalogue_todo.add(tk)
        elif t == "fill":
            self.stats["fills"] += 1
            self.fills.append((msg.get("trade_id"), msg.get("order_id"), msg.get("market_ticker"), msg.get("side"),
                               _cents(msg.get("yes_price_dollars")) if "yes_price_dollars" in msg else _num(msg.get("yes_price"), math.nan),
                               _num(msg.get("count_fp")) if "count_fp" in msg else _num(msg.get("count")), bool(msg.get("is_taker")), msg.get("action"),
                               _num(msg.get("ts"), time.time()), _num(msg.get("post_position"), math.nan), msg.get("subaccount"), json.dumps(msg, default=str)))
        elif t in ("user_order", "user_orders", "order"):
            self.stats["orders"] += 1
            o = msg.get("order") or msg
            if o.get("order_id"):
                self.order_updates.append((o.get("order_id"), o.get("status"), _num(o.get("fill_count_fp") if "fill_count_fp" in o else o.get("fill_count"), math.nan),
                                           _num(o.get("remaining_count_fp") if "remaining_count_fp" in o else o.get("remaining_count"), math.nan), json.dumps(o, default=str)))

    # ---- persistence ----
    def flush(self, before_minute: int) -> int:
        done = sorted(m for m in self.minutes if m < before_minute)
        if not done:
            return 0
        rows = []
        for m in done:
            ts = datetime.fromtimestamp(m, timezone.utc)
            for tk, r in self.minutes[m].items():
                f = lambda v: (v if math.isfinite(v) else None)
                rows.append((tk, ts, f(r.yes_bid), f(r.yes_ask), f(r.last), r.bid_size, r.ask_size, f(r.volume), f(r.oi), f(r.dollar_volume),
                             r.taker_yes, r.taker_no, r.n_trades, r.max_trade, r.block, f(r.ask_low), f(r.bid_high)))
        if rows:
            with transaction() as conn:
                conn.cursor().executemany(
                    "INSERT INTO kalshi_minutes (ticker, ts, yes_bid, yes_ask, last, bid_size, ask_size, volume_fp, open_interest_fp, dollar_volume, taker_buy_yes, taker_buy_no, "
                    "n_trades, max_trade, block_contracts, yes_ask_low, yes_bid_high) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (ticker, ts) DO UPDATE SET yes_bid = COALESCE(EXCLUDED.yes_bid, kalshi_minutes.yes_bid), yes_ask = COALESCE(EXCLUDED.yes_ask, kalshi_minutes.yes_ask), "
                    "last = COALESCE(EXCLUDED.last, kalshi_minutes.last), bid_size = EXCLUDED.bid_size, ask_size = EXCLUDED.ask_size, "
                    "volume_fp = COALESCE(EXCLUDED.volume_fp, kalshi_minutes.volume_fp), open_interest_fp = COALESCE(EXCLUDED.open_interest_fp, kalshi_minutes.open_interest_fp), "
                    "dollar_volume = COALESCE(EXCLUDED.dollar_volume, kalshi_minutes.dollar_volume), taker_buy_yes = kalshi_minutes.taker_buy_yes + EXCLUDED.taker_buy_yes, "
                    "taker_buy_no = kalshi_minutes.taker_buy_no + EXCLUDED.taker_buy_no, n_trades = kalshi_minutes.n_trades + EXCLUDED.n_trades, "
                    "max_trade = greatest(kalshi_minutes.max_trade, EXCLUDED.max_trade), block_contracts = kalshi_minutes.block_contracts + EXCLUDED.block_contracts, "
                    "yes_ask_low = least(kalshi_minutes.yes_ask_low, EXCLUDED.yes_ask_low), yes_bid_high = greatest(kalshi_minutes.yes_bid_high, EXCLUDED.yes_bid_high)", rows)
        for m in done:
            self.minutes.pop(m, None)
        self.stats["flushed_minutes"] += len(done); self.stats["flushed_rows"] += len(rows)
        newest = datetime.fromtimestamp(done[-1], timezone.utc)
        if self.flushed_through is None or newest > self.flushed_through:
            self.flushed_through = newest
        return len(rows)

    def flush_bound(self, now: float) -> int:
        cur = int(now // 60) * 60
        wm = int((self.max_event_s - FLUSH_GRACE_S) // 60) * 60
        if now - self.last_event_wall >= FLUSH_FALLBACK_S:
            wm = max(wm, int((now - FLUSH_FALLBACK_S) // 60) * 60)
        return min(cur, wm)

    def write_quotes(self) -> int:
        todo, self.dirty = self.dirty, set()
        rows = []
        for tk in todo:
            q = self.quotes.get(tk)
            if q is None:
                continue
            ts_s, yb, ya, bs, as_, px = q
            mid = (yb + ya) / 2.0 if math.isfinite(yb) and math.isfinite(ya) else None
            rows.append((tk, datetime.fromtimestamp(ts_s, timezone.utc), yb if math.isfinite(yb) else None, ya if math.isfinite(ya) else None, mid, bs, as_, px if math.isfinite(px) else None))
        if not rows:
            return 0
        try:
            with transaction() as conn:
                conn.cursor().executemany("INSERT INTO kalshi_quotes (ticker, ts, yes_bid, yes_ask, mid, bid_size, ask_size, last) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                                          "ON CONFLICT (ticker) DO UPDATE SET ts = EXCLUDED.ts, yes_bid = EXCLUDED.yes_bid, yes_ask = EXCLUDED.yes_ask, mid = EXCLUDED.mid, "
                                          "bid_size = EXCLUDED.bid_size, ask_size = EXCLUDED.ask_size, last = COALESCE(EXCLUDED.last, kalshi_quotes.last), updated_at = now()", rows)
        except Exception:
            self.dirty |= todo
            log.exception("quote upsert failed; %d quotes retried", len(todo))
            return 0
        return len(rows)

    def write_settlements(self) -> int:
        rows, self.settled = self.settled, []
        if not rows:
            return 0
        try:
            with transaction() as conn:
                for tk, result, ts_s, ev, close_ts in rows:
                    conn.execute("INSERT INTO kalshi_markets (ticker, status, result, settlement_ts, source) VALUES (%s,%s,%s,%s,'stream') ON CONFLICT (ticker) DO UPDATE SET "
                                 "status = EXCLUDED.status, result = COALESCE(EXCLUDED.result, kalshi_markets.result), settlement_ts = COALESCE(EXCLUDED.settlement_ts, kalshi_markets.settlement_ts), "
                                 "close_time = COALESCE(kalshi_markets.close_time, %s), updated_at = now()",
                                 (tk, ev, result if result in ("yes", "no") else None, datetime.fromtimestamp(ts_s, timezone.utc), datetime.fromtimestamp(close_ts, timezone.utc) if close_ts else None))
                    if result not in ("yes", "no"):
                        self.catalogue_todo.add(tk)                 # the result comes with the market fetch when the message lacks it
        except Exception:
            self.settled = rows + self.settled
            log.exception("settlement rows failed; %d retried", len(rows))
            return 0
        self.stats["settled"] += len(rows)
        return len(rows)

    def write_private(self) -> int:
        fills, self.fills = self.fills, []; orders, self.order_updates = self.order_updates, []
        if not fills and not orders:
            return 0
        try:
            with transaction() as conn:
                if fills:
                    conn.cursor().executemany(
                        "INSERT INTO kalshi_fills (trade_id, order_id, book, ticker, side, price_cents, count, fee_cents, is_taker, action, ts, post_position, raw) "
                        "SELECT %s, %s, o.book, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s FROM (SELECT %s::text AS oid) q LEFT JOIN kalshi_orders o ON o.order_id = q.oid "
                        "ON CONFLICT (trade_id) DO NOTHING",
                        [(tid, oid, tk, side, px, cnt, taker, act, datetime.fromtimestamp(ts_s, timezone.utc), pp if math.isfinite(pp) else None, raw, oid)
                         for tid, oid, tk, side, px, cnt, taker, act, ts_s, pp, sub, raw in fills if tid and tk])
                if orders:
                    conn.cursor().executemany("UPDATE kalshi_orders SET status = COALESCE(%s, status), fill_count = COALESCE(%s, fill_count), remaining = COALESCE(%s, remaining), "
                                              "response = %s::jsonb, updated_at = now() WHERE order_id = %s",
                                              [(st, fc if math.isfinite(fc) else None, rem if math.isfinite(rem) else None, raw, oid) for oid, st, fc, rem, raw in orders])
        except Exception:
            self.fills = fills + self.fills; self.order_updates = orders + self.order_updates
            log.exception("fill/order rows failed; retried")
            return 0
        return len(fills) + len(orders)


# ---------------------------------------------------------------- the catalogue (REST, in a thread)
def refresh_open_markets(rest=None, stop: threading.Event | None = None) -> int:
    """Every open non-combo market into ``kalshi_markets`` with its event and series (the engine's universe and metadata)."""
    from .client import rest as _rest
    from .history import _known_events, _known_series, ensure_catalogue
    rest = rest or _rest(); n = 0
    with transaction() as conn:
        known_e, known_s = _known_events(conn), _known_series(conn)
    for page in rest.markets(status="open", mve_filter="exclude"):
        if stop is not None and stop.is_set():
            break
        with transaction() as conn:
            D.upsert_markets(conn, page, "stream")
            ensure_catalogue(rest, conn, {m["event_ticker"] for m in page if m.get("event_ticker")}, known_e, known_s)
        n += len(page)
    return n


def fetch_markets(tickers: list[str], rest=None) -> int:
    """Markets the lifecycle channel announced (or settled without a result): one GET each, with event and series."""
    from .client import KalshiApiError, rest as _rest
    from .history import _known_events, _known_series, ensure_catalogue
    rest = rest or _rest(); n = 0
    with transaction() as conn:
        known_e, known_s = _known_events(conn), _known_series(conn)
    for tk in tickers:
        try:
            m = rest.market(tk) or {}                       # KalshiRest.market already unwraps {"market": ...}
        except KalshiApiError as e:
            log.warning("market %s: %s", tk, e); continue
        if not m.get("ticker"):
            continue
        with transaction() as conn:
            D.upsert_markets(conn, [m], "stream")
            if m.get("event_ticker"):
                ensure_catalogue(rest, conn, {m["event_ticker"]}, known_e, known_s)
        n += 1
    return n


def archive_old(keep_days: int) -> int:
    """Minute rows older than ``keep_days`` go to ``data/kalshi/stream/<day>.parquet`` (one file per UTC day), then out of
    the table: at ~9,000 quoted markets a minute the table would otherwise grow by ~13 M rows a day."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=keep_days)
    with transaction() as conn:
        days = [r["d"] for r in conn.execute("SELECT DISTINCT date_trunc('day', ts) AS d FROM kalshi_minutes WHERE ts < %s", (cutoff,)).fetchall()]
    n = 0
    for d in days:
        with transaction() as conn:
            rows = conn.execute("SELECT * FROM kalshi_minutes WHERE ts >= %s AND ts < %s ORDER BY ticker, ts", (d, d + timedelta(days=1))).fetchall()
            if rows:
                out = config.KALSHI_DIR / "stream"; out.mkdir(parents=True, exist_ok=True)
                path = out / f"{d:%Y-%m-%d}.parquet"; tmp = path.with_name(path.name + ".tmp")
                pq.write_table(pa.Table.from_pylist([dict(r) for r in rows]), tmp, compression="zstd"); tmp.replace(path)
                conn.execute("DELETE FROM kalshi_minutes WHERE ts >= %s AND ts < %s", (d, d + timedelta(days=1)))
                n += len(rows)
    return n


def _status(agg: Aggregator, extra: dict) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (STATUS_KEY, json.dumps({**agg.stats, **extra, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str)))
    except Exception:
        log.debug("kalshi stream status write failed", exc_info=True)


def auth_headers() -> dict | None:
    from kalshi_python_async.auth import KalshiAuth
    from .vendor.kalshi_client import _load_private_key
    key = _load_private_key()
    if not (config.KALSHI_API_KEY_ID and key):
        return None
    path = "/" + config.KALSHI_WS_URL.split("/", 3)[3] if config.KALSHI_WS_URL.count("/") >= 3 else "/trade-api/ws/v2"
    return KalshiAuth(config.KALSHI_API_KEY_ID, key).create_auth_headers("GET", path)


def subscribe_commands(private: bool) -> list[dict]:
    chans = list(PUBLIC_CHANNELS) + (list(PRIVATE_CHANNELS) if private else [])
    return [{"id": i, "cmd": "subscribe", "params": {"channels": [c]}} for i, c in enumerate(chans, start=1)]


def _logged(name: str, fn, *args):
    try:
        n = fn(*args)
        if n:
            log.info("%s: %d", name, n)
        return n
    except Exception:
        log.exception("%s failed", name)
        return 0


async def _run(stop: threading.Event | None, agg: Aggregator) -> None:
    import websockets
    ctx = ssl.create_default_context(cafile=certifi.where()); backoff = 1.0
    if auth_headers() is None:
        raise RuntimeError("the Kalshi websocket needs KALSHI_API_KEY_ID and a readable KALSHI_PRIVATE_KEY_PATH")
    private = config.KALSHI_SUBACCOUNT > 0 or config.KALSHI_LIVE_ENABLED
    last_status = last_flush = last_quotes = last_life = 0.0; last_cat = time.time() - CATALOGUE_S + 30; last_msg = 0.0; rate_n = 0; rate_t = time.time()
    bg: dict[str, asyncio.Task | None] = {"catalogue": None, "fetch": None, "archive": None}; last_archive = time.time() - ARCHIVE_S + 120
    while not (stop is not None and stop.is_set()):
        try:
            headers = auth_headers()                          # signed with the current timestamp: a reconnect with the first handshake's signature is refused (401)
            async with websockets.connect(config.KALSHI_WS_URL, additional_headers=headers, ssl=ctx, open_timeout=15, max_queue=None, ping_interval=20, ping_timeout=20) as ws:
                for cmd in subscribe_commands(private):
                    await ws.send(json.dumps(cmd))
                log.info("kalshi stream connected (%s)", "public + private channels" if private else "public channels"); backoff = 1.0
                while not (stop is not None and stop.is_set()):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        log.warning("kalshi stream: no messages for 30 s; reconnecting"); break
                    last_msg = time.time(); rate_n += 1
                    try:
                        e = orjson.loads(raw)
                    except orjson.JSONDecodeError:
                        continue
                    if isinstance(e, dict):
                        if e.get("type") == "error":
                            log.warning("kalshi stream error frame: %s", str(e)[:300])
                        else:
                            agg.ingest(e)
                    now = time.time()
                    if now - last_flush >= 1.0:
                        try:
                            agg.flush(agg.flush_bound(now))
                        except Exception:
                            log.exception("minute flush failed; rows kept for the next attempt")
                        last_flush = now
                    if now - last_quotes >= QUOTE_FLUSH_S:
                        agg.write_quotes(); agg.write_settlements(); agg.write_private(); last_quotes = now
                    if now - last_life >= LIFECYCLE_S and agg.catalogue_todo and (bg["fetch"] is None or bg["fetch"].done()):
                        todo = sorted(agg.catalogue_todo)[:200]; agg.catalogue_todo -= set(todo)
                        bg["fetch"] = asyncio.create_task(asyncio.to_thread(_logged, "markets fetched from the lifecycle channel", fetch_markets, todo)); last_life = now
                    if now - last_cat >= CATALOGUE_S and (bg["catalogue"] is None or bg["catalogue"].done()):
                        bg["catalogue"] = asyncio.create_task(asyncio.to_thread(_logged, "open markets refreshed", refresh_open_markets, None, stop)); last_cat = now
                    if now - last_archive >= ARCHIVE_S and (bg["archive"] is None or bg["archive"].done()):
                        bg["archive"] = asyncio.create_task(asyncio.to_thread(_logged, "archived old Kalshi minute rows", archive_old, config.KALSHI_MINUTES_KEEP_DAYS)); last_archive = now
                    if now - last_status >= STATUS_S:
                        _status(agg, {"events_per_s": rate_n / max(now - rate_t, 1e-9), "pending_minutes": len(agg.minutes), "lag_s": now - last_msg, "connected": True,
                                      "tickers_quoted": len(agg.quotes), "catalogue_todo": len(agg.catalogue_todo), "private": private,
                                      "flushed_through": agg.flushed_through.isoformat() if agg.flushed_through else None,
                                      "event_watermark": datetime.fromtimestamp(agg.max_event_s, timezone.utc).isoformat() if agg.max_event_s else None})
                        rate_n = 0; rate_t = now; last_status = now
        except Exception as e:
            agg.stats["reconnects"] += 1; log.warning("kalshi stream: %s; reconnecting in %.0fs", e, backoff)
            _status(agg, {"connected": False, "last_error": str(e)[:200]})
            await asyncio.sleep(backoff); backoff = min(60.0, backoff * 2)
    for t in bg.values():
        if t is not None:
            try:
                await t
            except Exception:
                pass
    try:
        agg.flush(int(time.time() // 60) * 60 + 60)
    except Exception:
        log.exception("final minute flush failed")
    agg.write_quotes(); agg.write_settlements(); agg.write_private()


def main(stop_event: threading.Event | None = None) -> None:
    setup("kalshi_stream"); agg = Aggregator()
    record_event("info", "kalshi_stream", "kalshi stream started", {"url": config.KALSHI_WS_URL})
    try:
        asyncio.run(_run(stop_event, agg))
    finally:
        _status(agg, {"connected": False, "stage": "stopped"})
        record_event("info", "kalshi_stream", "kalshi stream stopped", {k: agg.stats.get(k) for k in ("events", "tickers", "trades", "flushed_rows", "settled", "reconnects")})
