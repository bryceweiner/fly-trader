"""The Kalshi trading engine — the ``kalshi_runner`` worker (agent/minute_engine.py for prediction markets).

Every UTC minute, once the stream has written it (``kalshi_stream_status.flushed_through``), the minute's rows of
``kalshi_minutes`` become ``Bar``s fed to the same feature engine the corpus was built with (kalshi/features.py): one
``MarketState`` per market, one ``EventState`` per event (the siblings' quotes), metadata from ``kalshi_markets`` /
``kalshi_events`` / ``kalshi_series`` (kalshi/mature.load_meta). A market is scored on a minute it had a message on —
as a corpus row exists only for minutes with a candle — when it lies inside the entry window (``KALSHI_MIN_MINUTES_TO_CLOSE``
… ``KALSHI_MAX_DAYS_TO_CLOSE`` before close) and passes the eligibility gate (kalshi/decisions.eligible_mask); both sides
go to every book as ``K_COLS`` vectors. Books implement ``name``, ``on_minute(ctx)``, ``tickers_watched()``,
``maybe_reload()``, ``done`` and ``finish()``; each trades in its own savepoint. Warm-up feeds the last ``WARMUP_MIN``
minutes without trading; no entries while the stream is stale; missed minutes are replayed through the features only.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..db.apilog import record_event
from ..db.connection import connect, transaction
from ..logging_setup import setup
from . import stream as KS
from .decisions import eligible_mask
from .features import K_COLS, Bar, EventState, MarketMeta, MarketState, warm_fees
from .mature import load_meta
from .strategies import in_universe

log = logging.getLogger(__name__)
WARMUP_MIN = 1440
STREAM_STALE_S = 180
META_TTL_S = 3600
MODEL_CHECK_S = 600
IDLE_EVICT_S = 7 * 86400
KALSHI_LOCK_KEY = 0x6B616C5F72756E     # "kal_run": one Kalshi runner per database


@dataclass
class KMinute:
    """What every Kalshi book sees of one minute."""
    conn: object
    m0: datetime
    m1: datetime
    m1_epoch: float
    X: np.ndarray                      # [n, len(K_COLS)] eligible rows
    keys: list                         # [(ticker, side)] per row
    hold_s: np.ndarray                 # per row: seconds from the minute start to the market's close (the position's hold)
    metas: dict                        # ticker → MarketMeta (every market with a state)
    quotes: dict                       # ticker → (yes_bid, yes_ask, bid_size, ask_size) latest known
    extremes: dict                     # ticker → (yes_ask_low, yes_bid_high) of this minute (markets with a row)
    n_rows: int
    fresh: bool
    trade: bool
    engine: "KalshiMinuteEngine"
    results: dict = field(default_factory=dict)      # ticker → 'yes'|'no' for markets asked about (``settled``)

    @property
    def t_start(self) -> float:
        return self.m1_epoch - 60.0

    def settled(self, tickers) -> dict:
        """Results of the given markets that have resolved (from ``kalshi_markets``; the stream writes them)."""
        tickers = [t for t in set(tickers) if t not in self.results]
        if tickers:
            for r in self.conn.execute("SELECT ticker, result FROM kalshi_markets WHERE ticker = ANY(%s) AND result IN ('yes','no')", (tickers,)).fetchall():
                self.results[r["ticker"]] = r["result"]
        return self.results


class KalshiMinuteEngine:
    def __init__(self, books: list | None = None):
        self.books = list(books or [])
        self.states: dict[str, MarketState] = {}; self.events: dict[str, EventState] = {}
        self.metas: dict[str, MarketMeta] = {}; self.meta_at: dict[str, float] = {}; self.missing_meta: dict[str, float] = {}
        self.quotes: dict[str, tuple] = {}
        self.last_minute: float | None = None; self.last_sweep = time.time(); self.fees_warmed = False

    def has_book(self, name: str) -> bool:
        return any(b.name == name for b in self.books)

    # ---- metadata ----
    def _metas(self, conn, tickers: list[str]) -> None:
        now = time.time()
        todo = [t for t in tickers if (t not in self.metas or now - self.meta_at.get(t, 0) > META_TTL_S) and now - self.missing_meta.get(t, 0) > META_TTL_S]
        if not todo:
            return
        if not self.fees_warmed:
            warm_fees(conn); self.fees_warmed = True
        got = load_meta(conn, todo)
        for t in todo:
            if t in got:
                self.metas[t] = got[t]; self.meta_at[t] = now; self.missing_meta.pop(t, None)
            else:
                self.missing_meta[t] = now                      # not in the catalogue yet (the stream fetches lifecycle announcements)

    # ---- one minute ----
    def _rows(self, conn, m0: datetime) -> list[dict]:
        return [dict(r) for r in conn.execute("SELECT ticker, yes_bid, yes_ask, last, bid_size, ask_size, volume_fp, open_interest_fp, taker_buy_yes, taker_buy_no, n_trades, max_trade, "
                                              "block_contracts, yes_ask_low, yes_bid_high FROM kalshi_minutes WHERE ts = %s", (m0,)).fetchall()]

    @staticmethod
    def _bar(r: dict, t_end: float) -> Bar:
        f = lambda v: float(v) if v is not None and 0 < float(v) < 100 else math.nan
        return Bar(t_end, f(r["yes_bid"]), f(r["yes_ask"]), float(r["last"]) if r["last"] is not None else math.nan, float(r["bid_size"] or 0.0), float(r["ask_size"] or 0.0),
                   float(r["volume_fp"] or 0.0), float(r["open_interest_fp"] or 0.0), float(r["taker_buy_yes"] or 0.0), float(r["taker_buy_no"] or 0.0),
                   float(r["n_trades"] or 0.0), float(r["max_trade"] or 0.0), float(r["block_contracts"] or 0.0))

    def feed(self, conn, rows: list[dict], m1_epoch: float) -> list[str]:
        """Append this minute's bars to the states (and the events' sibling quotes); returns the tickers with a bar."""
        self._metas(conn, [r["ticker"] for r in rows])
        fed = []
        for r in rows:
            tk = r["ticker"]; bar = self._bar(r, m1_epoch)
            st = self.states.get(tk)
            if st is None:
                st = self.states[tk] = MarketState(tk)
            st.append(bar); fed.append(tk)
            self.quotes[tk] = (bar.yes_bid if math.isfinite(bar.yes_bid) else None, bar.yes_ask if math.isfinite(bar.yes_ask) else None, bar.bid_size, bar.ask_size)
            meta = self.metas.get(tk)
            if meta is not None and meta.event_ticker:
                ev = self.events.get(meta.event_ticker)
                if ev is None:
                    ev = self.events[meta.event_ticker] = EventState()
                ev.update(tk, bar, meta.strike)
        return fed

    def rows_for(self, fed: list[str], m1_epoch: float) -> tuple[np.ndarray, list, np.ndarray, int]:
        """K_COLS vectors of both sides of every fed market inside the entry window that passes the gate."""
        xs, keys, holds = [], [], []; n_rows = 0
        min_s = config.KALSHI_MIN_MINUTES_TO_CLOSE * 60.0; max_s = config.KALSHI_MAX_DAYS_TO_CLOSE * 86400.0
        for tk in fed:
            meta = self.metas.get(tk); st = self.states.get(tk)
            if meta is None or st is None or meta.close_ts is None:
                continue
            to_close = meta.close_ts - (m1_epoch - 60.0)
            if not (min_s <= to_close <= max_s):
                continue
            ev = self.events.get(meta.event_ticker) if meta.event_ticker else None
            for side in ("yes", "no"):
                xs.append(st.features(m1_epoch, meta, side, ev)); keys.append((tk, side)); holds.append(to_close); n_rows += 1
        if not xs:
            return np.zeros((0, len(K_COLS)), np.float32), [], np.zeros(0), 0
        X = np.nan_to_num(np.asarray(xs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0); holds = np.asarray(holds, dtype=np.float64)
        ok = eligible_mask(X, K_COLS, holds) & in_universe(X, K_COLS)
        return X[ok], [k for k, o in zip(keys, ok) if o], holds[ok], n_rows

    def _stream_through(self) -> float | None:
        with transaction() as conn:
            r = conn.execute("SELECT value->>'flushed_through' AS ft FROM ui_settings WHERE key = %s", (KS.STATUS_KEY,)).fetchone()
        return datetime.fromisoformat(r["ft"]).timestamp() if r and r["ft"] else None

    def ready_through(self, m1_epoch: float) -> float:
        ft = self._stream_through()
        return min(m1_epoch, ft + 60.0) if ft is not None else float("-inf")

    def stream_fresh(self, m0_epoch: float) -> bool:
        ft = self._stream_through()
        return ft is not None and m0_epoch - ft <= STREAM_STALE_S

    def warm_up(self, m1_epoch: float, minutes: int = WARMUP_MIN) -> int:
        n = 0
        with transaction() as conn:
            for k in range(minutes, 0, -1):
                t1 = m1_epoch - 60 * k
                rows = self._rows(conn, datetime.fromtimestamp(t1 - 60, timezone.utc))
                if rows:
                    self.feed(conn, rows, t1); n += len(rows)
        log.info("kalshi warm-up: %d market-minutes over the last %d minutes (%d markets, %d with metadata)", n, minutes, len(self.states), len(self.metas))
        return n

    def _sweep(self, now_s: float, held: set[str]) -> None:
        if now_s - self.last_sweep < 3600:
            return
        self.last_sweep = now_s
        idle = [t for t, s in self.states.items() if t not in held and (not s.ts or s.ts[-1] < now_s - IDLE_EVICT_S)]
        for t in idle:
            self.states.pop(t, None); self.metas.pop(t, None); self.meta_at.pop(t, None); self.quotes.pop(t, None)
        if idle:
            log.info("evicted %d market states idle for more than %d days", len(idle), IDLE_EVICT_S // 86400)

    def run_minute(self, m1_epoch: float, trade: bool = True) -> dict:
        m1 = datetime.fromtimestamp(m1_epoch, timezone.utc); m0 = datetime.fromtimestamp(m1_epoch - 60, timezone.utc)
        out: dict = {"minute": m1.isoformat()}
        with transaction() as conn:
            rows = self._rows(conn, m0)
            fed = self.feed(conn, rows, m1_epoch)
            X, keys, holds, n_rows = self.rows_for(fed, m1_epoch)
            extremes = {r["ticker"]: (r["yes_ask_low"], r["yes_bid_high"]) for r in rows}
            ctx = KMinute(conn=conn, m0=m0, m1=m1, m1_epoch=m1_epoch, X=X, keys=keys, hold_s=holds, metas=self.metas, quotes=self.quotes, extremes=extremes,
                          n_rows=n_rows, fresh=trade and self.stream_fresh(m1_epoch - 60), trade=trade, engine=self)
            out.update(markets_active=len(rows), in_window=n_rows // 2, eligible=len(keys)); held: set[str] = set()
            for b in list(self.books):
                try:
                    with conn.transaction():
                        out[b.name] = b.on_minute(ctx)
                        held |= set(b.tickers_watched())
                except Exception:
                    log.exception("%s: minute %s failed", b.name, m1.isoformat())
                    record_event("error", b.name, f"minute failed: {m1.isoformat()}")
            self._sweep(m1_epoch, held)
        for b in [b for b in self.books if b.done]:
            b.finish(); self.books.remove(b)
        return out


def _status(key: str, value: dict) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (key, json.dumps({**value, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str)))
    except Exception:
        log.debug("status write failed", exc_info=True)


def admit_books(engine: KalshiMinuteEngine) -> list:
    from . import fly_session
    new = []
    if not engine.has_book(fly_session.NAME):
        fb, why = fly_session.try_start()
        if fb is not None:
            new.append(fb)
        else:
            _status(fly_session.STATUS_KEY, {"stage": "not trading", "detail": why})
    return new


def acquire_lock():
    conn = connect(autocommit=True)
    try:
        ok = conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (KALSHI_LOCK_KEY,)).fetchone()["ok"]
    except Exception:
        conn.close(); raise
    if not ok:
        conn.close()
        raise RuntimeError("another Kalshi runner holds the lock; refusing to start a second one")
    return conn


def main(stop_event: threading.Event | None = None) -> None:
    setup("kalshi_runner")
    try:
        lock_conn = acquire_lock()
    except RuntimeError as e:
        log.error("%s", e); record_event("error", "kalshi_runner", str(e)); raise
    try:
        record_event("info", "kalshi_runner", "kalshi runner started", {"live": bool(config.KALSHI_LIVE_ENABLED)})
        _main(stop_event)
    finally:
        lock_conn.close()


def _main(stop_event: threading.Event | None) -> None:
    engine = KalshiMinuteEngine()
    while not (stop_event is not None and stop_event.is_set()):
        engine.books = admit_books(engine)
        if engine.books:
            break
        log.info("kalshi engine waiting: no book qualifies yet")
        (stop_event or threading.Event()).wait(MODEL_CHECK_S)
    else:
        return
    m_start = math.floor(time.time() / 60) * 60
    engine.warm_up(m_start); engine.last_minute = m_start - 60; last_check = time.time(); n_min = 0
    try:
        while not (stop_event is not None and stop_event.is_set()):
            now = time.time(); m1 = math.floor(now / 60) * 60
            if now - last_check >= MODEL_CHECK_S:
                last_check = now
                for b in engine.books:
                    try:
                        b.maybe_reload()
                    except Exception:
                        log.exception("%s: model reload check failed", b.name)
                try:
                    engine.books += admit_books(engine)
                except Exception:
                    log.exception("book admission failed")
            if engine.books and m1 > engine.last_minute and now - m1 >= 4.0 and (upto := engine.ready_through(m1)) > engine.last_minute:
                for minute in range(int(engine.last_minute) + 60, int(upto) + 1, 60):
                    try:
                        st = engine.run_minute(float(minute), trade=(minute == int(m1))); n_min += 1
                        if minute == int(m1) and n_min % 10 == 0:
                            log.info("kalshi minute %s: active %d in window %d eligible %d | %s", st["minute"], st.get("markets_active", 0), st.get("in_window", 0), st.get("eligible", 0),
                                     " | ".join(f"{b.name}: {(st.get(b.name) or {}).get('stage')} picks {(st.get(b.name) or {}).get('picks')}" for b in engine.books))
                    except Exception:
                        log.exception("kalshi minute %s failed", minute); record_event("error", "kalshi_runner", f"minute failed: {minute}")
                engine.last_minute = upto
            time.sleep(0.5)
    finally:
        for b in engine.books:
            try:
                b.finish()
            except Exception:
                log.exception("%s: finish failed", b.name)
