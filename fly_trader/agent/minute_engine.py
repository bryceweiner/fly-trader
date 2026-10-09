"""The trading engine on the live stream: every UTC minute, features once per mint, then every book trades on them.

Every UTC minute, once the stream has written it (``pumpstream_status.flushed_through``), the minute's rich candle per
mint (``pump_minutes``: open/high/low/close, SOL volume by side, buy/sell counts, distinct traders, quote reserve; the
same fields and filters ``train/mature.py`` builds from the archive) is fed to the live feature engine as two
synthetic trades, exactly as in training. Graduation time and creation/creator facts come from ``corpus_meta``, the
same source training uses. The full feature vector (``decisions.X_COLS``) of the rows in the models' universe
(``selector.in_universe``) that pass the eligibility gate (pool ≥ ``MIN_RESQ_SOL``, 15-minute volume ≥
``MIN_VOL_15M_SOL``, no scale break — a close 50× from the previous one or a reserve above 100,000 SOL — since the
mint's state began) goes to every book; each picks its model's columns by name.

Books — ``SelectorBook`` (agent/selector_session.py) and ``FlyBook`` (agent/fly_session.py) — implement ``name``,
``on_bars(t_start, bars)`` (every processed minute, warm-up included: open, close and exit cost of every traded mint),
``on_minute(ctx)``, ``open_mints(conn)``, ``maybe_reload()``, ``done`` and ``finish()``. Each book trades inside its own
savepoint, so one book's failure never undoes the other's minute. Books join when they qualify (checked every
``MODEL_CHECK_S``): the selector while it has a deployable model and has not handed its seat to a fly that is trading;
the fly once its design passed the replay and a deployable bootstrap exists. No entries or exits while the stream is
stale; missed minutes are replayed through the features without trading.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .. import config, markets
from ..db.apilog import record_event
from ..db.connection import transaction
from ..market.exit_cost import PUMP_SUPPLY
from ..market.exit_cost import cost_at_size
from ..market.features import FEATURES, FIDX, TokenMeta, TokenState
from ..train.corpus_meta import FEATURE_COLS as META_COLS
from ..train.decisions import MIN_RESQ_SOL, MIN_VOL_15M_SOL, X_COLS
from ..train.flow import FlowWindow, MarketWindow
from ..train.mature import MASKED
from ..train.selector import in_universe

log = logging.getLogger(__name__)
WARMUP_MIN = 1440            # training warms each day on the previous day's candles
IDLE_EVICT_S = 86400         # training has no history for a mint idle since before the previous day
STREAM_STALE_S = 180         # no entries or exits when the stream's newest complete minute is older than this
META_TTL_S = 600
MODEL_CHECK_S = 600          # how often books reload newer models and new books are admitted


class MintState:
    __slots__ = ("st", "meta", "hist", "decimals", "pool", "program_label", "graduated_at", "meta_row", "prev_close", "broken", "flow")

    def __init__(self, mint: str, decimals: int, pool: str | None, program_label: str | None, graduated_at: float | None, meta_row: dict | None):
        self.st = TokenState(mint); self.meta = TokenMeta(mint=mint, program_label=program_label or "Pump.fun Amm", graduated_at=graduated_at)
        self.hist: deque = deque(maxlen=200)       # (t_end, n_trades, n_traders) for the trailing hour
        self.decimals, self.pool, self.program_label, self.graduated_at, self.meta_row = decimals, pool, program_label, graduated_at, meta_row
        self.prev_close: float | None = None; self.broken = False
        self.flow = FlowWindow()                    # wallet flow inputs (train/flow.py, as mature.build_day)


@dataclass
class Minute:
    """What every book sees of one minute."""
    conn: object
    m0: datetime
    m1: datetime
    m1_epoch: float
    agg: dict
    X: np.ndarray            # [n, len(X_COLS)] eligible rows in the universe
    infos: list
    mints: list
    bars: dict               # mint -> (minute start, open, close, exit cost) for every traded mint
    prices: dict
    resqs: dict
    fees: dict
    fresh: bool
    trade: bool
    engine: "MinuteEngine"
    market: object = None     # fly_trader/markets.MarketSpec of these rows (None: Solana)

    @property
    def t_start(self) -> float:
        return self.m1_epoch - 60.0

    def last_resq(self, mint: str) -> float | None:
        return self.engine._last_resq(self.conn, mint)

    def mcap(self, mint: str, price: float | None) -> float | None:
        return self.engine._mcap(mint, price)


class MinuteEngine:
    """One market's minutes (``market``: Solana — pump_minutes, corpus_meta — or Robinhood Chain — rh_minutes, rh_meta,
    amounts in ETH). The engines of both markets share one list of books; each minute goes only to its market's books."""

    def __init__(self, books: list | None = None, live: bool = False, market=None):
        self.spec = market or markets.SOL; self.rh = self.spec.chain != "sol"
        # how old a minute may be and still be traded: Solana's feed flushes within seconds; the RH stream's minutes land
        # ~2–3 min late (confirmations, the index pass, the price marks), so on RH "the current minute" never arrives
        self.trade_lag_s = float(config.RH_TRADE_LAG_S) if self.rh else 0.0
        self.books = list(books or []); self.live = live
        self.states: dict[str, MintState] = {}
        self.meta_cache: dict[str, dict] = {}; self.last_sweep = time.time(); self.last_minute: float | None = None
        self.market = MarketWindow()                 # mkt_vol_1h over every traded mint (as mature._market_windows)

    def has_book(self, name: str) -> bool:
        return any(b.name == name for b in self.books)

    def tradeable(self, minute: float, m1: float) -> bool:
        """Whether minute ``minute`` (its end, epoch s) is recent enough to trade at wall-clock minute ``m1``."""
        return int(m1) - self.trade_lag_s <= int(minute) <= int(m1)

    def my_books(self) -> list:
        return [b for b in self.books if getattr(b, "chain", "sol") == self.spec.chain]

    # ---- lookups ----
    def _meta(self, conn, mint: str) -> dict | None:
        """``corpus_meta`` row (graduation time + creation/creator features), cached for 10 minutes: rows appear at
        graduation and the archive rebuild later replaces stream values, so the cache is refreshed."""
        hit = self.meta_cache.get(mint)
        if hit is None or time.time() - hit["at"] > META_TTL_S:
            if self.rh:                                  # Robinhood Chain: rh_meta, the same columns (rh/meta.py), plus the quote class
                r = conn.execute("SELECT graduated_at, supply, quote_class, " + ", ".join(META_COLS) + " FROM rh_meta WHERE mint = %s", (mint,)).fetchone()
            else:
                r = conn.execute("SELECT graduated_at, supply, " + ", ".join(META_COLS) + " FROM corpus_meta WHERE mint = %s", (mint,)).fetchone()
            hit = self.meta_cache[mint] = {"row": dict(r) if r else None, "at": time.time()}
        return hit["row"]

    def _state(self, conn, mint: str, pool: str | None, program_label: str | None) -> MintState:
        s = self.states.get(mint)
        if s is None:
            if self.rh:                                  # every Pons launch has 18 decimals
                s = MintState(mint, 18, pool, program_label, None, None); s.meta.ec_size = 0.1 * self.spec.k()
            else:
                t = conn.execute("SELECT decimals FROM tokens WHERE mint = %s", (mint,)).fetchone()
                s = MintState(mint, int((t or {}).get("decimals") or 6), pool, program_label, None, None)
            self.states[mint] = s
        row = self._meta(conn, mint)
        s.meta_row = row
        g = row["graduated_at"].timestamp() if row and row.get("graduated_at") else None
        if g != s.graduated_at:                     # same source as training: corpus_meta.graduated_at
            s.graduated_at = g; s.meta.graduated_at = g
        s.meta.supply = float(row["supply"]) if row and row.get("supply") else PUMP_SUPPLY   # as training (train/mature.build_day)
        return s

    def _last_resq(self, conn, mint: str) -> float | None:
        s = self.states.get(mint)
        if s is not None and s.st.last_res_quote_sol:
            return s.st.last_res_quote_sol
        if self.rh:
            r = conn.execute("SELECT resq_eth AS q FROM rh_minutes WHERE mint = %s AND resq_eth IS NOT NULL ORDER BY ts DESC LIMIT 1", (mint,)).fetchone()
        else:
            r = conn.execute("SELECT resq_sol AS q FROM pump_minutes WHERE mint = %s AND resq_sol IS NOT NULL ORDER BY ts DESC LIMIT 1", (mint,)).fetchone()
        return float(r["q"]) if r else None

    def _mcap(self, mint: str, price: float | None) -> float | None:
        """Market cap in SOL (price × supply), which sets the pool fee tier; None without a price."""
        s = self.states.get(mint)
        return float(price) * (s.meta.supply if s is not None else PUMP_SUPPLY) if price else None

    # ---- one minute ----
    def _aggregate(self, conn, m0: datetime, m1: datetime) -> dict[str, dict]:
        """PumpAPI minutes: every SOL-quoted PumpSwap pump.fun token, the same fields as the archive. Robinhood Chain: the
        rh_minutes of graduated Pons pools in ETH (rh/minutes.py), under the same keys."""
        if self.rh:
            rows = conn.execute("SELECT mint, pool_id, open, high, low, close, buy_eth, sell_eth, n_buys, n_sells, n_traders, resq_eth, fee_rate, "
                                "n_buyers, wash_eth, wash_buy_eth, top_sell_eth, insider_sell_eth, skill_buy FROM rh_minutes WHERE ts = %s", (m0,)).fetchall()
            return {r["mint"]: {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "buy": r["buy_eth"] or 0.0, "sell": r["sell_eth"] or 0.0,
                                "nb": r["n_buys"] or 0, "ns": r["n_sells"] or 0, "n_traders": int(r["n_traders"] or 0), "resq": r["resq_eth"],
                                "pool": r["pool_id"], "program_label": self.spec.program_label, "fee_rate": r["fee_rate"],
                                "n_buyers": r["n_buyers"], "wash_sol": r["wash_eth"], "wash_buy_sol": r["wash_buy_eth"], "top_sell_sol": r["top_sell_eth"],
                                "insider_sell_sol": r["insider_sell_eth"], "skill_buy": r["skill_buy"]} for r in rows if r["close"]}
        rows = conn.execute("SELECT mint, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol, fee_rate, "
                            "n_buyers, wash_sol, wash_buy_sol, top_sell_sol, insider_sell_sol, skill_buy FROM pump_minutes WHERE ts = %s", (m0,)).fetchall()
        return {r["mint"]: {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "buy": r["buy_sol"] or 0.0, "sell": r["sell_sol"] or 0.0,
                            "nb": r["n_buys"] or 0, "ns": r["n_sells"] or 0, "n_traders": int(r["n_traders"] or 0), "resq": r["resq_sol"],
                            "pool": r["pool_id"], "program_label": "Pump.fun Amm", "fee_rate": r["fee_rate"],
                            "n_buyers": r["n_buyers"], "wash_sol": r["wash_sol"], "wash_buy_sol": r["wash_buy_sol"], "top_sell_sol": r["top_sell_sol"],
                            "insider_sell_sol": r["insider_sell_sol"], "skill_buy": r["skill_buy"]} for r in rows if r["close"]}

    def _features(self, conn, mint: str, a: dict, t_end: float) -> tuple[np.ndarray, dict]:
        """The full feature vector (``X_COLS``) of a mint for the minute ending ``t_end``, and what trading needs."""
        s = self._state(conn, mint, a["pool"], a["program_label"])
        st, price, resq = s.st, float(a["close"]), a["resq"]
        if self.rh:
            s.meta.pool_fee = a.get("fee_rate")          # the Pons hook fee this minute charged (the cost model's fee)
        if (s.prev_close and not (1 / 50 <= price / s.prev_close <= 50)) or (resq is not None and resq > 1e5 * self.spec.k()):
            s.broken = True                          # train/decisions.py: a scale break makes the mint ineligible from that minute on
        s.prev_close = price
        if a["buy"] > 0:
            st.append(t_end - 2e-3, price, a["buy"], True, None, resq)
        if a["sell"] > 0 or a["buy"] <= 0:
            st.append(t_end - 1e-3, price, a["sell"], False, None, resq)
        s.hist.append((t_end, float(a["nb"] + a["ns"]), float(a["n_traders"])))
        s.flow.append(t_end, a["buy"], a["sell"], a.get("n_buyers"), a.get("wash_sol"), a.get("wash_buy_sol"), a.get("top_sell_sol"), a.get("insider_sell_sol"), a.get("skill_buy"))
        f, _mask = st.features(t_end, s.meta)
        ht = np.array([h[0] for h in s.hist]); hn = np.array([h[1] for h in s.hist]); htr = np.array([h[2] for h in s.hist])
        for k, w in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
            f[FIDX[f"logn_{k}"]] = math.log1p(float(hn[ht > t_end - w].sum()))
        for name in MASKED:
            f[FIDX[name]] = 0.0
        if s.graduated_at is None:
            f[FIDX["log_age_h"]] = 0.0
        hod = ((t_end - 60.0) % 86400) / 3600.0                     # minute start, as in train/decisions.py
        meta = s.meta_row or {}
        extra = {"traders_15m": float(htr[ht > t_end - 900].sum()), "traders_1h": float(htr[ht > t_end - 3600].sum()), "n_trades_1m": float(a["nb"] + a["ns"]),
                 "hod_s": math.sin(2 * math.pi * hod / 24), "hod_c": math.cos(2 * math.pi * hod / 24), "age_known": 1.0 if s.graduated_at is not None else 0.0,
                 "meta_known": 1.0 if meta.get("ttg_min") is not None else 0.0, **s.flow.features(t_end), "mkt_vol_1h": self.market.value(t_end)}
        by_name = {**{n: float(f[i]) for i, n in enumerate(FEATURES)}, **extra,
                   **{c: float(meta.get(c)) if meta.get(c) is not None else 0.0 for c in META_COLS}}
        if self.rh:                                      # the chain inputs (train/decisions.CHAIN_COLS)
            qc = meta.get("quote_class")
            by_name.update(chain_rh=1.0, qc_stable=float(qc == "stable"), qc_btc=float(qc == "btc"), qc_stock=float(qc == "stock"))
        x = np.nan_to_num(np.asarray([by_name.get(c, 0.0) for c in X_COLS], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        info = {"price": price, "open": float(a["open"]) if a.get("open") else price, "resq": resq, "logvol_15m": f[FIDX["logvol_15m"]],
                # "ec" is the training label's own cost model (market/exit_cost.cost_at_size), at the size the book trades
                "ec": float(cost_at_size(f[FIDX["exit_cost_0p1"]], resq, market=self.spec if self.rh else None)),
                "age_h": (t_end - s.graduated_at) / 3600 if s.graduated_at else None,
                "mcap": price * s.meta.supply, "fee_rate": a.get("fee_rate"), "decimals": s.decimals, "pool": s.pool, "program_label": s.program_label, "broken": s.broken,
                "skill_missing": a.get("skill_buy") is None, "curve_known": bool(meta.get("curve_known")), "chain": self.spec.chain,
                "quote_class": meta.get("quote_class")}
        return x, info

    def eligible(self, info: dict) -> bool:
        if self.rh:                                      # the same gates in the chain's unit (x K), as training (train/decisions)
            return not info["broken"] and info["resq"] is not None and info["resq"] >= self.spec.min_resq() and info["logvol_15m"] >= math.log1p(self.spec.min_vol_15m())
        return not info["broken"] and info["resq"] is not None and info["resq"] >= MIN_RESQ_SOL and info["logvol_15m"] >= math.log1p(MIN_VOL_15M_SOL)

    def _stream_through(self) -> float | None:
        with transaction() as conn:
            r = conn.execute("SELECT value->>'flushed_through' AS ft FROM ui_settings WHERE key = %s", (self.spec.stream_status_key,)).fetchone()
        return datetime.fromisoformat(r["ft"]).timestamp() if r and r["ft"] else None

    def ready_through(self, m1_epoch: float) -> float:
        """End of the newest minute (at most ``m1``) the stream has written; A minute is
        never read before it is written, so a lagging stream delays minutes instead of feeding them empty or partial."""
        ft = self._stream_through()
        return min(m1_epoch, ft + 60.0) if ft is not None else float("-inf")

    def stream_fresh(self, m0_epoch: float) -> bool:
        ft = self._stream_through()
        return ft is not None and m0_epoch - ft <= STREAM_STALE_S

    def _minute_rows(self, conn, agg: dict, m1_epoch: float) -> tuple[np.ndarray, list, list, dict]:
        xs, infos, mints, bars = [], [], [], {}
        self.market.add(m1_epoch, sum(float(a["buy"]) + float(a["sell"]) for a in agg.values()))
        for mint, a in agg.items():
            x, info = self._features(conn, mint, a, m1_epoch)
            bars[mint] = (m1_epoch - 60.0, info["open"], info["price"], info["ec"])
            if in_universe(x[None], X_COLS)[0] and self.eligible(info):        # the models' universe (train/selector.in_universe)
                xs.append(x); infos.append(info); mints.append(mint)
        return (np.stack(xs) if xs else np.zeros((0, len(X_COLS)), np.float32)), infos, mints, bars

    def warm_up(self, m1_epoch: float, minutes: int = WARMUP_MIN) -> int:
        """Feed the last ``minutes`` closed minutes through the feature states (and the books' bars) without trading."""
        n = 0
        with transaction() as conn:
            for k in range(minutes, 0, -1):
                t1 = m1_epoch - 60 * k
                agg = self._aggregate(conn, datetime.fromtimestamp(t1 - 60, timezone.utc), datetime.fromtimestamp(t1, timezone.utc))
                bars = {}
                self.market.add(t1, sum(float(a["buy"]) + float(a["sell"]) for a in agg.values()))
                for mint, a in agg.items():
                    _x, info = self._features(conn, mint, a, t1); n += 1
                    bars[mint] = (t1 - 60.0, info["open"], info["price"], info["ec"])
                for b in self.my_books():
                    b.on_bars(t1 - 60.0, bars)
        log.info("warm-up: %d token-minutes over the last %d minutes (%d tokens)", n, minutes, len(self.states))
        return n

    def _sweep(self, now_s: float, held: set[str]) -> None:
        if now_s - self.last_sweep < 3600:
            return
        self.last_sweep = now_s
        idle = [m for m, s in self.states.items() if m not in held and (s.st.last_ts is None or s.st.last_ts < now_s - IDLE_EVICT_S)]
        for m in idle:
            self.states.pop(m, None); self.meta_cache.pop(m, None)
        if idle:
            log.info("evicted %d mint states idle for more than %d h", len(idle), IDLE_EVICT_S // 3600)

    def run_minute(self, m1_epoch: float, trade: bool = True) -> dict:
        m1 = datetime.fromtimestamp(m1_epoch, timezone.utc); m0 = datetime.fromtimestamp(m1_epoch - 60, timezone.utc)
        out: dict = {"minute": m1.isoformat()}
        with transaction() as conn:
            agg = self._aggregate(conn, m0, m1)
            X, infos, mints, bars = self._minute_rows(conn, agg, m1_epoch)
            ctx = Minute(conn=conn, m0=m0, m1=m1, m1_epoch=m1_epoch, agg=agg, X=X, infos=infos, mints=mints, bars=bars,
                         prices={m: float(a["close"]) for m, a in agg.items()}, resqs={m: a["resq"] for m, a in agg.items()},
                         fees={m: a.get("fee_rate") for m, a in agg.items()}, fresh=trade and self.stream_fresh(m1_epoch - 60), trade=trade, engine=self,
                         market=self.spec)
            out.update(mints_traded=len(agg), eligible=len(mints)); held: set[str] = set()
            for b in list(self.my_books()):
                try:
                    with conn.transaction():                            # a savepoint per book
                        b.on_bars(m1_epoch - 60.0, bars)
                        out[b.name] = b.on_minute(ctx)
                        held |= set(b.open_mints(conn))
                except Exception:
                    log.exception("%s: minute %s failed", b.name, m1.isoformat())
                    record_event("error", b.name, f"minute failed: {m1.isoformat()}")
            self._sweep(m1_epoch, held)
        for b in [b for b in self.my_books() if b.done]:
            b.finish(); self.books.remove(b)
        return out



def release_stamp():
    """``ui_settings['release_applied_at']``: changes whenever ops/release.apply_release installs new models."""
    try:
        with transaction() as conn:
            r = conn.execute("SELECT value FROM ui_settings WHERE key = 'release_applied_at'").fetchone()
        return None if r is None else str(r["value"])
    except Exception:
        return None

def _status(key: str, value: dict) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (key, json.dumps({**value, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str)))
    except Exception:
        log.debug("status write failed", exc_info=True)


def admit_books(engine: MinuteEngine, live: bool) -> list:
    """Books that qualify now and are not running yet (the fly first: its presence decides whether the selector trades)."""
    from . import fly_session, handover
    from .selector_session import NoModel, SelectorBook
    new = []
    for m in markets.enabled():                         # Solana first: another chain's fly shares the Solana fly's brain
        fly_name = "fly" if m.chain == "sol" else f"fly_{m.chain}"; sel_name = "selector" if m.chain == "sol" else f"selector_{m.chain}"
        if not engine.has_book(fly_name) and not any(b.name == fly_name for b in new):
            brain = next((b for b in engine.books + new if b.name == "fly"), None) if m.chain != "sol" else None
            fb, why = fly_session.try_start(live, chain=m.chain, brain=brain)
            if fb is not None:
                new.append(fb)
            else:
                _status(m.fly_status_key, {"stage": "not trading", "detail": why, "chain": m.chain})
        fly_on = engine.has_book(fly_name) or any(b.name == fly_name for b in new)
        with transaction() as conn:
            handed = handover.state(conn, m.chain) is not None
        if not engine.has_book(sel_name) and not (handed and fly_on):
            try:
                new.append(SelectorBook(m.chain))
            except NoModel as e:
                _status(m.selector_status_key, {"stage": "waiting for a model", "detail": str(e), "chain": m.chain})
    return new


def main(stop_event: threading.Event | None = None, live: bool = False) -> None:
    """One engine per enabled chain (markets.enabled()), all sharing one list of books: each engine feeds its minutes, as
    its own stream writes them, to its chain's books only — a lagging chain never holds another chain's minutes back."""
    engines = [MinuteEngine(live=live, market=m) for m in markets.enabled()]
    lead = engines[0]; books: list = []
    for e in engines:
        e.books = books                                  # the same list object: admission and removal are seen by every engine
    while not (stop_event is not None and stop_event.is_set()):
        books += admit_books(lead, live)
        if books:
            break
        log.info("trading engine waiting: no book qualifies yet")
        (stop_event or threading.Event()).wait(MODEL_CHECK_S)
    else:
        return
    m_start = math.floor(time.time() / 60) * 60
    for e in engines:
        e.warm_up(m_start); e.last_minute = m_start - 60
    last_check = time.time(); n_min = 0
    last_release, last_stamp_check = release_stamp(), 0.0
    try:
        while not (stop_event is not None and stop_event.is_set()):
            now = time.time(); m1 = math.floor(now / 60) * 60
            if now - last_stamp_check >= 15.0:
                last_stamp_check, stamp = now, release_stamp()
                if stamp != last_release:              # a release was applied (ops/release.py): reload at once
                    last_release, last_check = stamp, 0.0
            if now - last_check >= MODEL_CHECK_S:
                last_check = now
                for b in books:
                    try:
                        b.maybe_reload()
                    except Exception:
                        log.exception("%s: model reload check failed", b.name)
                try:
                    books += admit_books(lead, live)
                except Exception:
                    log.exception("book admission failed")
            for engine in engines:
                if not engine.my_books() or m1 <= engine.last_minute or now - m1 < 4.0:
                    continue
                if (upto := engine.ready_through(m1)) <= engine.last_minute:        # only minutes this chain's stream has written
                    continue
                for minute in range(int(engine.last_minute) + 60, int(upto) + 1, 60):
                    try:
                        st = engine.run_minute(float(minute), trade=engine.tradeable(minute, m1))   # older minutes update features (and the fly's learning) only
                        n_min += 1
                        if engine.tradeable(minute, m1) and n_min % 10 == 0:
                            log.info("%s minute %s: traded %d eligible %d | %s", engine.spec.chain, st["minute"], st.get("mints_traded", 0), st.get("eligible", 0),
                                     " | ".join(f"{b.name}: picks {(st.get(b.name) or {}).get('picks')} open {(st.get(b.name) or {}).get('open')} wealth {(st.get(b.name) or {}).get('wealth')}"
                                                for b in engine.my_books()))
                    except Exception:
                        log.exception("%s minute %s failed", engine.spec.chain, minute); record_event("error", "engine", f"{engine.spec.chain} minute failed: {minute}")
                engine.last_minute = upto
            time.sleep(0.5)
    finally:
        for b in books:
            try:
                b.finish()
            except Exception:
                log.exception("%s: finish failed", b.name)