"""One feature engine for the Kalshi corpus and the live minute engine: per (market, minute, side) the inputs the
literature says carry information about a binary contract's settlement (plan of 2026-09-22, "Literature basis").

A market's state is a rolling window of minute bars (``Bar``: the YES top of book, sizes, contracts traded, open interest
and the taker flow by side) covering ``WINDOW_S`` (7 days). ``features(t_end, meta, side, event)`` returns the ``K_COLS``
vector for one side: every price is the side's own (a NO contract's ask is 100 minus the YES bid), so both sides share
one definition and one model. ``EventState`` holds the latest YES quote of every sibling market of an event, for the
sibling-implied probability, the count of near-certain legs and the strike-ladder residual. Category, recurrence, time
to close and the fee multiplier come from ``MarketMeta``. Fees are the vendored better_bot model
(``effective_price_cents``): warm its per-series cache from ``kalshi_series`` with ``warm_fees`` before building.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

from .vendor import kalshi_client as kc

KALSHI_FEATURE_VERSION = 1
WINDOW_S = 7 * 86400.0
WINDOWS = {"5m": 300.0, "15m": 900.0, "1h": 3600.0, "6h": 21600.0, "24h": 86400.0, "7d": WINDOW_S}
CATEGORIES = ("politics", "sports", "crypto", "economics", "weather", "entertainment", "science", "companies", "world", "health", "other")
SIDES = ("yes", "no")

K_COLS: list[str] = [
    # the side's quotes (cents) and the fee-inclusive price it would pay
    "side_yes", "yes_bid", "yes_ask", "mid", "spread", "last", "side_ask", "side_bid", "side_mid", "eff_price", "dist_mid", "fee_mult", "maker_fee",
    # the side's price history (cents) from the minute bars
    "ret_5m", "ret_15m", "ret_1h", "ret_6h", "ret_24h", "ret_7d", "rvol_1h", "rvol_24h", "dd_24h", "range_24h", "log_min_since_trade", "log_history_min",
    # taker flow on the side (contracts) and its imbalance
    "log_taker_side_5m", "log_taker_side_1h", "log_taker_side_24h", "taker_imb_5m", "taker_imb_1h", "taker_imb_24h", "log_trades_1h", "log_trades_24h",
    "log_max_trade_24h", "log_mean_trade_24h", "block_share_24h",
    # size and depth (top-of-book sizes are not in the archive's candles, so they are not inputs)
    "log_oi", "log_vol_24h", "log_vol_7d", "oi_change_24h",
    # time
    "log_h_to_close", "log_h_since_open", "life_frac", "last_day", "last_hour", "hod_s", "hod_c", "dow",
    # structure
    *[f"cat_{c}" for c in CATEGORIES], "is_recurring", "log_n_siblings", "mutually_exclusive", "sib_sum_bid", "sib_sum_ask", "implied_gap",
    "ladder_rank", "ladder_resid", "n_sib_ge90", "has_strike",
]
KIDX = {n: i for i, n in enumerate(K_COLS)}


class Bar(NamedTuple):
    t_end: float            # minute end (epoch s)
    yes_bid: float          # cents; NaN when no bid
    yes_ask: float
    last: float             # last trade price (cents) or NaN
    bid_size: float         # contracts at the YES bid (0 when unknown)
    ask_size: float
    volume: float           # contracts traded in the minute
    oi: float               # open interest at the minute's end
    taker_yes: float        # contracts bought by takers on the YES side in the minute
    taker_no: float
    n_trades: float
    max_trade: float
    block: float            # contracts in block trades


@dataclass
class MarketMeta:
    ticker: str
    event_ticker: str | None = None
    series_ticker: str | None = None
    category: str = "other"
    is_recurring: float = 0.0
    open_ts: float | None = None
    close_ts: float | None = None
    mutually_exclusive: float = 0.0
    fee_multiplier: float = 1.0
    maker_fee: float = 0.0
    strike: float | None = None


def warm_fees(conn) -> int:
    """The vendored fee model's per-series cache from ``kalshi_series`` (no network)."""
    n = 0
    for r in conn.execute("SELECT ticker, fee_type, fee_multiplier FROM kalshi_series").fetchall():
        kc._series_fee_cache[r["ticker"]] = {"fee_type": r["fee_type"], "fee_multiplier": float(r["fee_multiplier"] if r["fee_multiplier"] is not None else kc.DEFAULT_FEE_MULTIPLIER)}
        n += 1
    return n


def side_prices(bar: Bar, side: str) -> tuple[float, float, float]:
    """(ask, bid, mid) of ``side`` in cents; NaN where the book is one-sided."""
    if side == "yes":
        ask, bid = bar.yes_ask, bar.yes_bid
    else:
        ask, bid = 100.0 - bar.yes_bid, 100.0 - bar.yes_ask
    mid = (ask + bid) / 2.0 if math.isfinite(ask) and math.isfinite(bid) else (ask if math.isfinite(ask) else bid)
    return ask, bid, mid


def _mid(bar: Bar) -> float:
    if math.isfinite(bar.yes_bid) and math.isfinite(bar.yes_ask):
        return (bar.yes_bid + bar.yes_ask) / 2.0
    return bar.yes_ask if math.isfinite(bar.yes_ask) else bar.yes_bid


class EventState:
    """The latest YES quote and strike of every market of one event (the siblings)."""
    __slots__ = ("quotes", "strikes")

    def __init__(self):
        self.quotes: dict[str, tuple[float, float]] = {}
        self.strikes: dict[str, float] = {}

    def update(self, ticker: str, bar: Bar, strike: float | None = None) -> None:
        self.quotes[ticker] = (bar.yes_bid, bar.yes_ask)
        if strike is not None:
            self.strikes[ticker] = float(strike)

    def features(self, ticker: str) -> dict:
        others = [(b, a) for t, (b, a) in self.quotes.items() if t != ticker]
        bids = [b for b, _ in others if math.isfinite(b)]; asks = [a for _, a in others if math.isfinite(a)]
        out = {"n_siblings": float(len(others)), "sib_sum_bid": float(sum(bids)), "sib_sum_ask": float(sum(asks)),
               "n_sib_ge90": float(sum(1 for b, a in others if math.isfinite(b) and b >= 90.0)), "ladder_rank": 0.0, "ladder_resid": 0.0}
        st = self.strikes.get(ticker)
        if st is not None and len(self.strikes) > 1:
            ranked = sorted(self.strikes.items(), key=lambda kv: kv[1]); ks = [t for t, _ in ranked]
            i = ks.index(ticker); out["ladder_rank"] = i / max(len(ks) - 1, 1)
            # a higher strike must not price above a lower one ("above X" ladders): the amount by which this leg breaks that
            own = self.quotes.get(ticker); own_mid = (own[0] + own[1]) / 2.0 if own and math.isfinite(own[0]) and math.isfinite(own[1]) else float("nan")
            resid = 0.0
            if math.isfinite(own_mid):
                if i > 0:
                    lo = self.quotes.get(ks[i - 1])
                    if lo and math.isfinite(lo[0]) and math.isfinite(lo[1]):
                        resid = max(resid, own_mid - (lo[0] + lo[1]) / 2.0)
                if i < len(ks) - 1:
                    hi = self.quotes.get(ks[i + 1])
                    if hi and math.isfinite(hi[0]) and math.isfinite(hi[1]):
                        resid = max(resid, (hi[0] + hi[1]) / 2.0 - own_mid)
            out["ladder_resid"] = float(resid)
        return out


class MarketState:
    """A market's rolling minute bars; O(log n) window statistics by bisect over the bar ends (as market/features.py)."""
    __slots__ = ("ticker", "bars", "ts", "mids", "last_trade_ts", "first_ts")

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.bars: deque = deque(); self.ts: deque = deque(); self.mids: deque = deque()
        self.last_trade_ts: float | None = None; self.first_ts: float | None = None

    def append(self, bar: Bar) -> None:
        if self.ts and bar.t_end <= self.ts[-1]:
            return                                       # never rewinds: a late bar for a minute already seen is dropped
        self.bars.append(bar); self.ts.append(bar.t_end); self.mids.append(_mid(bar))
        if bar.n_trades > 0 or bar.volume > 0:
            self.last_trade_ts = bar.t_end
        if self.first_ts is None:
            self.first_ts = bar.t_end
        while self.ts and self.ts[0] <= bar.t_end - WINDOW_S - 60.0:
            self.bars.popleft(); self.ts.popleft(); self.mids.popleft()

    @property
    def last(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def _idx(self, t0: float) -> int:
        return bisect_left(self.ts, t0)

    def _mid_at_or_before(self, t: float) -> float:
        i = bisect_right(self.ts, t)
        if i == 0:
            return float("nan")
        return self.mids[i - 1]

    def features(self, t_end: float, meta: MarketMeta, side: str, event: EventState | None = None) -> list[float]:
        f = [0.0] * len(K_COLS)
        bar = self.last
        if bar is None:
            return f
        yes = side == "yes"; sgn = 1.0 if yes else -1.0
        ask, bid, mid_s = side_prices(bar, side)
        mid = _mid(bar)
        f[KIDX["side_yes"]] = 1.0 if yes else 0.0
        f[KIDX["yes_bid"]] = bar.yes_bid if math.isfinite(bar.yes_bid) else 0.0
        f[KIDX["yes_ask"]] = bar.yes_ask if math.isfinite(bar.yes_ask) else 100.0
        f[KIDX["mid"]] = mid if math.isfinite(mid) else 50.0
        f[KIDX["spread"]] = (bar.yes_ask - bar.yes_bid) if math.isfinite(bar.yes_ask) and math.isfinite(bar.yes_bid) else 99.0
        f[KIDX["last"]] = (bar.last if yes else 100.0 - bar.last) if math.isfinite(bar.last) else f[KIDX["mid"]]
        f[KIDX["side_ask"]] = ask if math.isfinite(ask) else 100.0
        f[KIDX["side_bid"]] = bid if math.isfinite(bid) else 0.0
        f[KIDX["side_mid"]] = mid_s if math.isfinite(mid_s) else 50.0
        f[KIDX["eff_price"]] = kc.effective_price_cents(min(max(f[KIDX["side_ask"]], 1.0), 99.0), meta.ticker) if math.isfinite(ask) else 100.0
        f[KIDX["dist_mid"]] = abs(f[KIDX["mid"]] - 50.0)
        f[KIDX["fee_mult"]] = meta.fee_multiplier; f[KIDX["maker_fee"]] = meta.maker_fee
        # history of the side's mid (cents); returns are differences, not ratios: a contract's scale is fixed
        m_now = mid if math.isfinite(mid) else 50.0
        for k in ("5m", "15m", "1h", "6h", "24h", "7d"):
            m_then = self._mid_at_or_before(t_end - WINDOWS[k])
            if not math.isfinite(m_then):
                m_then = self.mids[0] if math.isfinite(self.mids[0]) else m_now
            f[KIDX[f"ret_{k}"]] = sgn * (m_now - m_then)
        for k, w in (("1h", 3600.0), ("24h", 86400.0)):
            i = self._idx(t_end - w); seq = [m for m in list(self.mids)[max(i - 1, 0):] if math.isfinite(m)]
            if len(seq) > 1:
                d = [b - a for a, b in zip(seq[:-1], seq[1:])]; mu = sum(d) / len(d)
                f[KIDX[f"rvol_{k}"]] = math.sqrt(sum((x - mu) ** 2 for x in d) / len(d))
        i24 = self._idx(t_end - 86400.0); win = [m for m in list(self.mids)[i24:] if math.isfinite(m)]
        if win:
            side_win = [sgn * m + (0.0 if yes else 100.0) for m in win]; cur = sgn * m_now + (0.0 if yes else 100.0)
            f[KIDX["dd_24h"]] = cur - max(side_win); f[KIDX["range_24h"]] = max(side_win) - min(side_win)
        f[KIDX["log_min_since_trade"]] = math.log1p(max(0.0, (t_end - self.last_trade_ts) / 60.0)) if self.last_trade_ts else math.log1p(WINDOW_S / 60.0)
        f[KIDX["log_history_min"]] = math.log1p(max(0.0, (t_end - self.first_ts) / 60.0)) if self.first_ts else 0.0
        # flow
        bars = list(self.bars)
        for k in ("5m", "1h", "24h"):
            i = self._idx(t_end - WINDOWS[k]); w = bars[i:]
            ty = sum(b.taker_yes for b in w); tn = sum(b.taker_no for b in w); own = ty if yes else tn
            f[KIDX[f"log_taker_side_{k}"]] = math.log1p(own)
            f[KIDX[f"taker_imb_{k}"]] = sgn * (ty - tn) / (ty + tn) if ty + tn > 0 else 0.0
            if k in ("1h", "24h"):
                f[KIDX[f"log_trades_{k}"]] = math.log1p(sum(b.n_trades for b in w))
            if k == "24h":
                vol = sum(b.volume for b in w); nt = sum(b.n_trades for b in w)
                f[KIDX["log_max_trade_24h"]] = math.log1p(max((b.max_trade for b in w), default=0.0))
                f[KIDX["log_mean_trade_24h"]] = math.log1p(vol / nt) if nt > 0 else 0.0
                f[KIDX["block_share_24h"]] = sum(b.block for b in w) / vol if vol > 0 else 0.0
                f[KIDX["log_vol_24h"]] = math.log1p(vol)
                oi_then = w[0].oi if w else bar.oi
                f[KIDX["oi_change_24h"]] = math.log1p(bar.oi) - math.log1p(oi_then)
        f[KIDX["log_vol_7d"]] = math.log1p(sum(b.volume for b in bars))
        f[KIDX["log_oi"]] = math.log1p(bar.oi)
        # time
        h_close = max(0.0, (meta.close_ts - t_end) / 3600.0) if meta.close_ts else 24.0 * 365
        h_open = max(0.0, (t_end - meta.open_ts) / 3600.0) if meta.open_ts else 0.0
        f[KIDX["log_h_to_close"]] = math.log1p(h_close); f[KIDX["log_h_since_open"]] = math.log1p(h_open)
        f[KIDX["life_frac"]] = h_open / (h_open + h_close) if h_open + h_close > 0 else 0.0
        f[KIDX["last_day"]] = 1.0 if h_close <= 24.0 else 0.0; f[KIDX["last_hour"]] = 1.0 if h_close <= 1.0 else 0.0
        hod = ((t_end - 60.0) % 86400) / 3600.0
        f[KIDX["hod_s"]] = math.sin(2 * math.pi * hod / 24); f[KIDX["hod_c"]] = math.cos(2 * math.pi * hod / 24)
        f[KIDX["dow"]] = float(int((t_end - 60.0) // 86400 + 3) % 7)          # 0 = Monday (epoch day 0 was a Thursday)
        # structure
        f[KIDX[f"cat_{meta.category if meta.category in CATEGORIES else 'other'}"]] = 1.0
        f[KIDX["is_recurring"]] = meta.is_recurring; f[KIDX["mutually_exclusive"]] = meta.mutually_exclusive
        f[KIDX["has_strike"]] = 1.0 if meta.strike is not None else 0.0
        if event is not None:
            e = event.features(meta.ticker)
            f[KIDX["log_n_siblings"]] = math.log1p(e["n_siblings"]); f[KIDX["sib_sum_bid"]] = e["sib_sum_bid"]; f[KIDX["sib_sum_ask"]] = e["sib_sum_ask"]
            f[KIDX["n_sib_ge90"]] = e["n_sib_ge90"]; f[KIDX["ladder_rank"]] = e["ladder_rank"]; f[KIDX["ladder_resid"]] = e["ladder_resid"]
            if meta.mutually_exclusive and e["n_siblings"] > 0 and math.isfinite(ask):
                implied_yes = 100.0 - e["sib_sum_bid"]                  # what the siblings' bids leave for this leg
                f[KIDX["implied_gap"]] = sgn * (f[KIDX["yes_ask"]] - implied_yes) if yes else -(f[KIDX["yes_bid"]] - implied_yes)
        return f
