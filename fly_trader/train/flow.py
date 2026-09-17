"""Per-minute wallet flow: one definition for the archive (``train/mature.py``) and the live stream
(``ingest/pumpstream.py``), so the selector's new inputs are computed identically in training and live.

Per (mint, minute), over the minute's dominant-pool legs with a known trader (both paths drop the others): each
trader's bought and sold SOL → ``n_buyers`` (traders who bought), ``wash_sol`` (bought + sold SOL of traders who did
both in the minute: round trips, the wash-trading signature — Szwajcok et al. 2026), ``wash_buy_sol`` (their buy side),
``top_sell_sol`` (the largest single trader's selling), ``insider_sell_sol`` (selling by the token's creator, bundle
and early wallets: ``token_insiders``), ``skill_buy`` (bought SOL by wallet-skill decile 0–9 of the day's table,
index 10 = wallet without a score; None when the day has no table). The archive computes them in SQL
(``mature.aggregate_day``), the stream with ``minute_wallet_cols``; ``tests/test_flow_parity.py`` holds them equal.

``FlowWindow`` turns a mint's minutes into the derived model inputs (``FLOW_COLS`` + ``SKILL_COLS``), fed the same way
by ``mature.build_day`` and the live engine (``agent/minute_engine.py``); ``mkt_vol_1h`` is the market's total SOL volume
over the last hour (all mints), computed from the same minute rows in both paths.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

N_SKILL = 11                     # deciles 0..9 (9 = most skilled) + 10 = no score
UNKNOWN = 10
WINDOW_S = 3600.0
WINDOW_15M_S = 900               # the skill inputs' window (FlowWindow's r15)
FLOW_COLS = ["top_sell_share_1m", "insider_sell_share_1m", "log_insider_sell_15m", "buyers_ret", "buyers_slope",
             "org_imb_5m", "org_imb_15m", "wash_share_15m", "log_org_vol_15m"]
SKILL_COLS = [f"skill_top{q}_15m" for q in range(1, 10)] + ["skill_known_15m"]
MARKET_COLS = ["mkt_vol_1h"]
RAW_COLS = ["n_buyers", "wash_sol", "wash_buy_sol", "top_sell_sol", "insider_sell_sol", "skill_buy"]


def minute_wallet_cols(traders: dict, insiders=frozenset(), skill: dict | None = None) -> dict:
    """The raw wallet columns of one minute from ``traders``: wallet → [bought SOL, sold SOL]."""
    n_buyers = 0; wash = wash_b = top_sell = ins_sell = 0.0
    sk = None if skill is None else [0.0] * N_SKILL
    for w, (b, s) in traders.items():
        if b > 0:
            n_buyers += 1
        if b > 0 and s > 0:
            wash += b + s; wash_b += b
        top_sell = max(top_sell, s)
        if w in insiders:
            ins_sell += s
        if sk is not None:
            sk[skill.get(w, UNKNOWN)] += b
    return {"n_buyers": n_buyers, "wash_sol": wash, "wash_buy_sol": wash_b, "top_sell_sol": top_sell, "insider_sell_sol": ins_sell, "skill_buy": sk}


def _z(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


class FlowWindow:
    """A mint's last hour of minutes (only minutes it traded) → the derived flow inputs at a minute's end."""
    __slots__ = ("q",)

    def __init__(self):
        self.q: deque = deque()

    def append(self, t_end: float, buy: float, sell: float, n_buyers, wash_sol, wash_buy_sol, top_sell_sol, insider_sell_sol, skill_buy) -> None:
        sk = None if skill_buy is None else [_z(v) for v in skill_buy]
        self.q.append((t_end, _z(buy), _z(sell), _z(n_buyers), _z(wash_sol), _z(wash_buy_sol), _z(top_sell_sol), _z(insider_sell_sol), sk))
        while self.q and self.q[0][0] <= t_end - WINDOW_S:
            self.q.popleft()

    def features(self, t_end: float) -> dict:
        rows = [r for r in self.q if r[0] <= t_end]
        cur = rows[-1] if rows and rows[-1][0] == t_end else None

        def win(lo, hi=t_end):
            return [r for r in rows if lo < r[0] <= hi]

        def s(rs, k):
            return sum(r[k] for r in rs)

        r15, r5 = win(t_end - 900), win(t_end - 300)
        buy15, sell15, wash15 = s(r15, 1), s(r15, 2), s(r15, 4)
        out = {"top_sell_share_1m": cur[6] / cur[2] if cur and cur[2] > 0 else 0.0,
               "insider_sell_share_1m": cur[7] / cur[2] if cur and cur[2] > 0 else 0.0,
               "log_insider_sell_15m": math.log1p(s(r15, 7)),
               "buyers_ret": math.log1p(cur[3] if cur else 0.0) - math.log1p(s(win(t_end - 360, t_end - 60), 3) / 5.0),
               "buyers_slope": math.log1p(s(r5, 3)) - math.log1p(s(win(t_end - 600, t_end - 300), 3)),
               "wash_share_15m": wash15 / (buy15 + sell15) if buy15 + sell15 > 0 else 0.0,
               "log_org_vol_15m": math.log1p(max(buy15 + sell15 - wash15, 0.0))}
        for name, rs in (("org_imb_5m", r5), ("org_imb_15m", r15)):
            ob = s(rs, 1) - s(rs, 5); os_ = s(rs, 2) - (s(rs, 4) - s(rs, 5))
            out[name] = (ob - os_) / (ob + os_) if ob + os_ > 0 else 0.0
        sk = [0.0] * N_SKILL
        for r in r15:
            if r[8] is not None:
                for k in range(N_SKILL):
                    sk[k] += r[8][k]
        total = sum(sk); known = total - sk[UNKNOWN]
        for q in range(1, 10):
            out[f"skill_top{q}_15m"] = sum(sk[q:UNKNOWN]) / total if total > 0 else 0.0
        out["skill_known_15m"] = known / total if total > 0 else 0.0
        return out


def skill_features(mint: np.ndarray, ts_s: np.ndarray, skill_buy: np.ndarray) -> np.ndarray:
    """``FlowWindow``'s skill inputs (``SKILL_COLS``, in order) for many minute rows at once: row i's window is its mint's
    rows starting in (t_i − 900 s, t_i]; ``skill_buy`` [n, N_SKILL] with zeros where the minute had no table (FlowWindow
    skips those minutes)."""
    n = len(ts_s); out = np.zeros((n, len(SKILL_COLS)), np.float32)
    if not n:
        return out
    _, code = np.unique(np.asarray(mint).astype(str), return_inverse=True)
    key = code.astype(np.int64) * 10**10 + np.asarray(ts_s, dtype=np.int64)
    order = np.argsort(key, kind="stable"); ks = key[order]; sb = np.asarray(skill_buy, dtype=np.float64)[order]
    cs = np.vstack([np.zeros((1, N_SKILL)), np.cumsum(sb, axis=0)]); cn = np.r_[0, np.cumsum(sb.sum(1) > 0)]
    lo = np.searchsorted(ks, ks - WINDOW_15M_S, side="right"); hi = np.arange(1, n + 1)
    w = cs[hi] - cs[lo]; has = (cn[hi] - cn[lo]) > 0                    # a window without a table minute stays exactly 0
    total = w.sum(1); ok = has & (total > 0); res = np.zeros((n, len(SKILL_COLS)))
    for j, q in enumerate(range(1, 10)):
        res[ok, j] = w[ok, q:UNKNOWN].sum(1) / total[ok]
    res[ok, 9] = (total[ok] - w[ok, UNKNOWN]) / total[ok]
    out[order] = np.clip(res, 0.0, 1.0)
    return out


class MarketWindow:
    """The market's total SOL volume per minute (every mint's buy + sell) over the last hour → ``mkt_vol_1h``."""
    __slots__ = ("q",)

    def __init__(self):
        self.q: deque = deque()

    def add(self, t_end: float, vol: float) -> None:
        if self.q and self.q[-1][0] == t_end:
            self.q[-1] = (t_end, self.q[-1][1] + vol)
        else:
            self.q.append((t_end, vol))
        while self.q and self.q[0][0] <= t_end - WINDOW_S:
            self.q.popleft()

    def value(self, t_end: float) -> float:
        return math.log1p(sum(v for t, v in self.q if t_end - WINDOW_S < t <= t_end))
