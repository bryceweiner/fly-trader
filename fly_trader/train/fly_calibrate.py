"""The fly's buy line and sizing bands, recalibrated from its own scores and the realized outcomes — no training.

Every day (live: the first minute after 00:00 UTC; replay: every day boundary) on the scored minutes whose labels
resolved in the last ``WINDOW_DAYS`` days. Candidate lines: the selector's ``LINE_CANDIDATES`` plus the window's score
quantiles (a compressed score scale still gets a usable line); the line with the most total net profit (fixed size, one
position per token, the hold) over at least ``MIN_LINE_TRADES`` trades wins — the selector's rule — and a line that lost
money over the window never does; otherwise the previous line stays. Sizing: ``agent/sizing.build_table`` on the window's trades at that line (the previous table if too few).
The fly never trades the selector's line: its scores live on their own scale.

``rule`` picks the line (default ``total``; the memecoin fly passes ``RULE`` = ``config.FLY_CALIB_RULE``, the Kalshi
fly keeps the default):
- ``total``: the most total net profit over at least ``MIN_LINE_TRADES`` trades (the rule above). Volume wins: on fly
  #133's week it chose 414 trades at +2.8 % over 115 at +7.3 %, where its teacher took 13 at +8.5 %.
- ``selector``: the selector's own objective (train/strategies.better): admissible lines first (≥ ``WIN_MIN`` winners,
  profit factor ≥ ``PF_MIN``, ≥ ``MIN_LINE_TRADES`` trades), then those meeting the profit-factor bar, then total profit;
  a window with no line meeting at least the profit-factor bar keeps the previous line, as the selector would refuse.
- ``match``: the teacher's selectivity: the lowest line at which the fly takes no more trades over the window than its
  teacher would have (``target_trades``, one position per token over the hold); the ranking is the fly's own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agent import sizing
from .decisions import taken_idx
from .. import config
from .selector import LINE_CANDIDATES, MIN_EV, MIN_LINE_TRADES
from .strategies import PF_MIN, WIN_MIN

WINDOW_DAYS = 7
QUANTILES = (0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
RULE = config.FLY_CALIB_RULE
RULES = ("total", "selector", "match")


@dataclass
class Calibration:
    line: float
    sizing: list
    trades: int                 # trades at the chosen line in the window
    mean: float | None          # their mean net return
    total: float                # their summed net return (fixed size)
    changed: bool
    lines: list = field(default_factory=list)


def candidates(scores: np.ndarray) -> list[float]:
    s = np.asarray(scores, dtype=np.float64); s = s[np.isfinite(s)]
    q = np.quantile(s, QUANTILES).tolist() if len(s) else []
    return sorted({float(c) for c in (*LINE_CANDIDATES, *q)})


def _stats(r: np.ndarray) -> dict:
    g, l = float(r[r > 0].sum()), float(-r[r <= 0].sum())
    return {"trades": int(len(r)), "mean": float(r.mean()) if len(r) else None, "total": float(r.sum()),
            "win": float((r > 0).mean()) if len(r) else None, "pf": (g / l if l > 0 else (float("inf") if g > 0 else None))}


def match_line(ts, mint, horizon_s, scores, rows, target: int) -> float | None:
    """The lowest score line at which the taken trades over ``rows`` number at most ``target`` (bisection over the
    rows' own scores, highest first: the count only grows as the line falls, give or take a re-entry)."""
    s = np.sort(np.unique(scores[rows]))[::-1]
    if target <= 0 or not len(s):
        return float(s[0]) + 1e-9 if len(s) else None
    n = lambda k: len(taken_idx(ts, mint, horizon_s, rows[scores[rows] >= s[k]]))
    lo, hi = 0, len(s) - 1                       # n(lo) ≤ target is the invariant; find the largest such k
    if n(hi) <= target:
        return float(s[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n(mid) <= target:
            lo = mid
        else:
            hi = mid
    return float(s[lo])


def calibrate(ts: np.ndarray, mint: np.ndarray, horizon_s: float, scores: np.ndarray, rets: np.ndarray, prev_line: float | None = None,
              prev_sizing: list | None = None, min_trades: int = MIN_LINE_TRADES, rule: str | None = None,
              target_trades: int | None = None) -> Calibration:
    """Line and sizing from one window of (ts, mint, score, realized net return) rows. ``rule``: see the module
    docstring (default ``total``); ``match`` needs ``target_trades`` (the teacher's trades over the same window)."""
    rule = rule or "total"
    if rule not in RULES:
        raise ValueError(f"unknown calibration rule {rule!r}: one of {RULES}")
    scores = np.asarray(scores, dtype=np.float64); rets = np.asarray(rets, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(scores) & np.isfinite(rets))
    table = []
    for c in candidates(scores[rows]):
        tr = taken_idx(ts, mint, horizon_s, rows[scores[rows] >= c])
        table.append({"line": c, **_stats(rets[tr])})
    fallback = prev_line if prev_line is not None else MIN_EV
    if rule == "match":
        if target_trades is None:
            raise ValueError("the match rule needs target_trades (the teacher's trades over the window)")
        m = match_line(ts, mint, horizon_s, scores, rows, int(target_trades)) if len(rows) else None
        line = m if m is not None else fallback
    elif rule == "selector":
        # train/strategies.better: (admissible, profit-factor bar met, total); nothing meeting the bar keeps the old line
        key = lambda t: (t["trades"] >= min_trades and (t["pf"] or 0) >= PF_MIN and (t["win"] or 0) >= WIN_MIN,
                         t["trades"] >= min_trades and (t["pf"] or 0) >= PF_MIN, t["total"])
        ok = [t for t in table if key(t)[1] and t["total"] > 0]
        line = max(ok, key=key)["line"] if ok else fallback
    else:
        # A line must have made money over the window to be chosen: with a day of resolved rows the only candidate with
        # enough trades is the lowest line, and on 2026-09-23 it took the live fly from 4.1 % to 0.25 % on 119 losing trades.
        ok = [t for t in table if t["trades"] >= min_trades and t["total"] > 0]
        line = max(ok, key=lambda t: t["total"])["line"] if ok else fallback
    tr = taken_idx(ts, mint, horizon_s, rows[scores[rows] >= line]); r = rets[tr]
    table_s = sizing.build_table(scores[tr] - line, r) or list(prev_sizing or [])
    return Calibration(line=float(line), sizing=table_s, trades=int(len(r)), mean=float(r.mean()) if len(r) else None, total=float(r.sum()),
                       changed=prev_line is None or float(line) != float(prev_line), lines=table)
