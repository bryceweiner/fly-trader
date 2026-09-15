"""The fly's buy line and sizing bands, recalibrated from its own scores and the realized outcomes — no training.

Every day (live: the first minute after 00:00 UTC; replay: every day boundary) on the scored minutes whose labels
resolved in the last ``WINDOW_DAYS`` days. Candidate lines: the selector's ``LINE_CANDIDATES`` plus the window's score
quantiles (a compressed score scale still gets a usable line); the line with the most total net profit (fixed size, one
position per token, the hold) over at least ``MIN_LINE_TRADES`` trades wins — the selector's rule — otherwise the previous line
stays. Sizing: ``agent/sizing.build_table`` on the window's trades at that line (the previous table if too few).
The fly never trades the selector's line: its scores live on their own scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agent import sizing
from .decisions import taken_idx
from .selector import LINE_CANDIDATES, MIN_EV, MIN_LINE_TRADES

WINDOW_DAYS = 7
QUANTILES = (0.95, 0.98, 0.99, 0.995, 0.998, 0.999)


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


def calibrate(ts: np.ndarray, mint: np.ndarray, horizon_s: float, scores: np.ndarray, rets: np.ndarray, prev_line: float | None = None,
              prev_sizing: list | None = None, min_trades: int = MIN_LINE_TRADES) -> Calibration:
    """Line and sizing from one window of (ts, mint, score, realized net return) rows."""
    scores = np.asarray(scores, dtype=np.float64); rets = np.asarray(rets, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(scores) & np.isfinite(rets))
    table = []
    for c in candidates(scores[rows]):
        tr = taken_idx(ts, mint, horizon_s, rows[scores[rows] >= c]); r = rets[tr]
        table.append({"line": c, "trades": int(len(r)), "mean": float(r.mean()) if len(r) else None, "total": float(r.sum())})
    ok = [t for t in table if t["trades"] >= min_trades]
    line = max(ok, key=lambda t: t["total"])["line"] if ok else (prev_line if prev_line is not None else MIN_EV)
    tr = taken_idx(ts, mint, horizon_s, rows[scores[rows] >= line]); r = rets[tr]
    table_s = sizing.build_table(scores[tr] - line, r) or list(prev_sizing or [])
    return Calibration(line=float(line), sizing=table_s, trades=int(len(r)), mean=float(r.mean()) if len(r) else None, total=float(r.sum()),
                       changed=prev_line is None or float(line) != float(prev_line), lines=table)
