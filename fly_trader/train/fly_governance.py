"""The checks that roll the fly's plasticity back (live, hourly) — and are counted in the replay to measure false alarms.

They judge what learning did, so a channel that has learned nothing (drift at most ``DRIFT_MIN``: a learning rate of 0,
or a memory decayed back to its bootstrap) trips none: there is nothing for a rollback to undo. Otherwise any one trips
a rollback:
- shadow: over the last ``SHADOW_DAYS`` days of resolved picks (each arm at its own line, one position per token), the
  plastic fly's mean net return per trade is more than one standard error below its frozen shadow's (the bootstrap fly,
  D = 0, scoring the same minutes), with at least ``SHADOW_MIN_TRADES`` trades on each side;
- ic: over the last ``IC_HOURS`` hours (at least ``IC_MIN_ROWS`` resolved minutes), the rank correlation of its scores
  with the realized returns trails its frozen shadow's on the same rows by more than ``IC_MARGIN`` -- learning ranked
  worse than its bootstrap. Not merely below 0: a fly that scores as its teacher would then be rolled back for the
  teacher's own bad day (2026-10-11..13: ev, learning rate 0, rolled back three times at IC -0.10..-0.13, which froze
  learning). Without the shadow's rows, below 0;
- drift: ‖D‖_F / ‖w0‖_F above ``DRIFT_MAX``.
"""
from __future__ import annotations

import numpy as np

from .decisions import rank_corr, taken_idx

SHADOW_DAYS, SHADOW_MIN_TRADES = 3, 30
IC_HOURS, IC_MIN_ROWS, IC_SAMPLE = 24, 5000, 50_000
DRIFT_MAX = 0.5
DRIFT_MIN = 1e-3             # at most this far from its bootstrap, a channel has learned nothing a rollback could undo
IC_MARGIN = 0.02             # how far the plastic fly's IC must trail its frozen shadow's to count as learning's harm


def shadow_check(r_plastic: np.ndarray, r_frozen: np.ndarray, min_trades: int = SHADOW_MIN_TRADES) -> tuple[bool, dict]:
    rp, rf = np.asarray(r_plastic, dtype=np.float64), np.asarray(r_frozen, dtype=np.float64)
    d = {"n_plastic": int(len(rp)), "n_frozen": int(len(rf)), "mean_plastic": float(rp.mean()) if len(rp) else None, "mean_frozen": float(rf.mean()) if len(rf) else None}
    if len(rp) < min_trades or len(rf) < min_trades:
        return False, d
    se = float(np.sqrt(rp.var(ddof=1) / len(rp) + rf.var(ddof=1) / len(rf)))
    d["se"] = se
    return bool(rp.mean() - rf.mean() < -se), d


def ic_check(scores: np.ndarray, labels: np.ndarray, min_rows: int = IC_MIN_ROWS, sample: int = IC_SAMPLE, seed: int = 0) -> tuple[bool, float | None]:
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(y); s, y = s[ok], y[ok]
    if len(s) < min_rows:
        return False, None
    if len(s) > sample:
        i = np.random.default_rng(seed).choice(len(s), sample, replace=False); s, y = s[i], y[i]
    ic = rank_corr(s, y)
    return bool(ic is not None and ic < 0), ic


def drift_check(drift: float, cap: float = DRIFT_MAX) -> bool:
    return bool(drift > cap)


def picks_returns(ts: np.ndarray, mint: np.ndarray, horizon_s: float, scores: np.ndarray, lines: np.ndarray, rets: np.ndarray) -> np.ndarray:
    """Net returns of the trades an arm took: rows scored at/above the line in force, one position per token."""
    idx = np.flatnonzero(np.asarray(scores) >= np.asarray(lines))
    return np.asarray(rets)[taken_idx(ts, mint, horizon_s, idx)]


def check(shadow: tuple[np.ndarray, np.ndarray] | None, ic_rows: tuple[np.ndarray, np.ndarray] | None, drift: float,
          ic_frozen_rows: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """All three checks; ``triggers`` lists the ones that tripped. ``ic_frozen_rows``: the frozen shadow's scores and the
    labels of the same rows as ``ic_rows``. A channel at or below ``DRIFT_MIN`` trips nothing (what would have tripped is
    ``not_counted``)."""
    out = {"triggers": [], "drift": float(drift)}
    if shadow is not None:
        t, d = shadow_check(*shadow); out["shadow"] = d
        if t:
            out["triggers"].append("shadow")
    if ic_rows is not None:
        t, ic = ic_check(*ic_rows); out["ic"] = ic
        if ic_frozen_rows is not None:
            _, icf = ic_check(*ic_frozen_rows); out["ic_frozen"] = icf
            t = ic is not None and icf is not None and ic < icf - IC_MARGIN
        if t:
            out["triggers"].append("ic")
    if drift_check(drift):
        out["triggers"].append("drift")
    if float(drift) <= DRIFT_MIN and out["triggers"]:
        out["not_counted"], out["triggers"] = out["triggers"], []
    return out
