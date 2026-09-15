"""Decision points shared by the selector models: one row per eligible minute of the mature universe.

Rows come from ``train/mature.py`` feature parts (pump.fun-origin PumpSwap tokens of any age, 1-minute candles
fed through the live feature engine) joined with ``corpus_meta`` creation-time and creator columns. Eligible =
pool ≥ 20 SOL, 15-minute volume ≥ 5 SOL, contamination-free series, and a full label horizon ahead.
Label: net forward return over ``horizon_min`` (fill at the signal minute's close; ``fwd_pess`` fills at the
next traded minute's open) after trading costs; ``y`` = return above ``label_thr``. Costs default to the paper
broker's model on both sides (``exit_cost_0p1``: Jupiter fee by token age + pool fee + constant-product impact of a
0.1 SOL position; entry at the signal minute, exit at the exit minute) — the same costs the paper book pays. A flat
``fee`` round trip can be given instead (tests, sensitivity runs). ``random_trades`` is the no-skill baseline.
"""
from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config
from ..market.features import FEATURE_VERSION, FEATURES
from .corpus_features import _epoch_s
from .corpus_meta import FEATURE_COLS as META_COLS, load_features
from .mature import part_current

EXTRA_COLS = ["traders_15m", "traders_1h", "n_trades_1m", "hod_s", "hod_c", "age_known", "meta_known"]
X_COLS = FEATURES + EXTRA_COLS + META_COLS
MIN_RESQ_SOL, MIN_VOL_15M_SOL = 20.0, 5.0


@dataclass
class DecisionSet:
    X: np.ndarray          # [N, F] float32, raw (not standardized)
    y: np.ndarray          # [N] int8
    fwd: np.ndarray        # [N] float32 net forward return, close fill
    fwd_pess: np.ndarray   # [N] float32 net forward return, next-open fill
    day: np.ndarray        # [N] datetime.date
    ts: np.ndarray         # [N] float64 epoch seconds
    mint: np.ndarray       # [N] str
    cols: list[str]
    horizon_s: float

    @property
    def days(self) -> list[date]:
        return sorted(set(self.day.tolist()))

    def mask_days(self, lo: date | None = None, hi: date | None = None) -> np.ndarray:
        m = np.ones(len(self.y), bool)
        if lo is not None:
            m &= self.day >= lo
        if hi is not None:
            m &= self.day <= hi
        return m


def build(days: int | None = 45, horizon_min: int = 30, fee: float | None = None, label_thr: float = 0.03, feature_dir=None) -> DecisionSet:
    root = feature_dir or (config.CORPUS_DIR / "features_mature")
    files = sorted(glob.glob(str(root / "*" / "part.parquet")))
    if days:
        files = files[-days - 1:]
    if not files:
        raise RuntimeError("no mature feature parts; run build-mature")
    stale = [f for f in files if not part_current(f)]
    if stale:
        raise RuntimeError(f"{len(stale)} feature part(s) were built with another feature or aggregation version (e.g. {stale[0]}); run build-mature")
    need = ["mint", "ts", "open", "close", "resq", "age_h", "traders_15m", "traders_1h", "n_trades_1m"] + FEATURES
    df = pd.concat([pq.read_table(f, columns=need).to_pandas() for f in files], ignore_index=True).sort_values(["mint", "ts"]).reset_index(drop=True)
    # a mint whose series breaks scale (secondary pool in another quote, impossible reserve) is ineligible FROM THAT MINUTE ON —
    # never before it, so no decision row is judged with the token's future (rows before a dump keep their labels)
    jump = (df["close"] / df.groupby("mint")["close"].shift(1)).fillna(1.0)
    df["_off"] = ((jump > 50) | (jump < 1 / 50) | (df["resq"] > 1e5)).astype(np.int8)
    broken = df.groupby("mint")["_off"].cummax().to_numpy().astype(bool)
    df = df.merge(load_features(), on="mint", how="left")
    df["meta_known"] = df["ttg_min"].notna().astype(np.float32) if "ttg_min" in df else np.float32(0.0)
    H = horizon_min * 60.0
    ts = _epoch_s(df["ts"]); cl = df["close"].to_numpy(); op = df["open"].to_numpy()
    mints = df["mint"].to_numpy(); starts = np.r_[0, np.flatnonzero(mints[1:] != mints[:-1]) + 1, len(mints)]
    fwd = np.full(len(df), np.nan); fwdp = np.full(len(df), np.nan)
    ec = np.clip(df["exit_cost_0p1"].to_numpy(dtype=float), 0.0, 1.0) if fee is None else None   # one-side cost fraction per minute
    for s, e in zip(starts[:-1], starts[1:]):
        t = ts[s:e]; j = np.searchsorted(t, t + H, side="right") - 1 + s
        exit_px = cl[j]
        keep = (1 - ec[s:e]) * (1 - ec[j]) if fee is None else 1 - fee
        fwd[s:e] = exit_px / cl[s:e] * keep - 1
        c = cl[s:e]                                  # an exit on a one-minute print 50x off that reverts the next minute is a data artifact
        if e - s > 2:
            r_prev = np.r_[1.0, c[1:] / c[:-1]]; r_next = np.r_[c[1:] / c[:-1], 1.0]
            spike = ((r_prev > 50) & (r_next < 1 / 50)) | ((r_prev < 1 / 50) & (r_next > 50))
            fwd[s:e][spike[j - s]] = np.nan
        entry = cl[s:e].copy()                      # pessimistic entry: the next traded minute's open when it is within 2 minutes
        if e - s > 1:
            nxt_ok = np.r_[np.diff(t) <= 120.0, False]
            idx = np.flatnonzero(nxt_ok); entry[idx] = op[s + idx + 1]
        fwdp[s:e] = exit_px / entry * keep - 1
    hod = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hod_s"], df["hod_c"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    df["age_known"] = df["age_h"].notna().astype(np.float32)
    df["mayhem"] = df["mayhem"].astype(float) if "mayhem" in df else 0.0
    last_ts = df["ts"].max()
    elig = (np.isfinite(df["resq"]) & (df["resq"] >= MIN_RESQ_SOL) & (df["logvol_15m"] >= math.log1p(MIN_VOL_15M_SOL))
            & np.isfinite(fwd) & (df["ts"] < last_ts - pd.Timedelta(minutes=horizon_min + 5))).to_numpy() & ~broken
    df = df[elig].reset_index(drop=True); fwd = fwd[elig]; fwdp = fwdp[elig]; ts = ts[elig]
    X = df[X_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return DecisionSet(X=X, y=(fwd > label_thr).astype(np.int8), fwd=fwd.astype(np.float32), fwd_pess=fwdp.astype(np.float32),
                       day=df["ts"].dt.date.to_numpy(), ts=ts, mint=df["mint"].to_numpy(), cols=list(X_COLS), horizon_s=H)


def taken_rows(ds: DecisionSet, pick: np.ndarray) -> np.ndarray:
    """Row indices of the trades a picker takes (the same rule as ``trades_from_picks``)."""
    idx = np.flatnonzero(pick)
    if len(idx) == 0:
        return idx
    idx = idx[np.lexsort((ds.ts[idx], ds.mint[idx]))]; out = []; last_mint = None; last_t = -1e18
    for i in idx:
        if ds.mint[i] != last_mint:
            last_mint = ds.mint[i]; last_t = -1e18
        if ds.ts[i] >= last_t + ds.horizon_s:
            out.append(i); last_t = ds.ts[i]
    return np.asarray(out, dtype=int)


def trades_from_picks(ds: DecisionSet, pick: np.ndarray, returns: np.ndarray | None = None, with_days: bool = False):
    """Net returns of the trades a picker would take: one position per token, re-entry only after the hold.
    With ``with_days`` also returns the entry day of each trade (so per-day tables respect holds across midnight)."""
    r_all = ds.fwd_pess if returns is None else returns
    idx = np.flatnonzero(pick)
    if len(idx) == 0:
        return (np.array([], dtype=np.float32), np.array([], dtype=object)) if with_days else np.array([], dtype=np.float32)
    order = np.lexsort((ds.ts[idx], ds.mint[idx])); idx = idx[order]
    out = []; days = []; last_mint = None; last_t = -1e18
    for i in idx:
        if ds.mint[i] != last_mint:
            last_mint = ds.mint[i]; last_t = -1e18
        if ds.ts[i] >= last_t + ds.horizon_s:
            out.append(r_all[i]); days.append(ds.day[i]); last_t = ds.ts[i]
    r = np.asarray(out, dtype=np.float32)
    return (r, np.asarray(days, dtype=object)) if with_days else r


def random_trades(ds: DecisionSet, rows: np.ndarray, n_picks: int, seeds: int = 20) -> np.ndarray:
    """The no-skill baseline: ``n_picks`` random minutes among ``rows`` (the model's pick count on the same eligible
    minutes), traded with the same hold rule, fills and costs; ``seeds`` draws pooled."""
    idx = np.flatnonzero(rows)
    if n_picks <= 0 or len(idx) == 0:
        return np.array([], dtype=np.float32)
    out = []
    for s in range(seeds):
        pick = np.zeros(len(ds.y), bool); pick[np.random.default_rng(s).choice(idx, min(n_picks, len(idx)), replace=False)] = True
        out.append(trades_from_picks(ds, pick))
    return np.concatenate(out)


def _ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (what Spearman needs)."""
    _, inv, cnt = np.unique(np.asarray(a), return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.ravel()]


def rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman correlation (Pearson on average ranks); None when either side is constant or empty."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return None
    ra = _ranks(a); rb = _ranks(b); ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


def summarize(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0, "mean": None, "median": None, "win": None, "pf": None}
    g = float(r[r > 0].sum()); l = float(-r[r <= 0].sum())
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean()), "pf": (g / l) if l > 0 else None}   # None: no losing trade (JSON has no inf)


def evaluate(ds: DecisionSet, scores: np.ndarray, test: np.ndarray, threshold: float, label: str = "") -> dict:
    """Trades on rows of ``test`` whose score ≥ threshold; metrics per day and pooled."""
    pick = test & (scores >= threshold); r, rdays = trades_from_picks(ds, pick, with_days=True)
    per_day = {}
    for d in sorted(set(ds.day[test].tolist())):
        per_day[str(d)] = summarize(r[rdays == d])
    pooled = summarize(r); pooled["days_positive"] = sum(1 for v in per_day.values() if v["mean"] is not None and v["mean"] > 0); pooled["days"] = len(per_day)
    return {"label": label, "pooled": pooled, "per_day": per_day}
