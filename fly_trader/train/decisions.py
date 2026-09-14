"""Decision points shared by the selector models: one row per eligible minute of the mature universe.

Rows come from ``train/mature.py`` feature parts (pump.fun-origin PumpSwap tokens of any age, 1-minute candles
fed through the live feature engine) joined with ``corpus_meta`` creation-time and creator columns. Eligible =
pool ≥ 20 SOL, 15-minute volume ≥ 5 SOL, contamination-free series, and a full label horizon ahead.
Label: net forward return over ``horizon_min`` (fill at the signal minute's close; ``fwd_pess`` fills at the
next traded minute's open) after a ``fee`` round trip; ``y`` = return above ``label_thr``.
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
from ..market.features import FEATURES
from .corpus_meta import FEATURE_COLS as META_COLS, load_features

EXTRA_COLS = ["traders_15m", "traders_1h", "n_trades_1m", "hod_s", "hod_c", "age_known"]
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


def build(days: int | None = 45, horizon_min: int = 30, fee: float = 0.006, label_thr: float = 0.03, feature_dir=None) -> DecisionSet:
    root = feature_dir or (config.CORPUS_DIR / "features_mature")
    files = sorted(glob.glob(str(root / "*" / "part.parquet")))
    if days:
        files = files[-days - 1:]
    if not files:
        raise RuntimeError("no mature feature parts; run build-mature")
    need = ["mint", "ts", "open", "close", "resq", "age_h", "traders_15m", "traders_1h", "n_trades_1m"] + FEATURES
    df = pd.concat([pq.read_table(f, columns=need).to_pandas() for f in files], ignore_index=True).sort_values(["mint", "ts"]).reset_index(drop=True)
    jump = (df["close"] / df.groupby("mint")["close"].shift(1)).fillna(1.0)
    bad = set(df.loc[(jump > 50) | (jump < 1 / 50) | (df["resq"] > 1e5), "mint"])
    if bad:
        df = df[~df["mint"].isin(bad)].reset_index(drop=True)
    df = df.merge(load_features(), on="mint", how="left")
    H = horizon_min * 60.0
    ts = ((df["ts"] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(); cl = df["close"].to_numpy(); op = df["open"].to_numpy()
    mints = df["mint"].to_numpy(); starts = np.r_[0, np.flatnonzero(mints[1:] != mints[:-1]) + 1, len(mints)]
    fwd = np.full(len(df), np.nan); fwdp = np.full(len(df), np.nan)
    for s, e in zip(starts[:-1], starts[1:]):
        t = ts[s:e]; j = np.searchsorted(t, t + H, side="right") - 1 + s
        exit_px = cl[j]
        fwd[s:e] = exit_px / cl[s:e] * (1 - fee) - 1
        entry = cl[s:e].copy()
        if e - s > 1:
            gap = np.diff(t); nxt_ok = np.r_[gap <= 120.0, False]
            entry[nxt_ok] = op[s + 1:e + 1][nxt_ok[: e - s - 1].tolist() + [False]] if False else entry[nxt_ok]
            idx = np.flatnonzero(nxt_ok); entry[idx] = op[s + idx + 1]
        fwdp[s:e] = exit_px / entry * (1 - fee) - 1
    hod = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hod_s"], df["hod_c"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    df["age_known"] = df["age_h"].notna().astype(np.float32)
    df["mayhem"] = df["mayhem"].astype(float) if "mayhem" in df else 0.0
    last_ts = df["ts"].max()
    elig = (np.isfinite(df["resq"]) & (df["resq"] >= MIN_RESQ_SOL) & (df["logvol_15m"] >= math.log1p(MIN_VOL_15M_SOL))
            & np.isfinite(fwd) & (df["ts"] < last_ts - pd.Timedelta(minutes=horizon_min + 5))).to_numpy()
    df = df[elig].reset_index(drop=True); fwd = fwd[elig]; fwdp = fwdp[elig]; ts = ts[elig]
    X = df[X_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return DecisionSet(X=X, y=(fwd > label_thr).astype(np.int8), fwd=fwd.astype(np.float32), fwd_pess=fwdp.astype(np.float32),
                       day=df["ts"].dt.date.to_numpy(), ts=ts, mint=df["mint"].to_numpy(), cols=list(X_COLS), horizon_s=H)


def trades_from_picks(ds: DecisionSet, pick: np.ndarray, returns: np.ndarray | None = None) -> np.ndarray:
    """Net returns of the trades a picker would take: one position per token, re-entry only after the hold."""
    r_all = ds.fwd_pess if returns is None else returns
    idx = np.flatnonzero(pick)
    if len(idx) == 0:
        return np.array([], dtype=np.float32)
    order = np.lexsort((ds.ts[idx], ds.mint[idx])); idx = idx[order]
    out = []; last_mint = None; last_t = -1e18
    for i in idx:
        if ds.mint[i] != last_mint:
            last_mint = ds.mint[i]; last_t = -1e18
        if ds.ts[i] >= last_t + ds.horizon_s:
            out.append(r_all[i]); last_t = ds.ts[i]
    return np.asarray(out, dtype=np.float32)


def summarize(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0, "mean": None, "median": None, "win": None, "pf": None}
    g = float(r[r > 0].sum()); l = float(-r[r <= 0].sum())
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean()), "pf": (g / l) if l > 0 else float("inf")}


def evaluate(ds: DecisionSet, scores: np.ndarray, test: np.ndarray, threshold: float, label: str = "") -> dict:
    """Trades on rows of ``test`` whose score ≥ threshold; metrics per day and pooled."""
    pick = test & (scores >= threshold); r = trades_from_picks(ds, pick)
    per_day = {}
    for d in sorted(set(ds.day[test].tolist())):
        rd = trades_from_picks(ds, pick & (ds.day == d)); per_day[str(d)] = summarize(rd)
    pooled = summarize(r); pooled["days_positive"] = sum(1 for v in per_day.values() if v["mean"] is not None and v["mean"] > 0); pooled["days"] = len(per_day)
    return {"label": label, "pooled": pooled, "per_day": per_day}
