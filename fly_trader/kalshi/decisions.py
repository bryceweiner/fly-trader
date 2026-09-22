"""Decision points for the Kalshi selector and fly: one row per (market, minute, side) of the feature parts
(kalshi/mature.py), with the settlement labels both arms trade on.

Reuses ``train.decisions.DecisionSet`` unchanged: ``mint`` = ``"<ticker>:<side>"``, ``ts`` = minute start, ``day`` =
the minute's date, and — because every position is held to settlement — a per-row hold ``hold_s`` = settlement − ts
(``taken_idx`` / ``evaluate`` already take per-row holds). Labels, per dollar staked at the fee-inclusive price:
- ``fwd_pess`` (taker): ``(100·y − eff) / eff`` with ``eff = effective_price_cents(side ask)`` and ``y`` the settlement
  of the side (1 when it paid);
- ``fwd`` (reference): the same at the side's mid;
- ``fwd_h["maker"]``: rest at ``side ask − MAKER_DISCOUNT`` at the minute; filled iff the side's ask later fell strictly
  below the rest price (better_bot's traded-through rule, ``maker_paper.would_have_filled``) before close; the maker fee
  where the series charges one; NaN when unfilled.
Eligibility (the live engine applies the same, ``kalshi/engine.KalshiMinuteEngine.eligible``): both quotes present,
spread ≤ KALSHI_MAX_SPREAD_CENTS, side ask in 1..99, open interest ≥ KALSHI_MIN_OPEN_INTEREST, 24 h volume ≥
KALSHI_MIN_VOLUME_24H, settlement yes/no, time to close inside the entry window.
"""
from __future__ import annotations

import glob
import math
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config
from ..train.corpus_features import _epoch_s
from ..train.decisions import DecisionSet
from . import data as D
from .features import K_COLS, KIDX
from .mature import part_current
from .vendor import kalshi_client as kc

MAKER_DISCOUNT = 1                    # cents inside the ask a resting order sits (better_bot PAPER_MAKER_DISCOUNT_CENTS)
X_COLS = list(K_COLS)


def maker_fee_cents(price_cents, fee_mult, charged) -> np.ndarray:
    """Per-contract maker fee in cents where the series charges makers (KALSHI_MAKER_FEE_FACTOR·mult·p(1−p)·100), else 0."""
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.where(np.asarray(charged, dtype=float) > 0, config.KALSHI_MAKER_FEE_FACTOR * np.asarray(fee_mult, dtype=float) * p * (1 - p) * 100.0, 0.0)


def eligible_mask(X: np.ndarray, cols: list[str], to_close_s: np.ndarray) -> np.ndarray:
    c = {n: i for i, n in enumerate(cols)}
    return ((X[:, c["spread"]] <= config.KALSHI_MAX_SPREAD_CENTS) & (X[:, c["side_ask"]] >= 1.0) & (X[:, c["side_ask"]] <= 99.0)
            & (X[:, c["side_bid"]] >= 1.0) & (np.expm1(X[:, c["log_oi"]]) >= config.KALSHI_MIN_OPEN_INTEREST)
            & (np.expm1(X[:, c["log_vol_24h"]]) >= config.KALSHI_MIN_VOLUME_24H)
            & (to_close_s >= config.KALSHI_MIN_MINUTES_TO_CLOSE * 60.0) & (to_close_s <= config.KALSHI_MAX_DAYS_TO_CLOSE * 86400.0))


def build(days: int | None = None, feature_dir=None, label_thr: float = 0.0) -> DecisionSet:
    root = feature_dir or D.FEATURES_DIR
    dirs = sorted(glob.glob(str(root / "*")))
    if days:
        dirs = dirs[-days:]
    files = [f for d in dirs for f in sorted(glob.glob(str(root / d.split("/")[-1] / "part-*.parquet")))]
    if not files:
        raise RuntimeError("no Kalshi feature parts; run kalshi-build")
    stale = [f for f in files if not part_current(f)]
    if stale:
        raise RuntimeError(f"{len(stale)} Kalshi feature part(s) from another version (e.g. {stale[0]}); run kalshi-build")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df = df[df["result"].isin(["yes", "no"])].reset_index(drop=True)
    ts = _epoch_s(df["ts"])
    X = df[X_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    settled = np.where(np.isfinite(df["settled_ts"].to_numpy(dtype=float)), df["settled_ts"].to_numpy(dtype=float), df["close_ts"].to_numpy(dtype=float))
    to_close = df["close_ts"].to_numpy(dtype=float) - ts
    y = (df["result"].to_numpy() == df["side"].to_numpy()).astype(np.float64)
    eff = X[:, KIDX["eff_price"]].astype(np.float64); mid = np.clip(X[:, KIDX["side_mid"]].astype(np.float64), 1.0, 99.0)
    fwd_pess = (100.0 * y - eff) / eff
    fwd = (100.0 * y - mid) / mid
    rest = np.clip(np.round(X[:, KIDX["side_ask"]]) - MAKER_DISCOUNT, 1.0, 99.0)
    fut = df["fut_min_ask"].to_numpy(dtype=float)
    filled = np.isfinite(fut) & (fut < rest)
    m_eff = rest + maker_fee_cents(rest, X[:, KIDX["fee_mult"]], X[:, KIDX["maker_fee"]])
    maker = np.where(filled, (100.0 * y - m_eff) / m_eff, np.nan)
    elig = eligible_mask(X, X_COLS, to_close) & np.isfinite(fwd_pess) & (settled > ts)
    df = df[elig].reset_index(drop=True)
    ds = DecisionSet(X=X[elig], y=(fwd_pess[elig] > label_thr).astype(np.int8), fwd=fwd[elig].astype(np.float32), fwd_pess=fwd_pess[elig].astype(np.float32),
                     day=df["ts"].dt.date.to_numpy(), ts=ts[elig], mint=(df["ticker"] + ":" + df["side"]).to_numpy(), cols=list(X_COLS),
                     horizon_s=float(np.median(settled[elig] - ts[elig])) if elig.any() else 86400.0, fwd_h={"maker": maker[elig].astype(np.float32)})
    ds.hold_s = (settled[elig] - ts[elig]).astype(np.float64)          # per-row: every position is held to its own settlement
    ds.outcome = y[elig].astype(np.float32)
    ds.side = df["side"].to_numpy(); ds.ticker = df["ticker"].to_numpy(); ds.category = df["category"].to_numpy()
    return ds
