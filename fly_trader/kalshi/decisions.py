"""Decision points for the Kalshi selector and fly: one row per (market, minute, side) of the feature parts
(kalshi/mature.py), with the settlement labels both arms trade on.

Reuses ``train.decisions.DecisionSet`` unchanged: ``mint`` = ``"<ticker>:<side>"``, ``ts`` = minute start, ``day`` =
the minute's date, and — because every position is held to settlement — a per-row hold ``hold_s`` = settlement − ts
(``taken_idx`` / ``evaluate`` already take per-row holds). Labels, per dollar staked at the fee-inclusive price:
- ``fwd_pess`` (taker): ``(100·y − eff) / eff`` with ``eff = effective_price_cents(side ask)`` and ``y`` the settlement
  of the side (1 when it paid);
- ``fwd`` (reference): the same at the side's mid;
- ``fwd_h["maker"]``: the maker arm's order (kalshi/maker.py): a bid resting one tick above the side's bid, never at or
  through its ask (``maker_rest``), for the order's lifetime (``KALSHI_MAKER_TTL_H``, ending ``KALSHI_MAKER_QUIET_MIN``
  before close); filled (``maker_filled``) when the side's ask came down to it — every bid at that price, ours included,
  was taken — or a taker sold the side below it, or at it when our bid set a new best price and so stood first in its
  queue; the maker fee where the series charges one; NaN when unfilled. Until feature version 5 the order rested one tick
  inside the ask, for the market's whole life, and filled only when the ask later fell strictly below it: it paid away
  the spread makers earn and counted only the fills that came with an adverse move (2026-10-09).
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
from .features import K_COLS, KALSHI_FEATURE_VERSION, KIDX
from .mature import part_current
from .vendor import kalshi_client as kc

X_COLS = list(K_COLS)


def maker_rest(side_bid, side_ask) -> np.ndarray:
    """The maker arm's bid (cents): one tick above the side's best bid, at most one tick below its ask (``post_only``
    never crosses); with a one-tick spread it joins the bid's queue."""
    return np.clip(np.minimum(np.round(np.asarray(side_bid, dtype=float)) + 1.0, np.round(np.asarray(side_ask, dtype=float)) - 1.0), 1.0, 99.0)


def maker_filled(rest, side_bid, ask_low, sold_low) -> np.ndarray:
    """Would a bid at ``rest``, posted when the side's bid was ``side_bid``, have filled, given the lowest ask the side
    showed (``ask_low``) and the lowest price a taker sold the side at (``sold_low``) while it rested (NaN: none). The ask
    reaching ``rest`` means every bid at ``rest`` was taken; a taker sale below ``rest`` means the same; a sale at
    ``rest`` reaches us only when we were first there — our bid improved on the best bid, so nobody was ahead of us."""
    rest = np.asarray(rest, dtype=float); ask_low = np.asarray(ask_low, dtype=float); sold_low = np.asarray(sold_low, dtype=float)
    first = rest > np.round(np.asarray(side_bid, dtype=float))
    with np.errstate(invalid="ignore"):
        return (np.isfinite(ask_low) & (ask_low <= rest)) | (np.isfinite(sold_low) & ((sold_low < rest) | (first & (sold_low <= rest))))


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
    """The decision rows of the most recent ``days`` feature days (all of them when None). Each part is reduced to its
    eligible rows as it is read — float32 inputs, labels, integer market ids — so the whole corpus (tens of millions of
    rows) fits: reading every part into one pandas frame with its string columns peaked near three times the final size."""
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
    need = list(dict.fromkeys([*X_COLS, "ticker", "side", "ts", "category", "close_ts", "end_ts", "settled_ts", "result", "fill_ask_low", "fill_sold_low"]))
    tick_id: dict[str, int] = {}; tick_names: list[str] = []; cat_id: dict[str, int] = {}; cat_names: list[str] = []
    acc: dict[str, list] = {k: [] for k in ("X", "ts", "settled", "y", "fwd", "fwd_pess", "maker", "tick", "side", "cat", "ord")}
    for f in files:
        t = pq.read_table(f, columns=need)
        if not t.num_rows:
            continue
        res = np.asarray(t["result"].to_numpy(zero_copy_only=False), dtype=object); side = np.asarray(t["side"].to_numpy(zero_copy_only=False), dtype=object)
        ok = (res == "yes") | (res == "no")
        if not ok.any():
            continue
        X = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in X_COLS])
        X[~np.isfinite(X)] = 0.0
        ts = t["ts"].cast("int64").to_numpy().astype(np.float64) / 1000.0
        close = t["close_ts"].to_numpy(zero_copy_only=False).astype(np.float64); st_ = t["settled_ts"].to_numpy(zero_copy_only=False).astype(np.float64)
        end = t["end_ts"].to_numpy(zero_copy_only=False).astype(np.float64)
        settled = np.where(np.isfinite(st_), st_, end); to_close = close - ts     # window on the anchor; a label's timing on the realized end
        y = (res == side).astype(np.float64)
        eff = X[:, KIDX["eff_price"]].astype(np.float64); mid = np.clip(X[:, KIDX["side_mid"]].astype(np.float64), 1.0, 99.0)
        fwd_pess = (100.0 * y - eff) / eff; fwd = (100.0 * y - mid) / mid
        rest = maker_rest(X[:, KIDX["side_bid"]], X[:, KIDX["side_ask"]])
        filled = maker_filled(rest, X[:, KIDX["side_bid"]], t["fill_ask_low"].to_numpy(zero_copy_only=False).astype(np.float64),
                              t["fill_sold_low"].to_numpy(zero_copy_only=False).astype(np.float64))
        m_eff = rest + maker_fee_cents(rest, X[:, KIDX["fee_mult"]], X[:, KIDX["maker_fee"]])
        maker = np.where(filled, (100.0 * y - m_eff) / m_eff, np.nan)
        elig = ok & eligible_mask(X, X_COLS, to_close) & np.isfinite(fwd_pess) & (settled > ts)
        if not elig.any():
            continue
        tk = t["ticker"].to_numpy(zero_copy_only=False)[elig]; cats = t["category"].to_numpy(zero_copy_only=False)[elig]
        acc["X"].append(X[elig]); acc["ts"].append(ts[elig]); acc["settled"].append(settled[elig]); acc["y"].append(y[elig])
        acc["fwd"].append(fwd[elig].astype(np.float32)); acc["fwd_pess"].append(fwd_pess[elig].astype(np.float32)); acc["maker"].append(maker[elig].astype(np.float32))
        acc["tick"].append(np.fromiter((tick_id.setdefault(x, len(tick_id)) for x in tk), dtype=np.int64, count=len(tk)))
        acc["side"].append((side[elig] == "yes").astype(np.int8))
        acc["cat"].append(np.fromiter((cat_id.setdefault(x, len(cat_id)) for x in cats), dtype=np.int16, count=len(cats)))
        acc["ord"].append((np.floor(ts[elig] / 86400.0) + 719163).astype(np.int32))          # date.fromordinal(719163) is 1970-01-01
        del X, t
    if not acc["X"]:
        raise RuntimeError("no eligible Kalshi decision rows")
    cat = lambda k: np.concatenate(acc.pop(k))
    X = cat("X"); ts = cat("ts"); settled = cat("settled"); y = cat("y"); fwd = cat("fwd"); fwd_pess = cat("fwd_pess"); maker = cat("maker")
    tick = cat("tick"); side = cat("side"); catc = cat("cat"); ords = cat("ord")
    tick_names = np.array([None] * len(tick_id), dtype=object)
    for k_, v in tick_id.items():
        tick_names[v] = k_
    cat_names = np.array([None] * len(cat_id), dtype=object)
    for k_, v in cat_id.items():
        cat_names[v] = k_
    day_objs = {o: date.fromordinal(int(o)) for o in np.unique(ords)}
    day_arr = np.array([day_objs[o] for o in sorted(day_objs)], dtype=object)[np.searchsorted(np.array(sorted(day_objs)), ords)]
    ds = DecisionSet(X=X, y=(fwd_pess > label_thr).astype(np.int8), fwd=fwd, fwd_pess=fwd_pess, day=day_arr, ts=ts, mint=tick * 2 + side.astype(np.int64),
                     cols=list(X_COLS), horizon_s=float(np.median(settled - ts)), fwd_h={"maker": maker})
    ds._day_ord = ords
    ds.train_max_rows = int(config.KALSHI_TRAIN_MAX_ROWS) or None   # training rows per fit: a random subset (train/strategies.cap_rows)
    ds.hold_s = (settled - ts).astype(np.float64)                     # per-row: every position is held to its own settlement
    ds.data_version = f"kalshi-features-{KALSHI_FEATURE_VERSION}"     # in the walk-forward cache key: another feature version never reuses a fit
    ds.label_ts = (settled + 60.0).astype(np.float64)                 # when the outcome is known: training for a period may use only rows known before it
    ds.outcome = y.astype(np.float32)
    ds.side = np.array(["no", "yes"], dtype=object)[side]; ds.ticker = tick_names[tick]; ds.category = cat_names[catc]
    # the stack's bars (train/strategies.Score): a positive mean with a weekly sign test, not the memecoin PF ≥ 1.3 and 65 %
    # winners — a contract bought at 95c that pays 5c when it wins needs a 1.3 % edge for PF 1.3, about all the edge the
    # literature finds left on Kalshi, and its win rate is set by its price (operator decision, 2026-10-09)
    ds.gate = "significance"
    return ds
