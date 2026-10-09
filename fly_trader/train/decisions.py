"""Decision points shared by the selector models: one row per eligible minute of the mature universe.

Rows come from ``train/mature.py`` feature parts (pump.fun-origin PumpSwap tokens of any age, 1-minute candles
fed through the live feature engine) joined with ``corpus_meta`` creation-time and creator columns. Eligible =
pool ≥ ``MIN_RESQ_SOL``, 15-minute volume ≥ ``MIN_VOL_15M_SOL``, contamination-free series, and a full label horizon ahead.
Label: net forward return over ``horizon_min`` (fill at the signal minute's close; ``fwd_pess`` fills at the
next traded minute's open) after trading costs; ``y`` = return above ``label_thr``. Costs default to the paper
broker's model on both sides (``exit_cost_0p1``: the real fees — pool fee tier by market cap, Jupiter 10 bps, network
fee — + constant-product impact of a 0.1 SOL position; entry at the signal minute, exit at the exit minute) — the same costs the paper book pays. A flat
``fee`` round trip can be given instead (tests, sensitivity runs). ``random_trades`` is the no-skill baseline.
"""
from __future__ import annotations

import glob
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config
from ..market.features import FEATURE_VERSION, FEATURES
from .corpus_features import _epoch_s
from .corpus_meta import CURVE_COLS, FEATURE_COLS as META_COLS, load_features
from .flow import FLOW_COLS, MARKET_COLS, SKILL_COLS
from .mature import part_current

EXTRA_COLS = ["traders_15m", "traders_1h", "n_trades_1m", "hod_s", "hod_c", "age_known", "meta_known"]
# Inputs that carry no information (feature ablation 2026-09-15: the 120-min walk-forward over 149 days at real costs,
# repeated with each group removed; removing these lowered neither the rank correlation with net returns nor the
# out-of-sample profit): five stream features the archive cannot reproduce and the eleven Jupiter stats (always 0 — no
# history before 2026-09-12), realized volatility (IC 0.1221 → 0.1240, +0.29 → +0.38 SOL/day without it) and the
# creator's launch history (0.1221 → 0.1218, +0.29 → +0.29). All removed together (37 of 62 inputs): IC 0.1191,
# +0.50 SOL/day, +3.0 % per trade (all inputs: +1.5 %).
DROPPED_COLS = frozenset(["logsigners_15m", "logsigners_1h", "hawkes", "log_since_last", "vpin_15m",
                          "organic_score", "log_holders", "log_liq_usd", "top_holders_pct", "dev_balance_pct", "is_sus", "is_verified",
                          "net_buyers_1h", "holder_change_1h", "price_change_24h", "token2022",
                          "rvol_5m", "rvol_15m", "rvol_1h", "rvol_3h",
                          "prior_launches", "prior_grads", "prior_moon_share"])
# Input groups of the strategy stack (2026-09-15), each kept only if the walk-forward objective says so (train/selector.py):
# rug — graduation-time curve facts, the creator's rug history and insider selling; flow — wallet flow and market volume;
# skill — buying by skilled wallets. LEGACY_COLS are the 37 inputs of the single-strategy selector.
GROUPS = {"rug": [*CURVE_COLS, "curve_known", "prior_rug_share", "prior_known", "insider_sell_share_1m", "log_insider_sell_15m"],
          "flow": ["top_sell_share_1m", "buyers_ret", "buyers_slope", "org_imb_5m", "org_imb_15m", "wash_share_15m", "log_org_vol_15m", *MARKET_COLS],
          "skill": list(SKILL_COLS)}
_GROUPED = {c for g in GROUPS.values() for c in g}
LEGACY_COLS = [c for c in FEATURES + EXTRA_COLS + META_COLS if c not in DROPPED_COLS and c not in _GROUPED]
# One model family for both memecoin chains (branch rh-memecoins, 2026-09-28): amounts stay in each chain's own unit (SOL,
# ETH) and these inputs tell the rows apart — the chain, and on Robinhood Chain the class of the pool's quote asset (ETH
# is all zeros). Appended after the 64 Solana inputs, so every Solana value is unchanged (tests/test_solana_identity.py).
CHAIN_COLS = ["chain_rh", "qc_stable", "qc_btc", "qc_stock"]
X_COLS = LEGACY_COLS + [c for g in GROUPS.values() for c in g] + CHAIN_COLS
# Hold and gates fitted to the market at real costs (2026-09-15 sweep over 149 days: holds 10–240 min × age × pool ×
# 15-min volume × buy line, walk-forward): a 120-minute hold with pools ≥ 10 SOL and ≥ 5 SOL traded in 15 minutes (and
# tokens ≥ 6 h past graduation, train/selector.MIN_AGE_H) made money in every month and on 46–49 of 64 days in each
# half; the old 30 min / 20 SOL / 24 h lost money May–July.
HOLD_MIN = 120
MIN_RESQ_SOL, MIN_VOL_15M_SOL = 10.0, 5.0


def day_ords(ds) -> np.ndarray:
    """The rows' days as proleptic ordinals (int32), computed once per set: comparing tens of millions of date objects
    (``np.isin`` sorts them) cost minutes per walk-forward block on the Kalshi corpus."""
    o = getattr(ds, "_day_ord", None)
    if o is None or len(o) != len(ds.day):
        uniq = {}
        o = np.fromiter((uniq.setdefault(d, d.toordinal()) for d in ds.day), dtype=np.int32, count=len(ds.day))
        ds._day_ord = o
    return o


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
    fwd_h: dict = field(default_factory=dict)    # hold (min) → [N] float32 net return (next-open fill) over that hold; NaN past the data
    chain: np.ndarray | None = None               # [N] 'sol' | 'rh' (None: every row Solana)

    def chains(self) -> np.ndarray:
        return self.chain if self.chain is not None else np.full(len(self.y), "sol", dtype=object)

    @property
    def days(self) -> list[date]:
        return [date.fromordinal(int(o)) for o in np.unique(day_ords(self))]

    def subset(self, mask: np.ndarray) -> "DecisionSet":
        """The rows of ``mask`` (e.g. the days before a bootstrap), labels for every hold included."""
        m = np.asarray(mask, dtype=bool)
        return DecisionSet(X=self.X[m], y=self.y[m], fwd=self.fwd[m], fwd_pess=self.fwd_pess[m], day=self.day[m], ts=self.ts[m], mint=self.mint[m],
                           cols=list(self.cols), horizon_s=self.horizon_s, fwd_h={h: v[m] for h, v in self.fwd_h.items()},
                           chain=None if self.chain is None else self.chain[m])

    def mask_days(self, lo: date | None = None, hi: date | None = None) -> np.ndarray:
        m = np.ones(len(self.y), bool)
        if lo is not None:
            m &= self.day >= lo
        if hi is not None:
            m &= self.day <= hi
        return m


def _label_exit_cost(df) -> np.ndarray:
    """One side's real cost per row, at the size the book trades (market/exit_cost.cost_at_size). The live engine puts
    the same value in every bar it hands the fly (agent/minute_engine.py), so what the fly learns from and what it was
    trained on are one cost model. Robinhood Chain rows are priced at their own market's sizes (markets.RH)."""
    from ..market.exit_cost import cost_at_size
    ec = cost_at_size(df["exit_cost_0p1"].to_numpy(dtype=float), df["resq"].to_numpy(dtype=float))
    if "_chain" in df and (df["_chain"] == "rh").any():
        from ..markets import RH
        rh = (df["_chain"] == "rh").to_numpy()
        ec = np.asarray(ec, dtype=float).copy()
        ec[rh] = cost_at_size(df["exit_cost_0p1"].to_numpy(dtype=float)[rh], df["resq"].to_numpy(dtype=float)[rh], market=RH)
    return ec


def _roots(feature_dir) -> list[tuple]:
    """(feature part root, chain, part_current) per market: the given directory alone (Solana: tests, studies), else the
    Solana corpus plus Robinhood Chain's when it is enabled and built (rh/corpus.py)."""
    if feature_dir is not None:
        return [(feature_dir, "sol", part_current)]
    out = [(config.CORPUS_DIR / "features_mature", "sol", part_current)]
    if config.RH_ENABLED:
        from ..rh import corpus as rh_corpus
        if any(rh_corpus.FEAT_DIR.glob("*/part.parquet")):
            out.append((rh_corpus.FEAT_DIR, "rh", rh_corpus.part_current))
    return out


def build(days: int | None = 45, horizon_min: int = HOLD_MIN, fee: float | None = None, label_thr: float = 0.03, feature_dir=None, holds=None) -> DecisionSet:
    """``holds``: extra holds (minutes) whose labels are computed in the same pass (``DecisionSet.fwd_h``)."""
    need = ["mint", "ts", "open", "close", "resq", "age_h", "traders_15m", "traders_1h", "n_trades_1m"] + FEATURES + FLOW_COLS + SKILL_COLS + MARKET_COLS
    optional = set(FLOW_COLS + SKILL_COLS + MARKET_COLS)      # parts that predate the wallet-flow columns (tests' synthetic parts) read as 0
    roots = _roots(feature_dir); per_root = []
    for root, chain, current in roots:
        files = sorted(glob.glob(str(root / "*" / "part.parquet")))
        per_root.append((files, chain, current))
    if days:                                                  # the last ``days`` calendar days (+1 of warm-up) of the newest corpus
        last = max((f.split("/")[-2] for files, _, _ in per_root for f in files), default=None)
        if last is not None:
            lo = (date.fromisoformat(last) - pd.Timedelta(days=days)).isoformat()
            per_root = [([f for f in files if f.split("/")[-2] >= lo], c, cur) for files, c, cur in per_root]
    if not any(files for files, _, _ in per_root):
        raise RuntimeError("no mature feature parts; run build-mature")
    frames = []
    for files, chain, current in per_root:
        stale = [f for f in files if not current(f)]
        if stale:
            raise RuntimeError(f"{len(stale)} feature part(s) were built with another feature or aggregation version (e.g. {stale[0]}); run build-mature")
        for f in files:
            fr = pq.read_table(f, columns=[c for c in need if c not in optional or c in pq.read_schema(f).names]).to_pandas()
            fr["_chain"] = chain; frames.append(fr)
    df = pd.concat(frames, ignore_index=True).sort_values(["mint", "ts"]).reset_index(drop=True)
    kk = np.where(df["_chain"].to_numpy() == "rh", config.RH_ETH_PER_SOL, 1.0)      # ETH per SOL on RH rows: fixed-unit thresholds
    for c in optional:
        if c not in df:
            df[c] = 0.0
    # a mint whose series breaks scale (secondary pool in another quote, impossible reserve) is ineligible FROM THAT MINUTE ON —
    # never before it, so no decision row is judged with the token's future (rows before a dump keep their labels)
    jump = (df["close"] / df.groupby("mint")["close"].shift(1)).fillna(1.0)
    df["_off"] = ((jump > 50) | (jump < 1 / 50) | (df["resq"] > 1e5 * kk)).astype(np.int8)
    broken = df.groupby("mint")["_off"].cummax().to_numpy().astype(bool)
    meta = load_features()
    if (df["_chain"] == "rh").any():                          # each chain's launch facts, same columns (rh/meta.py)
        from ..rh.meta import load_features as rh_features
        rmeta = rh_features()
        with_q = rmeta.merge(_rh_quote_classes(), on="mint", how="left") if len(rmeta) else rmeta
        meta = pd.concat([meta, with_q], ignore_index=True)
    df = df.merge(meta, on="mint", how="left")
    df["meta_known"] = df["ttg_min"].notna().astype(np.float32) if "ttg_min" in df else np.float32(0.0)
    H = horizon_min * 60.0
    ts = _epoch_s(df["ts"]); cl = df["close"].to_numpy(); op = df["open"].to_numpy()
    mints = df["mint"].to_numpy(); starts = np.r_[0, np.flatnonzero(mints[1:] != mints[:-1]) + 1, len(mints)]
    fwd = np.full(len(df), np.nan); fwdp = np.full(len(df), np.nan)
    hold_list = sorted({int(h) for h in (holds or [])}); fh = {h: np.full(len(df), np.nan) for h in hold_list}
    ec = _label_exit_cost(df) if fee is None else None                 # one-side cost fraction per minute, at the size the book trades
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
        for hm in hold_list:                         # the same label for another hold: exit at the last close at or before t + hold
            jh = np.searchsorted(t, t + hm * 60.0, side="right") - 1 + s
            kh = (1 - ec[s:e]) * (1 - ec[jh]) if fee is None else 1 - fee
            v = cl[jh] / entry * kh - 1
            if e - s > 2:
                v[spike[jh - s]] = np.nan
            fh[hm][s:e] = v
    hod = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hod_s"], df["hod_c"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    df["age_known"] = df["age_h"].notna().astype(np.float32)
    df["mayhem"] = df["mayhem"].astype(float) if "mayhem" in df else 0.0
    last_ts = df["ts"].max()
    df["chain_rh"] = (df["_chain"] == "rh").astype(np.float32)
    qc = df["quote_class"] if "quote_class" in df else pd.Series(None, index=df.index)
    for c, cls in (("qc_stable", "stable"), ("qc_btc", "btc"), ("qc_stock", "stock")):
        df[c] = (qc == cls).astype(np.float32)
    vol_line = np.where(kk == 1.0, math.log1p(MIN_VOL_15M_SOL), np.log1p(MIN_VOL_15M_SOL * kk))      # Solana: exactly today's threshold
    elig = (np.isfinite(df["resq"]) & (df["resq"] >= MIN_RESQ_SOL * kk) & (df["logvol_15m"] >= vol_line)
            & np.isfinite(fwd) & (df["ts"] < last_ts - pd.Timedelta(minutes=horizon_min + 5))).to_numpy() & ~broken
    for hm in hold_list:                             # no label where that hold runs past the data
        fh[hm][(df["ts"] >= last_ts - pd.Timedelta(minutes=hm + 5)).to_numpy()] = np.nan
    chain = df["_chain"].to_numpy()[elig]
    df = df[elig].reset_index(drop=True); fwd = fwd[elig]; fwdp = fwdp[elig]; ts = ts[elig]
    X = df[X_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return DecisionSet(X=X, y=(fwd > label_thr).astype(np.int8), fwd=fwd.astype(np.float32), fwd_pess=fwdp.astype(np.float32),
                       day=df["ts"].dt.date.to_numpy(), ts=ts, mint=df["mint"].to_numpy(), cols=list(X_COLS), horizon_s=H,
                       fwd_h={hm: fh[hm][elig].astype(np.float32) for hm in hold_list}, chain=None if (chain == "sol").all() else chain.astype(object))


def _rh_quote_classes() -> pd.DataFrame:
    from ..db.connection import transaction
    with transaction() as conn:
        rows = conn.execute("SELECT mint, quote_class FROM rh_meta").fetchall()
    return pd.DataFrame(rows, columns=["mint", "quote_class"]) if rows else pd.DataFrame(columns=["mint", "quote_class"])


def taken_idx(ts: np.ndarray, mint: np.ndarray, horizon_s: float, idx: np.ndarray) -> np.ndarray:
    """Of the candidate row indices ``idx``, the ones a picker takes: one position per token, re-entry only once the hold
    has passed (the rule of ``trades_from_picks``), in (mint, ts) order. Vectorised: each taken pick jumps to the token's
    first candidate at least ``horizon_s`` later, so the loop runs once per trade of the busiest token, not once per row."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return idx
    idx = idx[np.lexsort((ts[idx], mint[idx]))]
    t = np.asarray(ts[idx], dtype=np.float64); m = mint[idx]
    hv = np.asarray(horizon_s, dtype=np.float64)       # a scalar, or one hold per row of ``ts`` (each position exits after its own)
    hs = np.full(len(idx), float(hv)) if hv.ndim == 0 else hv[idx]
    if not (hs > 0).all():         # a zero hold lets a token re-enter at the same minute: the frontier below would never advance
        raise ValueError(f"{int((hs <= 0).sum())} of {len(hs)} picked rows have a hold of zero or less; a position must be held for a positive time")
    first = np.r_[True, m[1:] != m[:-1]]; gid = np.cumsum(first) - 1; starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(idx)]
    span = float(t.max() - t.min()) + float(hs.max()) + 1.0
    key = gid * span + (t - t.min())                  # sorted: (token, time) in one exact float64 key
    nxt = np.searchsorted(key, key + hs, side="left")
    has_next = nxt < ends[gid]
    out = []; frontier = starts
    while len(frontier):
        out.append(frontier)
        frontier = nxt[frontier[has_next[frontier]]]
    return idx[np.sort(np.concatenate(out))]


def taken_rows(ds: DecisionSet, pick: np.ndarray, hold_s=None) -> np.ndarray:
    """Row indices of the trades a picker takes (the same rule as ``trades_from_picks``); ``hold_s``: per-row holds."""
    return taken_idx(ds.ts, ds.mint, ds.horizon_s if hold_s is None else hold_s, np.flatnonzero(pick))


def trades_from_picks(ds: DecisionSet, pick: np.ndarray, returns: np.ndarray | None = None, with_days: bool = False, hold_s=None):
    """Net returns of the trades a picker would take: one position per token, re-entry only after the hold (``hold_s``:
    per-row holds, each position exits after its own). With ``with_days`` also returns the entry day of each trade (so
    per-day tables respect holds across midnight)."""
    r_all = ds.fwd_pess if returns is None else returns
    tr = taken_idx(ds.ts, ds.mint, ds.horizon_s if hold_s is None else hold_s, np.flatnonzero(pick))
    r = np.asarray(r_all[tr], dtype=np.float32)
    return (r, np.asarray(ds.day[tr], dtype=object)) if with_days else r


def random_trades(ds: DecisionSet, rows: np.ndarray, n_picks: int, seeds: int = 20, hold_s=None, returns: np.ndarray | None = None) -> np.ndarray:
    """The no-skill baseline: ``n_picks`` random minutes among ``rows`` (the model's pick count on the same eligible
    minutes), traded with the same hold rule, fills and costs; ``seeds`` draws pooled."""
    idx = np.flatnonzero(rows)
    if n_picks <= 0 or len(idx) == 0:
        return np.array([], dtype=np.float32)
    out = []
    for s in range(seeds):
        pick = np.zeros(len(ds.y), bool); pick[np.random.default_rng(s).choice(idx, min(n_picks, len(idx)), replace=False)] = True
        out.append(trades_from_picks(ds, pick, returns=returns, hold_s=hold_s))
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


def evaluate(ds: DecisionSet, scores: np.ndarray, test: np.ndarray, threshold, label: str = "", hold_s=None, returns: np.ndarray | None = None) -> dict:
    """Trades on rows of ``test`` whose score ≥ threshold (a scalar or per-row lines); metrics per day and pooled."""
    pick = test & (scores >= threshold); r, rdays = trades_from_picks(ds, pick, returns=returns, with_days=True, hold_s=hold_s)
    per_day = {}
    for d in sorted(set(ds.day[test].tolist())):
        per_day[str(d)] = summarize(r[rdays == d])
    pooled = summarize(r); pooled["days_positive"] = sum(1 for v in per_day.values() if v["mean"] is not None and v["mean"] > 0); pooled["days"] = len(per_day)
    return {"label": label, "pooled": pooled, "per_day": per_day}
