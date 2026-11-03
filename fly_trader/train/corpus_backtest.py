"""Walk-forward event backtest of rule strategies on the corpus feature rows (``train/corpus_features.py``).

Per token, rows are the minutes it traded after graduation. A candidate entry at row i fills against the pool at
that minute's close, ``close·(1+impact)/(1−fee)`` (an AMM always fills at its current price); exits are evaluated
on every traded minute's close (trailing stop from the running peak, hard stop from the cost basis, time stop)
and fill at that close ``·(1−impact)·(1−fee)``. Busy minutes can move within the seconds it takes to land, so this
is mildly optimistic there; sparse minutes are exact. Impact is constant-product
against the reserve approximation carried in the rows. One position per token, cooldown after an exit.

Folds are graduation days: the first ``select_frac`` of days choose the configuration (highest mean net return
with at least ``min_trades`` trades), the remaining days score it; a random-entry baseline with the same entry
rate and exits is reported for the test days. Strategies live in ``STRATEGIES``; each is a list of named
candidate masks over the raw feature columns, crossed with the exit grid.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config
from ..market.features import FEATURE_VERSION
from .corpus_features import _epoch_s
from .corpus_meta import load_features

SIZE = config.MAX_POSITION_SOL
MIN_RESQ_SOL = 20.0      # entries only into pools with at least this much SOL (the live executability gate)
EXITS = list(itertools.product((0.05, 0.10, 0.20), (0.03, 0.08), (30, 120)))     # trail, hard, max hold (min)


def load(feature_dir=None, max_tokens: int | None = None, days: int | None = None, universe: str = "graduation") -> pd.DataFrame:
    root = feature_dir or (config.CORPUS_DIR / "features_mature" if universe == "mature" else config.CORPUS_FEATURES_DIR)
    files = sorted(glob.glob(str(root / "*" / "part*.parquet")))
    if days:
        cutoff = (pd.Timestamp.utcnow() - timedelta(days=days)).date().isoformat()
        files = [f for f in files if f.split("/")[-2] >= cutoff]
    if not files:
        raise SystemExit("no feature parts found")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df["day"] = df["ts"].dt.floor("D")
    # folds: graduation day for the graduation universe, calendar day for the mature universe
    df["grad_day"] = df["day"].dt.date.astype(str) if universe == "mature" else df.groupby("mint")["ts"].transform("min").dt.date.astype(str)
    if max_tokens:
        keep = df["mint"].drop_duplicates().iloc[:max_tokens]
        df = df[df["mint"].isin(keep)]
    # drop tokens whose series mixes price scales (secondary pools in other quotes) or carries impossible reserves
    df = df.sort_values(["mint", "ts"])
    jump = (df["close"] / df.groupby("mint")["close"].shift(1)).fillna(1.0)
    off = ((jump > 50) | (jump < 1 / 50) | (df["resq"] > 100000)).astype(np.int8)
    df["broken"] = off.groupby(df["mint"]).cummax().astype(bool)     # entries blocked from a token's first scale break on; exits still see the prices
    try:
        stale = sum(1 for f in files if (pq.read_schema(f).metadata or {}).get(b"fly_version", b"0") != str(FEATURE_VERSION).encode())
        if stale:
            print(f"(warning: {stale} of {len(files)} feature parts come from another feature version; rebuild for exact numbers)")
    except Exception as e:
        print(f"(warning: feature versions not checked: {e})")
    try:
        meta = load_features()
        if len(meta):
            df = df.merge(meta, on="mint", how="left")
    except Exception as e:
        print(f"(corpus_meta not merged: {e})")
    return df.sort_values(["mint", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------- candidate rules (raw feature units)
def _s1(d: pd.DataFrame, r: float, k: float, a: float) -> np.ndarray:
    v15, v1h = np.expm1(d["logvol_15m"].to_numpy()), np.expm1(d["logvol_1h"].to_numpy())
    return ((d["dd_1h"].to_numpy() >= -0.002) & (d["ret_15m"].to_numpy() >= r) & (v15 >= k * v1h / 4.0) & (d["vol_divergence"].to_numpy() > 0)
            & (d["age_h"].to_numpy() >= a) & ~d["has_trades"].to_numpy())


def _s2(d: pd.DataFrame, g: float, i: float, a: float) -> np.ndarray:
    s15, s1h = np.expm1(d["logsigners_15m"].to_numpy()), np.expm1(d["logsigners_1h"].to_numpy())
    return ((s15 >= g * s1h / 4.0) & (s15 >= 10) & (d["imb_15m"].to_numpy() >= i) & (d["net_buyers_1h"].to_numpy() > 0) & (d["ret_5m"].to_numpy() > 0)
            & (d["age_h"].to_numpy() >= a) & d["has_trades"].to_numpy())


def _s3(d: pd.DataFrame, top: float, pre_top: float, hc: float) -> np.ndarray:
    pre = d["pre_top10_pct"].to_numpy()
    return ((d["age_h"].to_numpy() >= 1.0) & (d["top_holders_pct"].to_numpy() <= top) & (np.isnan(pre) | (pre <= pre_top))
            & (d["holder_change_1h"].to_numpy() >= hc) & (d["net_buyers_1h"].to_numpy() > 0) & (d["ret_5m"].to_numpy() > 0)
            & (d["imb_5m"].to_numpy() > 0) & (d["dd_1h"].to_numpy() >= -0.05) & d["has_trades"].to_numpy())


def _grad_price(d: pd.DataFrame) -> np.ndarray:
    """First post-graduation close per mint, from candle rows only (sampled tokens also carry trade-path rows)."""
    c = d["close"].where(~d["has_trades"])
    return c.groupby(d["mint"]).transform("first").to_numpy()


def _s3b(d: pd.DataFrame, r: float, a_lo: float) -> np.ndarray:
    p0 = _grad_price(d)
    age = d["age_h"].to_numpy()
    return ((age >= a_lo) & (age <= 1.0) & (d["close"].to_numpy() >= p0) & (d["ret_15m"].to_numpy() >= r) & (d["logvol_15m"].to_numpy() >= math.log1p(5.0))
            & ~d["has_trades"].to_numpy())


def _s4(d: pd.DataFrame, age_min: float, rel_min: float, ret15_min: float) -> np.ndarray:
    p0 = _grad_price(d)
    return ((d["age_h"].to_numpy() * 60 >= age_min) & (d["close"].to_numpy() >= rel_min * p0) & (d["ret_15m"].to_numpy() >= ret15_min) & ~d["has_trades"].to_numpy())


EXITS_S4 = [(1.0, 1.0, 30), (1.0, 1.0, 60), (1.0, 1.0, 120), (0.20, 0.15, 60), (0.20, 0.15, 120), (0.10, 0.08, 60), (0.30, 0.25, 120)]   # (trail, hard, hold); 1.0 = never
def _s5(d: pd.DataFrame, ttg: float, dev_max: float, need_grad: bool, rug_max: float) -> np.ndarray:
    """Survivor entry restricted to organic launches and creators with a usable record (point-in-time columns)."""
    base = _s4(d, 60, 1.0, -1.0)
    m = base & (d["ttg_min"].to_numpy() >= ttg) & (d["dev_sol"].fillna(0).to_numpy() <= dev_max)
    if need_grad:
        m &= d["prior_grads"].fillna(0).to_numpy() >= 1
    if rug_max < 1.0:
        m &= ~(d["prior_rug_share"].fillna(0).to_numpy() > rug_max)
    return m


def _s6(d: pd.DataFrame, age_min_h: float, r1h: float, imb: float, dd: float) -> np.ndarray:
    """Mature momentum: token at least age_min_h old (or older than the archive), 1 h return and 15 m buy imbalance positive, near the 1 h high."""
    age = d["age_h"].to_numpy(); old = np.isnan(age) | (age >= age_min_h)
    return (old & (d["ret_1h"].to_numpy() >= r1h) & (d["ret_15m"].to_numpy() > 0) & (d["imb_15m"].to_numpy() >= imb) & (d["dd_1h"].to_numpy() >= dd)
            & (d["logvol_15m"].to_numpy() >= math.log1p(5.0)))


MATURE_STRATEGIES = {
    "S6 mature momentum": lambda d: [(f"age>={a}h ret1h>={r} imb15>={i} dd1h>={dd}", _s6(d, a, r, i, dd)) for a in (12, 48) for r in (0.05, 0.15) for i in (0.0, 0.3) for dd in (-0.05, -0.02)],
    "S6 control: mature, any minute with volume": lambda d: [(f"age>={a}h", (np.isnan(d["age_h"].to_numpy()) | (d["age_h"].to_numpy() >= a)) & (d["logvol_15m"].to_numpy() >= math.log1p(5.0))) for a in (12, 48)],
}
STRATEGIES = {
    "S5 organic survivor (creation-time + creator screens)": lambda d: [(f"ttg>={t}m dev<={dv} prior_grad={pg} rug<={rg}", _s5(d, t, dv, pg, rg)) for t in (10, 30, 60) for dv in (2.0, 1e9) for pg in (False, True) for rg in (0.5, 1.0)],
    "S4 survivor (age since graduation, price vs graduation price)": lambda d: [(f"age>={a}m rel>={r} ret15>={q}", _s4(d, a, r, q)) for a in (60, 120, 180) for r in (1.0, 1.5) for q in (-1.0, 0.0)],
    "S4 control: age only (no price filter)": lambda d: [(f"age>={a}m", _s4(d, a, 0.0, -1.0)) for a in (60, 180)],
    "S1 breakout+volume (candles, all tokens)": lambda d: [(f"r={r} k={k} age>={a}", _s1(d, r, k, a)) for r in (0.03, 0.06, 0.10) for k in (1.5, 3.0) for a in (0.5, 6.0)],
    "S2 participation flow (trade rows)": lambda d: [(f"g={g} i={i} age>={a}", _s2(d, g, i, a)) for g in (1.5, 3.0) for i in (0.2, 0.5) for a in (0.5, 6.0)],
    "S3 quality survivor (trade rows)": lambda d: [(f"top10<={t} pre_top10<={pt} holders_chg>={hc}", _s3(d, t, pt, hc)) for t in (60.0, 80.0) for pt in (90.0, 100.0) for hc in (0.0, 0.2)],
    "S3b graduation survivor (candles)": lambda d: [(f"ret15>={r} age>={a}", _s3b(d, r, a)) for r in (0.0, 0.05) for a in (0.25, 0.5)],
}


# ---------------------------------------------------------------- simulation
def simulate(d: pd.DataFrame, cand: np.ndarray, fee_side: float, trail: float, hard: float, max_hold_min: int, cooldown_min: int = 5) -> np.ndarray:
    """Returns array of (entry fold-day ordinal, net_return) per closed trade. Fills happen against the pool at the signal
    minute's close (an AMM always fills; the close is the pool's current price), entry ``close·(1+impact)/(1−fee)``,
    exit ``close·(1−impact)·(1−fee)``. Trailing and hard stops are checked on every traded minute's close; the time stop
    exits at the last close at or before ``entry + max_hold`` when the next traded minute is past it."""
    mints = d["mint"].to_numpy(); ts = _epoch_s(d["ts"])
    cl, resq = d["close"].to_numpy(), d["resq"].to_numpy()
    gday = pd.to_datetime(d["grad_day"]).map(pd.Timestamp.toordinal).to_numpy()
    imp = SIZE / (SIZE + np.where(np.isfinite(resq) & (resq > 0), resq, 5.0))
    hold_s = max_hold_min * 60; out = []
    starts = np.r_[0, np.flatnonzero(mints[1:] != mints[:-1]) + 1, len(mints)]
    for s, e in zip(starts[:-1], starts[1:]):
        first = np.flatnonzero(cand[s:e])
        if len(first) == 0:
            continue
        pos = False; basis = peak = 0.0; t_in = 0.0; d_in = 0; cd_until = -1.0
        for i in range(s + int(first[0]), e):
            t = ts[i]
            if pos and t - t_in > hold_s:        # time stop fell between traded minutes: the previous row is the last close before it
                j = i - 1
                out.append((d_in, cl[j] * (1 - imp[j]) * (1 - fee_side) / basis - 1)); pos = False; cd_until = t_in + hold_s + cooldown_min * 60
            if pos:
                peak = max(peak, cl[i])
                if (cl[i] / peak - 1 <= -trail) or (cl[i] / basis - 1 <= -hard) or (t - t_in >= hold_s):
                    out.append((d_in, cl[i] * (1 - imp[i]) * (1 - fee_side) / basis - 1)); pos = False; cd_until = t + cooldown_min * 60
            elif cand[i] and t >= cd_until:
                basis = cl[i] * (1 + imp[i]) / (1 - fee_side); peak = cl[i]; t_in = t; d_in = gday[i]; pos = True
        if pos:   # still open at the end of the rows: mark at the last close
            out.append((d_in, cl[e - 1] * (1 - imp[e - 1]) * (1 - fee_side) / basis - 1))
    return np.array(out, dtype=float).reshape(-1, 2)


def stats(r: np.ndarray) -> dict:
    if len(r) == 0:
        return dict(n=0, mean=np.nan, med=np.nan, win=np.nan, pf=np.nan, pnl=0.0, lo=np.nan, hi=np.nan, top3=np.nan)
    g = r[r > 0].sum(); l = -r[r <= 0].sum(); rng = np.random.default_rng(0)
    bs = np.array([rng.choice(r, len(r)).mean() for _ in range(1000)])
    return dict(n=len(r), mean=r.mean(), med=np.median(r), win=(r > 0).mean(), pf=g / l if l > 0 else np.inf, pnl=SIZE * r.sum(),
                lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), top3=SIZE * (r.sum() - np.sort(r)[-3:].sum()) if len(r) >= 3 else np.nan)


def fmt(s: dict) -> str:
    return (f"n={s['n']:5d} mean {s['mean']*100:+6.2f}% [{s['lo']*100:+.2f}..{s['hi']*100:+.2f}] median {s['med']*100:+6.2f}% win {s['win']*100:3.0f}% "
            f"PF {s['pf']:.2f} P&L {s['pnl']:+.3f} SOL (minus top-3 {s['top3']:+.3f})")


def run(df: pd.DataFrame, fee_side: float, select_frac: float = 0.6, min_trades: int = 30) -> None:
    days = sorted(df["grad_day"].unique()); n_sel = max(1, int(len(days) * select_frac))
    sel_days = set(days[:n_sel]); split_ord = pd.Timestamp(days[n_sel]).toordinal() if n_sel < len(days) else 10**9
    n_tok = df["mint"].nunique(); n_tr = df.loc[df["has_trades"], "mint"].nunique()
    print(f"\nfee/side {fee_side:.4f} | {n_tok:,} tokens ({n_tr:,} with trade rows) | {len(df):,} rows | days {days[0]}..{days[-1]} | selection {len(sel_days)} days, test {len(days)-n_sel} days", flush=True)
    gate = np.isfinite(df["resq"].to_numpy()) & (df["resq"].to_numpy() >= MIN_RESQ_SOL) & ~df["broken"].to_numpy()     # executability gate, as live
    for sname, make in STRATEGIES.items():
        rows = []; t0 = time.time()
        for vname, cand in make(df):
            cand = cand & gate
            if not cand.any():
                continue
            for trail, hard, mh in (EXITS_S4 if sname[:2] in ("S4", "S5", "S6") else EXITS):
                tr = simulate(df, cand, fee_side, trail, hard, mh)
                sel, tst = tr[tr[:, 0] < split_ord, 1], tr[tr[:, 0] >= split_ord, 1]
                rows.append((vname, trail, hard, mh, stats(sel), stats(tst), cand))
        if not rows:
            print(f"\n=== {sname}: no candidates"); continue
        ok = [r for r in rows if r[4]["n"] >= min_trades] or rows
        best = max(ok, key=lambda r: r[4]["mean"] if np.isfinite(r[4]["mean"]) else -9)
        vname, trail, hard, mh, ss, st, cand = best
        tested = [r for r in rows if r[5]["n"] >= 10]
        pos_frac = np.mean([r[5]["mean"] > 0 for r in tested]) if tested else np.nan
        p_sig = cand.sum() / max(1, gate.sum()); rb = []          # entry rate among the rows the gate lets through
        for seed in range(5):
            rc = (np.random.default_rng(seed).random(len(df)) < p_sig) & gate
            trr = simulate(df, rc, fee_side, trail, hard, mh); rb.append(stats(trr[trr[:, 0] >= split_ord, 1])["mean"])
        print(f"\n=== {sname} ===  {len(rows)} configs in {time.time()-t0:.0f}s; share positive on TEST (n>=10): {pos_frac*100:.0f}%")
        print(f"  chosen on selection: {vname} | trail {trail*100:.0f}% hard {hard*100:.0f}% max hold {mh} min")
        print(f"  selection: {fmt(ss)}\n  TEST     : {fmt(st)}")
        print(f"  random entries, same rate & exits, TEST mean: {np.nanmean(rb)*100:+.2f}%")
        for r in sorted(tested, key=lambda r: -r[5]["mean"])[:3]:
            print(f"    best-on-test (reference): {r[0]} trail {r[1]*100:.0f}% hard {r[2]*100:.0f}% hold {r[3]}m -> test n={r[5]['n']} mean {r[5]['mean']*100:+.2f}% PF {r[5]['pf']:.2f}; selection {r[4]['mean']*100:+.2f}%")
        sys.stdout.flush()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--max-tokens", type=int, default=None); ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fees", default="0.003,0.0055"); ap.add_argument("--only", default=None, help="substring filter on strategy names")
    ap.add_argument("--universe", default="graduation", choices=["graduation", "mature"]); a = ap.parse_args(argv)
    df = load(max_tokens=a.max_tokens, days=a.days, universe=a.universe)
    global STRATEGIES
    if a.universe == "mature":
        STRATEGIES = dict(MATURE_STRATEGIES)
    if a.only:
        STRATEGIES = {k: v for k, v in STRATEGIES.items() if a.only.lower() in k.lower()}
    for fee in (float(x) for x in a.fees.split(",")):
        run(df, fee)


if __name__ == "__main__":
    main()
