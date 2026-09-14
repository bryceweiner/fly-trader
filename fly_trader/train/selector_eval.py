"""Walk-forward evaluation of the mature-universe selector (the strategy candidate found 2026-09-14).

Per test day D: a gradient-boosted classifier is fit on eligible minutes from days before D−1 (one-day purge),
label = 30-minute forward return net of a 0.6 % round trip above +3 %; the top 1 % of D's minutes by score are
entered (one position per token, 30-minute hold), filled pessimistically at the next traded minute's open.
Eligible = pump.fun-origin PumpSwap tokens of any age with ≥ 20 SOL in the pool and ≥ 5 SOL of 15-minute volume.
First run (22 days, 9 test days): +5.8 % mean, +5.7 % median, 88 % win, PF 1.78, 1,237 trades, 9/9 days positive.
Run: ``fly-trader selector-eval`` (rebuild the mature universe first with ``build-mature``)."""
import glob, math, time, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from fly_trader import config
from fly_trader.market.features import FEATURES
from fly_trader.train.corpus_meta import load_features
t0 = time.time(); FEE = 0.006; H = 30 * 60
files = sorted(glob.glob(str(config.CORPUS_DIR / "features_mature" / "*" / "part.parquet")))
cols = ["mint", "ts", "open", "close", "resq", "age_h", "traders_15m", "traders_1h", "n_trades_1m"] + FEATURES
df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True).sort_values(["mint", "ts"]).reset_index(drop=True)
jump = (df["close"] / df.groupby("mint")["close"].shift(1)).fillna(1.0)
df["_off"] = ((jump > 50) | (jump < 1 / 50) | (df["resq"] > 1e5)).astype(np.int8)
df = df[~df.groupby("mint")["_off"].cummax().astype(bool)].reset_index(drop=True)     # past-only: rows before a scale break keep their labels
df = df.merge(load_features(), on="mint", how="left")
ts = ((df["ts"] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(); cl = df["close"].to_numpy(); op = df["open"].to_numpy(); mints = df["mint"].to_numpy()
fwd = np.full(len(df), np.nan); entry_pess = cl.copy(); gap_next = np.full(len(df), np.nan)
starts = np.r_[0, np.flatnonzero(mints[1:] != mints[:-1]) + 1, len(mints)]
for s, e in zip(starts[:-1], starts[1:]):
    t = ts[s:e]; j = np.searchsorted(t, t + H, side="right") - 1 + s
    fwd[s:e] = cl[j] / cl[s:e] * (1 - FEE) - 1
    nxt = np.arange(s + 1, e + 1); ok = nxt < e
    g = np.full(e - s, np.nan); g[ok] = t[1:][ok[:-1]] - t[:-1][ok[:-1]] if (e - s) > 1 else g[ok]
    gap_next[s:e] = g
    pe = cl[s:e].copy(); m = ok & (g <= 120)
    pe[m] = op[nxt[m]]
    entry_pess[s:e] = pe
fwd_pess = cl[np.clip(np.searchsorted(ts, ts + H, side="right") - 1, 0, len(ts) - 1)]  # placeholder, recomputed per token below
fwdp = np.full(len(df), np.nan)
for s, e in zip(starts[:-1], starts[1:]):
    t = ts[s:e]; j = np.searchsorted(t, t + H, side="right") - 1 + s
    fwdp[s:e] = cl[j] / entry_pess[s:e] * (1 - FEE) - 1
df["fwd"], df["fwd_pess"], df["gap_next"] = fwd, fwdp, gap_next
df["y"] = (fwd > 0.03).astype(int); df["day"] = df["ts"].dt.date
hod = df["ts"].dt.hour + df["ts"].dt.minute / 60; df["hod_s"], df["hod_c"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24); df["age_known"] = df["age_h"].notna().astype(float)
elig = np.isfinite(df["resq"]) & (df["resq"] >= 20) & (df["logvol_15m"] >= math.log1p(5.0)) & np.isfinite(df["fwd"]) & (df["ts"] < df["ts"].max() - pd.Timedelta(minutes=35))
df = df[elig].reset_index(drop=True)
X_cols = FEATURES + ["traders_15m", "traders_1h", "n_trades_1m", "hod_s", "hod_c", "age_known", "ttg_min", "dev_sol", "dev_share", "prior_launches", "prior_grads", "prior_rug_share", "mayhem"]
X = df[X_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(); y = df["y"].to_numpy()
days = sorted(df["day"].unique()); print(f"eligible rows {len(df):,} ({time.time()-t0:.0f}s)", flush=True)
def trades(sub, pick, col):
    out = []
    for mint, x in sub[pick].groupby("mint", sort=False):
        t = ((x["ts"] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(); r = x[col].to_numpy(); last = -1e18
        for i in range(len(t)):
            if t[i] >= last + H: out.append(r[i]); last = t[i]
    return np.array(out)
tot = {"fwd": [], "fwd_pess": []}; picks_all = []
for k, D in enumerate(days[-9:]):
    train = (df["day"] < D - pd.Timedelta(days=1)).to_numpy(); test = (df["day"] == D).to_numpy()
    tr_idx = np.flatnonzero(train); rng = np.random.default_rng(k)
    if len(tr_idx) > 2_500_000: tr_idx = rng.choice(tr_idx, 2_500_000, replace=False)
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6; Xn = (X - mu) / sd
    gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(Xn[tr_idx], y[tr_idx])
    s_te = gbm.predict_proba(Xn[test])[:, 1]; thr = np.quantile(gbm.predict_proba(Xn[tr_idx])[:, 1], 0.99)
    sub = df[test]; pick = s_te >= thr; picks_all.append(sub[pick])
    r1, r2 = trades(sub, pick, "fwd"), trades(sub, pick, "fwd_pess"); tot["fwd"].append(r1); tot["fwd_pess"].append(r2)
    print(f"{D}: top1% n={len(r1):4d} close-fill mean {r1.mean()*100:+6.2f}% | next-open fill mean {r2.mean()*100:+6.2f}% med {np.median(r2)*100:+5.2f}% win {(r2>0).mean()*100:3.0f}% | share with a trade in the next 2 min {np.isfinite(sub[pick].gap_next.to_numpy()).mean()*100:.0f}% (median gap {np.nanmedian(sub[pick].gap_next):.0f}s)", flush=True)
for k_, v in tot.items():
    r = np.concatenate(v); print(f"\nALL top1% [{k_}]: n={len(r)} mean {r.mean()*100:+.2f}% median {np.median(r)*100:+.2f}% win {(r>0).mean()*100:.0f}% PF {r[r>0].sum()/max(-r[r<=0].sum(),1e-9):.2f} | p10 {np.percentile(r,10)*100:+.1f}% p90 {np.percentile(r,90)*100:+.1f}%")
P = pd.concat(picks_all); A = df
show = ["ret_1m", "ret_5m", "ret_15m", "ret_1h", "imb_5m", "imb_15m", "logvol_15m", "logn_15m", "traders_15m", "dd_1h", "overext_1h", "resq", "age_h", "n_trades_1m"]
print("\npicked minutes vs all eligible (medians):"); print(pd.DataFrame({"picked": P[show].median(), "all": A[show].median()}).round(4).to_string())
print("picked: age known %.0f%% | median age h %.1f | hour-of-day distribution (UTC): %s" % (P.age_known.mean()*100, P.age_h.median(), P["ts"].dt.hour.value_counts().sort_index().to_dict()))
imp = permutation_importance(gbm, Xn[test][:30000], y[test][:30000], n_repeats=3, random_state=0, scoring="roc_auc")
top = pd.Series(imp.importances_mean, index=X_cols).sort_values(ascending=False).head(12); print("\npermutation importance (AUC drop, last fold):"); print(top.round(4).to_string())
print(f"\n({time.time()-t0:.0f}s)")
