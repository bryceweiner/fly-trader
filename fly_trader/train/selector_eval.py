"""Diagnostic walk-forward of the selector (``fly-trader selector-eval``): the same decision points, labels, purge and
model as ``train/selector.py``, plus close-fill vs pessimistic-fill returns, what the picks look like, and permutation
importance on the last fold. Uses ``train/decisions.py`` so its numbers can never drift from the strategy gate's.
Honest run on 2026-09-14 (46 days, 9 test days, after the look-ahead fixes): 993 trades, +1.9 % mean, +5.0 % median,
96 % win, PF 1.52, 7/9 days positive."""
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from fly_trader.train.decisions import build, summarize, trades_from_picks
from fly_trader.train.selector import fit

t0 = time.time()
ds = build(days=45)
days = ds.days
print(f"decision points {len(ds.y):,} over {len(days)} days, base rate {ds.y.mean()*100:.1f}%, universe mean {ds.fwd.mean()*100:+.2f}% ({time.time()-t0:.0f}s)", flush=True)
close_r, pess_r, picked = [], [], []
for k, D in enumerate(days[-9:]):
    train = ds.day < (D - timedelta(days=1)); test = ds.day == D
    m = fit(ds, train, 0.01, seed=k); s = np.zeros(len(ds.y)); s[test] = m.score(ds.X[test])
    pick = test & (s >= m.threshold)
    rc = trades_from_picks(ds, pick, returns=ds.fwd); rp = trades_from_picks(ds, pick)
    close_r.append(rc); pess_r.append(rp); picked.append(np.flatnonzero(pick))
    sm = summarize(rp)
    print(f"{D}: AUC {roc_auc_score(ds.y[test], s[test]):.3f} | n={sm['n']:4d} close-fill mean {rc.mean()*100 if len(rc) else 0:+6.2f}% | next-open fill mean {(sm['mean'] or 0)*100:+6.2f}% "
          f"median {(sm['median'] or 0)*100:+5.2f}% win {(sm['win'] or 0)*100:3.0f}%", flush=True)
for name, rs in (("close fill", close_r), ("next-open fill", pess_r)):
    r = np.concatenate(rs); sm = summarize(r)
    print(f"\nALL [{name}]: n={sm['n']} mean {sm['mean']*100:+.2f}% median {sm['median']*100:+.2f}% win {sm['win']*100:.0f}% PF {sm['pf'] if sm['pf'] is not None else float('inf'):.2f} "
          f"| p10 {np.percentile(r, 10)*100:+.1f}% p90 {np.percentile(r, 90)*100:+.1f}%")
X = pd.DataFrame(ds.X, columns=ds.cols); P = X.iloc[np.concatenate(picked)]
show = [c for c in ["ret_1m", "ret_5m", "ret_15m", "ret_1h", "imb_5m", "imb_15m", "logvol_15m", "logn_15m", "traders_15m", "dd_1h", "overext_1h", "log_liquidity_sol", "log_age_h", "n_trades_1m"] if c in ds.cols]
print("\npicked minutes vs all decision points (medians):"); print(pd.DataFrame({"picked": P[show].median(), "all": X[show].median()}).round(4).to_string())
test = ds.day == days[-1]; idx = np.flatnonzero(test)[:30000]
imp = permutation_importance(m.gbm, (ds.X[idx] - m.mean) / m.std, ds.y[idx], n_repeats=3, random_state=0, scoring="roc_auc")
print("\npermutation importance (AUC drop, last fold):"); print(pd.Series(imp.importances_mean, index=ds.cols).sort_values(ascending=False).head(12).round(4).to_string())
print(f"\n({time.time()-t0:.0f}s)")
