"""Diagnostic walk-forward of the selector (``fly-trader selector-eval``): the same folds as the strategy gate
(``train/selector.fold``: decision points, labels, purge, size guards and model), plus close-fill vs pessimistic-fill
returns, what the picks look like, and permutation importance on the last fold. Uses ``train/decisions.py`` so its
numbers can never drift from the strategy gate's."""
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fly_trader.train.decisions import build, summarize, trades_from_picks
from fly_trader.train.selector import fold


def _pct(v) -> str:
    return f"{v*100:+.2f}%" if v is not None else "-"


def main(days: int = 45, test_days: int = 9, top_frac: float = 0.01) -> None:
    t0 = time.time()
    ds = build(days=days)
    print(f"decision points {len(ds.y):,} over {len(ds.days)} days, base rate {ds.y.mean()*100:.1f}%, universe mean {ds.fwd.mean()*100:+.2f}% ({time.time()-t0:.0f}s)", flush=True)
    close_r, pess_r, picked = [], [], []; last = None
    for k, D in enumerate(ds.days[-test_days:]):
        fo = fold(ds, D, top_frac, seed=k)
        if fo is None:
            print(f"{D}: skipped (train or test too small, or one class)", flush=True); continue
        last = fo
        rc = trades_from_picks(ds, fo.pick, returns=ds.fwd); rp = trades_from_picks(ds, fo.pick)
        close_r.append(rc); pess_r.append(rp); picked.append(np.flatnonzero(fo.pick))
        sm = summarize(rp)
        print(f"{D}: AUC {fo.auc if fo.auc is not None else float('nan'):.3f} | n={sm['n']:4d} close-fill mean {_pct(summarize(rc)['mean'])} | "
              f"next-open fill mean {_pct(sm['mean'])} median {_pct(sm['median'])} win {_pct(sm['win'])}", flush=True)
    for name, rs in (("close fill", close_r), ("next-open fill", pess_r)):
        r = np.concatenate(rs) if rs else np.array([]); sm = summarize(r)
        if not sm["n"]:
            print(f"\nALL [{name}]: no trades"); continue
        print(f"\nALL [{name}]: n={sm['n']} mean {_pct(sm['mean'])} median {_pct(sm['median'])} win {_pct(sm['win'])} PF {sm['pf'] if sm['pf'] is not None else float('inf'):.2f} "
              f"| p10 {np.percentile(r, 10)*100:+.1f}% p90 {np.percentile(r, 90)*100:+.1f}%")
    X = pd.DataFrame(ds.X, columns=ds.cols); sel = np.concatenate(picked) if picked else np.array([], dtype=int)
    show = [c for c in ["ret_1m", "ret_5m", "ret_15m", "ret_1h", "imb_5m", "imb_15m", "logvol_15m", "logn_15m", "traders_15m", "dd_1h", "overext_1h", "log_liquidity_sol", "log_age_h", "n_trades_1m"] if c in ds.cols]
    if len(sel):
        print("\npicked minutes vs all decision points (medians):"); print(pd.DataFrame({"picked": X.iloc[sel][show].median(), "all": X[show].median()}).round(4).to_string())
    idx = np.flatnonzero(last.test)[:30000] if last is not None else np.array([], dtype=int)
    if len(idx) and len(np.unique(ds.y[idx])) == 2:
        m = last.model
        imp = permutation_importance(m.gbm, (ds.X[idx] - m.mean) / m.std, ds.y[idx], n_repeats=3, random_state=0, scoring="roc_auc")
        print(f"\npermutation importance (AUC drop, last fold {last.day}):"); print(pd.Series(imp.importances_mean, index=ds.cols).sort_values(ascending=False).head(12).round(4).to_string())
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
