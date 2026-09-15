"""Diagnostic walk-forward of the selector (``fly-trader selector-eval``): the same folds as the strategy gate
(``train/selector.fold``: decision points, purge, size guards, scaler and model), plus close-fill vs pessimistic-fill
returns at the default buy line, what the picks look like, and permutation importance on the last fold."""
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fly_trader.train.decisions import build, summarize, trades_from_picks
from fly_trader.train.selector import fold


def _pct(v) -> str:
    return f"{v*100:+.2f}%" if v is not None else "-"


def main(days: int | None = None, test_days: int = 21) -> None:
    t0 = time.time()
    ds = build(days=days)
    print(f"decision points {len(ds.y):,} over {len(ds.days)} days, universe mean {ds.fwd_pess.mean()*100:+.2f}% ({time.time()-t0:.0f}s)", flush=True)
    close_r, pess_r, picked = [], [], []; last = None
    for k, D in enumerate(ds.days[-test_days:]):
        fo = fold(ds, D, seed=k)
        if fo is None:
            print(f"{D}: skipped (train or test too small)", flush=True); continue
        last = fo; pick = fo.test & (fo.scores >= fo.model.threshold)
        rc = trades_from_picks(ds, pick, returns=ds.fwd); rp = trades_from_picks(ds, pick)
        close_r.append(rc); pess_r.append(rp); picked.append(np.flatnonzero(pick))
        sm = summarize(rp)
        print(f"{D}: IC {fo.ic if fo.ic is not None else float('nan'):.3f} | n={sm['n']:4d} close-fill mean {_pct(summarize(rc)['mean'])} | "
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
    if len(idx):
        m = last.model
        imp = permutation_importance(m.gbm, m.scaler.transform(ds.X[idx]), np.clip(ds.fwd_pess[idx], -1, 1), n_repeats=3, random_state=0)
        print(f"\npermutation importance (R² drop, last fold {last.day}):"); print(pd.Series(imp.importances_mean, index=ds.cols).sort_values(ascending=False).head(12).round(4).to_string())
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
