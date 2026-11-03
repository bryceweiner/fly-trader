"""Strategy study, step 2 -- saves NO model (the live selector switches itself to the newest saved one): the trainer's
own strategy fit (train/strategies.fit_strategy: hold, line and trigger thresholds chosen on the selection half; each
hold's out-of-sample scores from its own walk-forward) for capitulation, the strategy that trades, with the inputs the
deployed selector kept (rug, flow, skill), judged as the trainer judges (evaluation half) and on the 21-day holdout;
then the compared selector's own decisions on the holdout days after its training. No ev walk-forward: over every row
it costs the live console its memory. Reads step 1's decision set from disk, read-only.

    --holds 240              holds fitted by walk-forward (~35 min each on 6 threads; cached in data/corpus/stack_cache)
    --compare <snapshot id>  the selector to judge on the holdout (default: the one the live book trades)

Writes summary.json, cap.npz (every capitulation candidate: features, labels for every hold, scores for the fitted
holds), book.npz (the fresh capitulation book over every out-of-sample day), book_compare.npz and holdout_compare.json
into logs/strategy_study; tools/strategy_study/variants.py reads them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
OUT = REPO / "logs" / "strategy_study"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=[logging.FileHandler(OUT / "study.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("study")

import guard  # noqa: E402
guard.start(str(OUT / "guard.log"))

from fly_trader.train import progress  # noqa: E402
_last = [None]


def _stage(stage, *a, **k):                       # the console's training card must not show a research run: stages go to the log
    if stage != _last[0]:
        _last[0] = stage; log.info("stage: %s", stage)


progress.update = _stage
from fly_trader.agent.selector_session import pinned_snapshot  # noqa: E402
from fly_trader.db.connection import transaction  # noqa: E402
from fly_trader.train import decisions, selector, strategies  # noqa: E402
from fly_trader.train.decisions import taken_idx  # noqa: E402

EXTRA_HOLDS = (360, 480)
FEATURES = ("log_liquidity_sol", "logvol_15m", "logvol_1h", "logn_1h", "log_age_h", "traders_15m", "traders_1h", "mkt_vol_1h", "ret_1m", "ret_15m",
            "ret_1h", "ret_3h", "dd_1h", "dd_3h", "buyers_ret", "top_sell_share_1m", "imb_15m", "wash_share_15m", "exit_cost_0p1", "rvol_1h")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return f"array{x.shape}"
    if isinstance(x, date):
        return str(x)
    return x if isinstance(x, (str, int, float, bool, type(None))) else repr(x)[:200]


def load_ds() -> decisions.DecisionSet:
    R = np.load(OUT / "ds_rest.npz")
    od = R["day_ord"]; u = np.unique(od)
    day = np.array([date.fromordinal(int(o)) for o in u], dtype=object)[np.searchsorted(u, od)]
    ds = decisions.DecisionSet(X=np.load(OUT / "ds_X.npy", mmap_mode="r"), y=R["y"], fwd=R["fwd"], fwd_pess=R["fwd_pess"], day=day, ts=R["ts"],
                               mint=np.asarray(R["mint_u"], dtype=object)[R["mint_c"]], cols=[str(c) for c in R["cols"]],   # plain str: the walk-forward cache key
                               horizon_s=float(R["horizon_s"]), fwd_h={int(h): R[f"fwd_h_{int(h)}"] for h in R["holds"]},
                               chain=np.asarray(R["chain_u"], dtype=object)[R["chain_c"]] if "chain_u" in R.files else None)
    ds._day_ord = od
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--holds", default="240"); ap.add_argument("--compare", type=int, default=None)
    a = ap.parse_args()
    fit_holds = tuple(int(h) for h in a.holds.split(","))
    t0 = time.time(); ds = load_ds(); od = ds._day_ord
    log.info("loaded %d rows, %d days (%s..%s)", len(ds.y), len(ds.days), ds.days[0], ds.days[-1])
    sp = strategies.memecoin_spec()
    sp = dataclasses.replace(sp, strategies={**sp.strategies, "capitulation": {**sp.strategies["capitulation"], "holds": fit_holds}})
    uni = sp.in_universe(ds.X, ds.cols)
    cols = list(sp.legacy_cols) + [x for g in sp.groups for x in sp.groups[g]]
    holdout_days = list(ds.days[-strategies.HOLDOUT_DAYS:])                 # the days exactly as fit_stack splits them
    fit_days = [d for d in ds.days if d not in set(holdout_days)]
    tdays = sorted({d for blk in strategies.wf_blocks(fit_days) for d in blk})
    half = tdays[len(tdays) // 2]; day0 = tdays[0]
    tested = np.isin(od, [d.toordinal() for d in tdays]); sel = tested & (od < half.toordinal()); ev = tested & (od >= half.toordinal())
    HO = np.isin(od, [d.toordinal() for d in holdout_days])
    summary = {"rows": len(ds.y), "days": [str(ds.days[0]), str(ds.days[-1])], "holdout_days": [str(d) for d in holdout_days],
               "selection_days": [str(tdays[0]), str(tdays[tdays.index(half) - 1])], "evaluation_days": [str(half), str(tdays[-1])], "fits": []}
    f = strategies.fit_strategy(ds, "capitulation", cols, uni, sel, day0, None, spec=sp)
    if f is None or not f.selection:
        summary["fits"].append({"name": "capitulation", "passed": False, "reason": f.reason if f else "no candidates"})
    else:
        y_r = sp.label(ds, f.hold_min)
        evs = strategies.score_pick(ds, f.pick(ds, ev, uni), sp.hold_s(ds, f.hold_min), y_r, day0)
        hos = strategies.score_pick(ds, f.pick(ds, HO, uni), sp.hold_s(ds, f.hold_min), y_r, day0)
        f.evaluation = evs.dict(); f.passed = strategies.passes(evs, fallback=not f.selection.get("admissible"))
        summary["fits"].append({"name": "capitulation", "hold_min": f.hold_min, "line": f.line, "thr": jsonable(f.thr), "passed": f.passed,
                                "selection": jsonable(f.selection), "evaluation": jsonable(f.evaluation), "holdout": jsonable(hos.dict()), "trials": f.trials})
        log.info("capitulation: hold %d line %.4f thr %s | selection %s | evaluation %s | holdout %s", f.hold_min, f.line, f.thr, f.selection, f.evaluation, hos.dict())
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    chain = ds.chains()
    loose = uni & sp.base_mask("capitulation", ds.X, ds.cols, None); idx = np.flatnonzero(loose)
    arrs = {"idx": idx, "ts": ds.ts[idx], "chain": chain[idx].astype(str), "mint": ds.mint[idx].astype(str)}
    for c in FEATURES:
        if c in ds.cols:
            arrs["f_" + c] = np.asarray(ds.X[idx, ds.cols.index(c)])
    for H in tuple(strategies.HOLDS_MIN) + EXTRA_HOLDS:
        arrs[f"y_{H}"] = sp.label(ds, H)[idx]
    ci = np.asarray([ds.cols.index(x) for x in cols])
    for H in fit_holds:
        lab = sp.label(ds, H)
        arrs[f"s_{H}"] = strategies._cached(ds, cols, f"{sp.name}|capitulation|hold{H}", None, lambda H=H, lab=lab: sp.oos(ds, loose, lab, ci, H, None, f"capitulation {H}"))[idx]
    np.savez_compressed(OUT / "cap.npz", **arrs)
    if f is not None and f.selection:
        tr = taken_idx(ds.ts, ds.mint, sp.hold_s(ds, f.hold_min), np.flatnonzero(f.pick(ds, tested | HO, uni)))
        np.savez_compressed(OUT / "book.npz", ts=ds.ts[tr], ret=sp.label(ds, f.hold_min)[tr], chain=chain[tr].astype(str), mint=ds.mint[tr].astype(str))

    # the compared selector's own decisions (its final models, as the live engine makes them) on the holdout days after its training
    sid = a.compare
    if sid is None:
        with transaction() as conn:
            sid = pinned_snapshot() or (selector.latest_current(conn) or {}).get("id")
    m = selector.load_snapshot(int(sid))
    res = {"snapshot": int(sid), "trained_through": m.trained_through}
    rows_all, holds_all = [], []
    for d in holdout_days:
        if str(d) <= m.trained_through:
            continue
        rows = np.flatnonzero(od == d.toordinal())
        out = m.decide(np.asarray(ds.X[rows]), ds.cols, float(ds.ts[rows[0]]))
        sc = np.asarray(out["score"], float); thr = np.broadcast_to(np.asarray(out["threshold"], float), sc.shape)
        p = np.asarray(out["allow"], bool) & np.isfinite(sc) & (sc >= thr)
        hold = np.asarray(out["hold_s"], float) if np.ndim(out["hold_s"]) else np.full(len(rows), float(out["hold_s"]))
        rows_all.append(rows[p]); holds_all.append(hold[p])
    if rows_all:
        rows = np.concatenate(rows_all); hold_full = np.zeros(len(ds.y)); hold_full[rows] = np.concatenate(holds_all)
        t = taken_idx(ds.ts, ds.mint, hold_full, np.sort(rows))
        ret = np.array([ds.fwd_h[int(round(hold_full[r] / 60))][r] for r in t], dtype=float)
        for c in ("sol", "rh"):
            r = ret[(chain[t] == c) & np.isfinite(ret)]
            res[c] = {"n": int(len(r)), "mean": float(r.mean()) if len(r) else None, "win": float((r > 0).mean()) if len(r) else None,
                      "pf": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else None, "total": float(r.sum())}
        np.savez_compressed(OUT / "book_compare.npz", ts=ds.ts[t], ret=ret, chain=chain[t].astype(str), mint=ds.mint[t].astype(str))
    json.dump(res, open(OUT / "holdout_compare.json", "w"), indent=1)
    log.info("selector #%s on the holdout after %s: %s", sid, m.trained_through, res)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
