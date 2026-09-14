"""The selector: gradient-boosted classifier over decision points (``train/decisions.py``).

Walk-forward protocol (the strategy gate): for each of the last ``test_days`` days D, fit on eligible minutes
of days ≤ D−2 (one-day purge), score day D, trade the minutes above the threshold that marks the top
``top_frac`` of training scores, one position per token, ``horizon`` hold, pessimistic fills. The deployed model
is then fit on every day up to the last one and saved with its threshold, feature list and standardization
(``data/brain/selectors/selector_<ts>.joblib`` + ``brain_snapshots`` kind 'selector'). Runs as the ``train``
worker when ``training_params.regimen == 'selector'`` (default) and as ``fly-trader train-selector``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from . import progress as prog
from .decisions import DecisionSet, build, evaluate, random_trades, summarize, trades_from_picks

log = logging.getLogger(__name__)
SELECTOR_DIR = config.BRAIN_DIR / "selectors"


@dataclass
class SelectorModel:
    gbm: HistGradientBoostingClassifier
    mean: np.ndarray
    std: np.ndarray
    cols: list[str]
    threshold: float
    top_frac: float
    horizon_min: int
    trained_through: str
    metrics: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.gbm.predict_proba((X - self.mean) / self.std)[:, 1]


def fit(ds: DecisionSet, train: np.ndarray, top_frac: float, max_rows: int = 2_500_000, seed: int = 0) -> SelectorModel:
    idx = np.flatnonzero(train); rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    mean, std = ds.X[idx].mean(0), ds.X[idx].std(0) + 1e-6
    gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    gbm.fit((ds.X[idx] - mean) / std, ds.y[idx])
    thr = float(np.quantile(gbm.predict_proba((ds.X[idx] - mean) / std)[:, 1], 1 - top_frac))
    return SelectorModel(gbm=gbm, mean=mean, std=std, cols=ds.cols, threshold=thr, top_frac=top_frac, horizon_min=int(ds.horizon_s // 60),
                         trained_through=str(max(ds.day[train])))


@dataclass
class Fold:
    day: object
    model: SelectorModel
    test: np.ndarray       # [N] bool, rows of the test day
    scores: np.ndarray     # [N] float, scores on the test rows (0 elsewhere)
    pick: np.ndarray       # [N] bool, test rows at or above the threshold
    auc: float | None      # None when the test day has a single class


def fold(ds: DecisionSet, D, top_frac: float = 0.01, seed: int = 0, min_train: int = 50_000, min_test: int = 500) -> Fold | None:
    """One walk-forward step: fit on days ≤ D−2 (one-day purge), score day D. None when either side is too small or the
    training labels have a single class."""
    train = ds.day < (D - timedelta(days=1)); test = ds.day == D
    if train.sum() < min_train or test.sum() < min_test or len(np.unique(ds.y[train])) < 2:
        return None
    m = fit(ds, train, top_frac, seed=seed); full = np.zeros(len(ds.y)); full[test] = m.score(ds.X[test])
    auc = float(roc_auc_score(ds.y[test], full[test])) if len(np.unique(ds.y[test])) == 2 else None
    return Fold(day=D, model=m, test=test, scores=full, pick=test & (full >= m.threshold), auc=auc)


def _pct(v, fmt: str = "+.2f") -> str:
    return f"{v*100:{fmt}}%" if v is not None else "-"


def walk_forward(ds: DecisionSet, test_days: int = 9, top_frac: float = 0.01, stop: threading.Event | None = None) -> dict:
    days = ds.days; out = {"per_day": {}, "auc": {}}; trades = []; rand = []
    for k, D in enumerate(days[-test_days:]):
        if stop is not None and stop.is_set():
            break
        prog.update("selector walk-forward", k, test_days, day=str(D), force=True)
        fo = fold(ds, D, top_frac, seed=k)
        if fo is None:
            continue
        auc, m = fo.auc, fo.model; out["auc"][str(D)] = auc
        ev = evaluate(ds, fo.scores, fo.test, m.threshold)
        rr = random_trades(ds, fo.test, int(fo.pick.sum())); rs = summarize(rr)
        out["per_day"][str(D)] = {**ev["pooled"], "auc": auc, "threshold": m.threshold, "random_mean": rs["mean"]}
        r = ev["pooled"]; log.info("selector %s: AUC %s | top %.0f%%: n=%s mean %s median %s win %s PF %s | random mean %s", D, f"{auc:.3f}" if auc is not None else "-",
                                   top_frac * 100, r["n"], _pct(r["mean"]), _pct(r["median"]), _pct(r["win"], ".0f"), f"{r['pf']:.2f}" if r["pf"] is not None else "-", _pct(rs["mean"]))
        record_event("info", "selector", f"walk-forward {D}", {"day": str(D), "auc": auc, **{k_: v for k_, v in r.items()}, "random_mean": rs["mean"]})
        if r["n"]:
            trades.append(trades_from_picks(ds, fo.pick)); rand.append(rr)
    allr = np.concatenate(trades) if trades else np.array([])
    pooled = summarize(allr); pooled["days_positive"] = sum(1 for v in out["per_day"].values() if v["mean"] and v["mean"] > 0); pooled["days"] = len(out["per_day"])
    out["pooled"] = pooled
    out["random"] = summarize(np.concatenate(rand) if rand else np.array([]))     # same pick counts, costs and fills, no skill
    return out


def save(m: SelectorModel, run_id: str | None = None) -> tuple[Path, int]:
    SELECTOR_DIR.mkdir(parents=True, exist_ok=True)
    path = SELECTOR_DIR / f"selector_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.joblib"
    joblib.dump(m, path); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'selector',%s) RETURNING id",
                           (run_id, str(path), sha, json.dumps({"threshold": m.threshold, "top_frac": m.top_frac, "horizon_min": m.horizon_min,
                                                                "trained_through": m.trained_through, **m.metrics}, default=str)[:900])).fetchone()
    return path, int(row["id"])


def load_latest() -> SelectorModel | None:
    with transaction() as conn:
        r = conn.execute("SELECT path FROM brain_snapshots WHERE kind = 'selector' ORDER BY id DESC LIMIT 1").fetchone()
    return joblib.load(r["path"]) if r and Path(r["path"]).exists() else None


def main(days: int = 45, test_days: int = 9, top_frac: float = 0.01, horizon_min: int = 30, stop_event: threading.Event | None = None) -> dict:
    from ..ops.reset import reset_training_stats
    reset_training_stats("selector", reason="selector training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("selector: building decision points", 0, 1, force=True)
    t0 = time.time(); ds = build(days=days, horizon_min=horizon_min)
    prog.update("selector: decision points ready", 1, 1, force=True, rows=int(len(ds.y)), tokens=int(len(set(ds.mint.tolist()))), days=len(ds.days),
                base_rate=float(ds.y.mean()), universe_mean=float(ds.fwd.mean()))
    log.info("decision points: %d rows, %d days, base rate %.1f%%, universe mean %+.2f%% (%.0fs)", len(ds.y), len(ds.days), ds.y.mean() * 100, ds.fwd.mean() * 100, time.time() - t0)
    wf = walk_forward(ds, test_days=test_days, top_frac=top_frac, stop=stop_event)
    p = wf["pooled"]; log.info("selector walk-forward pooled: n=%s mean %s median %s win %s PF %s days positive %s/%s", p["n"],
                              f"{(p['mean'] or 0)*100:+.2f}%", f"{(p['median'] or 0)*100:+.2f}%", f"{(p['win'] or 0)*100:.0f}%", f"{p['pf']:.2f}" if p["pf"] else "-", p["days_positive"], p["days"])
    rb = wf["random"]; log.info("random baseline (same pick counts, costs, fills): n=%s mean %s median %s win %s", rb["n"], _pct(rb["mean"]), _pct(rb["median"]), _pct(rb["win"], ".0f"))
    prog.update("selector: walk-forward done", test_days, test_days, force=True, walk_forward=p, random_baseline=rb)
    if stop_event is not None and stop_event.is_set():
        return wf
    prog.update("selector: fitting deployable model", 0, 1, force=True)
    final = fit(ds, np.ones(len(ds.y), bool), top_frac, seed=99); final.metrics = {"walk_forward": p, "random_baseline": rb, "costs": "paper broker model", "auc_by_day": wf["auc"], "rows": int(len(ds.y)), "days": len(ds.days)}
    path, sid = save(final)
    record_event("info", "selector", f"selector saved (snapshot {sid})", {"path": str(path), "threshold": final.threshold, **p})
    prog.update("selector: saved", 1, 1, force=True, snapshot_id=sid, path=str(path), threshold=final.threshold)
    log.info("selector saved: %s (snapshot %d, threshold %.4f)", path, sid, final.threshold)
    return wf
