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
from .decisions import DecisionSet, build, evaluate

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


def walk_forward(ds: DecisionSet, test_days: int = 9, top_frac: float = 0.01, stop: threading.Event | None = None) -> dict:
    days = ds.days; out = {"per_day": {}, "auc": {}}; trades = []
    for k, D in enumerate(days[-test_days:]):
        if stop is not None and stop.is_set():
            break
        prog.update("selector walk-forward", k, test_days, day=str(D), force=True)
        train = ds.day < (D - timedelta(days=1)); test = ds.day == D
        if train.sum() < 50_000 or test.sum() < 500:
            continue
        m = fit(ds, train, top_frac, seed=k); s = m.score(ds.X[test])
        auc = float(roc_auc_score(ds.y[test], s)); out["auc"][str(D)] = auc
        full = np.zeros(len(ds.y)); full[test] = s
        ev = evaluate(ds, full, test, m.threshold); out["per_day"][str(D)] = {**ev["pooled"], "auc": auc, "threshold": m.threshold}
        r = ev["pooled"]; log.info("selector %s: AUC %.3f | top %.0f%%: n=%s mean %s median %s win %s PF %s", D, auc, top_frac * 100, r["n"],
                                   f"{r['mean']*100:+.2f}%" if r["mean"] is not None else "-", f"{r['median']*100:+.2f}%" if r["median"] is not None else "-",
                                   f"{r['win']*100:.0f}%" if r["win"] is not None else "-", f"{r['pf']:.2f}" if r["pf"] is not None else "-")
        record_event("info", "selector", f"walk-forward {D}", {"day": str(D), "auc": auc, **{k_: v for k_, v in r.items()}})
        if r["n"]:
            from .decisions import trades_from_picks
            trades.append(trades_from_picks(ds, test & (full >= m.threshold)))
    allr = np.concatenate(trades) if trades else np.array([])
    from .decisions import summarize
    pooled = summarize(allr); pooled["days_positive"] = sum(1 for v in out["per_day"].values() if v["mean"] and v["mean"] > 0); pooled["days"] = len(out["per_day"])
    out["pooled"] = pooled
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
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("selector: building decision points", 0, 1, force=True)
    t0 = time.time(); ds = build(days=days, horizon_min=horizon_min)
    prog.update("selector: decision points ready", 1, 1, force=True, rows=int(len(ds.y)), tokens=int(len(set(ds.mint.tolist()))), days=len(ds.days),
                base_rate=float(ds.y.mean()), universe_mean=float(ds.fwd.mean()))
    log.info("decision points: %d rows, %d days, base rate %.1f%%, universe mean %+.2f%% (%.0fs)", len(ds.y), len(ds.days), ds.y.mean() * 100, ds.fwd.mean() * 100, time.time() - t0)
    wf = walk_forward(ds, test_days=test_days, top_frac=top_frac, stop=stop_event)
    p = wf["pooled"]; log.info("selector walk-forward pooled: n=%s mean %s median %s win %s PF %s days positive %s/%s", p["n"],
                              f"{(p['mean'] or 0)*100:+.2f}%", f"{(p['median'] or 0)*100:+.2f}%", f"{(p['win'] or 0)*100:.0f}%", f"{p['pf']:.2f}" if p["pf"] else "-", p["days_positive"], p["days"])
    prog.update("selector: walk-forward done", test_days, test_days, force=True, walk_forward=p)
    if stop_event is not None and stop_event.is_set():
        return wf
    prog.update("selector: fitting deployable model", 0, 1, force=True)
    final = fit(ds, np.ones(len(ds.y), bool), top_frac, seed=99); final.metrics = {"walk_forward": p, "auc_by_day": wf["auc"], "rows": int(len(ds.y)), "days": len(ds.days)}
    path, sid = save(final)
    record_event("info", "selector", f"selector saved (snapshot {sid})", {"path": str(path), "threshold": final.threshold, **p})
    prog.update("selector: saved", 1, 1, force=True, snapshot_id=sid, path=str(path), threshold=final.threshold)
    log.info("selector saved: %s (snapshot %d, threshold %.4f)", path, sid, final.threshold)
    return wf
