"""The selector: a gradient-boosted model of each minute's net 30-minute return (``train/decisions.py``).

Strategy: the model predicts the net return of buying a token now and selling 30 minutes later, after the paper
broker's fees and price impact; it trains on every eligible minute. It trades only tokens at least ``MIN_AGE_H`` hours
past graduation (or graduated before the archive began) — fresh graduations are where the rugs and the 0.5 % Jupiter
fee are. ``in_universe`` is the one definition of that universe for training, backtest and the live engine.

Trading configuration, shared verbatim with the fly (``train/fly_selector.py``): features robust-scaled
(``train/scaling.py``); buy when the predicted net return clears the buy line; the line is the one (of
``LINE_CANDIDATES``) with the most total net profit on the first half of the walk-forward's out-of-sample days, and it
is judged on the second half only; same universe, gates, one position per token, hold, pessimistic fills, paper costs
and sizing procedure (``agent/sizing.py`` bands from the out-of-sample trades).

Walk-forward protocol (the strategy gate): every day after a 21-day warm-up is a test day, scored by a model fit on
every eligible minute of the days at least two days earlier (refit every 7 days). A model goes to work only if the
evaluation half made money after costs, beat random picks from the same universe and has at least 100 trades. The
deployed model is then fit on every day and saved with its line, scaler and sizing table
(``data/brain/selectors/selector_<ts>.joblib`` + ``brain_snapshots`` kind 'selector'). Runs inside the training
pipeline (``train/pipeline.py``) and as ``fly-trader train-selector``.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import config
from ..agent import sizing
from ..db.apilog import record_event
from ..db.connection import transaction
from ..market.features import FEATURE_VERSION
from . import progress as prog
from .decisions import DecisionSet, build, evaluate, random_trades, rank_corr, summarize, taken_rows
from .mature import AGG_VERSION
from .scaling import RobustScaler

log = logging.getLogger(__name__)
SELECTOR_DIR = config.BRAIN_DIR / "selectors"
# the data and model definitions a model was trained on; a model from other definitions is never loaded or shown
DATA_VERSION = {"agg": AGG_VERSION, "features": FEATURE_VERSION, "costs": "real-fees-1", "selector": "ev-aged-2", "fly": "flynet-1"}
WARMUP_DAYS, BLOCK_DAYS = 21, 7      # walk-forward: the first 21 days only train; every later day is tested, refit every 7 days
MIN_AGE_H = 24.0                     # trade only tokens at least this long past graduation (unknown age = graduated before the archive)
MIN_EV = 0.01                        # the buy line of a model before its walk-forward chose one
LINE_CANDIDATES = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
MIN_LINE_TRADES = 100
GBM_PARAMS = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=300, l2_regularization=1.0)
IC_SAMPLE = 200_000


def in_universe(X: np.ndarray, cols: list[str]) -> np.ndarray:
    """Rows the models may trade: age unknown (graduated before the archive) or at least ``MIN_AGE_H`` hours old."""
    X = np.atleast_2d(X); c = {n: i for i, n in enumerate(cols)}
    return (X[:, c["age_known"]] == 0) | (np.expm1(X[:, c["log_age_h"]]) >= MIN_AGE_H)


def is_current(meta: dict | None) -> bool:
    return bool(meta) and meta.get("data") == DATA_VERSION


def is_deployable(meta: dict | None) -> bool:
    """Trained on the current definitions AND its evaluation half made money after costs and beat random picks."""
    return is_current(meta) and bool(meta.get("deployable"))


def deploy_decision(p: dict, rb: dict) -> tuple[bool, str]:
    if (p.get("n") or 0) < 100:
        return False, f"too few backtest trades ({p.get('n') or 0})"
    if p.get("mean") is None or p["mean"] <= 0:
        return False, f"its backtest lost money ({(p.get('mean') or 0) * 100:+.2f}% per trade after costs)"
    if rb.get("mean") is not None and p["mean"] <= rb["mean"]:
        return False, "it did not beat random picks"
    return True, f"its backtest made {p['mean'] * 100:+.2f}% per trade after costs (random {(rb.get('mean') or 0) * 100:+.2f}%)"


@dataclass
class SelectorModel:
    gbm: HistGradientBoostingRegressor
    scaler: RobustScaler
    cols: list[str]
    threshold: float                            # the buy line: predicted net return
    horizon_min: int
    trained_through: str
    metrics: dict = field(default_factory=dict)
    sizing: list = field(default_factory=list)  # agent/sizing.py table from the backtest's out-of-sample trades

    def score(self, X: np.ndarray) -> np.ndarray:
        """Predicted net 30-minute return."""
        return self.gbm.predict(self.scaler.transform(X))

    def universe(self, X: np.ndarray) -> np.ndarray:
        return in_universe(X, self.cols)


def fit(ds: DecisionSet, train: np.ndarray, seed: int = 0) -> SelectorModel:
    """Regression of the net return (pessimistic fill, paper costs, clipped to ±100 %) on every eligible training minute."""
    idx = np.flatnonzero(train)
    scaler = RobustScaler.fit(ds.X[idx], seed=seed)
    gbm = HistGradientBoostingRegressor(**GBM_PARAMS, random_state=seed)
    gbm.fit(scaler.transform(ds.X[idx]), np.clip(ds.fwd_pess[idx], -1.0, 1.0))
    return SelectorModel(gbm=gbm, scaler=scaler, cols=list(ds.cols), threshold=MIN_EV, horizon_min=int(ds.horizon_s // 60), trained_through=str(max(ds.day[train])))


def ic(ds: DecisionSet, scores: np.ndarray, rows: np.ndarray, seed: int = 0) -> float | None:
    """Information coefficient: rank correlation of the scores with the realised net returns on (a sample of) ``rows``."""
    idx = np.flatnonzero(rows)
    if len(idx) > IC_SAMPLE:
        idx = np.random.default_rng(seed).choice(idx, IC_SAMPLE, replace=False)
    return rank_corr(scores[idx], ds.fwd_pess[idx]) if len(idx) > 1 else None


@dataclass
class Fold:
    day: object
    model: SelectorModel
    test: np.ndarray       # [N] bool, the tradable rows of the test days
    scores: np.ndarray     # [N] float, predicted net return on the test rows (NaN elsewhere)
    ic: float | None


def fold(ds: DecisionSet, D, seed: int = 0, min_train: int = 50_000, min_test: int = 500) -> Fold | None:
    """One walk-forward step: fit on every day before the first test day minus one (one-day purge), score the test
    day(s) ``D`` (a date or a list of dates). None when either side is too small."""
    test_days = list(D) if isinstance(D, (list, tuple)) else [D]
    train = ds.day < (min(test_days) - timedelta(days=1)); tix = np.flatnonzero(np.isin(ds.day, test_days))
    if train.sum() < min_train or len(tix) < min_test:
        return None
    m = fit(ds, train, seed=seed)
    tix = tix[m.universe(ds.X[tix])]; test = np.zeros(len(ds.y), bool); test[tix] = True
    scores = np.full(len(ds.y), np.nan); scores[tix] = m.score(ds.X[tix]) if len(tix) else scores[tix]
    return Fold(day=test_days[0], model=m, test=test, scores=scores, ic=ic(ds, scores, test, seed))


def choose_line(ds: DecisionSet, scores: np.ndarray, rows: np.ndarray) -> tuple[float, list[dict]]:
    """The buy line with the most total net profit (fixed size, one position per token) on ``rows``."""
    table = []
    for c in LINE_CANDIDATES:
        tr = taken_rows(ds, rows & (scores >= c)); r = ds.fwd_pess[tr]
        table.append({"line": c, "trades": int(len(r)), "mean": float(r.mean()) if len(r) else None, "total": float(r.sum())})
    ok = [t for t in table if t["trades"] >= MIN_LINE_TRADES]
    return (max(ok, key=lambda t: t["total"])["line"] if ok else MIN_EV), table


def _pct(v, fmt: str = "+.2f") -> str:
    return f"{v*100:{fmt}}%" if v is not None else "-"


def walk_forward(ds: DecisionSet, warmup_days: int = WARMUP_DAYS, block_days: int = BLOCK_DAYS, stop: threading.Event | None = None) -> dict:
    """Out-of-sample scores for every day after ``warmup_days`` (refit every ``block_days``); the buy line chosen on the
    first half of those days, everything judged on the second half."""
    days = ds.days; scores = np.full(len(ds.y), np.nan); tested = np.zeros(len(ds.y), bool); ics: dict[str, float | None] = {}
    blocks = [days[k:k + block_days] for k in range(warmup_days, len(days), block_days)]
    for k, blk in enumerate(blocks):
        if stop is not None and stop.is_set():
            break
        prog.update("selector walk-forward", k, len(blocks), day=f"{blk[0]}..{blk[-1]}", force=True)
        fo = fold(ds, blk, seed=k)
        if fo is None:
            continue
        scores[fo.test] = fo.scores[fo.test]; tested |= fo.test
        for D in blk:
            ics[str(D)] = ic(ds, scores, fo.test & (ds.day == D), k)
    tdays = sorted(set(ds.day[tested].tolist()))
    if len(tdays) < 2:
        empty = summarize(np.array([])); empty.update(days_positive=0, days=0)
        return {"per_day": {}, "ic": ics, "line": MIN_EV, "lines": [], "pooled": empty, "random": summarize(np.array([])), "sizing": [], "bankroll": {}}
    sel_days, ev_days = tdays[: len(tdays) // 2], tdays[len(tdays) // 2:]
    sel_mask = tested & np.isin(ds.day, sel_days); ev_mask = tested & np.isin(ds.day, ev_days)
    line, lines = choose_line(ds, scores, sel_mask)
    log.info("buy line %s chosen on %s..%s (total net profit per candidate: %s)", _pct(line), sel_days[0], sel_days[-1],
             ", ".join(f"{t['line'] * 100:.2f}%: {t['total']:+.1f} ({t['trades']})" for t in lines))
    out = {"per_day": {}, "ic": ics, "line": line, "lines": lines, "selection_days": [str(sel_days[0]), str(sel_days[-1])],
           "evaluation_days": [str(ev_days[0]), str(ev_days[-1])]}
    sel_set = set(sel_days)
    for D in tdays:
        td = tested & (ds.day == D); r = evaluate(ds, scores, td, line)["pooled"]
        rs = summarize(random_trades(ds, td, int((td & (scores >= line)).sum()))); half = "selection" if D in sel_set else "evaluation"
        out["per_day"][str(D)] = {**r, "ic": ics.get(str(D)), "line": line, "random_mean": rs["mean"], "half": half}
        log.info("selector %s [%s]: IC %s | predicted >= %s: n=%s mean %s median %s win %s PF %s | random %s", D, half,
                 f"{ics.get(str(D)):.3f}" if ics.get(str(D)) is not None else "-", _pct(line), r["n"], _pct(r["mean"]), _pct(r["median"]), _pct(r["win"], ".0f"),
                 f"{r['pf']:.2f}" if r["pf"] is not None else "-", _pct(rs["mean"]))
        record_event("info", "selector", f"walk-forward {D}", {"day": str(D), "ic": ics.get(str(D)), **r, "random_mean": rs["mean"], "half": half, "line": line})
    ev_rows = taken_rows(ds, ev_mask & (scores >= line))
    pooled = summarize(ds.fwd_pess[ev_rows]); pooled["days_positive"] = sum(1 for D in ev_days if (out["per_day"][str(D)]["mean"] or 0) > 0); pooled["days"] = len(ev_days)
    out["pooled"] = pooled
    out["random"] = summarize(random_trades(ds, ev_mask, int((ev_mask & (scores >= line)).sum())))
    # position sizing from the out-of-sample trades at the chosen line: bands of margin over the line -> growth-optimal fraction
    rows = taken_rows(ds, tested & (scores >= line)); margins = scores[rows] - line; rets = ds.fwd_pess[rows]
    resq = np.expm1(ds.X[rows, ds.cols.index("log_liquidity_sol")].astype(np.float64)) / 2.0
    out["sizing"] = sizing.build_table(margins, rets)
    out["bankroll"] = {"sized": sizing.simulate_bankroll(ds.ts[rows], ds.horizon_s, margins, rets, out["sizing"], res_quote=resq),
                       "fixed": sizing.simulate_bankroll(ds.ts[rows], ds.horizon_s, margins, rets, None, res_quote=resq)}
    return out


def save(m: SelectorModel, run_id: str | None = None) -> tuple[Path, int]:
    SELECTOR_DIR.mkdir(parents=True, exist_ok=True)
    path = SELECTOR_DIR / f"selector_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.joblib"
    joblib.dump(m, path); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'selector',%s) RETURNING id",
                           (run_id, str(path), sha, json.dumps({"threshold": m.threshold, "horizon_min": m.horizon_min, "trained_through": m.trained_through,
                                                                **m.metrics}, default=str))).fetchone()
    return path, int(row["id"])


def latest_current(conn) -> dict | None:
    """The newest deployable selector snapshot (current definitions, profitable backtest), or None."""
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = 'selector' ORDER BY id DESC").fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if is_deployable(meta) and Path(r["path"]).exists():
            return dict(r)
    return None


def load_latest() -> SelectorModel | None:
    with transaction() as conn:
        r = latest_current(conn)
    return joblib.load(r["path"]) if r else None


def main(days: int | None = None, horizon_min: int = 30, stop_event: threading.Event | None = None) -> dict:
    """``days``: None = every day of the corpus (the default; a number keeps only the most recent days)."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("selector", reason="selector training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("selector: building decision points", 0, 1, force=True)
    t0 = time.time(); ds = build(days=days, horizon_min=horizon_min)
    prog.update("selector: decision points ready", 1, 1, force=True, rows=int(len(ds.y)), tokens=int(len(set(ds.mint.tolist()))), days=len(ds.days),
                universe_mean=float(ds.fwd_pess.mean()))
    log.info("decision points: %d rows, %d days, universe mean %+.2f%% (%.0fs)", len(ds.y), len(ds.days), ds.fwd_pess.mean() * 100, time.time() - t0)
    wf = walk_forward(ds, stop=stop_event)
    p = wf["pooled"]; rb = wf["random"]
    log.info("selector walk-forward, evaluation half at line %s: n=%s mean %s median %s win %s PF %s days positive %s/%s | random %s", _pct(wf["line"]), p["n"],
             _pct(p["mean"]), _pct(p["median"]), _pct(p["win"], ".0f"), f"{p['pf']:.2f}" if p["pf"] else "-", p["days_positive"], p["days"], _pct(rb["mean"]))
    prog.update("selector: walk-forward done", 1, 1, force=True, walk_forward=p, random_baseline=rb, line=wf["line"])
    if stop_event is not None and stop_event.is_set():
        return wf
    prog.update("selector: fitting the deployable model on every day", 0, 1, force=True)
    deployable, why = deploy_decision(p, rb)
    final = fit(ds, np.ones(len(ds.y), bool), seed=99); final.threshold = wf["line"]; final.sizing = wf["sizing"]
    bk = wf["bankroll"]
    if bk:
        log.info("bankroll replay of the backtest trades: sized %.2fx (worst drawdown %.0f%%) vs fixed %g SOL %.2fx (worst drawdown %.0f%%)",
                 bk["sized"]["multiple"] or 0, bk["sized"]["max_drawdown"] * 100, config.MAX_POSITION_SOL, bk["fixed"]["multiple"] or 0, bk["fixed"]["max_drawdown"] * 100)
    final.metrics = {"walk_forward": p, "random_baseline": rb, "line": wf["line"], "lines": wf["lines"], "selection_days": wf.get("selection_days"),
                     "evaluation_days": wf.get("evaluation_days"), "costs": "real fees (pool fee by market cap, Jupiter 10 bps, network fee) + impact", "data": DATA_VERSION, "deployable": deployable, "deploy_reason": why,
                     "sizing": wf["sizing"], "bankroll": bk, "model": {"kind": "ev", "line": wf["line"], "min_age_h": MIN_AGE_H},
                     "ic_by_day": wf["ic"], "rows": int(len(ds.y)), "days": len(ds.days), "first_day": str(ds.days[0]), "last_day": str(ds.days[-1])}
    path, sid = save(final)
    record_event("info", "selector", f"selector saved (snapshot {sid})", {"path": str(path), "line": final.threshold, "deployable": deployable, "reason": why, **p})
    prog.update("selector: saved", 1, 1, force=True, snapshot_id=sid, path=str(path), line=final.threshold, deployable=deployable, deploy_reason=why)
    log.info("selector saved: %s (snapshot %d, line %s) — %s: %s", path, sid, _pct(final.threshold), "put to work" if deployable else "NOT put to work", why)
    return {**wf, "snapshot_id": sid, "deployable": deployable, "deploy_reason": why}
