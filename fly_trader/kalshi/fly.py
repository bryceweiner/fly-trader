"""The Kalshi fly: a second FlyNet on the optic lobes (brain/connectome.VISUAL_SUB — the memecoin senses dropped, VISUAL
kept), its features entering the photoreceptors (``photoreceptor_rows``), taught by the Kalshi selector's classifier
stack (kalshi/selector.py) and calibrated, snapshotted and governed by the memecoin fly's machinery (train/fly_selector.py)
under its own kind ('kalshi_fly_selector'), directory (``data/brain/kalshi_policies``) and version.

The head predicts **p̂(side pays)** per strategy (``scale`` 1: the readout is linear in probability, clipped to [0, 1] at
use) so the mushroom body's three-factor rule (brain/plastic.py: δ = r − ŷ, ``RET_CLIP`` 1) learns from the settlement
outcome y ∈ {0, 1} unchanged. The trading quantity is the **edge** p̂ − the arm's fee-inclusive price (kalshi/strategies
.effective): the fly's lines are minimum edges, its sizing bands are bands of edge over the line, and the calibration
labels are the arm's realised returns (taker at the ask, maker one tick inside it when filled).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import torch

from .. import config
from ..agent import sizing
from ..brain.connectome import Connectome, SubConnectome, VISUAL_SUB, photoreceptor_rows
from ..db.apilog import record_event
from ..db.connection import transaction
from ..train import fly_calibrate
from ..train import fly_selector as F
from ..train import progress as prog
from ..train import strategies as S
from ..train.decisions import DecisionSet, taken_idx
from ..train.scaling import RobustScaler
from . import decisions as KD
from . import selector as kselector
from .strategies import KALSHI, effective, in_universe, label, probability

log = logging.getLogger(__name__)

SCALE = 1.0
KIND = "kalshi_fly_selector"
SUBDIR = "kalshi_policies"
PREFIX = "kalshi_fly"
CALIB_DAYS, PURGE_DAYS = F.CALIB_DAYS, F.PURGE_DAYS
KALSHI_FLY_VERSION = {**kselector.DATA_VERSION, "fly": "visual-1", "plastic": "mb4-ch", "teacher": "deployed-1"}
TEACHER_NOTE = "a Kalshi stack refit on the fly's training days (no deployable Kalshi selector to copy)"
HOLD_CEILING_S = float(config.KALSHI_MAX_DAYS_TO_CLOSE) * 86400.0        # a position runs to its market's settlement; this is the longest
_GRAPH = None


def graph():
    """The Kalshi fly's sub-connectome (cached): the whole brain minus the memecoin afferents."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = SubConnectome(Connectome.load(), exclude=VISUAL_SUB)
    return _GRAPH


def net_kwargs(g=None) -> dict:
    return {"aff_rows": photoreceptor_rows(g if g is not None else graph()), "scale": SCALE}


class KalshiFlyModel(F.FlyModel):
    """A trained Kalshi fly: per strategy (each on its own dopamine channel) p̂ of the side paying; ``rules[name]["hold_min"]``
    is the strategy's **arm** ('taker' | 'maker'), not a hold — every position runs to settlement."""

    def arm(self, s: int) -> str:
        return str(self.rules[self.strategies[s]]["hold_min"])

    def hold_s(self, s: int) -> float:
        return HOLD_CEILING_S

    @torch.no_grad()
    def score_all(self, X: np.ndarray, batch: int | None = None) -> np.ndarray:
        """[B, S] p̂(side pays) per strategy, clipped to [0, 1]."""
        return np.clip(super().score_all(X, batch), 0.0, 1.0)

    def edge_all(self, X: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
        """[B, S] each strategy's edge: p̂ − its arm's fee-inclusive price (per dollar of payout)."""
        P = self.score_all(X) if P is None else P
        X = np.atleast_2d(X); out = np.empty_like(P)
        for j in range(len(self.strategies)):
            out[:, j] = P[:, j] - effective(X, self.cols, self.arm(j)) / 100.0
        return out

    def triggers(self, X: np.ndarray, cols: list[str]) -> np.ndarray:
        X = np.atleast_2d(X); out = np.zeros((len(X), len(self.strategies)), bool)
        for j, name in enumerate(self.strategies):
            r = self.rules[name]
            out[:, j] = KALSHI.base_mask(name, X, cols, r.get("high")) & S.trigger_mask(name, r.get("thr") or {}, X, cols, KALSHI)
        return out

    def universe(self, X: np.ndarray) -> np.ndarray:
        return in_universe(X, self.cols)


class KalshiTeacher(F.StackTeacher):
    """The Kalshi selector's stack as the fly's teacher: per strategy, on its candidates, the classifier's p̂ where the
    stack would trade and, where a filter blocks it, a p̂ one score-spread below the line (so the fly learns the decision)."""

    def __init__(self, models: dict, scaler: RobustScaler, cols: list[str]):
        self.models, self.scaler, self.cols = models, scaler, list(cols)
        self.strategies = list(models.get("strategies") or {})
        self.rules = {k: {"thr": dict(v["thr"]), "high": v["high"], "hold_min": str(v["hold_min"])} for k, v in models["strategies"].items()}
        self.lines = {k: float(v["line"]) for k, v in models["strategies"].items()}
        self.threshold = self.lines[self.strategies[0]] if self.strategies else 0.0
        self.combine = models.get("combine", "score"); self.eps: dict = {}

    def targets(self, X: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        per = S.decide_all(self.models, X, self.cols, ts, spec=KALSHI); X = np.atleast_2d(X)
        T = np.full((len(X), len(self.strategies)), np.nan); A = np.zeros_like(T, dtype=bool)
        for j, k in enumerate(self.strategies):
            d = per[k]; m = self.models["strategies"][k]; trig = d["trig"]; sc = d["score"]
            if k not in self.eps:
                al = sc[d["allow"] & np.isfinite(sc)]
                self.eps[k] = float(np.median(np.abs(al - np.median(al)))) if len(al) else 0.0
            if trig.any():
                ti = np.flatnonzero(trig); ci = [self.cols.index(c) for c in m["cols"]]
                p = probability(m, X[np.ix_(ti, ci)]); eff = effective(X[ti], self.cols, str(m["hold_min"])) / 100.0
                T[ti, j] = np.where(d["allow"][ti], p, np.minimum(p, eff + self.lines[k] - self.eps[k]))
            A[:, j] = d["allow"]
        return T, A


def _stack_teacher(ds: DecisionSet, train: np.ndarray, stop=None):
    sub = ds.subset(train)
    for a in ("hold_s", "outcome", "side", "ticker", "category"):
        setattr(sub, a, getattr(ds, a)[train])
    stack = S.fit_stack(sub, stop, holdout_days_n=0, spec=KALSHI)
    if not stack.fits:
        return None, stack
    return KalshiTeacher(S.final_models(sub, stack, spec=KALSHI), RobustScaler.fit(sub.X, seed=7), ds.cols), stack


def deployed_teacher(ds: DecisionSet, train: np.ndarray) -> tuple[KalshiTeacher | None, str, int | None]:
    """The newest deployable Kalshi selector as the teacher (its models are the ones the fly is judged against)."""
    with transaction() as conn:
        r = kselector.latest_current(conn)
    if r is None:
        return None, TEACHER_NOTE, None
    m = kselector.load_snapshot(int(r["id"]))
    if m is None or not (m.stack or {}).get("strategies"):
        return None, f"kalshi selector #{r['id']} could not be loaded; {TEACHER_NOTE}", None
    missing = sorted({c for v in m.stack["strategies"].values() for c in v["cols"]} - set(ds.cols))
    if missing:
        return None, f"kalshi selector #{r['id']} wants columns the corpus lacks ({', '.join(missing[:4])}); {TEACHER_NOTE}", None
    names = ", ".join(f"{k} (edge ≥ {v['line']:.3f}, {v['hold_min']})" for k, v in m.stack["strategies"].items())
    return KalshiTeacher(m.stack, RobustScaler.fit(ds.X[np.flatnonzero(train)], seed=7), list(ds.cols)), f"deployed kalshi selector #{r['id']}: {names}", int(r["id"])


def kalshi_decide(fly: KalshiFlyModel, X: np.ndarray, cols: list[str], edges: np.ndarray | None = None, hold_s: np.ndarray | None = None,
                  only: str | None = None, trig: np.ndarray | None = None) -> dict:
    """The fly's per-row decision: among the strategies whose trigger fires and whose edge clears the fly's line, the one
    with the larger sizing-band Kelly fraction or margin. ``hold_s``: each row's time to settlement (the position's hold);
    ``only``: an arm ('taker' | 'maker') restricts the choice to that arm's strategies (each arm trades its own book)."""
    X = np.atleast_2d(X); E = fly.edge_all(X) if edges is None else edges; trig = fly.triggers(X, cols) if trig is None else trig; n = len(X)
    strat = np.full(n, None, dtype=object); score = np.full(n, -1.0); thr = np.full(n, np.inf); tables = [[] for _ in range(n)]; arm = np.full(n, "", dtype=object)
    key = np.full(n, -np.inf)
    for j, name in enumerate(fly.strategies):
        if only is not None and fly.arm(j) != only:
            continue
        v = E[:, j]; line = fly.lines[name]; ok = trig[:, j] & (v >= line)
        kv = (np.array([((sizing.band_for(fly.sizings[name], m) or {}).get("kelly") or 0.0) for m in v - line]) + 1e-9 * v
              if fly.combine == "kelly" and fly.sizings[name] else v - line)
        take = ok & (kv > key); key = np.where(take, kv, key)
        for i in np.flatnonzero(take):
            strat[i] = name; score[i] = v[i]; thr[i] = line; tables[i] = fly.sizings[name]; arm[i] = fly.arm(j)
    hold = np.asarray(hold_s, dtype=np.float64) if hold_s is not None else np.full(n, HOLD_CEILING_S)
    return {"strategy": strat, "score": score, "hold_s": np.where(strat != None, hold, 0.0), "threshold": thr, "tables": tables, "arm": arm}   # noqa: E711


def bootstrap(ds: DecisionSet, Sday: date, epochs: int = F.EPOCHS, stop: threading.Event | None = None, seed: int = 0, g=None, device=None,
              teacher=None, teacher_note: str | None = None) -> tuple[KalshiFlyModel | None, dict]:
    """A Kalshi fly ready to trade from day ``Sday`` on: distilled from its teacher on the days before ``Sday − 8``, its
    edge lines and sizing calibrated on the week before ``Sday`` from the positions that had settled by then."""
    train_end = Sday - timedelta(days=CALIB_DAYS + PURGE_DAYS)
    train = ds.day < train_end; uni = in_universe(ds.X, ds.cols)
    if not train.any():
        raise RuntimeError(f"no training days before {train_end}")
    stack_info = None
    if teacher is None:
        prog.update("kalshi fly: fitting its teacher (the Kalshi stack, training days only)", 0, 1, force=True)
        teacher, stack = _stack_teacher(ds, train, stop)
        stack_info = {"components": stack.components, "deployable": stack.deployable, "reason": stack.reason}
        if teacher is None:
            return None, {"S": str(Sday), "gates_ok": False, "gate_failures": ["the teacher stack has no strategy on the training days"], "teacher_stack": stack_info}
    g = g if g is not None else graph()
    fly, fit_info = F.train_fly(ds, train & uni, teacher, epochs=epochs, stop=stop, seed=seed, graph=g, device=device, net_kwargs=net_kwargs(g),
                                model_cls=KalshiFlyModel, target_clip=(0.0, 1.0), horizon_min=0)
    if fly is None:
        return None, {"stopped": True, **fit_info}
    s_epoch = datetime(Sday.year, Sday.month, Sday.day, tzinfo=timezone.utc).timestamp()
    week = (ds.day >= Sday - timedelta(days=CALIB_DAYS)) & (ds.day < Sday) & uni
    prog.update("kalshi fly: calibrating its own edge lines", 0, 1, force=True)
    wi = np.flatnonzero(week)
    P = fly.score_all(ds.X[wi]) if len(wi) else np.zeros((0, len(fly.strategies)))
    E = fly.edge_all(ds.X[wi], P) if len(wi) else P; trig = fly.triggers(ds.X[wi], ds.cols) if len(wi) else np.zeros((0, len(fly.strategies)), bool)
    settled = ds.ts[wi] + ds.hold_s[wi] + 60.0 <= s_epoch                    # labels known at S
    per = {}
    for j, name in enumerate(fly.strategies):
        arm = fly.arm(j); y = label(ds, arm)
        known = trig[:, j] & settled; ci = wi[known]
        cal = fly_calibrate.calibrate(ds.ts[ci], ds.mint[ci], ds.hold_s[ci], E[known, j], y[ci])
        fly.lines[name], fly.sizings[name] = cal.line, cal.sizing
        per[name] = {"line": cal.line, "trades": cal.trades, "mean": cal.mean, "total": cal.total, "arm": arm}
    d = kalshi_decide(fly, ds.X[wi], ds.cols, E, ds.hold_s[wi]) if len(wi) else {"strategy": np.array([], dtype=object)}
    picked = np.array([x is not None for x in d["strategy"]], bool)
    ret = np.array([label(ds, fly.arm(fly.strategies.index(x)))[i] if x is not None else np.nan for i, x in zip(wi, d["strategy"])], dtype=np.float64)
    known = picked & settled & np.isfinite(ret)
    tr = taken_idx(ds.ts[wi], ds.mint[wi], ds.hold_s[wi], np.flatnonzero(known)); r = ret[tr]
    comb = {"trades": int(len(r)), "mean": float(np.mean(r)) if len(r) else None, "total": float(np.sum(r)) if len(r) else 0.0, "per_strategy": per}
    ci = wi[settled]
    di = ci if len(ci) <= F.DIAG_ROWS else np.sort(np.random.default_rng(seed).choice(ci, F.DIAG_ROWS, replace=False))
    t_sc = teacher.targets(ds.X[di], ds.ts[di])[0][:, 0] if len(di) else None
    if t_sc is not None:
        ok_t = np.isfinite(t_sc); di, t_sc = di[ok_t], t_sc[ok_t]
    diag = F.diagnose(fly, ds.X[di], t_sc) if len(di) else {"mbon_share": 0.0, "mbon_saturated": 1.0, "kc_active": 0.0, "rows": 0}
    ok, why = F.gate_check(diag)
    info = {"S": str(Sday), "train_through": str(train_end - timedelta(days=1)), "calibration_days": [str(Sday - timedelta(days=CALIB_DAYS)), str(Sday - timedelta(days=1))],
            "line": fly.threshold, "sizing": fly.sizing, "lines": fly.lines, "calibration": comb, "strategies": fly.strategies, "rules": fly.rules,
            "diagnostics": diag, "gates_ok": ok, "gate_failures": why, "fit": fit_info, "teacher": teacher_note or TEACHER_NOTE, "teacher_line": getattr(teacher, "threshold", None),
            "teacher_stack": stack_info, "photoreceptors": int(len(fly.net.aff_rows)), "neurons": int(fly.net.N)}
    log.info("kalshi fly bootstrap for %s: strategies %s, edge lines %s | calibration %d trades, mean %s | MBON share %.1f%% saturated %.0f%% | gates %s",
             Sday, fly.strategies, {k: round(v, 4) for k, v in fly.lines.items()}, comb["trades"], f"{comb['mean'] * 100:+.2f}%" if comb["mean"] is not None else "-",
             diag["mbon_share"] * 100, diag["mbon_saturated"] * 100, "ok" if ok else "; ".join(why))
    return fly, info


# ---------------------------------------------------------------- snapshots (the memecoin machinery under this fly's kind and version)
def is_current(meta: dict | None) -> bool:
    return F.is_current(meta, KALSHI_FLY_VERSION)


def deployable(meta: dict | None) -> tuple[bool, str]:
    return F.deployable(meta, KALSHI_FLY_VERSION)


def save(fly: KalshiFlyModel, metrics: dict, run_id: str | None = None):
    return F.save(fly, metrics, run_id, kind=KIND, subdir=SUBDIR, version=KALSHI_FLY_VERSION, prefix=PREFIX)


def load(path, g=None, device=None) -> KalshiFlyModel:
    return F.load(path, g if g is not None else graph(), device, model_cls=KalshiFlyModel)


def latest_current(conn) -> dict | None:
    return F.latest_current(conn, KIND, KALSHI_FLY_VERSION)


def latest_deployable(conn) -> dict | None:
    return F.latest_deployable(conn, KIND, KALSHI_FLY_VERSION)


def main(days: int | None = None, epochs: int | None = None, stop_event: threading.Event | None = None) -> dict:
    """Bootstrap a Kalshi fly to trade from tomorrow on, taught by the deployable Kalshi selector (or a refit stack)."""
    from ..ops.reset import reset_training_stats
    reset_training_stats(KIND, reason="kalshi fly bootstrap")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("kalshi fly: building decision points", 0, 1, force=True)
    ds = KD.build(days=days); Sday = ds.days[-1] + timedelta(days=1)
    teacher, note, taught_by = deployed_teacher(ds, ds.day < Sday - timedelta(days=CALIB_DAYS + PURGE_DAYS))
    log.info("kalshi fly bootstrap teacher: %s", note)
    fly, info = bootstrap(ds, Sday, epochs=F.EPOCHS if epochs is None else epochs, stop=stop_event, teacher=teacher, teacher_note=note)
    if fly is None:
        return {"stopped": True, **info}
    info["teacher_snapshot"] = taught_by
    record_event("info", KIND, "kalshi fly bootstrap", {k: info.get(k) for k in ("S", "lines", "calibration", "diagnostics", "gates_ok", "gate_failures", "teacher")})
    path, sid = save(fly, info)
    prog.update("kalshi fly: done", 1, 1, force=True, snapshot_id=sid, line=info["line"], gates_ok=info["gates_ok"], diagnostics=info["diagnostics"])
    return {**info, "snapshot_id": sid, "path": str(path)}
