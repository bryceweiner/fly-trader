"""The fly: the FlyWire central brain trained to imitate the selector, and traded exactly like it.

Network (``FlyNet``): the same robust-scaled features the selector sees (``train/scaling.py``, the selector's own
scaler) are injected in full — every feature, no bottleneck — into the 5,460 sensory (afferent) neurons through a
learned projection; activity propagates ``K_STEPS`` steps along the real, signed synapses (leaky tanh rate units,
synaptic magnitudes initialised from synapse counts, signs and topology fixed); the 1,303 descending neurons'
activity is normalised with running statistics (homeostasis: the signal reaching the output is weak, so it is
rescaled rather than lost); a non-saturating decoder (GELU) and a linear head give the predicted net return over the hold.
The head starts at the mean target.

Training (distillation, full budget): every training minute, in every epoch (``EPOCHS``), shuffled; target = the
teacher's predicted net return — the selector fit on the same training days with the same configuration; loss =
mean squared error weighted ``TOP_WEIGHT`` × on the minutes in the teacher's top 1 % of training scores, because the
top of the distribution is all that gets traded.

Trading and evaluation: identical to the selector — same universe (``selector.in_universe``), the same buy line (the
one the selector's walk-forward chose), gates, one position per token, hold, pessimistic fills, paper costs and sizing
procedure. On the last ``test_days`` days both models are scored out of sample: fly, selector (teacher) and random
picks, each score's rank correlation with the realised returns (IC), and how closely the fly copies the teacher
(``pick_overlap``: share of the teacher's picks the fly also picks; ``rank_corr``: of the two scores).
Snapshots: ``data/brain/policies/fly_<ts>.pt`` + ``brain_snapshots`` kind 'fly_selector'.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import config
from ..agent import sizing
from ..brain.connectome import AFFERENT_POPS, EFFERENT_POP, Connectome, SubConnectome
from ..db.apilog import record_event
from ..db.connection import transaction
from . import progress as prog
from . import selector
from .decisions import HOLD_MIN, DecisionSet, build, evaluate, random_trades, rank_corr, summarize, taken_rows
from .scaling import RobustScaler

log = logging.getLogger(__name__)

SCALE = 20.0                 # the head predicts the net return × 20 (±5 % ≈ ±1)
K_STEPS, LEAK, HIDDEN = 4, 0.5, 128
EPOCHS = 1                   # full passes over every training minute
BATCH = 512
TOP_WEIGHT = 10.0            # loss weight of the teacher's top 1 % of training minutes
LR_GRAPH, LR_HEAD = 1e-4, 1e-3
TEACHER_NOTE = "selector fit on the fly's training days, same configuration"


class FlyNet(nn.Module):
    """Scorer on the connectome: features → afferent neurons → K steps along the synapses → normalised descending
    neurons → decoder → predicted net return (× ``SCALE``)."""

    def __init__(self, graph, obs_dim: int, k_steps: int = K_STEPS, leak: float = LEAK, hidden: int = HIDDEN, device=None):
        super().__init__()
        dev = torch.device(device) if device else graph.device
        self.N, self.k_steps, self.leak, self.obs_dim, self.hidden = graph.N, k_steps, leak, obs_dim, hidden
        self.register_buffer("pre", graph.indices[1].to(dev)); self.register_buffer("post", graph.indices[0].to(dev))
        vals = graph.values_raw.float(); self.register_buffer("sign", torch.sign(vals).to(dev))
        s = float(getattr(graph, "s", 0.0) or 0.0) or 0.99 / float(getattr(graph, "spectral_radius_raw", None) or 2788.0)
        self.theta = nn.Parameter(torch.log(torch.expm1((vals.abs() * s).clamp(min=1e-6))).to(dev))   # softplus^-1(|w|)
        self.gain = nn.Parameter(torch.ones(self.N, device=dev)); self.bias = nn.Parameter(torch.zeros(self.N, device=dev))
        aff = [torch.arange(*graph.pop_ranges[p]) for p in AFFERENT_POPS if p in graph.pop_ranges]
        self.register_buffer("aff_rows", torch.cat(aff).to(dev)); self.register_buffer("eff_rows", torch.arange(*graph.pop_ranges[EFFERENT_POP]).to(dev))
        n_aff, n_eff = len(self.aff_rows), len(self.eff_rows)
        self.w_in = nn.Parameter(torch.randn(n_aff, obs_dim, device=dev) / obs_dim ** 0.5)       # every feature reaches every sensory neuron
        self.eff_norm = nn.BatchNorm1d(n_eff).to(dev)
        self.dec = nn.Sequential(nn.Linear(n_eff, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()).to(dev)
        self.head = nn.Linear(hidden, 1).to(dev)
        self.dev = dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        I_in = torch.zeros(self.N, B, device=self.dev).index_copy(0, self.aff_rows, self.w_in @ x.T)
        w = self.sign * Fn.softplus(self.theta); h = torch.zeros(self.N, B, device=self.dev)
        for _ in range(self.k_steps):
            msg = torch.zeros_like(h).index_add(0, self.post, h.index_select(0, self.pre) * w[:, None])
            h = torch.tanh(self.leak * h + self.gain[:, None] * msg + self.bias[:, None] + I_in)
        return self.head(self.dec(self.eff_norm(h.index_select(0, self.eff_rows).T))).squeeze(-1)

    def param_groups(self, lr_graph: float, lr_head: float) -> list[dict]:
        graph = [self.theta, self.gain, self.bias]; ids = {id(p) for p in graph}
        return [{"params": graph, "lr": lr_graph}, {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr_head}]


class FlyModel:
    """A trained fly as a trading model: the same interface as ``selector.SelectorModel``."""

    def __init__(self, net: FlyNet, scaler: RobustScaler, cols: list[str], threshold: float, horizon_min: int):
        self.net, self.scaler, self.cols, self.threshold, self.horizon_min = net, scaler, list(cols), threshold, horizon_min

    @torch.no_grad()
    def score(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        """Predicted net return over the hold."""
        self.net.eval(); out = np.empty(len(X), np.float64)
        for i in range(0, len(X), batch):
            out[i:i + batch] = (self.net(torch.tensor(self.scaler.transform(X[i:i + batch]), device=self.net.dev)) / SCALE).float().cpu().numpy()
        return out

    def universe(self, X: np.ndarray) -> np.ndarray:
        return selector.in_universe(X, self.cols)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _graph():
    return SubConnectome(Connectome.load(), exclude=("VISUAL",))


def train_fly(ds: DecisionSet, train: np.ndarray, teacher, epochs: int = EPOCHS, batch: int = BATCH, stop: threading.Event | None = None,
              seed: int = 0, graph=None, device=None) -> tuple[FlyModel | None, dict]:
    """Distil ``teacher`` (anything with ``score``, ``scaler`` and ``threshold``) into a FlyNet over every training minute."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = torch.device(device) if device else _device()
    net = FlyNet(graph if graph is not None else _graph(), obs_dim=len(ds.cols), device=dev)
    idx = np.flatnonzero(train); tgt = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), 500_000):
        tgt[i:i + 500_000] = teacher.score(ds.X[idx[i:i + 500_000]])
    tgt = np.clip(tgt, -1.0, 1.0); wts = np.where(tgt >= np.quantile(tgt, 0.99), TOP_WEIGHT, 1.0).astype(np.float32)
    with torch.no_grad():
        net.head.bias.fill_(float(tgt.mean()) * SCALE); net.head.weight.mul_(0.1)
    opt = torch.optim.Adam(net.param_groups(LR_GRAPH, LR_HEAD)); n_b = len(idx) // batch; total = max(1, epochs * n_b); hist = []; t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(idx)); tot = 0.0; net.train()
        for b in range(n_b):
            if stop is not None and stop.is_set():
                return None, {"epochs": hist, "stopped": True}
            sel = perm[b * batch:(b + 1) * batch]
            x = torch.tensor(teacher.scaler.transform(ds.X[idx[sel]]), device=dev)
            t = torch.tensor(tgt[sel] * SCALE, device=dev); w = torch.tensor(wts[sel], device=dev)
            loss = (w * (net(x) - t) ** 2).sum() / w.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tot += loss.item()
            if b % 200 == 0:
                done = ep * n_b + b + 1; el = time.time() - t0
                prog.update("fly: learning from the selector", done, total, epoch=ep, loss=tot / (b + 1), rows_per_s=done * batch / max(el, 1e-9),
                            eta_s=(total - done) * el / done)
        hist.append({"epoch": ep, "mse": tot / max(n_b, 1), "secs": time.time() - t0, "rows": n_b * batch})
        log.info("fly epoch %d: weighted mse %.4f over %d rows (%.0fs)", ep, tot / max(n_b, 1), n_b * batch, time.time() - t0)
    return FlyModel(net, teacher.scaler, ds.cols, teacher.threshold, int(ds.horizon_s // 60)), {"epochs": hist, "stopped": False, "rows": int(len(idx)), "top_weight": TOP_WEIGHT}


def agreement(fly_scores: np.ndarray, fly_thr: float, teacher_scores: np.ndarray, teacher_thr: float, rows: np.ndarray,
              sample: int = 200_000, seed: int = 3) -> dict:
    """How closely the student copies the teacher on ``rows``: share of the teacher's picks the fly also picks,
    and the rank correlation of the two scores on up to ``sample`` of those rows."""
    t_pick = rows & (teacher_scores >= teacher_thr); f_pick = rows & (fly_scores >= fly_thr)
    n_t = int(t_pick.sum()); n_both = int((t_pick & f_pick).sum())
    idx = np.flatnonzero(rows)
    if len(idx) > sample:
        idx = np.random.default_rng(seed).choice(idx, sample, replace=False)
    return {"pick_overlap": (n_both / n_t) if n_t else None, "rank_corr": rank_corr(fly_scores[idx], teacher_scores[idx]),
            "teacher_picks": n_t, "fly_picks": int(f_pick.sum()), "both_picks": n_both, "rank_rows": int(len(idx))}


def save(fly: FlyModel, metrics: dict, run_id: str | None = None) -> tuple[Path, int]:
    root = config.BRAIN_DIR / "policies"; root.mkdir(parents=True, exist_ok=True)
    path = root / f"fly_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt"
    torch.save({"state_dict": fly.net.state_dict(), "scaler": fly.scaler.state(), "cols": fly.cols, "threshold": fly.threshold, "horizon_min": fly.horizon_min,
                "config": {"k_steps": fly.net.k_steps, "leak": fly.net.leak, "hidden": fly.net.hidden, "obs_dim": fly.net.obs_dim, "scale": SCALE}, "metrics": metrics}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'fly_selector',%s) RETURNING id",
                           (run_id, str(path), sha, json.dumps(metrics, default=str))).fetchone()
    return path, int(row["id"])


def main(days: int | None = None, test_days: int = 21, line: float | None = None, epochs: int = EPOCHS, horizon_min: int = HOLD_MIN,
         stop_event: threading.Event | None = None) -> dict:
    """Train on every day before the last ``test_days`` (one-day purge), score both models on the last ``test_days``
    at the same buy line (``line``: the selector's walk-forward choice)."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("fly_selector", reason="fly training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("fly: building decision points", 0, 1, force=True)
    ds = build(days=days, horizon_min=horizon_min); dl = ds.days; D_cut = dl[-test_days]
    train = ds.day < (D_cut - timedelta(days=1)); test_all = ds.day >= D_cut
    prog.update("fly: fitting its teacher (the selector, same configuration)", 0, 1, force=True)
    teacher = selector.fit(ds, train, seed=7); teacher.threshold = selector.MIN_EV if line is None else float(line); thr = teacher.threshold
    tix = np.flatnonzero(test_all); tix = tix[teacher.universe(ds.X[tix])]; test = np.zeros(len(ds.y), bool); test[tix] = True
    gs = np.full(len(ds.y), np.nan); gs[tix] = teacher.score(ds.X[tix])
    log.info("fly: %d rows, train %s..%s, test %s..%s (%d tradable rows), line %.2f%%", len(ds.y), dl[0], D_cut - timedelta(days=2), D_cut, dl[-1], len(tix), thr * 100)
    fly, fit_info = train_fly(ds, train, teacher, epochs=epochs, stop=stop_event)
    if fly is None:
        return {"stopped": True}
    prog.update("fly: scoring the test days", 0, 1, force=True)
    fs = np.full(len(ds.y), np.nan); fs[tix] = fly.score(ds.X[tix])
    g_ev = evaluate(ds, gs, test, thr, "selector"); f_ev = evaluate(ds, fs, test, thr, "fly")
    rnd = summarize(random_trades(ds, test, max(1, int((test & (fs >= thr)).sum()))))
    agree = agreement(fs, thr, gs, thr, test)
    rows = taken_rows(ds, test & (fs >= thr)); size_table = sizing.build_table(fs[rows] - thr, ds.fwd_pess[rows])   # the fly's own certainty bands
    f_mean = f_ev["pooled"]["mean"]
    verdict = {"fly": {"ic": selector.ic(ds, fs, test), **f_ev["pooled"]}, "gbm": {"ic": selector.ic(ds, gs, test), **g_ev["pooled"]}, "random": rnd, "line": thr,
               "fly_beats_gbm": bool((f_mean if f_mean is not None else -1) > (g_ev["pooled"]["mean"] if g_ev["pooled"]["mean"] is not None else -1)),
               "fly_positive": bool(f_mean is not None and f_mean > 0), "agreement": agree, "per_day_fly": f_ev["per_day"], "per_day_gbm": g_ev["per_day"],
               "fit": fit_info, "test_days": [str(D_cut), str(dl[-1])], "sizing": size_table}
    log.info("fly: %s | selector: %s | agreement %s | positive %s", f_ev["pooled"], g_ev["pooled"], agree, verdict["fly_positive"])
    record_event("info", "fly_selector", "fly (distilled) vs gbm (single split)",
                 {k: verdict[k] for k in ("fly", "gbm", "random", "agreement", "fly_positive", "fly_beats_gbm", "line")})
    path, sid = save(fly, {**{k: verdict[k] for k in ("fly", "gbm", "random", "fly_beats_gbm", "fly_positive", "agreement", "test_days", "sizing", "line", "fit")},
                           "teacher": TEACHER_NOTE, "data": selector.DATA_VERSION})
    verdict["snapshot_id"] = sid
    prog.update("fly: done", 1, 1, force=True, snapshot_id=sid, verdict={k: verdict[k] for k in ("fly", "gbm", "random", "agreement", "fly_positive", "fly_beats_gbm")})
    return verdict
