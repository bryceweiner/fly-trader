"""The fly as the selector: the connectome policy (``brain/policy.py``) trained as a scorer on the same decision
points, labels and walk-forward protocol as the gradient-boosted selector (``train/selector.py``).

Each eligible minute is one column of the connectome batch: standardized features → learned encoder → afferent
populations → k message-passing steps over the signed, count-initialised synapses → descending-neuron decoder →
one logit (``mu``). Loss: class-weighted binary cross-entropy on ``y`` (net 30-minute return above +3 %). The
hidden state starts at zero for every minute (stateless scoring), so training and live scoring are identical.
Evaluation reuses ``decisions.evaluate``: top ``top_frac`` of training scores as the threshold, one position per
token, pessimistic fills. The fly must beat the gradient-boosted selector under this protocol to take its seat.
Snapshots: ``data/brain/policies/flysel_<ts>.pt`` + ``brain_snapshots`` kind 'fly_selector'.
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
from sklearn.metrics import roc_auc_score

from .. import config
from ..brain.policy import ConnectomePolicy, SubConnectome
from ..db.apilog import record_event
from ..db.connection import transaction
from . import progress as prog
from .decisions import DecisionSet, build, evaluate, random_trades, summarize, trades_from_picks

log = logging.getLogger(__name__)


class FlyScorer:
    def __init__(self, ds: DecisionSet, train: np.ndarray, subgraph: str = "central", k_steps: int = 4, device: str | None = None):
        from ..brain.lif import Connectome
        c = Connectome.load()
        self.graph = SubConnectome(c, exclude=("VISUAL",)) if subgraph == "central" else c
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.cols = ds.cols
        idx = np.flatnonzero(train)
        self.mean = ds.X[idx].mean(0).astype(np.float32); self.std = (ds.X[idx].std(0) + 1e-6).astype(np.float32)
        self.policy = ConnectomePolicy(self.graph, obs_dim=ds.X.shape[1], k_steps=k_steps, device=self.dev)
        p = float(ds.y[idx].mean()); self.pos_weight = torch.tensor((1 - p) / max(p, 1e-3), device=self.dev)
        self.threshold = 0.5

    def logits(self, X: np.ndarray) -> torch.Tensor:
        obs = torch.tensor((X - self.mean) / self.std, device=self.dev)
        h = self.policy.init_hidden(obs.shape[0])
        return self.policy(obs, h).mu

    @torch.no_grad()
    def score(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        self.policy.eval(); out = []
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(self.logits(X[i:i + batch])).float().cpu().numpy())
        self.policy.train()
        return np.concatenate(out) if out else np.array([])

    def fit(self, ds: DecisionSet, train: np.ndarray, epochs: int = 2, rows_per_epoch: int = 600_000, batch: int = 256,
            lr_graph: float = 1e-4, lr_heads: float = 1e-3, stop: threading.Event | None = None, seed: int = 0) -> dict:
        idx_all = np.flatnonzero(train); rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.policy.param_groups(lr_graph, lr_heads))
        hist = []; t0 = time.time(); step = 0
        for ep in range(epochs):
            idx = rng.choice(idx_all, min(rows_per_epoch, len(idx_all)), replace=False); n_b = len(idx) // batch
            tot = 0.0
            for b in range(n_b):
                if stop is not None and stop.is_set():
                    return {"epochs": hist, "stopped": True}
                bi = idx[b * batch:(b + 1) * batch]
                logit = self.logits(ds.X[bi]); y = torch.tensor(ds.y[bi].astype(np.float32), device=self.dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_weight)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0); opt.step()
                tot += float(loss); step += 1
                if b % 100 == 0:
                    prog.update("fly selector: training", ep * n_b + b, epochs * n_b, epoch=ep, loss=tot / (b + 1), elapsed_s=time.time() - t0)
            hist.append({"epoch": ep, "bce": tot / max(n_b, 1), "secs": time.time() - t0}); log.info("fly selector epoch %d: bce %.4f (%.0fs)", ep, tot / max(n_b, 1), time.time() - t0)
        # threshold = top-frac quantile of training scores (on a sample)
        return {"epochs": hist, "stopped": False}

    def set_threshold(self, ds: DecisionSet, train: np.ndarray, top_frac: float, sample: int = 400_000, seed: int = 1) -> float:
        idx = np.flatnonzero(train); rng = np.random.default_rng(seed)
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        self.threshold = float(np.quantile(self.score(ds.X[idx]), 1 - top_frac)); return self.threshold

    def save(self, metrics: dict, run_id: str | None = None) -> tuple[Path, int]:
        root = config.BRAIN_DIR / "policies"; root.mkdir(parents=True, exist_ok=True)
        path = root / f"flysel_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt"
        torch.save({"state_dict": self.policy.state_dict(), "mean": self.mean, "std": self.std, "cols": self.cols, "threshold": self.threshold,
                    "k_steps": self.policy.k_steps, "obs_dim": self.policy.obs_dim, "connectome_sha256": getattr(self.graph, "content_sha256", None), "metrics": metrics}, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with transaction() as conn:
            row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'fly_selector',%s) RETURNING id",
                               (run_id, str(path), sha, json.dumps(metrics, default=str)[:900])).fetchone()
        return path, int(row["id"])


def main(days: int = 45, test_days: int = 9, top_frac: float = 0.01, horizon_min: int = 30, epochs: int = 2, rows_per_epoch: int = 600_000,
         stop_event: threading.Event | None = None) -> dict:
    """Single-split comparison: train on days ≤ D_cut−2, score the last ``test_days`` days, same protocol for the GBM."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("fly_selector", reason="fly selector training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("fly selector: building decision points", 0, 1, force=True)
    ds = build(days=days, horizon_min=horizon_min); dl = ds.days; D_cut = dl[-test_days]
    train = ds.day < (D_cut - timedelta(days=1)); test = ds.day >= D_cut
    log.info("fly selector: %d rows, train days %s..%s, test days %s..%s", len(ds.y), dl[0], D_cut - timedelta(days=2), D_cut, dl[-1])
    # reference: the gradient-boosted selector under the identical single split
    from .selector import fit as gbm_fit
    prog.update("fly selector: fitting the gradient-boosted reference", 0, 1, force=True)
    g = gbm_fit(ds, train, top_frac, seed=7); gs = np.zeros(len(ds.y)); gs[test] = g.score(ds.X[test])
    g_ev = evaluate(ds, gs, test, g.threshold, "gbm"); g_auc = float(roc_auc_score(ds.y[test], gs[test]))
    log.info("reference GBM (single split): AUC %.3f | %s", g_auc, g_ev["pooled"])
    record_event("info", "fly_selector", "reference gbm single-split", {"auc": g_auc, **g_ev["pooled"]})
    fly = FlyScorer(ds, train)
    prog.update("fly selector: training", 0, 1, force=True, graph={"N": fly.graph.N, "edges": int(fly.graph.indices.shape[1]), "params": fly.policy.describe()["params"]},
                reference_gbm={"auc": g_auc, **g_ev["pooled"]})
    fit_info = fly.fit(ds, train, epochs=epochs, rows_per_epoch=rows_per_epoch, stop=stop_event)
    thr = fly.set_threshold(ds, train, top_frac)
    fs = np.zeros(len(ds.y)); fs[test] = fly.score(ds.X[test]); f_auc = float(roc_auc_score(ds.y[test], fs[test]))
    f_ev = evaluate(ds, fs, test, thr, "fly")
    rnd = summarize(random_trades(ds, test, int((test & (fs >= thr)).sum())))          # no-skill baseline at the fly's pick count
    verdict = {"fly": {"auc": f_auc, **f_ev["pooled"]}, "gbm": {"auc": g_auc, **g_ev["pooled"]}, "random": rnd,
               "fly_beats_gbm": bool((f_ev["pooled"]["mean"] or -1) > (g_ev["pooled"]["mean"] or -1)),
               "per_day_fly": f_ev["per_day"], "per_day_gbm": g_ev["per_day"], "fit": fit_info, "test_days": [str(D_cut), str(dl[-1])]}
    log.info("fly selector: AUC %.3f | %s | beats GBM: %s", f_auc, f_ev["pooled"], verdict["fly_beats_gbm"])
    record_event("info", "fly_selector", "fly vs gbm (single split)", {"fly": verdict["fly"], "gbm": verdict["gbm"], "random": rnd, "fly_beats_gbm": verdict["fly_beats_gbm"]})
    path, sid = fly.save({k: verdict[k] for k in ("fly", "gbm", "random", "fly_beats_gbm", "test_days")})
    prog.update("fly selector: done", 1, 1, force=True, snapshot_id=sid, verdict={k: verdict[k] for k in ("fly", "gbm", "random", "fly_beats_gbm")})
    return verdict
