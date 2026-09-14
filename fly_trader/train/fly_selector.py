"""The fly as the selector: the connectome policy (``brain/policy.py``) distilled from the gradient-boosted selector
(``train/selector.py``) on the same decision points and single walk-forward split.

Each eligible minute is one column of the connectome batch: standardized features → learned encoder → afferent
populations → k message-passing steps over the signed, count-initialised synapses → descending-neuron decoder →
one logit (``mu``). The hidden state starts at zero for every minute (stateless scoring), so training and live
scoring are identical.

Target (knowledge distillation): the teacher is the selector fit on the fly's own training split (days ≤ D_cut−2),
so the test days stay out of sample for both. The fly's target for each training minute is the teacher's
probability for that minute; loss = binary cross-entropy on those soft targets (no class weighting). The fly
imitates the selector rather than learning the raw labels ``y`` (net 30-minute return above +3 %).
``FlyScorer.fit(teacher=None)`` keeps the old label-based loss (class-weighted BCE on ``y``).

Evaluation on the last ``test_days`` days, unchanged protocol (``decisions.evaluate``): each model trades the test
minutes at or above the top ``top_frac`` quantile of its own training scores, one position per token, pessimistic
fills, paper costs. Reported: fly (student), gbm (teacher), and random (same pick count as the fly). Agreement
measures how well the student copied the teacher on the test days: ``pick_overlap`` = share of the teacher's test
picks the fly also picks; ``rank_corr`` = Spearman correlation of fly vs teacher scores on (a sample of) test rows.
``fly_positive`` = the fly's pooled mean per trade is above zero; ``fly_beats_gbm`` = it beats the teacher's.
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
from typing import Callable

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

TEACHER_NOTE = "selector fit on the fly's training split"


def _ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (what Spearman needs)."""
    _, inv, cnt = np.unique(np.asarray(a), return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.ravel()]


def rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman correlation (Pearson on average ranks); None when either side is constant or empty."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return None
    ra = _ranks(a); rb = _ranks(b); ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


def agreement(fly_scores: np.ndarray, fly_thr: float, teacher_scores: np.ndarray, teacher_thr: float, rows: np.ndarray,
              sample: int = 200_000, seed: int = 3) -> dict:
    """How closely the student copies the teacher on ``rows``: share of the teacher's picks the fly also picks,
    and Spearman correlation of the two scores on up to ``sample`` of those rows."""
    t_pick = rows & (teacher_scores >= teacher_thr); f_pick = rows & (fly_scores >= fly_thr)
    n_t = int(t_pick.sum()); n_both = int((t_pick & f_pick).sum())
    idx = np.flatnonzero(rows)
    if len(idx) > sample:
        idx = np.random.default_rng(seed).choice(idx, sample, replace=False)
    return {"pick_overlap": (n_both / n_t) if n_t else None, "rank_corr": rank_corr(fly_scores[idx], teacher_scores[idx]),
            "teacher_picks": n_t, "fly_picks": int(f_pick.sum()), "both_picks": n_both, "rank_rows": int(len(idx))}


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
            lr_graph: float = 1e-4, lr_heads: float = 1e-3, stop: threading.Event | None = None, seed: int = 0,
            teacher: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
        """With ``teacher`` (X → probabilities, e.g. ``SelectorModel.score``): distillation, BCE on the teacher's
        probabilities as soft targets (no pos_weight). Without: class-weighted BCE on the labels ``y``."""
        idx_all = np.flatnonzero(train); rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.policy.param_groups(lr_graph, lr_heads))
        hist = []; t0 = time.time(); step = 0; target = "teacher" if teacher is not None else "labels"
        for ep in range(epochs):
            idx = rng.choice(idx_all, min(rows_per_epoch, len(idx_all)), replace=False); n_b = len(idx) // batch
            tgt = None
            if teacher is not None:        # the teacher's probabilities for this epoch's sampled rows, scored in chunks
                ch = 200_000
                tgt = np.concatenate([np.asarray(teacher(ds.X[idx[i:i + ch]]), dtype=np.float32) for i in range(0, n_b * batch, ch)]) if n_b else None
            tot = 0.0
            for b in range(n_b):
                if stop is not None and stop.is_set():
                    return {"epochs": hist, "stopped": True, "target": target}
                bi = idx[b * batch:(b + 1) * batch]
                logit = self.logits(ds.X[bi])
                if tgt is None:
                    y = torch.tensor(ds.y[bi].astype(np.float32), device=self.dev)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_weight)
                else:
                    t = torch.tensor(tgt[b * batch:(b + 1) * batch], device=self.dev)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, t)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0); opt.step()
                tot += loss.item(); step += 1
                if b % 100 == 0:
                    prog.update("fly selector: training", ep * n_b + b, epochs * n_b, epoch=ep, loss=tot / (b + 1), target=target, elapsed_s=time.time() - t0)
            hist.append({"epoch": ep, "bce": tot / max(n_b, 1), "secs": time.time() - t0})
            log.info("fly selector epoch %d: bce %.4f vs %s (%.0fs)", ep, tot / max(n_b, 1), target, time.time() - t0)
        return {"epochs": hist, "stopped": False, "target": target}

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
                               (run_id, str(path), sha, json.dumps(metrics, default=str))).fetchone()
        return path, int(row["id"])


def main(days: int | None = None, test_days: int = 21, top_frac: float = 0.01, horizon_min: int = 30, epochs: int = 2, rows_per_epoch: int = 600_000,
         stop_event: threading.Event | None = None) -> dict:
    """Single split: fit the selector (teacher) on days ≤ D_cut−2, distill the fly from its scores on the same days,
    evaluate both (and a random baseline) on the last ``test_days`` days."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("fly_selector", reason="fly selector training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("fly selector: building decision points", 0, 1, force=True)
    ds = build(days=days, horizon_min=horizon_min); dl = ds.days; D_cut = dl[-test_days]
    train = ds.day < (D_cut - timedelta(days=1)); test = ds.day >= D_cut
    log.info("fly selector: %d rows, train days %s..%s, test days %s..%s", len(ds.y), dl[0], D_cut - timedelta(days=2), D_cut, dl[-1])
    # teacher: the gradient-boosted selector fit on the fly's own training split (test days out of sample for both)
    from .selector import DATA_VERSION, fit as gbm_fit
    prog.update("fly selector: fitting the teacher (gradient-boosted selector)", 0, 1, force=True)
    g = gbm_fit(ds, train, top_frac, seed=7); gs = np.zeros(len(ds.y)); gs[test] = g.score(ds.X[test])
    g_ev = evaluate(ds, gs, test, g.threshold, "gbm"); g_auc = float(roc_auc_score(ds.y[test], gs[test]))
    log.info("teacher GBM (single split): AUC %.3f | %s", g_auc, g_ev["pooled"])
    record_event("info", "fly_selector", "teacher gbm single-split", {"auc": g_auc, **g_ev["pooled"]})
    fly = FlyScorer(ds, train)
    prog.update("fly selector: distilling the fly from the selector", 0, 1, force=True,
                graph={"N": fly.graph.N, "edges": int(fly.graph.indices.shape[1]), "params": fly.policy.describe()["params"]},
                reference_gbm={"auc": g_auc, **g_ev["pooled"]})
    fit_info = fly.fit(ds, train, epochs=epochs, rows_per_epoch=rows_per_epoch, stop=stop_event, teacher=g.score)
    thr = fly.set_threshold(ds, train, top_frac)
    fs = np.zeros(len(ds.y)); fs[test] = fly.score(ds.X[test]); f_auc = float(roc_auc_score(ds.y[test], fs[test]))
    f_ev = evaluate(ds, fs, test, thr, "fly")
    rnd = summarize(random_trades(ds, test, int((test & (fs >= thr)).sum())))          # no-skill baseline at the fly's pick count
    agree = agreement(fs, thr, gs, g.threshold, test)
    f_mean = f_ev["pooled"]["mean"]
    verdict = {"fly": {"auc": f_auc, **f_ev["pooled"]}, "gbm": {"auc": g_auc, **g_ev["pooled"]}, "random": rnd,
               "fly_beats_gbm": bool((f_mean or -1) > (g_ev["pooled"]["mean"] or -1)), "fly_positive": bool(f_mean is not None and f_mean > 0),
               "agreement": agree, "per_day_fly": f_ev["per_day"], "per_day_gbm": g_ev["per_day"], "fit": fit_info,
               "test_days": [str(D_cut), str(dl[-1])]}
    log.info("fly selector: AUC %.3f | %s | agreement %s | positive %s | beats GBM: %s", f_auc, f_ev["pooled"], agree, verdict["fly_positive"], verdict["fly_beats_gbm"])
    record_event("info", "fly_selector", "fly (distilled) vs gbm (single split)",
                 {k: verdict[k] for k in ("fly", "gbm", "random", "agreement", "fly_positive", "fly_beats_gbm")})
    path, sid = fly.save({**{k: verdict[k] for k in ("fly", "gbm", "random", "fly_beats_gbm", "fly_positive", "agreement", "test_days")},
                          "teacher": TEACHER_NOTE, "data": DATA_VERSION})
    verdict["snapshot_id"] = sid
    prog.update("fly selector: done", 1, 1, force=True, snapshot_id=sid,
                verdict={k: verdict[k] for k in ("fly", "gbm", "random", "agreement", "fly_positive", "fly_beats_gbm")})
    return verdict
