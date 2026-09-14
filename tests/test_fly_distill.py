"""The fly is distilled from the selector: fit() with a teacher trains on the teacher's probabilities, and the
agreement helpers (pick overlap, numpy Spearman) are correct. The connectome is swapped for a tiny stand-in."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fly_trader.brain import lif
from fly_trader.train import fly_selector
from fly_trader.train.decisions import DecisionSet


class TinyPolicy(nn.Module):
    """Same surface as ConnectomePolicy as FlyScorer uses it."""

    def __init__(self, graph, obs_dim, k_steps=4, device=None):
        super().__init__()
        self.obs_dim = obs_dim; self.k_steps = k_steps
        self.enc = nn.Linear(obs_dim, 16); self.dec = nn.Linear(16, 1)

    def init_hidden(self, batch):
        return torch.zeros(1, batch)

    def forward(self, obs, h):
        return SimpleNamespace(mu=self.dec(torch.tanh(self.enc(obs))).squeeze(-1), h=h)

    def param_groups(self, lr_graph, lr_heads):
        return [{"params": list(self.enc.parameters()), "lr": lr_graph}, {"params": list(self.dec.parameters()), "lr": lr_heads}]


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(lif.Connectome, "load", staticmethod(lambda *a, **k: SimpleNamespace(N=1)))
    monkeypatch.setattr(fly_selector, "SubConnectome", lambda c, exclude=(): c)
    monkeypatch.setattr(fly_selector, "ConnectomePolicy", TinyPolicy)
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)


def _ds(n=6000, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, f)) * np.array([1, 2, 0.5, 3, 1, 1]) + 5).astype(np.float32)
    y = (rng.random(n) < 0.05).astype(np.int8)             # labels unrelated to the teacher: only the teacher can be learned
    days = np.array([date(2026, 9, 1 + i % 4) for i in range(n)], dtype=object)
    return DecisionSet(X=X, y=y, fwd=np.zeros(n, np.float32), fwd_pess=np.zeros(n, np.float32), day=days, ts=np.arange(n, dtype=np.float64) * 60,
                       mint=np.array([f"m{i % 50}" for i in range(n)]), cols=[f"f{i}" for i in range(f)], horizon_s=1800.0)


def _teacher(X):
    z = (X - 5) / np.array([1, 2, 0.5, 3, 1, 1])
    return 1 / (1 + np.exp(-(2.0 * z[:, 0] - 1.5 * z[:, 2] + 0.5 * z[:, 3])))


def test_fit_with_teacher_imitates_the_teacher(tiny):
    torch.manual_seed(0)
    ds = _ds(); train = np.ones(len(ds.y), bool)
    fly = fly_selector.FlyScorer(ds, train, device="cpu")
    before = fly_selector.rank_corr(fly.score(ds.X), _teacher(ds.X))
    info = fly.fit(ds, train, epochs=8, rows_per_epoch=len(ds.y), batch=128, lr_graph=1e-2, lr_heads=1e-2, teacher=_teacher)
    assert info["target"] == "teacher" and not info["stopped"]
    assert info["epochs"][-1]["bce"] < info["epochs"][0]["bce"]
    s = fly.score(ds.X); t = _teacher(ds.X)
    rc = fly_selector.rank_corr(s, t)
    assert rc > 0.95 and rc > (before or 0)
    assert np.abs(s - t).mean() < 0.08                       # soft targets: calibrated to the teacher's probabilities, not to y's 5 %
    thr = fly.set_threshold(ds, train, 0.05)
    t_thr = float(np.quantile(t, 0.95))
    a = fly_selector.agreement(s, thr, t, t_thr, train)
    assert a["pick_overlap"] > 0.6 and a["teacher_picks"] > 0


def test_fit_without_teacher_keeps_the_label_path(tiny, monkeypatch):
    ds = _ds(n=1024); train = np.ones(len(ds.y), bool)
    fly = fly_selector.FlyScorer(ds, train, device="cpu")
    seen = []
    orig = torch.nn.functional.binary_cross_entropy_with_logits
    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits",
                        lambda logit, target, **k: seen.append(k.get("pos_weight") is not None) or orig(logit, target, **k))
    info = fly.fit(ds, train, epochs=1, rows_per_epoch=512, batch=128)
    assert info["target"] == "labels" and seen and all(seen)       # class-weighted BCE on y
    seen.clear(); fly.fit(ds, train, epochs=1, rows_per_epoch=512, batch=128, teacher=_teacher)
    assert seen and not any(seen)                                    # soft targets: no pos_weight


def test_rank_corr_matches_spearman_with_ties():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 20, 500).astype(float); b = a + rng.normal(scale=5, size=500); b[::7] = b[0]
    ref = pd.Series(a).corr(pd.Series(b), method="spearman")
    assert fly_selector.rank_corr(a, b) == pytest.approx(ref, abs=1e-12)
    x = rng.random(100)
    assert fly_selector.rank_corr(x, np.exp(3 * x)) == pytest.approx(1.0)
    assert fly_selector.rank_corr(x, -x) == pytest.approx(-1.0)
    assert fly_selector.rank_corr(x, np.ones(100)) is None
    assert fly_selector.rank_corr(np.array([1.0]), np.array([2.0])) is None
    assert list(fly_selector._ranks(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_agreement_overlap_counts_only_rows_in_the_mask():
    rows = np.array([True] * 8 + [False] * 2)
    t = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .95, .95])      # teacher picks rows 0,1,2 (8,9 are out of the mask)
    f = np.array([.9, .1, .7, .8, .1, .1, .1, .1, .99, .99])      # fly picks rows 0,2,3
    a = fly_selector.agreement(f, 0.5, t, 0.5, rows)
    assert a["teacher_picks"] == 3 and a["fly_picks"] == 3 and a["both_picks"] == 2
    assert a["pick_overlap"] == pytest.approx(2 / 3)
    assert a["rank_rows"] == 8 and a["rank_corr"] == pytest.approx(pd.Series(f[:8]).corr(pd.Series(t[:8]), method="spearman"))
    assert fly_selector.agreement(f, 0.5, t, 2.0, rows)["pick_overlap"] is None     # teacher picks nothing
    big = np.ones(1000, bool); s = np.random.default_rng(0).random(1000)
    assert fly_selector.agreement(s, 0.5, s, 0.5, big, sample=100)["rank_rows"] == 100
