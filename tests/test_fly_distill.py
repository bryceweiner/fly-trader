"""The fly: FlyNet on a tiny stand-in connectome learns to imitate a teacher, is built without saturating or
bottlenecked parts, starts at the mean target, and the agreement / rank-correlation helpers are correct."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fly_trader.train import fly_selector
from fly_trader.train.decisions import DecisionSet, _ranks, rank_corr
from fly_trader.train.scaling import RobustScaler


def _graph(n=60, seed=0):
    """30 sensory neurons (ORN_FOOD) → 20 interneurons → 10 descending neurons, random signed synapses."""
    g = torch.Generator().manual_seed(seed); pre, post = [], []
    for a, b in ((range(0, 30), range(30, 50)), (range(30, 50), range(50, 60)), (range(0, 30), range(50, 60))):
        for j in b:
            for i in torch.randint(a.start, a.stop, (6,), generator=g).tolist():
                pre.append(i); post.append(j)
    vals = (torch.rand(len(pre), generator=g) * 5 + 1) * torch.where(torch.rand(len(pre), generator=g) < 0.8, 1.0, -1.0)
    return SimpleNamespace(N=n, indices=torch.tensor([post, pre]), values_raw=vals, pop_ranges={"ORN_FOOD": (0, 30), "DESCENDING": (50, 60)},
                           device=torch.device("cpu"), s=0.05, spectral_radius_raw=None)


def _ds(n=6000, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.standard_t(3, size=(n, f)) * np.array([1, 2, 0.5, 300, 1, 1]) + 5).astype(np.float32)      # fat tails, one huge-scale feature
    days = np.array([date(2026, 9, 1 + i % 4) for i in range(n)], dtype=object)
    return DecisionSet(X=X, y=np.zeros(n, np.int8), fwd=np.zeros(n, np.float32), fwd_pess=np.zeros(n, np.float32), day=days, ts=np.arange(n, dtype=np.float64) * 60,
                       mint=np.array([f"m{i % 50}" for i in range(n)]), cols=[f"f{i}" for i in range(f)], horizon_s=1800.0)


class Teacher:
    def __init__(self, ds):
        self.scaler = RobustScaler.fit(ds.X); self.threshold = 0.01

    def score(self, X):
        z = self.scaler.transform(X)
        return 0.02 * z[:, 0] - 0.015 * z[:, 2] + 0.01 * z[:, 3]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)


def test_flynet_has_no_saturating_or_bottleneck_parts_and_starts_at_the_mean():
    ds = _ds(); t = Teacher(ds)
    net = fly_selector.FlyNet(_graph(), obs_dim=len(ds.cols), device="cpu")
    assert tuple(net.w_in.shape) == (30, len(ds.cols))                                  # every feature reaches every sensory neuron
    assert isinstance(net.eff_norm, nn.BatchNorm1d)                                     # homeostasis on the descending neurons
    assert not any(isinstance(m, nn.Tanh) for m in net.dec.modules())                  # the decoder cannot saturate
    fly, info = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=1, batch=500, graph=_graph(), device="cpu")
    assert info["rows"] == len(ds.y) and info["epochs"][0]["rows"] == len(ds.y)          # the full budget: every training row
    start, _ = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=0, graph=_graph(), device="cpu")
    mean_target = float(np.clip(t.score(ds.X), -1, 1).mean())
    assert float(start.net.head.bias) == pytest.approx(mean_target * fly_selector.SCALE, rel=1e-4)   # starts at the mean target


def test_train_fly_imitates_the_teacher_on_the_same_signals():
    torch.manual_seed(0)
    ds = _ds(); train = np.ones(len(ds.y), bool); t = Teacher(ds)
    fly, info = fly_selector.train_fly(ds, train, t, epochs=6, batch=200, graph=_graph(), device="cpu")
    assert not info["stopped"] and info["epochs"][-1]["mse"] < info["epochs"][0]["mse"]
    s, g = fly.score(ds.X), t.score(ds.X)
    assert rank_corr(s, g) > 0.9                                                         # the fat-tailed feature does not blind it
    b = np.polyfit(g, s, 1)[0]
    assert 0.6 < b < 1.4                                                                  # on the teacher's scale, not squashed
    assert fly.scaler is t.scaler and fly.threshold == t.threshold                        # same signals, same buy line
    a = fly_selector.agreement(s, np.quantile(s, 0.95), g, np.quantile(g, 0.95), train)
    assert a["pick_overlap"] > 0.6


def test_rank_corr_matches_spearman_with_ties():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 20, 500).astype(float); b = a + rng.normal(scale=5, size=500); b[::7] = b[0]
    assert rank_corr(a, b) == pytest.approx(pd.Series(a).corr(pd.Series(b), method="spearman"), abs=1e-12)
    x = rng.random(100)
    assert rank_corr(x, np.exp(3 * x)) == pytest.approx(1.0) and rank_corr(x, -x) == pytest.approx(-1.0)
    assert rank_corr(x, np.ones(100)) is None and rank_corr(np.array([1.0]), np.array([2.0])) is None
    assert list(_ranks(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_agreement_overlap_counts_only_rows_in_the_mask():
    rows = np.array([True] * 8 + [False] * 2)
    t = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .95, .95])
    f = np.array([.9, .1, .7, .8, .1, .1, .1, .1, .99, .99])
    a = fly_selector.agreement(f, 0.5, t, 0.5, rows)
    assert a["teacher_picks"] == 3 and a["fly_picks"] == 3 and a["both_picks"] == 2 and a["pick_overlap"] == pytest.approx(2 / 3)
    assert fly_selector.agreement(f, 0.5, t, 2.0, rows)["pick_overlap"] is None


def test_robust_scaler_tames_fat_tails_and_keeps_order():
    X = np.c_[np.r_[np.ones(999), 1e6], np.arange(1000.0)]
    sc = RobustScaler.fit(X); Z = sc.transform(X)
    assert np.abs(Z).max() < 25                                                          # a 1e6 outlier no longer swamps the units
    assert (np.diff(Z[:, 1]) > 0).all()                                                  # monotone: tree splits are unchanged
    assert np.allclose(RobustScaler.from_state(sc.state()).transform(X), Z)
