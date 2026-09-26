"""The fly: FlyNet on a tiny stand-in connectome (with a mushroom body) learns to imitate a teacher, is built without
saturating or bottlenecked parts, starts at the mean target, keeps a sparse KC code, reaches its output through the MBON
readout, and is bootstrapped with its own buy line; the agreement / rank-correlation helpers are correct."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fly_trader.train import fly_selector
from fly_trader.train.decisions import DecisionSet, _ranks, rank_corr
from fly_trader.train.scaling import RobustScaler

POPS = {"KC": (0, 20), "MBON_APP": (20, 22), "MBON_AV": (22, 24), "MBON_OTHER": (24, 26), "ORN_FOOD": (26, 56), "DESCENDING": (70, 80)}


def _graph(seed=0):
    """20 KCs, 6 MBONs, 30 sensory neurons (ORN_FOOD) → KCs and 14 interneurons → 10 descending neurons; MBONs → DNs;
    KC→MBON synapses only in the dense block (as the build stores them)."""
    g = torch.Generator().manual_seed(seed); pre, post = [], []
    for a, b in ((range(26, 56), range(0, 20)), (range(26, 56), range(56, 70)), (range(56, 70), range(70, 80)), (range(20, 26), range(70, 80)),
                 (range(26, 56), range(70, 80))):
        for j in b:
            for i in torch.randint(a.start, a.stop, (6,), generator=g).tolist():
                pre.append(i); post.append(j)
    vals = (torch.rand(len(pre), generator=g) * 5 + 1) * torch.where(torch.rand(len(pre), generator=g) < 0.8, 1.0, -1.0)
    W = torch.randint(1, 12, (20, 6), generator=g).float() * (torch.rand(20, 6, generator=g) < 0.4)
    return SimpleNamespace(N=80, indices=torch.tensor([post, pre]), values_raw=vals, pop_ranges=dict(POPS), W_KM0=W, M_KM=W > 0,
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
    assert net.theta_km.numel() == int(_graph().M_KM.sum())                            # only the synapses the connectome has
    fly, info = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=1, batch=500, graph=_graph(), device="cpu")
    assert info["rows"] == len(ds.y) and info["epochs"][0]["rows"] == len(ds.y)          # the full budget below the sample cap
    start, _ = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=0, graph=_graph(), device="cpu")
    mean_target = float(np.clip(t.score(ds.X), -1, 1).mean())
    assert float(start.net.head.bias.detach()) == pytest.approx(mean_target * fly_selector.SCALE, rel=1e-4)   # starts at the mean target


def test_kc_code_is_sparse_and_the_mbon_readout_is_the_plastic_path():
    torch.manual_seed(0)
    net = fly_selector.FlyNet(_graph(), obs_dim=6, device="cpu").eval()
    x = torch.randn(64, 6)
    with torch.no_grad():
        y_dn, u0, k = net.forward_parts(x)
        assert ((k > 0).sum(1) <= net.k_active).all() and (k >= 0).all()             # APL-like winner-take-all, non-negative rates
        assert torch.allclose(net.readout(y_dn, u0, k), net(x), atol=1e-5)           # frozen fly = D None
        assert torch.allclose(net.readout(y_dn, u0, k, torch.zeros(20, 6)), net(x), atol=1e-5)
        D = torch.randn(20, 6)
        moved = net.readout(y_dn, u0, k, D) - net(x)
        assert moved.abs().max() > 0                                                  # a KC→MBON change reaches the output
        assert torch.allclose(net.readout(y_dn, u0, k, D * ~net.mask), net(x), atol=1e-5)   # but only on existing synapses
    assert (net.c[:2] > 0).all() and (net.c[2:4] < 0).all()                           # approach +, avoid −


def test_train_fly_imitates_the_teacher_on_the_same_signals():
    torch.manual_seed(0)
    ds = _ds(); train = np.ones(len(ds.y), bool); t = Teacher(ds)
    fly, info = fly_selector.train_fly(ds, train, t, epochs=6, batch=200, graph=_graph(), device="cpu")
    assert not info["stopped"] and info["epochs"][-1]["mse"] < info["epochs"][0]["mse"]
    s, g = fly.score(ds.X), t.score(ds.X)
    assert rank_corr(s, g) > 0.9                                                         # the fat-tailed feature does not blind it
    b = np.polyfit(g, s, 1)[0]
    assert 0.6 < b < 1.4                                                                  # on the teacher's scale, not squashed
    assert fly.scaler is t.scaler                                                         # same signals
    top_t, top_f = g >= np.quantile(g, 0.95), s >= np.quantile(s, 0.95)
    assert (top_t & top_f).sum() / top_t.sum() > 0.6                                     # it picks what the teacher picks
    d = fly_selector.diagnose(fly, ds.X[:2000], g[:2000])
    assert 0.0 <= d["mbon_share"] <= 1.0 and 0.0 <= d["kc_active"] <= fly_selector.KC_ACTIVE + 1e-6 and d["slope"] == pytest.approx(b, rel=0.2)


def test_normalisation_is_recalibrated_over_many_rows_not_the_last_batches():
    torch.manual_seed(0)
    ds = _ds(); t = Teacher(ds)
    fly, _ = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=1, batch=500, graph=_graph(), device="cpu")
    assert fly.net.eff_norm.momentum is None and int(fly.net.eff_norm.num_batches_tracked) == -(-len(ds.y) // fly_selector.BATCH)


def _market_ds(days=12, per_day=1000, seed=0):
    """Rows with the universe columns and a label that follows f0: something a fly can calibrate a line on."""
    rng = np.random.default_rng(seed); n = days * per_day
    X = rng.normal(size=(n, 6)).astype(np.float32); X[:, 4] = 0.0; X[:, 5] = np.log1p(24.0)     # age_known 0: in the universe
    fwd = (0.03 * X[:, 0] + rng.normal(scale=0.02, size=n)).astype(np.float32)
    d0 = datetime(2026, 9, 1, tzinfo=timezone.utc); k = np.arange(n)
    ts = np.array([(d0 + timedelta(days=int(i // per_day), minutes=int(i % per_day))).timestamp() for i in k])
    day = np.array([(d0 + timedelta(days=int(i // per_day))).date() for i in k], dtype=object)
    return DecisionSet(X=X, y=(fwd > 0.03).astype(np.int8), fwd=fwd, fwd_pess=fwd, day=day, ts=ts, mint=np.array([f"m{i % 400}" for i in k]),
                       cols=["f0", "f1", "f2", "f3", "age_known", "log_age_h"], horizon_s=1800.0)


def test_bootstrap_gives_the_fly_its_own_line_from_labels_known_before_S():
    torch.manual_seed(0)
    ds = _market_ds(); S = ds.days[-1] + timedelta(days=1)
    fly, info = fly_selector.bootstrap(ds, S, graph=_graph(), device="cpu")
    assert fly.threshold == info["line"] and fly.sizing == info["sizing"]
    assert info["calibration"]["trades"] >= 100 or info["line"] == fly_selector.selector.MIN_EV
    assert info["train_through"] == str(S - timedelta(days=9)) and info["calibration_days"] == [str(S - timedelta(days=7)), str(S - timedelta(days=1))]
    assert set(info["diagnostics"]) >= {"mbon_share", "mbon_saturated", "kc_active", "slope"} and isinstance(info["gates_ok"], bool)


def test_the_fly_is_taught_by_the_selector_the_book_trades(monkeypatch):
    """Until 2026-09-21 the bootstrap refit its own stack and distilled that. Selection is unstable enough that the
    refit was a different trading system (deployed ev at 0.046 over 120 min vs a refit capitulation at 0.010 over 240),
    so the fly was never a copy of the selector it was judged against."""
    import contextlib

    from fly_trader.agent import selector_session
    ds = _market_ds(); train = ds.day < ds.days[-3]
    ev = {"cols": ["f0", "f1"], "line": 0.046, "hold_min": 120, "thr": {}, "high": None}
    monkeypatch.setattr(selector_session, "pinned_snapshot", lambda: 60)
    monkeypatch.setattr(fly_selector.selector, "load_snapshot", lambda i: SimpleNamespace(stack={"strategies": {"ev": ev}, "combine": "score"}))
    t, note, sid = fly_selector.deployed_teacher(ds, train)
    assert sid == 60 and "#60" in note and t.strategies == ["ev"] and t.rules["ev"]["hold_min"] == 120 and t.lines["ev"] == 0.046

    monkeypatch.setattr(fly_selector, "transaction", lambda: contextlib.nullcontext(None))
    monkeypatch.setattr(selector_session, "pinned_snapshot", lambda: None)
    monkeypatch.setattr(fly_selector.selector, "latest_current", lambda conn: None)
    assert fly_selector.deployed_teacher(ds, train) == (None, fly_selector.TEACHER_NOTE, None)      # says so rather than pretending

    monkeypatch.setattr(selector_session, "pinned_snapshot", lambda: 61)
    monkeypatch.setattr(fly_selector.selector, "load_snapshot", lambda i: SimpleNamespace(stack={"strategies": {"ev": {**ev, "cols": ["f0", "nope"]}}}))
    t2, note2, sid2 = fly_selector.deployed_teacher(ds, train)
    assert t2 is None and sid2 is None and "nope" in note2                                          # never distil inputs the corpus lacks


def test_the_teacher_never_saw_the_week_the_fly_calibrates_on(monkeypatch):
    """Fly #107 distilled selector #100, fit on every day through 2026-09-23, and calibrated its line and sizing on
    2026-09-17..23: the teacher's in-sample fit made that week look like +2.79 % a trade, and its biggest bets lost
    live. The teacher is now the same stack refit without its holdout days, used only when that fit ends before the week."""
    from datetime import timedelta

    from fly_trader.agent import selector_session
    ds = _market_ds(); train = ds.day < ds.days[-3]; wk = ds.days[-1]; ok = str(wk - timedelta(days=fly_selector.PURGE_DAYS + 1))
    ev = {"cols": ["f0", "f1"], "line": 0.046, "hold_min": 120, "thr": {}, "high": None}
    blind = {"models": {"strategies": {"ev": {**ev, "line": 0.05}}, "combine": "score"}, "fit_through": ok, "holdout_days": [ok, str(wk)]}
    monkeypatch.setattr(selector_session, "pinned_snapshot", lambda: 100)
    load = lambda **kw: monkeypatch.setattr(fly_selector.selector, "load_snapshot", lambda i: SimpleNamespace(stack={"strategies": {"ev": ev}, "combine": "score"}, **kw))

    load(blind=blind, trained_through=str(wk))
    t, note, sid = fly_selector.deployed_teacher(ds, train, calib_start=wk)
    assert sid == 100 and t.lines["ev"] == 0.05 and "blind" in note                    # the holdout-blind refit, same strategy

    load(blind={**blind, "fit_through": str(wk - timedelta(days=fly_selector.PURGE_DAYS))}, trained_through=str(wk))
    t, note, sid = fly_selector.deployed_teacher(ds, train, calib_start=wk)
    assert t is None and sid is None and "saw the calibration week" in note             # its labels would spill into the week

    load(trained_through=ok)                                                            # an old snapshot fit before the week
    t, note, sid = fly_selector.deployed_teacher(ds, train, calib_start=wk)
    assert sid == 100 and t.lines["ev"] == 0.046 and "before the calibration week" in note

    load(trained_through=str(wk))                                                       # an old snapshot that saw it
    t, note, sid = fly_selector.deployed_teacher(ds, train, calib_start=wk)
    assert t is None and "no blind models" in note
    assert fly_selector.deployed_teacher(ds, train)[2] == 100                           # without a week to protect: as before


def test_training_stops_when_an_epoch_buys_less_than_its_own_noise(monkeypatch):
    """One epoch left a per-row error of ~3.9 % net return against a 4.6 % line, so the ordering near the line was
    mostly noise. The count is now the data's to choose: passes end when the held-out error stops moving by more than
    its standard error, and the epoch that generalised best is the one kept."""
    monkeypatch.setattr(fly_selector, "VAL_MIN_ROWS", 100); monkeypatch.setattr(fly_selector, "VAL_ROWS", 500)
    torch.manual_seed(0)
    ds = _ds(); t = Teacher(ds)
    fly, info = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=12, batch=500, graph=_graph(), device="cpu")
    assert info["val_rows"] == 500 and info["epoch_cap"] == 12 and fly_selector.MIN_EPOCHS <= info["epochs_run"] <= 12
    assert all("val_mse" in e and "val_se" in e for e in info["epochs"])
    assert info["val_mse"] == min(e["val_mse"] for e in info["epochs"])                             # the best epoch, not the last
    assert (info["early_stop"] is None) == (info["epochs_run"] == 12)
    small, si = fly_selector.train_fly(ds, np.ones(len(ds.y), bool), t, epochs=0, graph=_graph(), device="cpu")
    assert si["val_rows"] == 0 and si["early_stop"] is None and si["epochs_run"] == 0              # nothing to hold out, nothing to stop


def test_rank_corr_matches_spearman_with_ties():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 20, 500).astype(float); b = a + rng.normal(scale=5, size=500); b[::7] = b[0]
    assert rank_corr(a, b) == pytest.approx(pd.Series(a).corr(pd.Series(b), method="spearman"), abs=1e-12)
    x = rng.random(100)
    assert rank_corr(x, np.exp(3 * x)) == pytest.approx(1.0) and rank_corr(x, -x) == pytest.approx(-1.0)
    assert rank_corr(x, np.ones(100)) is None and rank_corr(np.array([1.0]), np.array([2.0])) is None
    assert list(_ranks(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_robust_scaler_tames_fat_tails_and_keeps_order():
    X = np.c_[np.r_[np.ones(999), 1e6], np.arange(1000.0)]
    sc = RobustScaler.fit(X); Z = sc.transform(X)
    assert np.abs(Z).max() < 25                                                          # a 1e6 outlier no longer swamps the units
    assert (np.diff(Z[:, 1]) > 0).all()                                                  # monotone: tree splits are unchanged
    assert np.allclose(RobustScaler.from_state(sc.state()).transform(X), Z)


def test_the_scaler_never_touches_the_callers_array():
    """transform() copies; only transform_into() writes, and the walk-forwards hand it arrays they own."""
    X = np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32); before = X.copy()
    sc = RobustScaler.fit(X)
    Z = sc.transform(X)
    assert np.array_equal(X, before) and not np.shares_memory(Z, X)
    Y = X.copy()
    assert sc.transform_into(Y) is Y and np.allclose(Y, Z)
