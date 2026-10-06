"""Combined training on both chains: a unit-bearing trigger means the same on both (exact K scaling, Solana untouched);
the fly keeps its own line per chain; the replay judges each chain on its own trades; the selector's holdout per chain."""
import math

import numpy as np
import pytest
import torch

from fly_trader import config
from fly_trader.train import decisions, fly_replay, fly_selector, strategies
from tests.test_fly_distill import _graph, _market_ds


def test_unit_bearing_trigger_is_exactly_rescaled_on_rh_rows(monkeypatch):
    monkeypatch.setattr(config, "RH_ETH_PER_SOL", 0.05)
    cols = ["log_org_vol_15m", "chain_rh"]
    X = np.array([[np.log1p(10.0), 0.0], [np.log1p(0.5), 1.0]], dtype=np.float32)     # 10 SOL, and 0.5 ETH = 10 SOL at K = 0.05
    v = strategies.trigger_value("breakout", "log_org_vol_15m", X, cols)
    assert v[0] == X[0, 0] and v[1] == pytest.approx(np.log1p(10.0), rel=1e-6)
    Xs = X[:1]; assert strategies.trigger_value("breakout", "log_org_vol_15m", Xs, cols) is not None
    assert np.shares_memory(strategies.trigger_value("breakout", "log_org_vol_15m", Xs, cols), Xs)     # Solana-only: untouched


def _two_chain_ds(days=16, per_day=600):
    ds = _market_ds(days=days, per_day=per_day)
    rng = np.random.default_rng(3)
    half = np.array(["rh" if m.endswith("7") or m.endswith("3") else "sol" for m in ds.mint], dtype=object)   # a fixed share of tokens on RH
    if "chain_rh" not in ds.cols:
        ds.cols = list(ds.cols) + ["chain_rh"]; ds.X = np.c_[ds.X, (half == "rh").astype(np.float32)]
    ds.chain = half
    first_rh = ds.days[3]                                          # RH starts later than Solana
    keep = (half == "sol") | (ds.day >= first_rh)
    return ds.subset(keep), rng


def test_fly_lines_per_chain_and_replay_verdict_per_chain(monkeypatch):
    monkeypatch.setattr(fly_replay, "other_chain_lead", lambda: 3); monkeypatch.setattr(fly_replay, "MIN_TRADES", 3)   # a small synthetic market
    ds, _ = _two_chain_ds()
    torch.manual_seed(0)
    fly, boot = fly_selector.bootstrap(ds, ds.days[9], graph=_graph(), device="cpu")
    per = boot["calibration"]["per_strategy"]
    assert any(k.endswith("@rh") for k in per) and "rh" in fly.chain_lines                    # its own RH line, calibrated on RH rows
    X = ds.X[:5].copy(); X[:, ds.cols.index("chain_rh")] = 1.0
    d = fly_selector.fly_decide(fly, X, ds.cols)
    assert all(t in (np.inf, *fly.chain_lines["rh"].values()) for t in d["threshold"])          # RH rows face the RH line
    out = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    assert set(out["chains"]) == {"sol", "rh"} and all("passed" in v and "evaluation" in v for v in out["chains"].values())
    assert out["passed"] == any(v["passed"] for v in out["chains"].values())
    assert all(v["frozen"] is not None for v in out["chains"].values())                  # each chain's own frozen book, reported


def test_solana_only_replay_has_no_chain_split():
    ds = _market_ds(days=16, per_day=600)
    torch.manual_seed(0)
    fly, boot = fly_selector.bootstrap(ds, ds.days[9], graph=_graph(), device="cpu")
    assert fly.chain_lines == {}
    out = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    assert "chains" not in out


def test_row_weights_take_one_line_per_row():
    from fly_trader.brain import plastic
    sc = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    assert plastic.row_weights(sc, torch.tensor([0.15, 0.25])).tolist() == [[0, 1, 1], [0, 0, 1]]
    assert plastic.row_weights(sc, torch.tensor([[0.05, 0.25, 0.25], [0.2, 0.2, 0.4]])).tolist() == [[1, 0, 1], [0, 1, 0]]


def test_a_nan_calibration_mean_never_deploys_the_fly():
    from fly_trader.train import fly_selector
    meta = {"data": fly_selector.FLY_VERSION, "gates_ok": True, "calibration": {"trades": 10 ** 4, "mean": float("nan")}}
    ok, why = fly_selector.deployable(meta, fly_selector.FLY_VERSION)
    assert not ok and "lost money" in why


def test_rejudging_one_chain_keeps_the_others_verdict():
    from fly_trader.train import fly_replay
    old = {"data": {"v": 1}, "S": "2026-08-25", "passed": True,
           "chains": {"sol": {"passed": True, "reason": "sol ok"}, "rh": {"passed": False, "reason": "rh stale teacher"}}}
    new = {"data": {"v": 1}, "S": "2026-09-10", "finished_at": "x",
           "chains": {"sol": {"passed": False, "reason": "short window"}, "rh": {"passed": True, "reason": "rh ok"}}}
    m = fly_replay.merge_chains({"value": old}, new, ("rh",))
    assert m["chains"]["sol"] == {"passed": True, "reason": "sol ok"}                     # Solana keeps its verdict
    assert m["chains"]["rh"]["passed"] and m["chains"]["rh"]["S"] == "2026-09-10" and m["passed"]
    assert fly_replay.merge_chains({"value": {**old, "data": {"v": 0}}}, new, ("rh",)) is new      # other definitions: replaced whole
