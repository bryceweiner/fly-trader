"""Model files load without running code: weights-only torch checkpoints, skops selectors, and a release refuses the rest."""
import os

import numpy as np
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from fly_trader.train import model_io
from fly_trader.train.scaling import RobustScaler
from fly_trader.train.selector import SelectorModel

MARK = "fly_model_io_pwned"


class Evil:
    def __reduce__(self):
        return (exec, (f"import os; os.environ[{MARK!r}] = '1'",))


@pytest.fixture(autouse=True)
def clean_env():
    os.environ.pop(MARK, None)
    yield
    os.environ.pop(MARK, None)


def _selector():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)); y = X[:, 0] * 0.1
    sc = RobustScaler.fit(X)
    gbm = HistGradientBoostingRegressor(max_iter=5).fit(sc.transform(X), y)
    return SelectorModel(gbm=gbm, scaler=sc, cols=["a", "b", "c"], threshold=0.01, horizon_min=15, trained_through="2026-09-01"), X


def test_torch_round_trip(tmp_path):
    p = tmp_path / "fly.pt"
    model_io.torch_save({"state_dict": {"w": torch.ones(2)}, "cols": ["a"], "config": {"k": 3, "x": None}}, p)
    assert model_io.torch_load(p)["config"]["k"] == 3
    model_io.check_file(p)


def test_torch_save_refuses_what_would_not_load_safely(tmp_path):
    p = tmp_path / "bad.pt"
    with pytest.raises(model_io.UnsafeModel):
        model_io.torch_save({"x": Evil()}, p)
    assert not p.exists()


def test_crafted_pickle_never_runs(tmp_path):
    p = tmp_path / "evil.pt"
    torch.save({"state_dict": {}, "x": Evil()}, p)       # what an attacker would ship
    with pytest.raises(Exception):
        model_io.torch_load(p)
    with pytest.raises(model_io.UnsafeModel):
        model_io.check_file(p)
    assert MARK not in os.environ


def test_selector_skops_round_trip(tmp_path):
    m, X = _selector()
    p = tmp_path / "selector.skops"
    model_io.save_selector(m, p)
    m2 = model_io.load_selector(p)
    assert np.allclose(m.score(X), m2.score(X)) and m2.cols == m.cols
    model_io.check_file(p)


def test_selector_with_foreign_type_is_refused(tmp_path):
    m, _ = _selector()
    m.metrics = {"x": Evil()}
    with pytest.raises(Exception):
        model_io.save_selector(m, tmp_path / "s.skops")
    assert MARK not in os.environ


def test_pickle_formats_refused(tmp_path):
    for name in ("s.joblib", "s.pkl", "s.pickle"):
        p = tmp_path / name; p.write_bytes(b"\x80\x04N.")
        with pytest.raises(model_io.UnsafeModel):
            model_io.check_file(p)
    with pytest.raises(model_io.UnsafeModel):
        model_io.load_selector(tmp_path / "s.joblib")
