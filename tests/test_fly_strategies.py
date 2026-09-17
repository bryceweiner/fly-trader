"""The fly with several strategies: one head per strategy distilled from the stack's per-strategy targets on that
strategy's candidates; each strategy calibrates its own line; in the replay a strategy's tags are due only at its own
hold and teach only its own channel; the verdict carries one configuration per strategy; save/load keeps the rules."""
import math
from datetime import timedelta

import numpy as np
import pytest
import torch

from fly_trader.brain import plastic
from fly_trader.train import fly_replay, fly_selector
from tests.test_fly_distill import _graph, _market_ds

HOLDS = {"ev": 30, "capitulation": 10}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(fly_replay.prog, "update", lambda *a, **k: None)


def _ds(days=16, per_day=600):
    ds = _market_ds(days=days, per_day=per_day)
    rng = np.random.default_rng(3)
    ret = (0.01 * rng.normal(size=len(ds.y))).astype(np.float32)
    ds.X = np.column_stack([ds.X, ret, np.zeros(len(ds.y), np.float32)]).astype(np.float32); ds.cols = ds.cols + ["ret_1m", "insider_sell_share_1m"]
    ds.fwd_h = {30: ds.fwd_pess, 10: (-2.0 * ret + rng.normal(scale=0.01, size=len(ret))).astype(np.float32)}
    return ds


class _Teacher(fly_selector.StackTeacher):
    """A stand-in for the fitted stack: ev follows f0, capitulation (ret_1m < 0) follows −ret_1m; rows below the line blocked."""

    def __init__(self, ds):
        self.cols = list(ds.cols); self.scaler = fly_selector.RobustScaler.fit(ds.X, seed=7); self.strategies = list(HOLDS)
        self.rules = {k: {"thr": {}, "high": None, "hold_min": h} for k, h in HOLDS.items()}
        self.lines = {"ev": 0.02, "capitulation": 0.01}; self.threshold = 0.02; self.combine = "score"; self.eps = {}

    def targets(self, X, ts):
        c = self.cols; ev = 0.03 * X[:, c.index("f0")]; cap = -2.0 * X[:, c.index("ret_1m")]
        T = np.column_stack([ev, np.where(X[:, c.index("ret_1m")] < 0, cap, np.nan)])
        A = np.isfinite(T) & (T >= np.array([self.lines["ev"], self.lines["capitulation"]]))
        return T, A


def _boot(ds, day=9, epochs=1):
    torch.manual_seed(0)
    return fly_selector.bootstrap(ds, ds.days[day], epochs=epochs, graph=_graph(), device="cpu", teacher=_Teacher(ds))


def test_one_head_per_strategy_with_its_own_line(tmp_path, db_conn, monkeypatch):
    ds = _ds(); fly, info = _boot(ds, epochs=40)                                     # 600 training rows: enough steps to learn
    assert fly.strategies == ["ev", "capitulation"] and fly.net.n_strategies == 2 and len(fly.net.heads) == 2
    assert set(info["calibration"]["per_strategy"]) == {"ev", "capitulation"} and info["lines"] == fly.lines
    trig = fly.triggers(ds.X[:500], ds.cols)
    assert trig[:, 0].all() and (trig[:, 1] == (ds.X[:500, ds.cols.index("ret_1m")] < 0)).all()
    V = fly.score_all(ds.X[:2000])
    c0 = np.corrcoef(V[:, 0], ds.X[:2000, 0])[0, 1]; c1 = np.corrcoef(V[:, 1], -ds.X[:2000, ds.cols.index("ret_1m")])[0, 1]
    assert c0 > 0.3 and c1 > 0.3                                                     # each head learned its own strategy's value
    monkeypatch.setattr(fly_selector.config, "BRAIN_DIR", tmp_path)
    path, sid = fly_selector.save(fly, {})
    with fly_selector.transaction() as conn:
        conn.execute("DELETE FROM brain_snapshots WHERE id = %s", (sid,))
    back = fly_selector.load(path, graph=_graph(), device="cpu")
    assert back.rules == fly.rules and back.lines == pytest.approx(fly.lines) and np.allclose(back.score_all(ds.X[:200]), V[:200], atol=1e-5)
    d = fly_selector.fly_decide(fly, ds.X[:2000], ds.cols, V)
    for i in np.flatnonzero(d["strategy"] != None):                                  # noqa: E711
        j = fly.strategies.index(d["strategy"][i])
        assert trig_ok(fly, ds, i, j) and V[i, j] >= fly.lines[d["strategy"][i]] and d["hold_s"][i] == HOLDS[d["strategy"][i]] * 60.0


def trig_ok(fly, ds, i, j):
    return bool(fly.triggers(ds.X[i:i + 1], ds.cols)[0, j])


def test_replay_tags_are_due_at_their_own_hold_and_teach_their_own_channel(monkeypatch):
    ds = _ds(); fly, boot = _boot(ds)
    learn = fly.net.learn.cpu().numpy(); seen = {0: 0, 1: 0}
    real = plastic.PlasticBank.update

    def spy(self, tags, r, t):
        holds = np.array([HOLDS[fly.strategies[s]] * 60.0 for s in tags.s])
        assert (tags.ts + holds + fly_replay.LABEL_LAG_S <= t + 1e-6).all()          # never a label from the future, per hold
        for (row, s), s2 in zip(tags.keys, tags.s):
            assert s == s2; seen[int(s)] += 1
            if s == 1:
                assert ds.X[row, ds.cols.index("ret_1m")] < 0                         # capitulation tags only on its candidates
        self.decay_to(t); before = self.D.clone(); out = real(self, tags, r, t)       # forgetting (time) moves every channel; learning must not
        moved = ((self.D - before).abs().sum((0, 1)) > 0).cpu().numpy()
        present = set(int(x) for x in tags.s)
        others = np.zeros(learn.shape[1], bool)
        for s3 in set(range(learn.shape[0])) - present:
            others |= learn[s3]
        assert not (moved & others).any()                                              # a strategy's outcome leaves the others' columns alone
        return out

    monkeypatch.setattr(plastic.PlasticBank, "update", spy)
    out = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    assert seen[0] > 0 and seen[1] > 0
    assert set(out["per_strategy"]) == {"ev", "capitulation"} and out["strategies"] == ["ev", "capitulation"]
    assert {v["hold_min"] for v in out["per_strategy"].values()} == {30, 10} and isinstance(out["passed"], bool)
    for v in out["per_strategy"].values():
        assert v["chosen"] is None or v["chosen"]["alpha"] > 0


def test_bootstrap_line_uses_labels_known_before_S():
    ds = _ds(); S = ds.days[9]; fly, info = _boot(ds)
    assert info["calibration_days"] == [str(S - timedelta(days=7)), str(S - timedelta(days=1))]
    assert all(v["hold_min"] == HOLDS[k] for k, v in info["calibration"]["per_strategy"].items())
