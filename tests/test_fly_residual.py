"""The fly that scores as its teacher plus a learned correction (config.FLY_SCORE = teacher, 2026-10-11): at bootstrap its
score is the teacher's exactly (PlasticBank.head_for), only its KC->MBON plasticity moves it, and the replay's frozen arm
is the teacher itself. The distilled copy it replaces ranked the selector's top 1 % of candidates only ~0.6-correlated."""
import math

import numpy as np
import pytest
import torch

from fly_trader import config
from fly_trader.brain import plastic
from fly_trader.train import fly_replay, fly_selector
from tests.test_fly_distill import _graph
from tests.test_fly_strategies import _Teacher, _ds


class _ScoringTeacher(_Teacher):
    """The synthetic stack teacher of test_fly_strategies with its own raw scores (ev follows f0, capitulation -ret_1m)."""

    def raw_scores(self, X, ts, names):
        c = self.cols; out = np.full((len(X), len(names)), np.nan)
        for j, k in enumerate(names):
            if k == "ev":
                out[:, j] = 0.03 * X[:, c.index("f0")]
            elif k == "capitulation":
                r = X[:, c.index("ret_1m")]; out[:, j] = np.where(r < 0, -2.0 * r, np.nan)
        return out


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(fly_replay.prog, "update", lambda *a, **k: None)


def test_at_bootstrap_the_score_is_the_teachers_and_only_learning_moves_it():
    torch.manual_seed(0)
    net = fly_selector.FlyNet(_graph(), obs_dim=6, device="cpu", n_strategies=2)
    bank = plastic.PlasticBank(net, [(0.0, math.inf), (1e-3, math.inf)], fly_selector.SCALE)
    with torch.no_grad():
        _, u0, k = net.forward_parts_all(torch.randn(64, 6))
    target = torch.linspace(-0.3, 0.3, 64)
    y = bank.head_for(target, u0)
    assert torch.allclose(bank.frozen(y, u0), target, atol=1e-6)                       # the frozen fly is the teacher
    sc, _ = bank.predict(y, u0, k)
    assert torch.allclose(sc, target[None].expand(2, -1), atol=1e-6)                   # D = 0: every configuration agrees
    bank.D[1] = 0.01 * bank.w0 * bank.mask                                             # some learning in configuration 1
    sc, _ = bank.predict(y, u0, k)
    base, _ = bank.predict(torch.zeros_like(y), u0, k)
    frozen0 = bank.frozen(torch.zeros_like(y), u0)
    assert torch.allclose(sc[0], target, atol=1e-6)                                    # the frozen arm stays the teacher
    assert torch.allclose(sc[1] - target, base[1] - frozen0, atol=1e-5)                 # the plastic arm: teacher + its own change
    assert (sc[1] - target).abs().max() > 0


def test_the_replay_starts_from_the_teacher_and_its_verdict_says_so(monkeypatch):
    monkeypatch.setattr(config, "FLY_SCORE", "teacher"); monkeypatch.setattr(config, "FLY_TEACHER_GATE", True)
    ds = _ds(); torch.manual_seed(0)
    fly, boot = fly_selector.bootstrap(ds, ds.days[9], epochs=1, graph=_graph(), device="cpu", teacher=_ScoringTeacher(ds))
    seen = {}
    real = fly_replay._verdict

    def capture(ds_, order, ts_o, live_from, scores, *a, **k):
        seen.update(order=order, scores=scores.copy())
        return real(ds_, order, ts_o, live_from, scores, *a, **k)
    monkeypatch.setattr(fly_replay, "_verdict", capture)
    out = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    tsc = fly.teacher.raw_scores(ds.X[seen["order"]], ds.ts[seen["order"]], fly.strategies)
    for s_ in range(len(fly.strategies)):
        has = np.isfinite(seen["scores"][0, s_])
        assert has.any() and np.allclose(seen["scores"][0, s_][has], tsc[has, s_], atol=1e-5)      # the frozen arm is the teacher
        assert not (np.isfinite(seen["scores"][0, s_]) & ~np.isfinite(tsc[:, s_])).any()           # nothing the teacher does not score
    assert out["data"] == {**fly_selector.FLY_VERSION, "score": "teacher"} == fly_selector.verdict_version()


def test_the_own_scoring_verdicts_keep_their_definitions(monkeypatch):
    monkeypatch.setattr(config, "FLY_SCORE", "own")
    assert fly_selector.verdict_version() == fly_selector.FLY_VERSION                 # the verdicts stored before it stay valid for it
