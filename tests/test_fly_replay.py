"""The fly replay: bootstrapped once, it learns minute by minute from labels it could have known at the time, its
chosen configuration does not depend on the evaluation half, and the verdict uses the selector's deploy rule.
Also the rollback checks it counts."""
import math

import numpy as np
import pytest
import torch

from fly_trader.brain import plastic
from fly_trader.train import fly_governance as gov, fly_replay, fly_selector
from tests.test_fly_distill import _graph, _market_ds


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(fly_replay.prog, "update", lambda *a, **k: None)


def _boot(ds, start_day):
    torch.manual_seed(0)
    return fly_selector.bootstrap(ds, ds.days[start_day], graph=_graph(), device="cpu")


def test_replay_learns_only_from_known_labels_and_judges_with_the_deploy_rule(monkeypatch):
    ds = _market_ds(days=16, per_day=600); fly, boot = _boot(ds, 9)
    seen = []
    real = plastic.PlasticBank.update

    def spy(self, tags, r, t):
        assert (tags.ts + ds.horizon_s + fly_replay.LABEL_LAG_S <= t + 1e-6).all()        # never a label from the future
        seen.append(len(tags)); return real(self, tags, r, t)

    monkeypatch.setattr(plastic.PlasticBank, "update", spy)
    out = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    assert sum(seen) > 0 and out["configs"][0]["alpha"] == 0.0 and len(out["configs"]) == 2
    assert isinstance(out["passed"], bool) and "reason" in out and out["data"] == fly_selector.FLY_VERSION
    if out.get("chosen"):
        assert out["chosen"]["alpha"] > 0                                                   # the frozen fly is a comparison, never the pick


def test_choice_does_not_depend_on_evaluation_half_labels():
    ds = _market_ds(days=16, per_day=600); fly, boot = _boot(ds, 9)
    cfg = [(0.0, math.inf), (1e-3, 1.0), (3e-3, 30.0)]
    a = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=cfg, save_verdict=False)
    days_r = ds.days[9:]; ev = np.isin(ds.day, days_r[len(days_r) // 2:])
    ds.fwd_pess[ev] = -ds.fwd_pess[ev]                                                    # the evaluation half's future changes
    b = fly_replay.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=cfg, save_verdict=False)
    assert [t["total"] for t in a["configs"]] == pytest.approx([t["total"] for t in b["configs"]])
    assert (a.get("chosen") or {}).get("config") == (b.get("chosen") or {}).get("config")


def test_rollback_checks():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0.01, 0.02, 200), rng.normal(-0.01, 0.02, 200)
    assert gov.shadow_check(bad, good)[0] and not gov.shadow_check(good, bad)[0]
    assert not gov.shadow_check(bad[:10], good)[0]                                        # too few trades to judge
    s = rng.normal(size=10000)
    assert gov.ic_check(s, -s + rng.normal(size=10000))[0] and not gov.ic_check(s, s)[0]
    assert gov.ic_check(s[:100], -s[:100]) == (False, None)
    assert gov.drift_check(0.6) and not gov.drift_check(0.4)
    out = gov.check((bad, good), (s, -s), 0.7)
    assert out["triggers"] == ["shadow", "ic", "drift"]
