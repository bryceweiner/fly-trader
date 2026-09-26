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
    assert out["learning"] and sum(d["n"] for d in out["learning"]) > 0          # the replay records what its learning did
    assert all(len(d["drift"]) == len(out["configs"]) for d in out["learning"])   # per configuration, every day
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


def test_unresolved_rows_are_not_trades_and_a_nan_verdict_still_stores():
    """A 240-minute hold has no label at the end of the corpus (decisions.build leaves NaN). Counting those rows put
    n=3195 mean=NaN in a finished replay's own judgement and then lost five hours of work to a JSON error."""
    import json
    import numpy as np
    from fly_trader.train import progress as prog
    ds = _market_ds(days=14, per_day=400)
    lab = ds.fwd_pess.copy(); lab[-200:] = np.nan                       # the tail has no resolved label
    ds.fwd_h = {30: lab}
    order = np.argsort(ds.ts, kind="stable"); ts_o = ds.ts[order]
    scores = np.full((1, 1, len(order)), 1.0, np.float32)
    pick, hold, ret = fly_replay._book(ds, order, ts_o, scores, {}, [0], [0.0], [30], np.ones(len(order), bool))
    assert pick.any() and np.isfinite(ret[pick]).all()                  # every taken row has a known outcome
    assert not pick[np.flatnonzero(~np.isfinite(lab))].any()            # and the unresolved tail was not taken
    assert json.loads(json.dumps(prog._finite({"frozen": {"n": 3195, "mean": float("nan")}})))["frozen"]["mean"] is None


def test_the_replay_writes_its_trades_with_the_sizing_table_in_force(monkeypatch, tmp_path):
    """Sizing rules are judged on the fly's own out-of-sample trades: the replay keeps them, one position per token,
    each with its margin over the line in force, its return, its pool's reserve and the table the book sized it with."""
    import json
    ds = _market_ds(days=14, per_day=400); ds.cols = list(ds.cols) + ["log_liquidity_sol"]
    ds.X = np.c_[ds.X, np.full(len(ds.y), np.log1p(200.0))].astype(ds.X.dtype)
    ds.fwd_h = {30: ds.fwd_pess.copy()}
    order = np.argsort(ds.ts, kind="stable"); ts_o = ds.ts[order]
    scores = np.linspace(-1, 1, len(order), dtype=np.float32)[None, None, :]
    t_mid = float(np.median(ds.ts)); old, new = [{"lo": 0.0, "kelly": 0.1}], [{"lo": 0.0, "kelly": 0.4}]
    monkeypatch.setattr(fly_replay, "TRADES_DIR", tmp_path)
    path = fly_replay._dump_trades(ds, order, ts_o, scores, {}, {"s": [(-math.inf, old), (t_mid, new)]}, [0], [0.0], [30], ["s"],
                                   np.ones(len(order), bool), ds.day[order], list(ds.days[:7]))
    d = json.loads(open(path).read()); tr = d["trades"]
    assert tr and all(t["margin"] >= 0 and t["strategy"] == "s" and t["hold_s"] == 1800 for t in tr)
    assert [t["ts"] for t in tr] == sorted(t["ts"] for t in tr) and all(abs(t["resq"] - 100.0) < 1e-3 for t in tr)
    assert all(d["tables"][t["table"]] == (new if t["ts"] >= t_mid else old) for t in tr)
    by = {}
    for t in tr:                                                        # one position per token until its hold has passed
        assert t["ts"] >= by.get(t["mint"], -math.inf); by[t["mint"]] = t["ts"] + t["hold_s"]


def test_every_configuration_shares_the_frozen_flys_line(monkeypatch):
    """Per-configuration lines let the frozen arm take 4,216 selection-half trades to the chosen arm's 2,609 with the
    weights no more than 5 % apart: the arms traded different slices because of their lines, not their weights. One line,
    calibrated on the frozen fly, leaves the weights as the only difference between arms."""
    from types import SimpleNamespace
    calls = []

    def fake_cal(ts, mint, hold_s, sc, y, prev_line=None, prev_sizing=None):
        calls.append(len(sc)); return SimpleNamespace(line=float(np.mean(sc)), sizing=[{"lo": 0.0, "kelly": 0.1}])
    monkeypatch.setattr(fly_replay.fly_calibrate, "calibrate", fake_cal)
    n, C, NS = 200, 3, 2
    ts = np.arange(n, dtype=np.float64) * 60.0
    ds = SimpleNamespace(ts=ts, mint=np.array([f"m{i % 7}" for i in range(n)]), fwd_h={}, fwd_pess=np.zeros(n, np.float32))
    scores = np.random.default_rng(0).normal(size=(C, NS, n))               # every configuration scores differently
    lines = np.full((C, NS), 0.02); sizings = [[[] for _ in range(NS)] for _ in range(C)]; line_log = {}
    fly_replay._recalibrate(ds, np.arange(n), ts, scores, lines, sizings, float(ts[-1]) + 61.0, [60.0, 60.0], [10, 30], line_log)
    assert calls == [n, n]                                                   # once per strategy, on every resolved row
    for s in range(NS):
        assert lines[0, s] == pytest.approx(np.mean(scores[0, s]))           # the frozen configuration's scores
        assert (lines[:, s] == lines[0, s]).all() and all(line_log[(c, s)][-1] == (float(ts[-1]) + 61.0, lines[0, s]) for c in range(C))
        assert all(sizings[c][s] == sizings[0][s] for c in range(C))
