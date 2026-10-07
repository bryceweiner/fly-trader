"""The fly buys only what its teacher (the deployed selector stack) buys: the selector's strategy, trigger, line, win filter,
gates, dump veto and fail-closed rules decide every row; the fly's learning can drop a teacher trade (its plastic score
more than FLY_LEARNED_VETO below its bootstrap score) but never add one. 2026-10-08: ungated, the live fly bought rows the
selector rejected (-25.9 % per row) and missed rows it allowed (+16.0 %)."""
from types import SimpleNamespace

import numpy as np

from fly_trader import config
from fly_trader.agent import fly_session


def test_the_fly_trades_the_selectors_set_and_only_learning_can_drop_a_trade(monkeypatch):
    monkeypatch.setattr(config, "FLY_LEARNED_VETO", 0.02)
    teacher = {"strategy": np.array(["capitulation", "capitulation", None, "capitulation", "ev", "ev"], dtype=object),
               "score": np.array([0.10, 0.12, -1.0, 0.11, 0.20, 0.20]),
               "allow": np.array([True, False, False, True, True, True]),
               "reason": np.array(["trade", "dump veto (P(≤-17 %) ≥ 0.43)", "no strategy fires", "trade", "trade", "trade"], dtype=object)}
    monkeypatch.setattr(fly_session.strategies, "decide", lambda models, X, cols, t: {k: v.copy() for k, v in teacher.items()})
    me = SimpleNamespace(names=["ev", "capitulation"], holds={"ev": 7200.0, "capitulation": 14400.0},
                         sizing={"plastic": {"ev": [{"lo": 0}], "capitulation": [{"lo": 1}]}}, teacher_live={"strategies": {}}, teacher_groups={"skill"})
    infos = [{}, {}, {}, {}, {"skill_missing": True}, {}]
    ctx = SimpleNamespace(mints=[f"m{i}" for i in range(6)], infos=infos, X=np.zeros((6, 3)), t_start=0.0)
    V = np.array([[-np.inf, 0.05], [-np.inf, 0.30], [0.9, 0.9], [-np.inf, 0.05], [0.4, -np.inf], [0.4, -np.inf]])   # the fly's own (plastic) scores
    F = np.array([[np.nan, 0.06], [np.nan, 0.30], [0.9, 0.9], [np.nan, 0.08], [0.4, np.nan], [0.4, np.nan]])        # its bootstrap scores
    d = fly_session.FlyBook._gated(me, ctx, V, F)
    assert d["allow"].tolist() == [True, False, False, False, False, True]
    assert d["strategy"][0] == "capitulation" and d["hold_s"][0] == 14400.0          # the teacher's strategy and hold, at a fly score below its own line
    assert d["reason"][1].startswith("selector: dump veto")                          # the selector's filter binds the fly
    assert d["strategy"][2] is None and np.isinf(d["threshold"][2])                 # no teacher trade: no fly trade, however high the fly scores it
    assert d["reason"][3].startswith("learned veto")                                 # 0.05 < 0.08 - 0.02: its learning drops the trade
    assert d["reason"][4].startswith("selector: fail closed")                        # the selector's live fail-closed rules bind it too
    assert (d["score"] >= d["threshold"])[[0, 1, 3, 4, 5]].all()                     # every teacher row reaches the book, allowed or recorded as blocked


def test_the_replay_judges_the_same_gated_fly(monkeypatch):
    from fly_trader.train import fly_replay
    monkeypatch.setattr(config, "FLY_LEARNED_VETO", 0.02)
    n = 5
    ds = SimpleNamespace(y=np.zeros(n))
    order = np.arange(n); mask_pos = np.array([True, True, True, True, False])
    Sf = np.array([0.05, 0.30, 0.90, 0.05, 0.50]); Lf = np.full(n, 0.10)              # configuration c's (plastic) scores and its line
    S0 = np.array([0.06, 0.30, 0.90, 0.08, 0.50])                                       # the frozen fly's scores
    monkeypatch.setattr(fly_replay, "_arm", lambda *a, **k: (S0, Lf))
    gate = np.array([[True], [False], [False], [True], [True]])                         # the teacher's decisions
    ok = fly_replay._gated_ok(ds, order, None, np.zeros((2, 1, n)), None, 1, 0, Sf, Lf, mask_pos, gate, None, None, 0.1)
    assert ok.tolist() == [True, False, False, False, False]      # the teacher's row below the fly's line: taken; the fly's best rows the teacher rejects: not;
    #                                                                 a learned veto (0.05 < 0.08 - 0.02): dropped; outside the judged half: not
    assert fly_replay._gated_ok(ds, order, None, np.zeros((2, 1, n)), None, 1, 0, Sf, Lf, mask_pos, None, None, None, 0.1).tolist() == (Sf >= Lf).tolist()   # no gate: its line
