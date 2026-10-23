"""The fly buys only what its teacher (the deployed selector stack) buys: the selector's strategy, trigger, line, win filter,
gates, dump veto and fail-closed rules decide every row; the fly's learning can drop a teacher trade (its plastic score
more than FLY_LEARNED_VETO below its bootstrap score) but never add one. 2026-10-08: ungated, the live fly bought rows the
selector rejected (-25.9 % per row) and missed rows it allowed (+16.0 %)."""
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from fly_trader import config
from fly_trader.agent import fly_session


def test_the_fly_trades_the_selectors_set_and_only_learning_can_drop_a_trade(monkeypatch):
    monkeypatch.setattr(config, "FLY_LEARNED_VETO", 0.02)
    teacher = {"strategy": np.array(["capitulation", "capitulation", None, "capitulation", "ev", "ev"], dtype=object),
               "score": np.array([0.10, 0.12, -1.0, 0.11, 0.20, 0.20]),
               "allow": np.array([True, False, False, True, True, True]),
               "reason": np.array(["trade", "dump veto (P(≤-17 %) ≥ 0.43)", "no strategy fires", "trade", "trade", "trade"], dtype=object),
               "threshold": np.array([0.069, 0.069, np.inf, 0.069, 0.115, 0.115]),
               "tables": [[{"lo": 0.0, "kelly": 0.13}], [{"lo": 0.0, "kelly": 0.15}], [], [{"lo": 0.0, "kelly": 0.13}], [{"lo": 0.0, "kelly": 0.2}], [{"lo": 0.0, "kelly": 0.2}]]}
    monkeypatch.setattr(fly_session.strategies, "decide", lambda models, X, cols, t: {k: (v.copy() if hasattr(v, "copy") else list(v)) for k, v in teacher.items()})
    me = SimpleNamespace(names=["ev", "capitulation"], holds={"ev": 7200.0, "capitulation": 14400.0},
                         sizing={"plastic": {"ev": [{"lo": 0}], "capitulation": [{"lo": 1}]}}, teacher_live={"strategies": {}}, teacher_groups={"skill"})
    infos = [{}, {}, {}, {}, {"skill_missing": True}, {}]
    ctx = SimpleNamespace(mints=[f"m{i}" for i in range(6)], infos=infos, X=np.zeros((6, 3)), t_start=0.0)
    V = np.array([[-np.inf, 0.05], [-np.inf, 0.30], [0.9, 0.9], [-np.inf, 0.05], [0.4, -np.inf], [0.4, -np.inf]])   # the fly's own (plastic) scores
    F = np.array([[np.nan, 0.06], [np.nan, 0.30], [0.9, 0.9], [np.nan, 0.08], [0.4, np.nan], [0.4, np.nan]])        # its bootstrap scores
    me._teacher_decision = lambda c: fly_session.FlyBook._teacher_decision(me, c)
    d = fly_session.FlyBook._gated(me, ctx, V, F)
    assert d["allow"].tolist() == [True, False, False, False, False, True]
    assert d["strategy"][0] == "capitulation" and d["hold_s"][0] == 14400.0          # the teacher's strategy and hold, at a fly score below its own line
    assert d["score"][0] == 0.10 and d["threshold"][0] == 0.069 and d["tables"][0] == [{"lo": 0.0, "kelly": 0.13}]   # sized as the selector sizes it
    assert d["reason"][1].startswith("selector: dump veto")                          # the selector's filter binds the fly
    assert d["strategy"][2] is None and np.isinf(d["threshold"][2])                 # no teacher trade: no fly trade, however high the fly scores it
    assert d["reason"][3].startswith("learned veto")                                 # 0.05 < 0.08 - 0.02: its learning drops the trade
    assert d["vetoed"].tolist() == [False, False, False, True, False, False]         # only its learning's drop is a veto, not the selector's blocks
    assert d["reason"][4].startswith("selector: fail closed")                        # the selector's live fail-closed rules bind it too
    assert (d["score"] >= d["threshold"])[[0, 1, 3, 4, 5]].all()                     # every teacher row reaches the book, allowed or recorded as blocked


def test_until_the_fly_wins_its_race_the_wallet_trades_the_selectors_decision(monkeypatch):
    """Whoever holds the seat decides what the wallet buys. Until the fly has won its race (no handover), a teacher trade its
    learning vetoes stays out of its paper book -- what the race measures -- but goes live as the selector decided it; once
    the fly holds the seat its veto binds the wallet too. A row the selector itself blocks never goes live."""
    d = {"allow": np.array([True, False, False]), "vetoed": np.array([False, True, False]), "score": np.array([0.10, 0.11, 0.12]),
         "threshold": np.full(3, 0.069), "hold_s": np.full(3, 14400.0), "strategy": np.array(["capitulation"] * 3, dtype=object), "tables": [[]] * 3,
         "reason": np.array(["trade", "learned veto: score 0.020 < bootstrap 0.050 - 0.02", "selector: dump veto (P(≤-17 %) ≥ 0.43)"], dtype=object)}
    planned, sent = [], []
    monkeypatch.setattr(fly_session.paper_trading, "plan_minute", lambda ctx, **kw: planned.append(kw) or {"beat_id": 7, "entries": ["the plan's entries"], "rebuy": {}})
    for seat in (False, True):
        me = SimpleNamespace(_commands=lambda conn, t: None, _fast=lambda: True, _decide=lambda c: {"decision": d, "picks": 2}, race_started_at=0.0, beat_no=0,
                             BOOK="paper_fly", run_id="run", broker=None, KIND="fly", names=["capitulation"], holds={"capitulation": 14400.0}, has_seat=seat,
                             trades_live=lambda conn: True, _go_live=lambda c, beat, entries, rebuy: sent.append((beat, entries)) or {"sent": len(entries)})
        ctx = SimpleNamespace(conn=None, m1=datetime(2026, 10, 22, tzinfo=timezone.utc), m1_epoch=0.0, lag_s=0.0, trade=True, fresh=True,
                              mints=["KEEPpump", "VETOpump", "DUMPpump"], infos=[{}] * 3)
        fly_session.FlyBook.on_minute(me, ctx)
    before, after = planned
    assert before["allow"].tolist() == [True, False, False]          # the paper book: the fly's decision, its learned veto included
    assert before["live_allow"].tolist() == [True, True, False]      # the wallet: the seat holder's, the selector's
    assert after["live_allow"] is None                                # the fly holds the seat: its decision for both
    assert sent == [(7, ["the plan's entries"])] * 2                  # the wallet's orders come from that one plan


def test_the_live_phase_scores_only_the_teachers_trades_and_decides_as_the_full_scoring(monkeypatch):
    """Live money first (CLAUDE.md): before the wallet's orders the fly makes one pass of its teacher's stack and scores
    only the rows the teacher would trade (its learned veto's input); every row's score, its tag and the full forward wait
    for after the orders, on the same pass. 2026-10-19: the live phase scored every row and ran the stack three times."""
    import torch
    monkeypatch.setattr(config, "FLY_LEARNED_VETO", 0.02); monkeypatch.setattr(config, "FLY_TEACHER_GATE", True); monkeypatch.setattr(config, "FLY_SCORE", "teacher")
    inf = np.inf; T, F_ = True, False
    per = {"ev": {"trig": np.array([T, T, F_, T, F_]), "score": np.array([0.20, 0.05, -inf, 0.30, -inf]), "allow": np.array([T, F_, F_, T, F_]),
                  "why": np.array([""] * 5, dtype=object), "line": 0.10, "hold_min": 120, "sizing": []},
           "capitulation": {"trig": np.array([F_, F_, T, F_, T]), "score": np.array([-inf, -inf, 0.15, -inf, 0.12]), "allow": np.array([F_, F_, T, F_, F_]),
                            "why": np.array(["", "", "", "", "dump veto"], dtype=object), "line": 0.10, "hold_min": 240, "sizing": []}}
    passes = []
    monkeypatch.setattr(fly_session.strategies, "decide_all", lambda models, X, cols, t, spec=None: passes.append(len(X)) or {k: {**v} for k, v in per.items()})
    forwards = []

    class Fly:
        lines, sizings = {}, {}

        def parts_all_h(self, X):
            forwards.append(("all_h", len(X))); B = len(X)
            return torch.zeros(B, 2), torch.as_tensor(X[:, :1], dtype=torch.float32), torch.zeros(B, 4), torch.zeros(B, 6)

        def parts_all(self, X):
            forwards.append(("all", len(X))); B = len(X)
            return torch.zeros(B, 2), torch.as_tensor(X[:, :1], dtype=torch.float32), torch.zeros(B, 4)

        def triggers(self, X, cols):
            return np.ones((len(X), 2), bool)

    bank = SimpleNamespace(dev="cpu", fill=torch.zeros(2, dtype=torch.bool), head_for=lambda target, u0, s=None: target,
                           predict=lambda y, u0, k, s=None: ((y - 0.05 * u0[:, 0])[None], None), frozen=lambda y, u0, s=None: y)
    pushed = []
    X = np.zeros((5, 3), np.float32); X[3, 0] = 1.0                                      # row 3: its learning lowered it 0.05 below its bootstrap score

    def book():
        b = object.__new__(fly_session.FlyBook)
        b.__dict__.update(names=["ev", "capitulation"], holds={"ev": 7200.0, "capitulation": 14400.0}, idx=np.arange(3), chain="sol", boot_id=1,
                          lines={"plastic": {"ev": 0.1, "capitulation": 0.1}, "frozen": {"ev": 0.1, "capitulation": 0.1}},
                          sizing={"plastic": {"ev": [], "capitulation": []}}, teacher_live={"strategies": {}}, teacher_groups=set(), teacher_models=None,
                          capture_ok=False, nu_set=True, fill_choice=False, fly=Fly(), bank=bank, watch={}, bars={}, _scored_rows=[], _activity=None, _live_pass=None,
                          pending=SimpleNamespace(push=lambda keys, *a, **k: pushed.append(len(keys))), fly_lock=threading.Lock())
        return b

    ctx = SimpleNamespace(mints=[f"m{i}" for i in range(5)], infos=[{}] * 5, X=X, t_start=0.0, m1="minute", bars={}, conn=None)
    live = book(); assert live._fast()
    d = live._decide(ctx)["decision"]
    assert passes == [5] and forwards == [("all", 3)] and not pushed and not live._scored_rows   # one pass; the fly scored the teacher's 3 trades; no tags yet
    assert d["allow"].tolist() == [True, False, True, False, False] and d["reason"][3].startswith("learned veto")
    live._score(None, ctx, live_pass=live._live_pass[1:])                             # after the orders (after_minute)
    assert passes == [5] and forwards[1:] == [("all_h", 5)] and sum(pushed) == 5 and len(live._scored_rows) == 5   # the same pass; every triggered row tagged
    passes.clear(); full = book(); assert full._score(None, ctx)["decision"]["allow"].tolist() == d["allow"].tolist()   # the full scoring decides the same
    assert passes == [5, 5]                                                            # it ran the stack twice (decision, teacher scores)


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


def test_without_its_teacher_the_gated_fly_enters_nothing():
    d = fly_session.no_teacher(4)
    assert not d["allow"].any() and all(r.startswith("fail closed") for r in d["reason"]) and np.isinf(d["threshold"]).all()


def test_a_release_keeps_the_fly_pointing_at_its_teacher():
    from fly_trader.ops.release import remap_teacher
    note = {"teacher_snapshot": 154, "lines": {"ev": 0.16}}
    assert remap_teacher(note, {154: 1, 157: 2}) == {"teacher_snapshot": 1, "lines": {"ev": 0.16}}   # the shipped selector's id here
    assert remap_teacher({"teacher_snapshot": 1}, {154: 1, 157: 2}) is None                          # already remapped (a re-applied release)
    assert remap_teacher({"lines": {}}, {154: 1}) is None and remap_teacher({"teacher_snapshot": 154}, {154: 154}) is None
