"""The strategy stack: every fit records the best setting it reached even when none clears the bars; the objective (win rate, profit factor, trade count; weekly totals recorded, not required) orders settings with
admissible ones first; a filter must beat random removal; on a planted edge the stack finds an admissible ev strategy,
records every component with a reason, and the live decision reproduces the fitted rule."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fly_trader.train import strategies as S
from fly_trader.train.decisions import X_COLS, DecisionSet


def _ds(days=42, per_day=600, seed=0):
    rng = np.random.default_rng(seed); n = days * per_day
    X = np.zeros((n, len(X_COLS)), np.float32); c = {k: i for i, k in enumerate(X_COLS)}
    X[:, c["age_known"]] = 0.0
    sig = rng.normal(size=n).astype(np.float32); X[:, c["imb_5m"]] = sig
    X[:, c["ret_1m"]] = rng.normal(0, 0.02, n); X[:, c["mkt_vol_1h"]] = rng.normal(5, 1, n)
    d0 = datetime(2026, 7, 1, tzinfo=timezone.utc); k = np.arange(n)
    ts = np.array([(d0 + timedelta(days=int(i // per_day), minutes=int(i % per_day) * 2)).timestamp() for i in k])
    day = np.array([(d0 + timedelta(days=int(i // per_day))).date() for i in k], dtype=object)
    good = sig > 1.2
    fwd = np.where(good, np.where(rng.random(n) < 0.8, 0.05, -0.04), rng.normal(-0.01, 0.03, n)).astype(np.float32)
    fh = {h: fwd.copy() for h in S.HOLDS_MIN}
    return DecisionSet(X=X, y=(fwd > 0).astype(np.int8), fwd=fwd, fwd_pess=fwd, day=day, ts=ts, mint=np.array([f"m{i % 150}" for i in k]),
                       cols=list(X_COLS), horizon_s=7200.0, fwd_h=fh)


def test_score_sign_test_and_ordering(monkeypatch):
    monkeypatch.setattr(S, "_min_trades", lambda: 5)
    ds = _ds(days=21, per_day=50)
    r = np.where(np.arange(len(ds.y)) % 10 < 7, 0.02, -0.01).astype(np.float32)
    s = S.score_trades(ds, np.arange(len(ds.y)), r, ds.day[0])
    assert s.win == pytest.approx(0.7) and s.pf == pytest.approx((0.7 * 0.02) / (0.3 * 0.01)) and s.weeks == 3 and s.weeks_pos == 3
    assert not s.sign_ok and s.base_ok                                # the weekly statistic is recorded, no longer a bar
    a = S.Score(n=200, win=0.7, pf=2.0, total=5.0, weeks=8, weeks_pos=8, sign_ok=True)
    b = S.Score(n=200, win=0.5, pf=2.0, total=50.0, weeks=8, weeks_pos=8, sign_ok=True)
    assert a.admissible and not b.admissible and b.base_ok
    assert S.better(a, b) and not S.better(b, a)                      # admissible beats a larger fallback total
    assert S.passes(b, fallback=True) and not S.passes(b, fallback=False)


def test_filter_must_beat_random_removal():
    ds = _ds(days=21, per_day=40)
    before = np.ones(len(ds.y), bool); after = before & (ds.fwd_pess > 0)          # removes exactly the losers
    ok, info = S.beats_random_removal(ds, before, after, 7200.0, ds.fwd_pess, ds.day[0])
    assert ok and info["filtered_total"] > info["random_total"]


def test_stack_finds_the_planted_edge_and_decides_like_it(monkeypatch):
    monkeypatch.setattr(S, "STRATEGIES", {"ev": {**S.STRATEGIES["ev"], "holds": (30, 120)}})    # two holds: the scan runs, the test stays quick
    monkeypatch.setattr(S, "GROUPS", {})
    import fly_trader.train.strategies as mod
    monkeypatch.setattr(mod, "LEGACY_COLS", ["imb_5m", "ret_1m", "age_known", "log_age_h"])
    ds = _ds(days=100, per_day=400)                                   # enough weeks per half for the walk-forward
    st = S.fit_stack(ds)
    names = [c["name"] for c in st.components]
    assert "strategy: ev" in names and "dump veto" in names and "hour and market gates" in names and any(n.startswith("meta-label") for n in names)
    ev = next(c for c in st.components if c["name"] == "strategy: ev")
    assert ev["passed"] and ev["selection"]["win"] >= S.WIN_MIN and ev["evaluation"]["win"] >= S.WIN_MIN
    assert st.deployable and not st.fallback
    models = S.final_models(ds, st)
    X = ds.X[:200]; d = S.decide(models, X, ds.cols, float(ds.ts[0]))
    good = X[:, ds.cols.index("imb_5m")] > 1.2
    assert (d["strategy"][good] == "ev").mean() > 0.8 and (d["strategy"][~good] == None).mean() > 0.8   # noqa: E711
    assert set(np.unique(d["reason"])) <= {"trade", "no strategy fires", "win filter (meta-label)"} | {r for r in d["reason"] if "veto" in r or "gated" in r}


def test_a_fit_that_clears_nothing_records_what_it_reached(monkeypatch):
    """A dropped component must say how close it came: the most profitable setting tried, whether or not it passed."""
    monkeypatch.setattr(S, "STRATEGIES", {"ev": {**S.STRATEGIES["ev"], "holds": (120,)}})
    monkeypatch.setattr(S, "PF_MIN", 99.0)                            # nothing can clear the profit factor
    ds = _ds(days=60, per_day=200)                                    # past the walk-forward warmup, so settings really are scored
    uni = np.ones(len(ds.y), bool); sel = (ds.day >= ds.days[25]) & (ds.day < ds.days[45])   # inside the scored region, past the warmup
    f = S.fit_strategy(ds, "ev", ["imb_5m", "ret_1m", "age_known", "log_age_h"], uni, sel, ds.day[0])
    assert f.selection == {} and f.trials > 0 and "no setting met" in f.reason
    assert f.best_seen and f.best_seen["n"] > 0 and f.best_seen["win"] is not None    # the numbers survive the failure
    assert f.best_seen["total"] != 0.0                                                # a setting that traded, not the empty pick


def test_empty_inputs_are_skipped_and_groups_are_judged_at_one_hold(monkeypatch):
    """Walk-forwards are the expensive part: an input group with no signal costs none, and the group comparison runs at a
    single hold, while each strategy still fits its hold from the whole grid."""
    monkeypatch.setattr(S, "STRATEGIES", {"ev": {**S.STRATEGIES["ev"], "holds": (30, 120)}})
    monkeypatch.setattr(S, "GROUPS", {"empty": ["age_known"], "market": ["mkt_vol_1h"]})     # age_known never varies here
    import fly_trader.train.strategies as mod
    monkeypatch.setattr(mod, "LEGACY_COLS", ["imb_5m", "ret_1m", "log_age_h"])
    calls = []; real = S.fit_strategy
    monkeypatch.setattr(S, "fit_strategy", lambda *a, **k: (calls.append((a[1], k.get("holds"))), real(*a, **k))[1])
    st = S.fit_stack(_ds(days=100, per_day=150))
    empty = next(c for c in st.components if c["name"] == "inputs: empty")
    assert not empty["passed"] and "empty in this corpus" in empty["reason"] and empty["trials"] == 0
    n_inputs = sum(1 for c in st.components if c["name"].startswith("inputs:") and c["trials"])
    assert all(h == (S.EV_HOLD,) for _, h in calls[:1 + n_inputs])            # the baseline and every group trial: one hold
    assert any(h is None for _, h in calls[1 + n_inputs:])                    # the strategies themselves scan the whole grid
