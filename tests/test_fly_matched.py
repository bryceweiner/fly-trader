"""Matched mode (2026-10-23): the fly is taught by a given stack on exactly the rows that stack was fit on, its lines
matched to the teacher's own trade count there (no calibration week, no purge day), it is scored beside the selector on
the held-out days with the selector's rules, and a matched fly races on that out-of-sample evidence. The default
bootstrap is untouched (tests/test_fly_strategies.py, tests/test_fly_distill.py)."""
import shutil
from datetime import timedelta

import numpy as np
import pytest
import torch

from fly_trader import config
from fly_trader.train import decisions, fly_selector, strategies
from tests.test_fly_distill import _graph
from tests.test_fly_strategies import HOLDS, _ds, _Teacher


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(config, "FLY_SCORE", "own")


def _models(ds, fit):
    """A two-strategy stack fitted on ``fit`` (strategies.final_models' shape): ev on every row, capitulation on falling minutes."""
    cols = ["f0", "f1", "f2", "f3"]; ci = [ds.cols.index(c) for c in cols]; rows = np.flatnonzero(fit)
    out = {"strategies": {}, "veto": None, "combine": "score", "spec": "memecoin"}
    for k, H in HOLDS.items():
        r = rows if k == "ev" else rows[ds.X[rows, ds.cols.index("ret_1m")] < 0]
        gbm, sc = strategies.fit_regressor(ds.X[np.ix_(r, ci)], ds.fwd_h[H][r], seed=99)
        out["strategies"][k] = {"gbm": gbm, "scaler": sc, "cols": cols, "line": 0.005, "hold_min": H, "thr": {}, "high": None, "meta": None,
                                "hours": [], "regimes": [], "sizing": [], "selection": {}, "evaluation": {}}
    return out


def test_the_fit_rows_are_final_models_rows():
    ds = _ds(); hold = ds.days[-3:]
    keep = strategies.fit_keep(ds, hold)
    assert not keep[np.isin(ds.day, hold)].any() and keep[~np.isin(ds.day, hold)].all()
    assert strategies.fit_keep(ds, None).all()


def test_a_matched_fly_learns_on_its_teachers_rows_with_lines_matched_to_its_trades():
    torch.manual_seed(0)
    ds = _ds(); hold = ds.days[-4:]; fit = strategies.fit_keep(ds, hold)
    fly, info = fly_selector.bootstrap_matched(ds, _Teacher(ds), fit, epochs=2, graph=_graph(), device="cpu")
    assert info["mode"] == fly_selector.MATCHED and info["calibration"] is None and info["calibration_days"] is None
    assert info["train_days"] == [str(ds.days[0]), str(hold[0] - timedelta(days=1))] and info["S"] == str(hold[0])
    per = info["line_fit"]["per_strategy"]
    for k in HOLDS:
        assert per[k]["teacher_trades"] > 0 and per[k]["trades"] <= per[k]["teacher_trades"]        # the match rule: no more trades than its teacher
        assert fly.lines[k] == pytest.approx(per[k]["line"])
    assert info["line_fit"]["days"] == info["train_days"]                                          # matched on the training days, not on a later week


def test_a_matched_fly_skips_rows_its_teacher_had_no_label_for():
    torch.manual_seed(0)
    ds = _ds(); y30 = ds.fwd_h[30].copy(); y30[ds.day == ds.days[5]] = np.nan; ds.fwd_h[30] = y30
    fit = strategies.fit_keep(ds, ds.days[-4:]); uni = fly_selector.selector.in_universe(ds.X, ds.cols)
    _, info = fly_selector.bootstrap_matched(ds, _Teacher(ds), fit, epochs=1, graph=_graph(), device="cpu")
    cap = ds.X[:, ds.cols.index("ret_1m")] < 0
    assert info["fit"]["training_rows"] == int((fit & uni & (np.isfinite(y30) | (cap & np.isfinite(ds.fwd_h[10])))).sum())


def test_the_oos_report_scores_both_books_on_the_holdout_with_the_selectors_rules():
    torch.manual_seed(0)
    ds = _ds(); hold = ds.days[-4:]; fit = strategies.fit_keep(ds, hold); models = _models(ds, fit)
    teacher, fit2, note = fly_selector.matched_teacher(ds, models, exclude_days=hold)
    assert (fit2 == fit).all() and "blind to" in note
    fly, _ = fly_selector.bootstrap_matched(ds, teacher, fit, epochs=1, graph=_graph(), device="cpu")
    rep = fly_selector.oos_report(ds, models, fly, hold)
    sel = rep["chains"]["all"]["selector"]; rec = strategies.score_holdout(ds, models, hold)
    assert (sel["all"]["n"], sel["all"]["mean"], sel["all"]["random_mean"]) == (rec["n"], rec["mean"], rec["random_mean"])   # the stack's own record
    assert sum(sel[k]["n"] for k in HOLDS) == sel["all"]["n"]                                      # per strategy: the book's trades by strategy
    fl = rep["chains"]["all"]["fly"]
    assert set(fl) == {"all", *HOLDS} and sum(fl[k]["n"] for k in HOLDS) == fl["all"]["n"]
    assert rep["days"] == [str(hold[0]), str(hold[-1])] and set(rep["chains"]) == {"all", "sol"}
    alone = fly_selector.oos_report(ds, models, None, hold)                       # matched-pair --no-blind-fly
    assert set(alone["chains"]["all"]) == {"selector"} and alone["chains"]["all"]["selector"]["all"]["n"] == sel["all"]["n"]


def test_a_matched_fly_races_on_its_out_of_sample_evidence(monkeypatch):
    monkeypatch.setattr(config, "HANDOVER_MIN_TRADES", 30)
    base = {"mode": fly_selector.MATCHED, "data": fly_selector.FLY_VERSION, "gates_ok": True, "calibration": None}
    ok = lambda ev: fly_selector.deployable({**base, "oos": {"evidence": ev}})[0]        # noqa: E731
    assert ok({"trades": 40, "mean": 0.012})
    assert not ok({"trades": 40, "mean": -0.01}) and not ok({"trades": 12, "mean": 0.05}) and not ok({"trades": 40, "mean": float("nan")})
    assert not fly_selector.deployable(base)[0]                                                   # no report, no race
    assert not fly_selector.deployable({**base, "gates_ok": False, "gate_failures": ["x"], "oos": {"evidence": {"trades": 99, "mean": 0.1}}})[0]
    assert fly_selector.deployable({"data": fly_selector.FLY_VERSION, "gates_ok": True, "calibration": {"trades": 40, "mean": 0.01}})[0]   # the default: its week


def test_a_pinned_set_ends_on_its_last_day(tmp_path, db_conn):
    from tests.test_solana_identity import HOLDS as H, _part
    one = _part(tmp_path / "one"); two = _part(tmp_path / "two"); shutil.copytree(two / "2026-09-01", two / "2026-09-02")
    a = decisions.build(days=None, feature_dir=one, holds=H); b = decisions.build(days=None, feature_dir=two, holds=H, last_day="2026-09-01")
    assert b.days == a.days and len(b.y) == len(a.y) and np.array_equal(a.X, b.X, equal_nan=True)


def test_without_the_test_fly_the_full_fly_races_on_the_selectors_holdout_picks(tmp_path, monkeypatch):
    """--no-blind-fly (2026-10-24): no test fly is trained, and the full fly's out-of-sample evidence is the blind
    selector's own holdout picks. 2026-10-25: the last step read the freed test-fly variable, so the run died after saving
    the full fly, which then had no evidence to race on."""
    from datetime import date
    from types import SimpleNamespace
    from fly_trader.train import matched_pair
    last = date(2026, 10, 22); book = {"n": 40, "mean": 0.05, "total": 2.0, "win": 0.6, "random_mean": -0.04}
    sel = SimpleNamespace(blind={"holdout_days": ["2026-10-22", "2026-10-22"], "models": "blind"}, stack="full",
                          metrics={"holdout": {"n": 40, "mean": 0.05}, "deployable": True, "components": []})
    trained, saved = [], []

    def fly(ds, out, sub, models, exclude_days, epochs, match_days, stop):
        (out / sub).mkdir(exist_ok=True); trained.append((sub, models))
        return SimpleNamespace(), {"mode": fly_selector.MATCHED, "gates_ok": True, "lines": {"capitulation": 0.1}}, None
    monkeypatch.setattr(matched_pair, "_quiet_progress", lambda: None); monkeypatch.setattr(config, "HANDOVER_MIN_TRADES", 30)
    monkeypatch.setattr(decisions, "build", lambda **k: SimpleNamespace(days=[date(2026, 10, 21), last], y=np.zeros(3)))
    monkeypatch.setattr(matched_pair, "_selector", lambda ds, out, stop: (sel, [last]))
    monkeypatch.setattr(fly_selector, "oos_report", lambda ds, models, fly, hold: {"days": ["2026-10-22", "2026-10-22"], "rows": 3,
                                                                                    "chains": {"all": {"selector": {"all": book}}}})
    monkeypatch.setattr(matched_pair, "report_text", lambda rep: "report"); monkeypatch.setattr(matched_pair, "_fly", fly)
    monkeypatch.setattr(matched_pair, "_save_fly", lambda f, info, path: saved.append(info))
    out = matched_pair.main(tmp_path, "2026-10-22", epochs=1, blind_fly=False)
    assert trained == [("fly_full", "full")]                                             # the full fly only, taught by the full stack
    assert saved[-1]["oos"]["source"] == "the blind selector's own picks on the holdout days (no test fly)"
    assert saved[-1]["oos"]["evidence"]["trades"] == 40 and saved[-1]["calibration"]["trades"] == 40
    assert out["fly_full"]["deployable"] and (tmp_path / "summary.json").exists()
