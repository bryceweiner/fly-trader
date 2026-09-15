"""Training pipeline: schedule, waiting for data, the deploy decision and the build-completeness check."""
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from fly_trader.train import mature, pipeline
from fly_trader.train.selector import DATA_VERSION, deploy_decision, is_deployable


def test_schedule_is_seven_days_or_on_request():
    now = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
    assert pipeline.due({}, now)                                                                  # never ran
    assert not pipeline.due({"last_run_at": (now - timedelta(days=6)).isoformat()}, now)
    assert pipeline.due({"last_run_at": (now - timedelta(days=7)).isoformat()}, now)
    assert pipeline.due({"last_run_at": now.isoformat(), "run_requested": True}, now)             # Retrain now
    assert not pipeline.due({"run_requested": True, "retry_after": (now + timedelta(minutes=5)).isoformat()}, now)


def test_run_waits_for_the_feature_build(monkeypatch):
    monkeypatch.setattr(mature, "build_complete", lambda: (False, "3 day(s) of features still to build"))
    with pytest.raises(pipeline.NotReady, match="features still to build"):
        pipeline.run()


def test_run_trains_selector_then_fly(monkeypatch):
    calls = []; saved = {}
    monkeypatch.setattr(mature, "build_complete", lambda: (True, "ok"))
    monkeypatch.setattr(pipeline, "state", lambda: dict(saved))
    monkeypatch.setattr(pipeline, "_save", lambda **kv: saved.update(kv) or dict(saved))
    monkeypatch.setattr(pipeline, "record_event", lambda *a, **k: None)
    from fly_trader.train import fly_selector, selector
    monkeypatch.setattr(selector, "main", lambda stop_event=None: calls.append("selector") or {"snapshot_id": 7, "deployable": False, "line": 0.005})
    monkeypatch.setattr(fly_selector, "main", lambda stop_event=None, line=None: calls.append(("fly", line)) or {"snapshot_id": 8})
    out = pipeline.run()
    assert calls == ["selector", ("fly", 0.005)] and out["selector_snapshot"] == 7 and out["fly_snapshot"] == 8 and out["stage"] == "done"   # same line
    assert datetime.fromisoformat(out["next_run_at"]) - datetime.fromisoformat(out["last_run_at"]) == timedelta(days=7)


def test_only_a_profitable_selector_that_beats_random_is_deployable():
    assert deploy_decision({"n": 500, "mean": 0.01}, {"mean": -0.03})[0]
    assert not deploy_decision({"n": 500, "mean": -0.005}, {"mean": -0.03})[0]                    # lost money
    assert not deploy_decision({"n": 500, "mean": 0.01}, {"mean": 0.02})[0]                       # no better than random
    assert not deploy_decision({"n": 20, "mean": 0.05}, {"mean": -0.03})[0]                       # too few trades
    assert is_deployable({"data": DATA_VERSION, "deployable": True}) and not is_deployable({"data": {"agg": 1}, "deployable": True})


def test_build_complete_needs_every_feature_day(tmp_path, monkeypatch):
    monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature"); monkeypatch.setattr(mature, "MATURE_FEAT_DIR", tmp_path / "feat")
    monkeypatch.setattr(mature, "days_ready", lambda: [])
    t = pa.table({"mint": ["A"], "close": [1.0]})
    for d in ("2026-09-01", "2026-09-02"):
        mature.write_part(t, tmp_path / "mature" / f"{d}.parquet", mature.AGG_VERSION)
    ok, why = mature.build_complete()
    assert not ok and "features still to build" in why                                            # 09-02 has no feature part
    mature.write_part(t, tmp_path / "feat" / "2026-09-02" / "part.parquet", mature.FEATURE_VERSION, {"fly_agg": mature.AGG_VERSION})
    assert mature.build_complete()[0]
    mature.write_part(t, tmp_path / "feat" / "2026-09-02" / "part.parquet", mature.FEATURE_VERSION)    # built before the aggregation stamp
    assert not mature.build_complete()[0]
