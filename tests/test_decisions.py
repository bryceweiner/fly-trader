"""Decision points: labels, pessimistic fills, the past-only scale-break rule, holds and per-day accounting."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader.market.features import FEATURES
from fly_trader.train import decisions
from fly_trader.train.mature import SCHEMA


def _part(tmp_path, rows, version=None):
    from fly_trader.market.features import FEATURE_VERSION
    from fly_trader.train.mature import write_part
    write_part(pa.Table.from_pylist(rows, schema=SCHEMA), tmp_path / "2026-09-01" / "part.parquet", FEATURE_VERSION if version is None else version)
    return tmp_path


def _row(mint, ts, close, resq=100.0, open_=None, logvol_15m=3.0):
    r = {"mint": mint, "ts": ts, "phase": "amm", "t_rel_min": 10, "open": open_ if open_ is not None else close, "high": close, "low": close, "close": close,
         "volume_sol": 10.0, "resq": resq, "age_h": 2.0, "has_trades": False, "mask": 0,
         "pre_minutes": 0.0, "pre_trades": 0.0, "pre_buyers": 0.0, "pre_vol_sol": 0.0, "pre_top10_pct": 0.0,
         "traders_15m": 5.0, "traders_1h": 10.0, "n_trades_1m": 2.0}
    for f in FEATURES:
        r[f] = 0.0
    r["logvol_15m"] = logvol_15m
    return r


def test_labels_fills_and_past_only_rule(tmp_path):
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    rows = []
    # token A: flat 1.0 for 40 minutes, then +10 % at minute 20 relative to a decision at minute 0
    for k in range(80):
        px = 1.0 if k < 20 else 1.10
        rows.append(_row("A", t0 + timedelta(minutes=k), px, open_=px * 1.01 if k == 1 else px))
    # token B: normal for 30 minutes, then a 100x print at minute 30 (scale break), then normal again
    for k in range(80):
        px = 1.0 if k != 30 else 100.0
        rows.append(_row("B", t0 + timedelta(minutes=k), px))
    ds = decisions.build(days=None, feature_dir=_part(tmp_path, rows), horizon_min=30, fee=0.0)
    a = ds.mint == "A"; b = ds.mint == "B"
    # A at minute 0 sees the close at minute 30 (1.10): +10 %; pessimistic fill uses minute 1's open (1.01)
    i0 = np.flatnonzero(a & (ds.ts == t0.timestamp()))[0]
    assert ds.fwd[i0] == pytest.approx(0.10, abs=1e-6)
    assert ds.fwd_pess[i0] == pytest.approx(1.10 / 1.01 - 1, abs=1e-6)
    assert ds.y[i0] == 1
    # the last 35 minutes of the file have no full horizon: ineligible
    assert ds.ts.max() <= (t0 + timedelta(minutes=80 - 36)).timestamp()
    # B: rows before the break are eligible and their labels see the 100x print (no leak in either direction);
    # rows from the break onward are gone
    tb = ds.ts[b]
    assert tb.max() < (t0 + timedelta(minutes=30)).timestamp()
    assert not (b & (ds.ts == t0.timestamp())).any()            # its 30-minute exit lands on the reverting 100x print: not a label
    ib5 = np.flatnonzero(b & (ds.ts == (t0 + timedelta(minutes=5)).timestamp()))[0]
    assert ds.fwd[ib5] == pytest.approx(0.0, abs=1e-6)          # its mark (minute 35) is back at 1.0


def test_model_costs_on_both_sides_and_random_baseline(tmp_path):
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    rows = []
    for k in range(80):                                   # flat price; entry minutes cost 1 %, the rest 2 % one-way
        r = _row("C", t0 + timedelta(minutes=k), 1.0); r["exit_cost_0p1"] = 0.01 if k < 10 else 0.02; rows.append(r)
    ds = decisions.build(days=None, feature_dir=_part(tmp_path, rows), horizon_min=30)   # default: the paper broker's cost model
    i0 = np.flatnonzero(ds.ts == t0.timestamp())[0]
    assert ds.fwd[i0] == pytest.approx(0.99 * 0.98 - 1, abs=1e-6)          # entry cost at the signal minute, exit cost at the exit minute
    rr = decisions.random_trades(ds, np.ones(len(ds.y), bool), 3, seeds=2)
    assert len(rr) >= 2 and (rr < 0).all()                                  # a flat market loses exactly the costs
    assert len(decisions.random_trades(ds, np.ones(len(ds.y), bool), 0)) == 0


def test_trades_respect_hold_and_summaries():
    ts = np.array([0.0, 60.0, 120.0, 1800.0, 1900.0, 0.0], dtype=float)
    ds = decisions.DecisionSet(X=np.zeros((6, 1), np.float32), y=np.zeros(6, np.int8), fwd=np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1], np.float32),
                               fwd_pess=np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1], np.float32),
                               day=np.array([datetime(2026, 9, 1).date()] * 5 + [datetime(2026, 9, 2).date()], dtype=object),
                               ts=ts, mint=np.array(["A", "A", "A", "A", "A", "B"]), cols=["x"], horizon_s=1800.0)
    r = decisions.trades_from_picks(ds, np.ones(6, bool))
    assert list(r) == pytest.approx([0.1, 0.4, -0.1])            # A re-enters only after the 30-minute hold; B once
    s = decisions.summarize(np.array([0.1, 0.2], np.float32))
    assert s["pf"] is None and s["win"] == 1.0                   # no losing trade → profit factor undefined, never inf
    ev = decisions.evaluate(ds, np.ones(6), np.ones(6, bool), 0.5)
    assert ev["pooled"]["n"] == 3 and ev["per_day"]["2026-09-01"]["n"] == 2 and ev["per_day"]["2026-09-02"]["n"] == 1


def test_stale_feature_parts_are_refused(tmp_path):
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    rows = [_row("A", t0 + timedelta(minutes=k), 1.0) for k in range(80)]
    with pytest.raises(RuntimeError, match="feature version"):
        decisions.build(days=None, feature_dir=_part(tmp_path, rows, version=1), horizon_min=30, fee=0.0)
