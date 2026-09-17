"""Per-strategy holds: labels for several holds computed at once equal single-hold builds, and the one-position-per-token
rule with per-row holds equals the scalar rule when holds are equal and respects each position's own exit otherwise."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fly_trader.market.features import FEATURES
from fly_trader.train import decisions, mature
from fly_trader.train.decisions import taken_idx
from fly_trader.train.flow import FLOW_COLS, MARKET_COLS, SKILL_COLS


def _part(tmp_path):
    rng = np.random.default_rng(1); base = datetime(2026, 9, 1, tzinfo=timezone.utc); rows = []
    for mint in ("Apump", "Bpump"):
        price = 1.0
        for o in sorted(rng.choice(600, 300, replace=False).tolist()):
            price *= float(np.exp(rng.normal(0, 0.02)))
            rows.append({**{f: 0.0 for f in FEATURES + FLOW_COLS + SKILL_COLS + MARKET_COLS}, "mint": mint, "ts": base + timedelta(minutes=int(o)),
                         "open": price * float(np.exp(rng.normal(0, 0.004))), "close": price, "resq": 50.0, "age_h": 30.0, "traders_15m": 10.0, "traders_1h": 40.0,
                         "n_trades_1m": 3.0, "logvol_15m": 3.0, "exit_cost_0p1": float(rng.uniform(0.005, 0.02))})
    root = tmp_path / "feat"
    mature.write_part(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), root / "2026-09-01" / "part.parquet", mature.FEATURE_VERSION, {"fly_agg": mature.AGG_VERSION})
    return root


def test_multi_hold_labels_equal_single_hold_builds(tmp_path):
    root = _part(tmp_path)
    ds = decisions.build(days=None, horizon_min=30, feature_dir=root, holds=[10, 30, 60])
    assert np.allclose(ds.fwd_h[30], ds.fwd_pess, equal_nan=True)
    for H in (10, 60):
        one = decisions.build(days=None, horizon_min=H, feature_dir=root)
        key = pd.Series(one.fwd_pess, index=pd.MultiIndex.from_arrays([one.mint, one.ts]))
        got = pd.Series(ds.fwd_h[H], index=pd.MultiIndex.from_arrays([ds.mint, ds.ts]))
        common = key.index.intersection(got.index)
        assert len(common) > 100 and np.allclose(got[common].to_numpy(), key[common].to_numpy(), equal_nan=True)


def test_per_row_holds_in_taken_idx():
    rng = np.random.default_rng(0); n = 2000
    ts = rng.integers(0, 20000, n).astype(float) * 60; mint = rng.choice([f"m{i}" for i in range(30)], n); idx = rng.choice(n, 900, replace=False)
    assert np.array_equal(taken_idx(ts, mint, np.full(n, 1800.0), idx), taken_idx(ts, mint, 1800.0, idx))
    hold = np.where(rng.random(n) < 0.5, 600.0, 14400.0)
    got = taken_idx(ts, mint, hold, idx)
    by = {}
    for i in got:                                                        # no taken entry falls inside an earlier position's own hold
        by.setdefault(mint[i], []).append(i)
    for m, ii in by.items():
        ii = sorted(ii, key=lambda i: ts[i])
        for a, b in zip(ii, ii[1:]):
            assert ts[b] >= ts[a] + hold[a]


def test_trades_and_random_with_per_row_holds(tmp_path):
    root = _part(tmp_path)
    ds = decisions.build(days=None, horizon_min=30, feature_dir=root, holds=[10, 30])
    pick = np.isfinite(ds.fwd_h[10]); hold = np.full(len(ds.y), 600.0)
    r = decisions.trades_from_picks(ds, pick, returns=ds.fwd_h[10], hold_s=hold)
    assert len(r) == len(taken_idx(ds.ts, ds.mint, hold, np.flatnonzero(pick))) and len(r) > len(decisions.trades_from_picks(ds, pick))
    ev = decisions.evaluate(ds, np.where(pick, 1.0, np.nan), pick, 0.5, hold_s=hold, returns=ds.fwd_h[10])
    assert ev["pooled"]["n"] == len(r)
