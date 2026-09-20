"""The live fly learns from exactly the label it was trained on: its resolver equals ``decisions.build``'s ``fwd_pess``
row for row (next-open entry within two minutes else the close, last close at or before t + hold, exit cost at both
ends), drops an exit after a 50× jump or without the row's own bar; tags can be filtered."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from fly_trader.agent.fly_session import resolve_label
from fly_trader.brain import plastic
from fly_trader.market.features import FEATURES
from fly_trader.train import decisions, mature


def _part(tmp_path):
    rng = np.random.default_rng(0)
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = []
    for mint in ("Apump", "Bpump"):
        offs = sorted(rng.choice(400, 220, replace=False).tolist())      # traded minutes with irregular gaps (some > 2 minutes)
        price = 1.0
        for o in offs:
            price *= float(np.exp(rng.normal(0, 0.02)))
            rows.append({**{f: 0.0 for f in FEATURES}, "mint": mint, "ts": base + timedelta(minutes=int(o)), "open": price * float(np.exp(rng.normal(0, 0.005))),
                         "close": price, "resq": 50.0, "age_h": 30.0, "traders_15m": 10.0, "traders_1h": 40.0, "n_trades_1m": 3.0, "logvol_15m": 3.0,
                         "exit_cost_0p1": float(rng.uniform(0.005, 0.02))})
    df = pd.DataFrame(rows)
    root = tmp_path / "feat"
    mature.write_part(pa.Table.from_pandas(df, preserve_index=False), root / "2026-09-01" / "part.parquet", mature.FEATURE_VERSION, {"fly_agg": mature.AGG_VERSION})
    return df, root


def test_resolver_equals_the_training_label(tmp_path):
    df, root = _part(tmp_path)
    ds = decisions.build(days=None, horizon_min=10, feature_dir=root)
    assert len(ds.y) > 100
    from fly_trader.market.exit_cost import cost_at_size          # the engine reprices before it hands a bar to the fly
    bars = {m: [(t.timestamp(), o, c, cost_at_size(e, rq)) for t, o, c, e, rq in
                zip(g["ts"], g["open"], g["close"], g["exit_cost_0p1"], g["resq"])]
            for m, g in df.sort_values("ts").groupby("mint")}
    got = np.array([resolve_label(bars[m], t, ds.horizon_s) for m, t in zip(ds.mint, ds.ts)], dtype=np.float64)
    assert np.allclose(got, ds.fwd_pess, rtol=1e-5, atol=1e-6)
    gaps = np.array([next((b[0] for b in bars[m] if b[0] > t), np.inf) - t for m, t in zip(ds.mint, ds.ts)])
    assert (gaps > 120).any() and (gaps <= 120).any()                               # both entry rules were exercised


def test_resolver_refuses_jumps_and_missing_rows():
    bars = [(0.0, 1.0, 1.0, 0.01), (60.0, 1.0, 1.0, 0.01), (120.0, 1.0, 60.0, 0.01)]
    assert resolve_label(bars, 0.0, 120.0) is None                                  # a 60× jump into the exit minute
    assert resolve_label(bars, 30.0, 120.0) is None                                 # the row's own minute is not in the bars
    ok = [(0.0, 1.0, 1.0, 0.01), (300.0, 2.0, 1.1, 0.02)]
    assert resolve_label(ok, 0.0, 600.0) == pytest.approx(1.1 / 1.0 * 0.99 * 0.98 - 1)   # next trade 5 min later: entry at the close


def test_tags_subset_keeps_the_chosen_rows():
    k = torch.rand(3, 8); t = plastic.Tags(keys=["a", "b", "c"], ts=np.array([0.0, 1.0, 2.0]), y_dn=torch.tensor([1.0, 2.0, 3.0]), u0=torch.rand(3, 4), k=k,
                                           w=torch.tensor([[1.0, 10.0, 1.0]]))
    s = t.subset(np.array([True, False, True]))
    assert s.keys == ["a", "c"] and list(s.ts) == [0.0, 2.0] and s.y_dn.tolist() == [1.0, 3.0] and torch.equal(s.k, k[[0, 2]]) and s.w.tolist() == [[1.0, 1.0]]
