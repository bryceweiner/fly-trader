"""The combined decision set: Solana and Robinhood Chain rows in their own units, told apart by the chain inputs; RH
eligibility and the scale-break rule scaled by K (ETH per SOL); RH labels priced with the Pons cost model at RH sizes;
each chain's launch facts merged into its own rows."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fly_trader import config
from fly_trader.market.features import FEATURES
from fly_trader.train import decisions

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _part(root, mint, resq, logvol, n=200, ec=0.01, px0=1e-6, drift=0.001, more=()):
    """One day's part holding ``mint`` and any ``more`` (mint, resq, logvol) on the same price path."""
    from fly_trader.market.features import FEATURE_VERSION
    from fly_trader.train.mature import AGG_VERSION, SCHEMA, write_part
    rows = []
    for mint, resq, logvol in [(mint, resq, logvol), *more]:
        rows += _rows(mint, resq, logvol, n, ec, px0, drift)
    write_part(pa.Table.from_pylist(rows, schema=SCHEMA), root / T0.date().isoformat() / "part.parquet", FEATURE_VERSION,
               {"fly_agg": AGG_VERSION, "fly_chain": "rh", "fly_rh_agg": 1})
    return root


def _rows(mint, resq, logvol, n, ec, px0, drift):
    rows = []; px = px0
    for k in range(n):
        px *= 1 + drift
        r = {"mint": mint, "ts": T0 + timedelta(minutes=k), "phase": "amm", "t_rel_min": k, "open": px, "high": px, "low": px, "close": px, "volume_sol": 1.0,
             "resq": resq, "age_h": 7.0, "has_trades": True, "mask": 0, "pre_minutes": 0.0, "pre_trades": 0.0, "pre_buyers": 0.0, "pre_vol_sol": 0.0,
             "pre_top10_pct": 0.0, "traders_15m": 3.0, "traders_1h": 9.0, "n_trades_1m": 2.0}
        for f in FEATURES:
            r[f] = 0.0
        r["logvol_15m"] = logvol; r["exit_cost_0p1"] = ec
        rows.append(r)
    return rows


@pytest.fixture
def two_chains(tmp_path, monkeypatch):
    k = 0.05; monkeypatch.setattr(config, "RH_ETH_PER_SOL", k); monkeypatch.setattr(config, "RH_LABEL_SIZE_ETH", 0.5 * k)
    sol, rh = tmp_path / "sol", tmp_path / "rh"
    _part(sol, "Spump", resq=30.0, logvol=np.log1p(20.0))
    _part(rh, "0xaaa", resq=30.0 * k, logvol=np.log1p(20.0 * k),            # the same market in ETH: eligible
          more=[("0xbbb", 9.0 * k, np.log1p(20.0 * k))])                   # a 9-SOL-equivalent pool: below the gate
    from fly_trader.rh import corpus as rh_corpus
    monkeypatch.setattr(decisions, "_roots", lambda fd: [(sol, "sol", decisions.part_current), (rh, "rh", rh_corpus.part_current)])
    meta_cols = ["mint"] + decisions.META_COLS
    monkeypatch.setattr(decisions, "load_features", lambda: pd.DataFrame(columns=meta_cols))
    import fly_trader.rh.meta as rhm
    monkeypatch.setattr(rhm, "load_features", lambda: pd.DataFrame([{"mint": "0xaaa", "ttg_min": 30.0, "dev_sol": 0.01, "mayhem": 0.0, "curve_known": 1.0}]))
    monkeypatch.setattr(decisions, "_rh_quote_classes", lambda: pd.DataFrame([{"mint": "0xaaa", "quote_class": "stock"}, {"mint": "0xbbb", "quote_class": "eth"}]))
    return k


def test_both_chains_one_set(two_chains):
    ds = decisions.build(days=None, horizon_min=30, holds=[30])
    ch = ds.chains()
    assert set(ch) == {"sol", "rh"} and set(ds.mint[ch == "rh"]) == {"0xaaa"}             # the shallow RH pool is gated out like a 9-SOL pool
    c = {n: ds.X[:, ds.cols.index(n)] for n in decisions.CHAIN_COLS}
    assert (c["chain_rh"] == (ch == "rh")).all() and (c["qc_stock"][ch == "rh"] == 1).all() and (c["qc_stock"][ch == "sol"] == 0).all()
    assert ds.X[ch == "rh", ds.cols.index("ttg_min")].max() == 30.0                        # RH launch facts on RH rows only
    assert ds.subset(ch == "rh").chains().tolist() == ["rh"] * int((ch == "rh").sum())


def test_rh_labels_pay_the_pons_costs_at_rh_sizes(two_chains, monkeypatch):
    ds = decisions.build(days=None, horizon_min=30, holds=[30])
    ch = ds.chains()
    # the same price path on both chains: the RH label differs only by its costs (Pons fee model at RH sizes)
    sol_r, rh_r = np.nanmedian(ds.fwd[ch == "sol"]), np.nanmedian(ds.fwd[ch == "rh"])
    from fly_trader.market.exit_cost import cost_at_size
    from fly_trader.markets import RH
    assert cost_at_size(0.01, 30.0 * two_chains, market=RH) != pytest.approx(cost_at_size(0.01, 30.0))
    assert rh_r != pytest.approx(sol_r) and np.isfinite(rh_r)


def test_solana_only_without_rh_parts(tmp_path, monkeypatch):
    sol = tmp_path / "sol"; _part(sol, "Spump", resq=30.0, logvol=np.log1p(20.0))
    ds = decisions.build(days=None, feature_dir=sol, horizon_min=30)
    assert ds.chain is None and (ds.X[:, ds.cols.index("chain_rh")] == 0).all()
