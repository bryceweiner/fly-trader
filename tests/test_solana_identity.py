"""Solana behaviour pinned before Robinhood Chain was added (branch rh-memecoins, 2026-09-28).

Every shared function the second chain touches — decision rows, labels and eligibility (train/decisions.build), the
feature engine (market/features.TokenState), trading costs (market/exit_cost), sizing (agent/sizing) and the live
engine's feature vector (agent/minute_engine) — is run on fixed synthetic Solana inputs and compared with the values the
code produced before any RH change (tests/fixtures/solana_golden.npz). Columns are compared by name, so inputs appended
for the second chain do not count; every Solana value must stay bit-identical.

Regenerate only on purpose: SOLANA_GOLDEN_REGEN=1 pytest tests/test_solana_identity.py
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from fly_trader import config
from fly_trader.agent import sizing
from fly_trader.market import exit_cost
from fly_trader.market.features import FEATURES, TokenMeta, TokenState
from fly_trader.train import decisions

GOLDEN = Path(__file__).parent / "fixtures" / "solana_golden.npz"
BASE_COLS = list(decisions.X_COLS[:64])          # the 64 Solana inputs as of 2026-09-28
HOLDS = (30, 120, 240)
T0 = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)


def _part(tmp_path):
    from fly_trader.market.features import FEATURE_VERSION
    from fly_trader.train.mature import AGG_VERSION, SCHEMA, write_part
    rng = np.random.default_rng(20260928); rows = []
    for m in range(6):
        px = 1e-6 * (1 + m); resq = 20.0 + 15 * m
        for k in range(420):
            px *= float(np.exp(rng.normal(0, 0.03))); resq = max(5.0, resq * float(np.exp(rng.normal(0, 0.01))))
            r = {"mint": f"M{m}pump", "ts": T0 + timedelta(minutes=k), "phase": "amm", "t_rel_min": k, "open": px * (1 + rng.normal(0, 0.005)),
                 "high": px * 1.01, "low": px * 0.99, "close": px, "volume_sol": float(rng.uniform(0, 30)), "resq": resq, "age_h": 6.5 + k / 60,
                 "has_trades": True, "mask": 0, "pre_minutes": 0.0, "pre_trades": 0.0, "pre_buyers": 0.0, "pre_vol_sol": 0.0, "pre_top10_pct": 0.0,
                 "traders_15m": float(rng.integers(0, 40)), "traders_1h": float(rng.integers(0, 120)), "n_trades_1m": float(rng.integers(0, 12))}
            for f in FEATURES:
                r[f] = float(rng.normal(0, 1))
            r["logvol_15m"] = float(abs(rng.normal(2.5, 1))); r["log_liquidity_sol"] = float(np.log1p(2 * resq)); r["exit_cost_0p1"] = float(rng.uniform(0.004, 0.03))
            rows.append(r)
    write_part(pa.Table.from_pylist(rows, schema=SCHEMA), tmp_path / "2026-09-01" / "part.parquet", FEATURE_VERSION, {"fly_agg": AGG_VERSION})
    return tmp_path


def _features_stream():
    rng = np.random.default_rng(7); st = TokenState("Fpump"); meta = TokenMeta("Fpump", program_label="Pump.fun Amm", graduated_at=1_790_000_000.0 - 7200.0)
    out = []; t = 1_790_000_000.0; px = 2e-6
    for k in range(600):
        t += float(rng.uniform(1, 20)); px *= float(np.exp(rng.normal(0, 0.01)))
        st.append(t, px, float(rng.uniform(0.01, 3)), bool(rng.random() < 0.55), f"w{int(rng.integers(0, 40))}", float(40 + rng.normal(0, 3)))
        if k % 50 == 49:
            f, _ = st.features(t, meta); out.append(np.asarray(f, dtype=np.float64))
    return np.stack(out)


def _costs():
    ec = np.linspace(0.003, 0.05, 9); r = np.array([0.5, 5, 20, 80, 400, 5000.0, np.nan, -1.0, 0.0])
    grid = np.array([[exit_cost.cost_at_size(e, q) for q in r] for e in ec], dtype=np.float64)
    fees = np.array([exit_cost.fee_fraction(v, m) for v in (0.02, 0.1, 0.5, 2.0) for m in (50.0, 500.0, 5000.0, 1e5, None)])
    full = np.array([exit_cost.exit_cost_fraction(v, q, m, lbl) for v in (0.05, 0.5, 3.0) for q in (10.0, 100.0, None)
                     for m in (100.0, 30000.0) for lbl in ("Pump.fun Amm", "Meteora DLMM")])
    lsz = np.array([exit_cost.label_size(q) for q in (None, 1.0, 10.0, 25.0, 1000.0)])
    return grid, fees, full, lsz


def _sizes():
    table = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.5, "kelly": 0.3}, {"lo": 0.01, "n": 100, "mean": 0.05, "win": 0.6, "kelly": 0.7}]
    out = []
    for flat in (False, True):
        for tab in (table, []):
            for bank in (0.5, 5.3, 50.0, 500.0):
                for cash in (0.31, 2.0, bank):
                    for resq in (None, 3.0, 40.0, 1000.0):
                        for sc in (0.1, 0.105, 0.2):
                            out.append(sizing.size_position(sc, 0.1, tab, bank, cash, resq, flat=flat)[0])
    return np.array(out)


def _engine_vectors(db_conn):
    from fly_trader.agent import minute_engine as me
    s = me.MinuteEngine([]); rng = np.random.default_rng(3); t = 1_790_000_040.0; px = 1e-6; xs = []; infos = []
    for k in range(90):
        px *= float(np.exp(rng.normal(0, 0.02)))
        a = {"open": px * 0.99, "high": px * 1.02, "low": px * 0.98, "close": px, "buy": float(rng.uniform(0, 8)), "sell": float(rng.uniform(0, 8)),
             "nb": int(rng.integers(0, 9)), "ns": int(rng.integers(0, 9)), "n_traders": int(rng.integers(1, 12)), "resq": float(30 + k),
             "pool": "GOLDPOOL", "program_label": "Pump.fun Amm"}
        x, info = s._features(db_conn, "GOLDENpump", a, t + 60.0 * k)
        xs.append([x[decisions.X_COLS.index(c)] for c in BASE_COLS]); infos.append([info["price"], info["resq"], info["ec"], float(info["broken"])])
    return np.asarray(xs, dtype=np.float64), np.asarray(infos, dtype=np.float64)


def _compute(db_conn, tmp_path):
    monkey = {"LABEL_SIZE_SOL": 0.5, "MAX_POOL_SHARE": 0.02, "GAS_RESERVE_SOL": 0.30, "MAX_POSITION_FRACTION": 0.10, "KELLY_FRACTION": 0.25,
              "MIN_POSITION_SOL": 0.02, "MAX_POSITION_SOL": 0.1}
    for k, v in monkey.items():
        setattr(config, k, v)
    ds = decisions.build(days=None, feature_dir=_part(tmp_path), holds=HOLDS)
    X = np.stack([ds.X[:, ds.cols.index(c)] for c in BASE_COLS], axis=1).astype(np.float64)
    out = {"X": X, "ts": ds.ts.astype(np.float64), "mint": np.array([str(m) for m in ds.mint]), "y": ds.y.astype(np.float64),
           "fwd": np.asarray(ds.fwd, np.float64), "fwd_pess": np.asarray(ds.fwd_pess, np.float64)}
    for h in HOLDS:
        out[f"fwd_h{h}"] = np.asarray(ds.fwd_h[h], np.float64)
    out["features"] = _features_stream()
    out["cost_grid"], out["fees"], out["exit_full"], out["label_size"] = _costs()
    out["sizes"] = _sizes()
    out["engine_x"], out["engine_info"] = _engine_vectors(db_conn)
    return out


def test_solana_is_bit_identical(db_conn, tmp_path, monkeypatch):
    for k in ("LABEL_SIZE_SOL", "MAX_POOL_SHARE", "GAS_RESERVE_SOL", "MAX_POSITION_FRACTION", "KELLY_FRACTION", "MIN_POSITION_SOL", "MAX_POSITION_SOL"):
        monkeypatch.setattr(config, k, getattr(config, k))          # restored after the test
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint LIKE 'M_pump' OR mint = 'GOLDENpump'")
    db_conn.commit()
    got = _compute(db_conn, tmp_path)
    if os.environ.get("SOLANA_GOLDEN_REGEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(GOLDEN, **got)
        pytest.skip("golden regenerated")
    want = np.load(GOLDEN, allow_pickle=False)
    assert set(want.files) == set(got), set(want.files) ^ set(got)
    for k in want.files:
        a, b = want[k], got[k]
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if a.dtype.kind in "fc":
            assert np.array_equal(a, b, equal_nan=True), f"{k}: {int((~np.isclose(a, b, equal_nan=True, rtol=0, atol=0)).sum())} values differ"
        else:
            assert np.array_equal(a, b), k
