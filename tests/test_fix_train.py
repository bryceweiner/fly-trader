"""Review fixes in the feature engine and the train pipeline (compaction, window highs, exit-cost size, mature
graduation refresh, causal replay band, backtest exits and baseline, corpus part commits, selector folds)."""
import math
import random
from datetime import date, datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader import config
from fly_trader.market.exit_cost import exit_cost_fraction
from fly_trader.market.features import FEATURE_VERSION, FEATURES, FIDX, TokenMeta, TokenState
from fly_trader.train import corpus_backtest as cb
from fly_trader.train import corpus_features as cf
from fly_trader.train import mature, selector
from fly_trader.train.decisions import DecisionSet
from fly_trader.train.replay_assemble import band_amm


# ---------------------------------------------------------------- market/features.py
def test_compaction_keeps_window_start_price():
    st = TokenState("M"); t = 0.0
    for i in range(20500):
        st.append(t, 1.0, 0.1, True, f"s{i % 50}", 100.0); t += 1.0
    t += 3 * 3600 + 300
    st.append(t, 2.0, 0.1, True, "new", 100.0)              # triggers compaction
    assert len(st.ts) == 2                                   # the last pre-window trade survives
    f, mask = st.features(t, TokenMeta("M"))
    assert f[FIDX["ret_3h"]] == pytest.approx(math.log(2.0)) and mask & (1 << FIDX["ret_3h"])
    assert f[FIDX["logsigners_1h"]] == pytest.approx(math.log1p(1))
    assert st.cum_vol[-1] == pytest.approx(0.2)


def test_window_highs_exact_on_long_windows_and_after_compaction():
    random.seed(1); st = TokenState("M"); t = 0.0; lp = 0.0; compacted = False
    for i in range(50000):
        lp += random.gauss(0, 0.01); st.append(t, math.exp(lp), 0.1, i % 2 == 0, None, 100.0); t += 0.5
        compacted |= st.n_since_compact == 0
        if i % 4999 == 0 and i:
            f, _ = st.features(t, TokenMeta("M"))
            for k, w in (("1h", 3600.0), ("3h", 10800.0)):
                j = next(n for n, x in enumerate(st.ts) if x >= t - w)
                assert f[FIDX[f"dd_{k}"]] == pytest.approx(st.logp[-1] - max(st.logp[j:]), abs=1e-12)
    assert compacted


def test_exit_cost_uses_fixed_size(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION_SOL", 7.0)
    st = TokenState("M"); st.append(0.0, 1.0, 1.0, True, None, 30.0)
    f, _ = st.features(60.0, TokenMeta("M", graduated_at=-3600.0, program_label="Pump.fun Amm"))
    assert f[FIDX["exit_cost_0p1"]] == pytest.approx(exit_cost_fraction(0.1, 30.0, 3600.0 / 3600 + 60 / 3600, "Pump.fun Amm"))


# ---------------------------------------------------------------- train/mature.py
def _candles(path, day: date, mints):
    t0 = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc); rows = []
    for m in mints:
        for k in range(30):
            rows.append({"mint": m, "ts": t0 + timedelta(minutes=k), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0 + k / 100, "buy_sol": 1.0, "sell_sol": 0.5,
                         "n_buys": 2, "n_sells": 1, "n_traders": 2, "resq_sol": 50.0})
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_mature_part_records_known_graduations_and_refreshes(tmp_path, monkeypatch):
    monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature"); monkeypatch.setattr(mature, "MATURE_FEAT_DIR", tmp_path / "feat")
    (tmp_path / "mature").mkdir(); d = date(2026, 9, 2)
    _candles(tmp_path / "mature" / "2026-09-01.parquet", date(2026, 9, 1), ["A", "B"]); _candles(tmp_path / "mature" / "2026-09-02.parquet", d, ["A", "B"])
    grads = {"A": datetime(2026, 8, 30, tzinfo=timezone.utc)}
    assert mature.build_day(d, grads=grads) > 0
    part = tmp_path / "feat" / "2026-09-02" / "part.parquet"
    assert mature.part_version(part) == FEATURE_VERSION and mature.part_known(part) == 1
    t = pq.read_table(part, columns=["mint", "age_h"]).to_pandas()
    assert t.loc[t["mint"] == "A", "age_h"].notna().all() and t.loc[t["mint"] == "B", "age_h"].isna().all()
    more = {**grads, "B": datetime(2026, 8, 31, tzinfo=timezone.utc)}
    assert not mature._knows_more(part, grads)
    assert not mature._knows_more(part, more)                   # one more dated mint is below the rebuild bar (no rebuild storm)
    monkeypatch.setattr(mature, "KNOWN_REBUILD_MIN", 1)
    assert mature._knows_more(part, more)
    # a replaced part stays readable until its successor lands; the old one is kept aside
    monkeypatch.setattr(mature, "STALE_DIR", tmp_path / "stale")
    mature._archive(part, "feat_2026-09-02")
    assert part.exists() and len(list((tmp_path / "stale").glob("feat_2026-09-02_v*.parquet"))) == 1
    # a part from before the metadata key: the known count comes from its age_h column
    legacy = tmp_path / "legacy.parquet"; tab = pq.read_table(part)
    pq.write_table(tab.replace_schema_metadata({b"fly_version": str(FEATURE_VERSION).encode()}), legacy)
    assert mature.part_known(legacy) == 1


# ---------------------------------------------------------------- train/replay_assemble.py
def test_replay_band_is_causal():
    con = duckdb.connect()
    con.execute("""CREATE TEMP TABLE t0 AS SELECT 'M' AS mint, TIMESTAMP '2026-09-01' + to_seconds(i) AS ts, i AS slot, 'pump-amm' AS pool,
                   CASE WHEN i < 10 THEN 1.0 ELSE exp(ln(10000.0) * (i - 9) / 291.0) END AS price FROM range(300) r(i)""")
    con.execute("INSERT INTO t0 VALUES ('M', TIMESTAMP '2026-09-01' + to_seconds(150), 150, 'pump-amm', 1e9), "
                "('M', TIMESTAMP '2026-08-31 23:50', 0, 'pump', 1e-9)")
    band_amm(con, "t0", "t")
    assert con.execute("SELECT count(*) FROM t WHERE price = 1.0").fetchone()[0] == 10     # a later 50x+ run no longer deletes graduation-time legs
    assert con.execute("SELECT count(*) FROM t WHERE price >= 1e9").fetchone()[0] == 0      # the off-scale leg is dropped
    assert con.execute("SELECT count(*) FROM t WHERE pool = 'pump'").fetchone()[0] == 1     # curve legs are not banded
    assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 301


# ---------------------------------------------------------------- train/corpus_backtest.py
def _bt_frame(minutes, closes, t0, resq=1e12, broken=None):
    ts = [t0 + timedelta(minutes=m) for m in minutes]
    df = pd.DataFrame({"mint": "A", "ts": pd.to_datetime(ts, utc=True), "close": closes, "resq": resq, "has_trades": False})
    df["grad_day"] = df["ts"].dt.date.astype(str); df["broken"] = False if broken is None else broken
    return df


def test_time_stop_between_traded_minutes_and_entry_day_fold():
    t0 = datetime(2026, 9, 1, 23, 58, tzinfo=timezone.utc)
    df = _bt_frame([0, 1, 2, 200], [1.0, 1.1, 1.2, 5.0], t0)
    cand = np.array([True, False, False, False])
    tr = cb.simulate(df, cand, 0.0, 1.0, 1.0, 30)
    assert tr.shape == (1, 2)
    assert tr[0, 0] == date(2026, 9, 1).toordinal()          # entry day, although the exit is on the 2nd
    assert tr[0, 1] == pytest.approx(0.2, abs=1e-6)          # last close before the 30-minute stop, not minute 200's 5.0
    df = _bt_frame([0, 10, 30, 31], [1.0, 1.1, 1.3, 2.0], t0)
    assert cb.simulate(df, cand, 0.0, 1.0, 1.0, 30)[0, 1] == pytest.approx(0.3, abs=1e-6)   # a traded minute right at the stop exits there


def test_random_baseline_respects_gates(monkeypatch):
    t0 = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc); frames = []
    for j in range(12):
        f = _bt_frame(list(range(200)), list(1.0 + 0.001 * np.arange(200)), t0 + timedelta(days=j), resq=np.where(np.arange(200) % 2, 50.0, 5.0),
                      broken=np.arange(200) >= 150)
        f["mint"] = f"M{j}"; frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    gate = (df["resq"].to_numpy() >= cb.MIN_RESQ_SOL) & ~df["broken"].to_numpy(); seen = []
    real = cb.simulate
    monkeypatch.setattr(cb, "simulate", lambda d, cand, *a, **k: (seen.append(cand.copy()), real(d, cand, *a, **k))[1])
    monkeypatch.setattr(cb, "STRATEGIES", {"S9 all": lambda d: [("all", np.ones(len(d), bool))]})
    cb.run(df, 0.003, min_trades=1)
    assert len(seen) == len(cb.EXITS) + 5
    assert all(not (c & ~gate).any() for c in seen)          # strategy and random entries only where the gate allows
    assert seen[-1].any()


# ---------------------------------------------------------------- train/corpus_features.py
class _Tx:
    def __init__(self, fail=False, pending=()):
        self.fail, self.pending, self.rows = fail, list(pending), None

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self

    def execute(self, *a):
        return self

    def fetchall(self):
        return self.pending

    def executemany(self, sql, rows):
        if self.fail:
            raise RuntimeError("db down")
        self.rows = rows


def _fake_rows(mint):
    g = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    return [cf._row(mint, g + 60 * k, g, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, False, [0.0] * len(FEATURES), 0, {}) for k in range(3)], 0


def test_corpus_features_failures_and_atomic_commit(tmp_path, monkeypatch):
    g = datetime(2026, 9, 1, tzinfo=timezone.utc)
    todo = [{"mint": m, "graduated_at": g, "candle_path": None, "trade_path": None} for m in ("GOOD", "BAD")]
    monkeypatch.setattr(config, "CORPUS_FEATURES_DIR", tmp_path); monkeypatch.setattr(cf, "_failed", set())
    monkeypatch.setattr(cf, "build_token", lambda m, *a: (_ for _ in ()).throw(ValueError("corrupt")) if m == "BAD" else _fake_rows(m))
    monkeypatch.setattr(cf, "_pending", lambda limit: todo)
    monkeypatch.setattr(cf, "transaction", _Tx(fail=True))
    with pytest.raises(RuntimeError):
        cf.build_batch()
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]      # failed insert: no part left behind, so no duplicate later
    tx = _Tx(); monkeypatch.setattr(cf, "transaction", tx)
    assert cf.build_batch() == 2
    assert [r[0] for r in tx.rows] == ["GOOD"] and "BAD" in cf._failed  # the failure is not registered
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].name.startswith("part-") and files[0].suffix == ".parquet"
    assert pq.read_table(files[0]).num_rows == 3 and tx.rows[0][4] == str(files[0])
    monkeypatch.undo(); monkeypatch.setattr(cf, "_failed", {"BAD"})
    monkeypatch.setattr(cf, "transaction", _Tx(pending=[{"mint": "BAD"}, {"mint": "X"}, {"mint": "Y"}]))
    assert [r["mint"] for r in cf._pending(1)] == ["X"]              # skipped in this process


# ---------------------------------------------------------------- train/selector.py
def _ds(n_days=5, per_day=400, seed=0):
    rng = np.random.default_rng(seed); n = n_days * per_day
    X = rng.normal(size=(n, 3)).astype(np.float32); y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int8)
    day = np.array([date(2026, 9, 1) + timedelta(days=k) for k in range(n_days) for _ in range(per_day)], dtype=object)
    ts = np.array([datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() + 60 * (i % per_day) for i, d in enumerate(day)])
    fwd = np.where(y == 1, 0.05, -0.02).astype(np.float32)
    return DecisionSet(X=X, y=y, fwd=fwd, fwd_pess=fwd, day=day, ts=ts, mint=np.array([f"M{i % 7}" for i in range(n)]), cols=["a", "b", "c"], horizon_s=1800.0)


def test_selector_fold_guards():
    ds = _ds(); days = ds.days
    fo = selector.fold(ds, days[-1], 0.05, min_train=100, min_test=10)
    assert fo is not None and fo.auc is not None and fo.auc > 0.7 and fo.pick.any() and not (fo.pick & ~fo.test).any()
    assert selector.fold(ds, days[0], 0.05, min_train=100, min_test=10) is None                  # empty train
    assert selector.fold(ds, days[-1], 0.05) is None                                                # below the default sizes
    ds.y[ds.day == days[-1]] = 0
    assert selector.fold(ds, days[-1], 0.05, min_train=100, min_test=10).auc is None              # single-class test day: no crash
    from fly_trader.train import selector_eval                                                      # importable: runs only as a script
    assert selector_eval.fold is selector.fold
