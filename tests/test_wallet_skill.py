"""Wallet skill: markouts per wallet and hold are the net return of holding (buys) or the move stepped aside from
(sells); the table for day D only uses wallet days up to D − GAP_DAYS (no look-ahead); scores are cut into deciles."""
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader import config
from fly_trader.train import mature, wallet_skill

D = date(2026, 9, 5)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    monkeypatch.setattr(mature, "MATURE_FEAT_DIR", tmp_path / "feat")
    t0 = datetime(D.year, D.month, D.day, 12, 0, tzinfo=timezone.utc)
    (tmp_path / "replay" / D.isoformat()).mkdir(parents=True)
    pq.write_table(pa.table({"ts": pa.array([t0 + timedelta(seconds=10), t0 + timedelta(seconds=20)], pa.timestamp("ms", tz="UTC")), "pool": ["pump-amm"] * 2,
                             "mint": ["Spump"] * 2, "trader": ["BUYER", "SELLER"], "side": pa.array([1, -1], pa.int8()), "sol": [2.0, 1.0], "price": [1.0, 1.0]}),
                   tmp_path / "replay" / D.isoformat() / "12_trades.parquet")
    for d in (D, D + timedelta(days=1)):
        mins = [datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=k) for k in range(0, 300)]
        close = [1.0 if k < 10 else 1.2 for k in range(300)]
        mature.write_part(pa.table({"mint": ["Spump"] * 300, "ts": pa.array(mins, pa.timestamp("us", tz="UTC")), "close": close}), tmp_path / "mature" / f"{d}.parquet", mature.AGG_VERSION)
        mature.write_part(pa.table({"mint": ["Spump"] * 300, "ts": pa.array(mins, pa.timestamp("us", tz="UTC")), "exit_cost_0p1": [0.01] * 300}),
                          tmp_path / "feat" / d.isoformat() / "part.parquet", 3)


def test_markouts_and_no_lookahead(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert wallet_skill.build_wallet_day(D, out_dir=tmp_path / "wd") > 0
    t = pq.read_table(tmp_path / "wd" / f"{D}.parquet").to_pandas().set_index(["wallet", "h"])
    assert t.loc[("BUYER", 30), "sol_m"] == pytest.approx(2.0 * (1.2 * 0.99 * 0.99 - 1))     # held past the +20 % move, costs both sides
    assert t.loc[("BUYER", 10), "sol_m"] == pytest.approx(2.0 * (1.2 * 0.99 * 0.99 - 1))
    assert t.loc[("SELLER", 30), "sol_m"] == pytest.approx(-1.0 * 0.2)                        # sold before a +20 % move
    assert wallet_skill.window_days(D + timedelta(days=wallet_skill.GAP_DAYS), 3) == [D - timedelta(days=2), D - timedelta(days=1), D]
    assert all(x <= D for x in wallet_skill.window_days(D + timedelta(days=wallet_skill.GAP_DAYS), 30))
    assert wallet_skill.skill_table(D + timedelta(days=wallet_skill.GAP_DAYS - 1), 30, 1, day_dir=tmp_path / "wd") is None   # day D is not yet usable
    tab = wallet_skill.skill_table(D + timedelta(days=wallet_skill.GAP_DAYS), 30, 1, day_dir=tmp_path / "wd").to_pandas().set_index("wallet")
    assert tab.loc["BUYER", "bucket"] > tab.loc["SELLER", "bucket"]


def test_decile_buckets_cover_0_to_9():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0); n = 1000
    wd = pd.DataFrame({"wallet": [f"w{i}" for i in range(n)], "h": 60, "n": 1, "sol": rng.uniform(0.1, 5, n), "sol_m": rng.normal(0, 1, n)})
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        pq.write_table(pa.Table.from_pandas(wd, preserve_index=False), pathlib.Path(td) / f"{D}.parquet")
        tab = wallet_skill.skill_table(D + timedelta(days=wallet_skill.GAP_DAYS), 60, 1, day_dir=pathlib.Path(td)).to_pandas()
    counts = tab["bucket"].value_counts()
    assert sorted(counts.index) == list(range(10)) and counts.min() == counts.max() == n // 10


def test_daily_writes_each_table_once_its_window_exists(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(wallet_skill, "DAY_DIR", tmp_path / "wd"); monkeypatch.setattr(mature, "SKILL_DIR", tmp_path / "skill")
    monkeypatch.setattr(wallet_skill, "CONFIG_FILE", tmp_path / "skill" / "config.json")
    assert wallet_skill.daily() == 1 and not list((tmp_path / "skill").glob("*.parquet"))       # no (h, L) frozen: the wallet day only
    wallet_skill.freeze(30, 1)
    f = tmp_path / "skill" / f"{D + timedelta(days=wallet_skill.GAP_DAYS)}.parquet"
    assert wallet_skill.daily() == 1 and [p.name for p in (tmp_path / "skill").glob("*.parquet")] == [f.name]   # only the day whose window exists
    assert wallet_skill.table_version(f) == "h30-L1" and wallet_skill.daily() == 0
    wallet_skill.freeze(10, 1)
    assert wallet_skill.daily() == 1 and wallet_skill.table_version(f) == "h10-L1"                 # a new (h, L) rewrites it
