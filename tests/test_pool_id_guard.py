"""The pool_id parity guard: an archive day with an hour file lacking the column is never aggregated (the live stream's
blocked-pool filter and dominant-pool choice need it), build_complete says why, and refetch marks those hours."""
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ingest import replay_pull
from fly_trader.train import mature


def _hour(path, with_pool_id: bool):
    cols = {"ts": pa.array([datetime(2026, 9, 1, 0, 0, 5, tzinfo=timezone.utc)], pa.timestamp("ms", tz="UTC")), "slot": [1], "pool": ["pump-amm"],
            "mint": ["Apump"], "trader": ["W"], "side": pa.array([1], pa.int8()), "sol": [1.0], "tokens": [10.0], "price": [0.1], "quote_in_pool": [50.0]}
    if with_pool_id:
        cols["pool_id"] = ["P1"]
    path.parent.mkdir(parents=True, exist_ok=True); pq.write_table(pa.table(cols), path)


def test_day_without_pool_id_is_not_aggregated_and_hours_are_marked(tmp_path, monkeypatch, db_conn):
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    _hour(tmp_path / "replay" / "2026-09-01" / "00_trades.parquet", True)
    _hour(tmp_path / "replay" / "2026-09-01" / "01_trades.parquet", False)
    _hour(tmp_path / "replay" / "2026-09-02" / "00_trades.parquet", True)
    assert mature.days_missing_pool_id() == ["2026-09-01"]
    assert mature.aggregate_day(datetime(2026, 9, 1).date()) == 0 and not (tmp_path / "mature" / "2026-09-01.parquet").exists()
    assert replay_pull.refetch_missing_pool_id() == 1
    with transaction() as conn:
        r = conn.execute("SELECT status FROM replay_hours WHERE hour = %s", (datetime(2026, 9, 1, 1, tzinfo=timezone.utc),)).fetchone()
        conn.execute("DELETE FROM replay_hours WHERE hour = %s", (datetime(2026, 9, 1, 1, tzinfo=timezone.utc),))
    assert r["status"] == "refetch"
