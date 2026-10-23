"""Directly created (custom) PumpSwap pools are excluded from the universe: live stream, training aggregates, archive backfill."""
from datetime import date, datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from fly_trader import config
from fly_trader.ingest import pumpstream
from fly_trader.train import corpus_meta, mature

WSOL = pumpstream.WSOL


def _leg(mint, ts_ms, price, pool_id, action="buy"):
    return {"action": action, "pool": "pump-amm", "mint": mint, "timestamp": ts_ms, "price": price, "quoteMint": WSOL, "quoteInPool": 80.0,
            "txSigner": "t", "breakdown": [{"action": action, "trader": "t", "quoteAmount": 1.0, "tokenAmount": 1.0}], "poolId": pool_id}


def _create_pool(mint, ts_ms, pool_id, by="custom"):
    e = {"action": "createPool", "pool": "pump-amm", "mint": mint, "timestamp": ts_ms, "poolId": pool_id, "quoteMint": WSOL}
    if by is not None:
        e["poolCreatedBy"] = by
    return e


def test_stream_drops_custom_pool_legs(db_conn):
    agg = pumpstream.Aggregator(); m = 1_800_000_000
    agg.ingest(_leg("Apump", (m + 1) * 1000, 1.0, "MIGRATED"))                      # a migrated pool (no createPool here): kept
    agg.ingest(_create_pool("Bpump", (m + 2) * 1000, "CUSTOM_T1"))
    agg.ingest(_create_pool("Cpump", (m + 2) * 1000, "CUSTOM_T2", by=None))       # missing poolCreatedBy counts as custom
    agg.ingest(_create_pool("Dpump", (m + 2) * 1000, "MIGR_T3", by="pump"))       # created by a pump.fun migration: not blocked
    for pid, mint in (("CUSTOM_T1", "Bpump"), ("CUSTOM_T2", "Cpump"), ("MIGR_T3", "Dpump"), ("MIGRATED", "Apump")):
        agg.ingest(_leg(mint, (m + 5) * 1000, 1.0, pid)); agg.ingest(_leg(mint, (m + 6) * 1000, 1.0, pid, "sell"))
    assert agg.blocked == {"CUSTOM_T1", "CUSTOM_T2"} and agg.stats["dropped_custom_pool"] == 4
    assert set(agg.minutes[m]) == {"Apump", "Dpump"} and agg.row(m, "Apump").nb == 2
    try:
        assert agg.write_pools() == 2 and agg.pending_pools == []
        rows = db_conn.execute("SELECT pool_id, mint, created_by, source, ts FROM pump_pools WHERE pool_id LIKE 'CUSTOM_T%' ORDER BY 1").fetchall()
        assert [(r["pool_id"], r["mint"], r["created_by"], r["source"]) for r in rows] == [("CUSTOM_T1", "Bpump", "custom", "stream"), ("CUSTOM_T2", "Cpump", None, "stream")]
        assert rows[0]["ts"] == datetime.fromtimestamp(m + 2, timezone.utc)
    finally:
        db_conn.execute("DELETE FROM pump_pools WHERE pool_id LIKE 'CUSTOM_T%'"); db_conn.commit()


def test_stream_loads_blocked_set(db_conn):
    db_conn.execute("INSERT INTO pump_pools (pool_id, mint, created_by, source) VALUES ('LOAD_T1', 'Lpump', 'custom', 'archive'), ('LOAD_T2', 'Lother', 'custom', 'archive') "
                    "ON CONFLICT DO NOTHING"); db_conn.commit()
    try:
        agg = pumpstream.Aggregator(); pumpstream._load_blocked(agg)
        assert "LOAD_T1" in agg.blocked and "LOAD_T2" not in agg.blocked     # a mint that is not pump.fun's never reaches the pool check
        agg.ingest(_leg("Lpump", 1_800_000_001_000, 1.0, "LOAD_T1"))
        assert agg.stats["dropped_custom_pool"] == 1 and not agg.minutes
    finally:
        db_conn.execute("DELETE FROM pump_pools WHERE pool_id LIKE 'LOAD_T%'"); db_conn.commit()


def test_aggregate_day_excludes_custom_pools(db_conn, tmp_path, monkeypatch):
    d = date(2031, 1, 1); t0 = datetime(2031, 1, 1, tzinfo=timezone.utc).timestamp()
    rows = []
    for k in range(3):                  # GOOD: minutes 0..2; AGG_BAD (custom) minutes 5..7 for the same mint, and a mint only in AGG_BAD
        rows.append(("Xpump", t0 + 60 * k + 1, "GOOD", 1.0)); rows.append(("Xpump", t0 + 60 * (5 + k) + 1, "AGG_BAD", 1.5)); rows.append(("Ypump", t0 + 60 * k + 1, "AGG_BAD", 2.0))
    tab = pa.table({"ts": pa.array([datetime.fromtimestamp(r[1], timezone.utc) for r in rows], pa.timestamp("ms", tz="UTC")), "slot": pa.array(range(len(rows)), pa.int64()),
                    "pool": ["pump-amm"] * len(rows), "mint": [r[0] for r in rows], "trader": ["t"] * len(rows), "side": pa.array([1] * len(rows), pa.int8()),
                    "sol": [1.0] * len(rows), "price": [r[3] for r in rows], "quote_in_pool": [80.0] * len(rows), "quote_mint": [WSOL] * len(rows),
                    "pool_id": [r[2] for r in rows]})
    (tmp_path / "replay" / d.isoformat()).mkdir(parents=True)
    pq.write_table(tab, tmp_path / "replay" / d.isoformat() / "00_trades.parquet")
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    db_conn.execute("INSERT INTO pump_pools (pool_id, mint, created_by, source) VALUES ('AGG_BAD', 'Ypump', 'custom', 'archive') ON CONFLICT DO NOTHING"); db_conn.commit()
    try:
        assert mature.aggregate_day(d) == 3
        out = pq.read_table(tmp_path / "mature" / f"{d.isoformat()}.parquet").to_pandas()
        assert set(out["mint"]) == {"Xpump"} and (out["close"] == 1.0).all() and len(out) == 3
        assert mature.part_version(tmp_path / "mature" / f"{d.isoformat()}.parquet") == mature.AGG_VERSION == 5
    finally:
        db_conn.execute("DELETE FROM pump_pools WHERE pool_id = 'AGG_BAD'"); db_conn.execute("DELETE FROM mature_days WHERE day = %s", (d,)); db_conn.commit()


def test_archive_backfill_idempotent(db_conn, tmp_path, monkeypatch):
    day = tmp_path / "replay" / "2031-01-02"; day.mkdir(parents=True)
    t = lambda s: datetime(2031, 1, 2, 0, 0, s, tzinfo=timezone.utc)
    ev = pa.table({"ts": pa.array([t(5), t(1), t(2), t(3), t(4)], pa.timestamp("ms", tz="UTC")), "sig": ["s1", "s2", "s3", "s4", "s5"],
                   "action": ["createPool", "createPool", "createPool", "migrate", "createPool"], "pool": ["pump-amm", "pump-amm", "pump-amm", "pump-amm", "raydium-cpmm"],
                   "mint": ["Apump", "Apump", "Bpump", "Cpump", "Dpump"], "pool_id": ["BF_C1", "BF_C1", "BF_C2", "BF_M1", "BF_R1"],
                   "pool_created_by": ["custom", "custom", None, "pump", "custom"]})
    pq.write_table(ev, day / "00_events.parquet")
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay")
    db_conn.execute("DELETE FROM ui_settings WHERE key = 'pump_pools_backfill'"); db_conn.execute("DELETE FROM pump_pools WHERE pool_id LIKE 'BF_%'"); db_conn.commit()
    try:
        assert corpus_meta.backfill_pump_pools() == 2
        assert corpus_meta.backfill_pump_pools() == 0                         # re-run: nothing new
        rows = db_conn.execute("SELECT pool_id, mint, created_by, ts, source FROM pump_pools WHERE pool_id LIKE 'BF_%' ORDER BY 1").fetchall()
        assert [(r["pool_id"], r["mint"], r["created_by"], r["source"]) for r in rows] == [("BF_C1", "Apump", "custom", "archive"), ("BF_C2", "Bpump", None, "archive")]
        assert rows[0]["ts"] == t(1)                                             # the pool's first createPool
        assert "BF_C1" in corpus_meta.blocked_pool_ids(db_conn)
    finally:
        db_conn.execute("DELETE FROM ui_settings WHERE key = 'pump_pools_backfill'"); db_conn.execute("DELETE FROM pump_pools WHERE pool_id LIKE 'BF_%'"); db_conn.commit()


def test_stream_keeps_pumpfun_graduates_without_the_suffix(monkeypatch):
    """A pump.fun graduate whose mint lacks the "pump" suffix is a universe coin once its migration is seen; any other
    mint without the suffix is not."""
    agg = pumpstream.Aggregator(); m = 1_800_000_000
    monkeypatch.setattr(agg, "write_creates", lambda: 0); monkeypatch.setattr(agg, "_graduation", lambda e: None)
    agg.ingest({"action": "migrate", "pool": "pump-amm", "mint": "GradNoSuffix", "timestamp": (m + 1) * 1000, "poolId": "GRAD_POOL", "signature": "g1"})
    assert agg.pumpfun("GradNoSuffix") and agg.pumpfun("Apump") and not agg.pumpfun("OtherNoSuffix")
    for mint, pid in (("GradNoSuffix", "GRAD_POOL"), ("OtherNoSuffix", "OTHER_POOL")):
        agg.ingest(_leg(mint, (m + 5) * 1000, 1.0, pid))
    assert set(agg.minutes[m]) == {"GradNoSuffix"}


def test_stream_loads_graduates_without_the_suffix(db_conn):
    db_conn.execute("INSERT INTO corpus_meta (mint, graduated_at) VALUES ('GradLoadNoSuffix', now()) ON CONFLICT DO NOTHING")
    db_conn.execute("INSERT INTO pump_pools (pool_id, mint, created_by, source) VALUES ('LOAD_G1', 'GradLoadNoSuffix', 'custom', 'archive') ON CONFLICT DO NOTHING")
    db_conn.commit()
    try:
        agg = pumpstream.Aggregator(); pumpstream._load_blocked(agg)
        assert "GradLoadNoSuffix" in agg.grads and "LOAD_G1" in agg.blocked    # its custom pools are blocked like any universe coin's
        agg.ingest(_leg("GradLoadNoSuffix", 1_800_000_001_000, 1.0, "LOAD_G1"))
        assert agg.stats["dropped_custom_pool"] == 1 and not agg.minutes
    finally:
        db_conn.execute("DELETE FROM pump_pools WHERE pool_id = 'LOAD_G1'"); db_conn.execute("DELETE FROM corpus_meta WHERE mint = 'GradLoadNoSuffix'")
        db_conn.commit()


def test_aggregate_day_includes_pumpfun_graduates_without_the_suffix(db_conn, tmp_path, monkeypatch):
    d = date(2031, 1, 3); t0 = datetime(2031, 1, 3, tzinfo=timezone.utc).timestamp()
    rows = [(mint, t0 + 60 * k + 1, pid) for k in range(3) for mint, pid in (("Xpump", "P_X"), ("GradAggNoSuffix", "P_G"), ("OtherAggNoSuffix", "P_O"))]
    tab = pa.table({"ts": pa.array([datetime.fromtimestamp(r[1], timezone.utc) for r in rows], pa.timestamp("ms", tz="UTC")), "slot": pa.array(range(len(rows)), pa.int64()),
                    "pool": ["pump-amm"] * len(rows), "mint": [r[0] for r in rows], "trader": ["t"] * len(rows), "side": pa.array([1] * len(rows), pa.int8()),
                    "sol": [1.0] * len(rows), "price": [1.0] * len(rows), "quote_in_pool": [80.0] * len(rows), "quote_mint": [WSOL] * len(rows),
                    "pool_id": [r[2] for r in rows]})
    (tmp_path / "replay" / d.isoformat()).mkdir(parents=True)
    pq.write_table(tab, tmp_path / "replay" / d.isoformat() / "00_trades.parquet")
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    db_conn.execute("INSERT INTO corpus_meta (mint, graduated_at) VALUES ('GradAggNoSuffix', now()) ON CONFLICT DO NOTHING"); db_conn.commit()
    try:
        assert mature.aggregate_day(d) == 6
        out = pq.read_table(tmp_path / "mature" / f"{d.isoformat()}.parquet").to_pandas()
        assert set(out["mint"]) == {"Xpump", "GradAggNoSuffix"}               # a mint pump.fun never graduated stays out
    finally:
        db_conn.execute("DELETE FROM corpus_meta WHERE mint = 'GradAggNoSuffix'"); db_conn.execute("DELETE FROM mature_days WHERE day = %s", (d,)); db_conn.commit()


def test_corpus_versions_never_retire_a_model():
    """The corpus build versions say when the files are rebuilt, not what a model means: a selector, fly or replay verdict
    that still carries them (every one trained before 2026-10-23 does) stays current, so a rebuild retires nothing; a
    changed, missing or extra definition still does (fly #251's ``distil`` must not make it the live fly)."""
    from fly_trader.train import fly_replay, fly_selector, selector
    assert "agg" not in selector.DATA_VERSION and "rh_agg" not in selector.DATA_VERSION
    old = {**selector.DATA_VERSION, "agg": 4, "rh_agg": 1}
    assert selector.is_current({"data": old}) and not selector.is_current({"data": {**old, "features": -1}})
    assert fly_selector.is_current({"data": {**fly_selector.FLY_VERSION, "agg": 4, "rh_agg": 1}})
    assert not fly_selector.is_current({"data": {k: v for k, v in fly_selector.FLY_VERSION.items() if k != "fly"}})
    assert not fly_selector.is_current({"data": {**fly_selector.FLY_VERSION, "agg": 4, "distil": "x"}})     # any other extra key still retires a fly
    row = {"value": {"data": {**fly_selector.verdict_version(), "agg": 4, "rh_agg": 1}, "chains": {"rh": {"passed": True}}}}
    merged = fly_replay.merge_chains(row, {"data": fly_selector.verdict_version(), "chains": {"sol": {"passed": True}}, "S": "x"}, ("sol",))
    assert set(merged["chains"]) == {"rh", "sol"} and merged["passed"]       # the stored verdict's other chain is kept
