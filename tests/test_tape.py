"""swap_tape round trips against fly_trader_test (conftest forces DATABASE_URL; db_conn rolls back)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from fly_trader.db import schema
from fly_trader.ingest import swap_decoder as sd
from fly_trader.ingest import tape
from fly_trader.ingest.tape import COLUMNS, TapeRow

FX = Path(__file__).parent / "fixtures"
POOL = "TapeTestPool11111111111111111111111111111111"
MINT = "TapeTestMint11111111111111111111111111111111"


def _watch_pool(conn, pool=POOL, mint=MINT, quote_mint=sd.WSOL_MINT, base_decimals=6, quote_decimals=9):
    conn.execute(
        "INSERT INTO watch_pools (pool, mint, quote_mint, program_label, base_vault, quote_vault, base_decimals, "
        "quote_decimals, source, active) VALUES (%s, %s, %s, 'Pump.fun Amm', 'bv', 'qv', %s, %s, 'test', true) "
        "ON CONFLICT (pool) DO UPDATE SET base_decimals = EXCLUDED.base_decimals, quote_decimals = EXCLUDED.quote_decimals",
        (pool, mint, quote_mint, base_decimals, quote_decimals))


def _row(i: int, ts: datetime, side: int, **kw) -> TapeRow:
    base = dict(ts=ts, slot=446_000_000 + i, sig=f"sig{i}", tx_index=i, pool=POOL, mint=MINT, side=side,
                amount_base=1_000_000 * (i + 1), amount_quote=250_000_000 * (i + 1), price_sol=0.00025,
                signer=f"signer{i}", res_base=10**19 + i, res_quote=5_000_000_000_000, program_label="Pump.fun Amm",
                price_quote=0.00025, quote_mint=sd.WSOL_MINT)
    base.update(kw)
    return TapeRow(**base)


def test_insert_and_tail_roundtrip(db_conn):
    _watch_pool(db_conn)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = tape.latest_id(db_conn)
    rows = [_row(0, now, 1), _row(1, now + timedelta(seconds=1), -1),
            _row(2, now + timedelta(seconds=2), 0, amount_quote=0, price_sol=None, price_quote=None,
                 amount_base=10**19 + 7)]  # > int64: exercises numeric(30,0)
    assert tape.insert_rows(db_conn, rows, commit=False) == 3

    got = tape.tail(db_conn, start)
    assert [g["sig"] for g in got] == ["sig0", "sig1", "sig2"]
    assert got[0]["id"] < got[1]["id"] < got[2]["id"]
    for g, r in zip(got, rows):
        assert set(COLUMNS) <= set(g) and "id" in g
        assert g["ts"] == r.ts and g["slot"] == r.slot and g["tx_index"] == r.tx_index
        assert (g["pool"], g["mint"], g["side"], g["signer"], g["program_label"]) == (POOL, MINT, r.side, r.signer, r.program_label)
        assert int(g["amount_base"]) == r.amount_base and g["amount_quote"] == r.amount_quote
        assert int(g["res_base"]) == r.res_base and g["res_quote"] == r.res_quote
        assert g["price_sol"] == r.price_sol
        # runner-facing extras via the watch_pools join
        assert g["quote_mint"] == sd.WSOL_MINT and g["quote_decimals"] == 9 and g["base_decimals"] == 6
    assert isinstance(got[2]["amount_base"], Decimal) and int(got[2]["amount_base"]) == 10**19 + 7
    assert got[0]["price_quote"] == pytest.approx((250_000_000 / 1e9) / (1_000_000 / 1e6))
    assert got[2]["price_quote"] is None  # lp row with a zero quote leg

    assert tape.latest_id(db_conn) == got[-1]["id"]
    assert tape.last_swap_ts(db_conn) == rows[-1].ts
    assert [g["sig"] for g in tape.tail(db_conn, start, limit=2)] == ["sig0", "sig1"]
    assert [g["sig"] for g in tape.tail(db_conn, got[1]["id"])] == ["sig2"]
    assert tape.tail(db_conn, got[-1]["id"]) == []


def test_decoded_fixture_rows_round_trip(db_conn):
    manifest = json.loads((FX / "manifest.json").read_text())
    p = manifest["pools"]["pumpswap_cate"]
    pv = sd.PoolVaults(pool=p["pool"], mint=p["mint"], quote_mint=p["quote_mint"], base_vault=p["base_vault"],
                       quote_vault=p["quote_vault"], base_decimals=p["base_decimals"],
                       quote_decimals=p["quote_decimals"], program_label=p["program_label"])
    _watch_pool(db_conn, pool=p["pool"], mint=p["mint"])
    rows = sd.decode_transaction(json.loads((FX / "pumpswap_buy.json").read_text()), {p["pool"]: pv})
    assert len(rows) == 1
    start = tape.latest_id(db_conn)
    # the fixture's blockTime hour may have no partition in the test database: insert_rows creates it
    tape.insert_rows(db_conn, rows, commit=False)
    got = tape.tail(db_conn, start)
    assert len(got) == 1 and got[0]["side"] == 1 and got[0]["pool"] == p["pool"]
    assert got[0]["price_sol"] == pytest.approx(rows[0].price_sol)
    assert got[0]["price_quote"] == pytest.approx(rows[0].price_quote, rel=1e-9)
    assert got[0]["quote_decimals"] == 9


def test_insert_creates_missing_hourly_partition(db_conn):
    ts = (datetime.now(timezone.utc) + timedelta(hours=30)).replace(minute=0, second=0, microsecond=0)
    name = schema._partition_name("swap_tape", ts, "hour")
    exists = lambda: db_conn.execute("SELECT 1 FROM pg_class WHERE relname = %s", (name,)).fetchone() is not None
    assert not exists()
    tape.forget_partitions()
    start = tape.latest_id(db_conn)
    assert tape.insert_rows(db_conn, [_row(9, ts, 1)], commit=False) == 1
    assert exists()
    assert [g["sig"] for g in tape.tail(db_conn, start)] == ["sig9"]


def test_tail_beyond_end_is_empty(db_conn):
    assert tape.tail(db_conn, 10**15) == []


def test_partition_listing_helpers(db_conn):
    attached = tape.list_partitions(db_conn, "swap_tape")
    assert attached and all(a.startswith("swap_tape_") for a in attached)
    assert all(tape._partition_start(a, "swap_tape", "hour") is not None for a in attached)


def test_archive_detaches_and_never_drops_then_purge_requires_confirm(tmp_path):
    """archive_partitions exports an old swap_tape partition to Parquet and DETACHES it (row counts must
    match); purge_archived drops the detached table only with confirm=True. Uses its own committed
    connections to the test database (partition DDL must be visible to the archive connection)."""
    from fly_trader.db.connection import connect
    from tests.conftest import TEST_URL
    old = (datetime.now(timezone.utc) - timedelta(hours=80)).replace(minute=0, second=0, microsecond=0)
    name = schema._partition_name("swap_tape", old, "hour")
    with connect(TEST_URL) as conn:
        conn.execute("DELETE FROM swap_tape WHERE pool = %s", (POOL,))
        conn.commit()
        _watch_pool(conn)
        tape.forget_partitions()
        tape.insert_rows(conn, [_row(1, old + timedelta(minutes=5), 1), _row(2, old + timedelta(minutes=6), -1)])
        assert name in tape.list_partitions(conn, "swap_tape")
    actions = tape.archive_partitions(url=TEST_URL, archive_dir=tmp_path)
    mine = [a for a in actions if a["partition"] == name]
    assert len(mine) == 1 and mine[0]["detached"] and mine[0]["rows"] == 2 and mine[0]["exported"]
    parquet = tmp_path / "swap_tape" / f"{name}.parquet"
    assert parquet.exists()
    import pyarrow.parquet as pq
    table = pq.read_table(parquet)
    assert table.num_rows == 2 and set(table.column("sig").to_pylist()) == {"sig1", "sig2"}
    with connect(TEST_URL) as conn:
        assert name not in tape.list_partitions(conn, "swap_tape")
        assert name in tape.list_detached(conn, "swap_tape")           # detached, still present
        assert conn.execute(f"SELECT count(*) AS n FROM {name}").fetchone()["n"] == 2
    # a second archive pass is a no-op for it (already detached) and a dry-run purge keeps it
    assert not [a for a in tape.archive_partitions(url=TEST_URL, archive_dir=tmp_path) if a["partition"] == name]
    dry = [a for a in tape.purge_archived(False, url=TEST_URL, archive_dir=tmp_path) if a["partition"] == name]
    assert dry and dry[0]["safe"] and not dry[0]["dropped"]
    with connect(TEST_URL) as conn:
        assert name in tape.list_detached(conn, "swap_tape")
    wet = [a for a in tape.purge_archived(True, url=TEST_URL, archive_dir=tmp_path) if a["partition"] == name]
    assert wet and wet[0]["dropped"]
    with connect(TEST_URL) as conn:
        assert name not in tape.list_detached(conn, "swap_tape")
        assert conn.execute("SELECT 1 FROM pg_class WHERE relname = %s", (name,)).fetchone() is None
