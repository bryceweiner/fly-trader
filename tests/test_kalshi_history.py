"""The Kalshi corpus stays bounded: the dataset seed keeps only the most traded markets of each close day above the volume
floor, the exchange refresh applies the same floor, and pending rows beyond the bounds are skipped, never fetched."""
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.kalshi import history as H


def _clean():
    with transaction() as c:
        c.execute("DELETE FROM kalshi_corpus WHERE ticker LIKE 'TSTH-%'"); c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTH-%'")
        c.execute("DELETE FROM ui_settings WHERE key = %s", (H.SEED_KEY,))


def test_seed_keeps_the_most_traded_markets_per_day_above_the_floor(tmp_path, monkeypatch):
    _clean()
    monkeypatch.setattr(config, "KALSHI_DATASET_DIR", str(tmp_path)); monkeypatch.setattr(config, "KALSHI_MIN_MARKET_VOLUME", 100.0); monkeypatch.setattr(config, "KALSHI_MARKETS_PER_DAY", 2)
    monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2025-01-01"); monkeypatch.setattr(H.D, "TRADES_DIR", tmp_path / "trades_out")
    d = tmp_path / "data" / "kalshi"; (d / "markets").mkdir(parents=True); (d / "trades").mkdir()
    day = datetime(2026, 3, 1, 12, 0)
    rows = [("TSTH-A", 5000.0, day), ("TSTH-B", 300.0, day + timedelta(hours=1)), ("TSTH-C", 200.0, day + timedelta(hours=2)),        # day 1: A, B kept; C beyond the cap
            ("TSTH-D", 50.0, day + timedelta(days=1)), ("TSTH-E", 900.0, day + timedelta(days=1, hours=3)),                             # day 2: D below the floor, E kept
            ("TSTH-F", 9999.0, datetime(2024, 6, 1))]                                                                                     # before the start
    pq.write_table(pa.table({"ticker": [r[0] for r in rows], "event_ticker": ["TSTH-EV"] * 6, "market_type": ["binary"] * 6, "title": ["t"] * 6, "yes_sub_title": [""] * 6, "no_sub_title": [""] * 6,
                             "status": ["settled"] * 6, "result": ["yes", "no", "yes", "no", "yes", "no"], "open_time": [r[2] - timedelta(days=2) for r in rows], "close_time": [r[2] for r in rows],
                             "volume": [r[1] for r in rows], "open_interest": [0.0] * 6}), d / "markets" / "markets_0.parquet")
    pq.write_table(pa.table({"trade_id": ["1", "2", "3"], "ticker": ["TSTH-A", "TSTH-A", "TSTH-C"], "count": [5.0, 7.0, 1.0], "yes_price": [40, 42, 50], "no_price": [60, 58, 50],
                             "taker_side": ["yes", "no", "yes"], "created_time": [day - timedelta(hours=5), day - timedelta(hours=4), day - timedelta(hours=1)]}), d / "trades" / "trades_0.parquet")
    out = H.seed_from_dataset()
    assert out == {"markets": 3, "trades": 1}                                       # A, B, E; only A has trades
    with transaction() as c:
        got = {r["ticker"]: dict(r) for r in c.execute("SELECT ticker, status, volume, trade_path, trades FROM kalshi_corpus WHERE ticker LIKE 'TSTH-%'").fetchall()}
    assert set(got) == {"TSTH-A", "TSTH-B", "TSTH-E"} and got["TSTH-A"]["volume"] == 5000.0 and got["TSTH-A"]["trades"] == 2 and got["TSTH-B"]["trade_path"] is None
    assert H.seed_from_dataset() == {"skipped": "already seeded"}
    _clean()


def test_prune_marks_pending_rows_beyond_the_bounds_as_skipped(monkeypatch):
    _clean(); monkeypatch.setattr(config, "KALSHI_MIN_MARKET_VOLUME", 100.0); monkeypatch.setattr(config, "KALSHI_MARKETS_PER_DAY", 1)
    day = datetime(2026, 4, 1, tzinfo=timezone.utc)
    with transaction() as c:
        for tk, vol, close, raw in (("TSTH-P1", 500.0, day, None), ("TSTH-P2", None, day + timedelta(hours=1), '{"volume_fp": "800.00"}'), ("TSTH-P3", None, day + timedelta(hours=2), '{"volume": 20}')):
            c.execute("INSERT INTO kalshi_markets (ticker, status, source, raw) VALUES (%s, 'settled', 'test', %s::jsonb)", (tk, raw))
            c.execute("INSERT INTO kalshi_corpus (ticker, status, close_time, volume) VALUES (%s, 'pending', %s, %s)", (tk, close, vol))
    assert H.prune_pending() == 2
    with transaction() as c:
        got = {r["ticker"]: (r["status"], r["volume"]) for r in c.execute("SELECT ticker, status, volume FROM kalshi_corpus WHERE ticker LIKE 'TSTH-P%'").fetchall()}
    assert got["TSTH-P2"] == ("pending", 800.0)                     # volume read from the raw JSON; the day's most traded stays
    assert got["TSTH-P1"][0] == "skipped" and got["TSTH-P3"] == ("skipped", 20.0)
    assert H.market_volume({"volume_fp": "12.50"}) == 12.5 and H.market_volume({"volume": 3}) == 3.0 and H.market_volume({}) is None
    _clean()
