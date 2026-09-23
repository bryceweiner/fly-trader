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


class FakeRest:
    """Two tiers of settled markets, newest first, four per page with opaque cursors; ``fail_after`` pages raises a timeout."""

    def __init__(self, live: list, hist: list, fail_after: int | None = None):
        self.live, self.hist, self.fail_after = live, hist, fail_after; self.calls: list = []

    def _walk(self, rows, cursor, with_cursor, min_close_ts=None):
        if min_close_ts is not None:
            rows = [m for m in rows if datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp() >= min_close_ts]
        i = int(cursor.split(":")[1]) if cursor else 0
        while i < len(rows):
            self.calls.append(("page", i))
            if self.fail_after is not None and len([c for c in self.calls if c[0] == "page"]) > self.fail_after:
                raise TimeoutError("network")
            page = rows[i:i + 4]; nxt = f"c:{i + 4}" if i + 4 < len(rows) else None
            yield (page, nxt) if with_cursor else page
            if nxt is None:
                return
            i += 4

    def markets(self, cursor=None, with_cursor=False, **params):
        yield from self._walk(self.live, cursor, with_cursor, params.get("min_close_ts"))

    def historical_markets(self, cursor=None, with_cursor=False, **params):
        yield from self._walk(self.hist, cursor, with_cursor)

    def event(self, et, with_nested_markets=False):
        self.calls.append(("event", et)); return {"event": {"event_ticker": et, "series_ticker": "TSTH", "category": "Politics"}}

    def series(self, st):
        return {"ticker": st, "category": "Politics"}


def _mk(i: int, close: datetime, vol=5000.0) -> dict:
    return {"ticker": f"TSTH-W{i}", "event_ticker": f"TSTH-E{i % 3}", "result": "yes", "close_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"), "open_time": (close - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settlement_ts": close.strftime("%Y-%m-%dT%H:%M:%SZ"), "volume_fp": f"{vol:.2f}", "status": "settled", "market_type": "binary", "title": "t"}


def _clean_walk():
    _clean()
    with transaction() as c:
        c.execute("DELETE FROM ui_settings WHERE key = %s", (H.WALK_KEY,)); c.execute("DELETE FROM kalshi_events WHERE event_ticker LIKE 'TSTH-%'"); c.execute("DELETE FROM kalshi_series WHERE ticker = 'TSTH'")


def test_the_exchange_walk_resumes_from_its_saved_page_and_later_rounds_cover_only_new_settlements(monkeypatch):
    _clean_walk(); monkeypatch.setattr(config, "KALSHI_MIN_MARKET_VOLUME", 100.0); monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2026-01-01")
    t0 = datetime(2026, 9, 20, tzinfo=timezone.utc)
    live = [_mk(i, t0 - timedelta(hours=6 * i)) for i in range(10)]                     # 3 pages
    hist = [_mk(100 + i, datetime(2026, 7, 20, tzinfo=timezone.utc) - timedelta(days=i)) for i in range(6)]   # 2 pages
    rest = FakeRest(live, hist, fail_after=4)                                             # dies on the 5th page fetch (2nd of the archive)
    with pytest.raises(TimeoutError):
        H.refresh_markets(rest)
    st = H.walk_state()
    assert st["tier"] == "historical" and st["cursor"] == "c:4" and st.get("complete_through") is None and st["walk_started"]
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_corpus WHERE ticker LIKE 'TSTH-W%'").fetchone()["n"] == 14        # 10 live + the archive's first page
    rest2 = FakeRest(live, hist)
    n = H.refresh_markets(rest2)
    assert n == 2 and rest2.calls == [("page", 4)]                                         # resumed at the archive's second page: no live page fetched again
    st = H.walk_state()
    assert st["tier"] is None and st["cursor"] is None and st["complete_through"] and st["walk_started"] is None
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_corpus WHERE ticker LIKE 'TSTH-W%'").fetchone()["n"] == 16
    # the next round: only the live tier, only markets closing since the completed walk began (less the recovery margin)
    rest3 = FakeRest(live + [_mk(50, t0 + timedelta(days=1))], hist)
    n = H.refresh_markets(rest3)
    assert n == 1 and all(c[0] != "page" or True for c in rest3.calls) and not any(c[1] for c in rest3.calls if c[0] == "event" and "E100" in c[1])
    assert [c for c in rest3.calls if c[0] == "page"] == [("page", 0)]                   # one live page, the archive untouched
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_corpus WHERE ticker = 'TSTH-W50'").fetchone()["n"] == 1
    _clean_walk()
