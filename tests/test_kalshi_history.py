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
    _clean_walk(); monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2026-01-01")
    monkeypatch.setattr(H, "sample_rates", lambda conn=None: {"rates": {}})                # every event sampled: this test is about the walk
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
    rest3 = FakeRest([_mk(50, datetime.now(timezone.utc).replace(microsecond=0))] + live, hist)     # settled after the walk began: newest first
    n = H.refresh_markets(rest3)
    assert n == 1
    assert [c for c in rest3.calls if c[0] == "page"] == [("page", 0)]                   # one live page, the archive untouched
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_corpus WHERE ticker = 'TSTH-W50'").fetchone()["n"] == 1
    _clean_walk()


def test_parallel_fill_finishes_markets_and_leaves_network_failures_pending(tmp_path, monkeypatch):
    import threading, time as _t
    _clean(); monkeypatch.setattr(H.D, "CANDLES_DIR", tmp_path / "c"); monkeypatch.setattr(H.D, "TRADES_DIR", tmp_path / "t"); monkeypatch.setattr(H, "TRANSIENT_WAIT_S", 0.0)
    close = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with transaction() as c:
        for i in range(12):
            c.execute("INSERT INTO kalshi_corpus (ticker, status, close_time, open_time, settled_ts) VALUES (%s, 'pending', %s, %s, %s)",
                      (f"TSTH-F{i}", close - timedelta(minutes=i), close - timedelta(days=2), close))
    live = {"n": 0, "max": 0}; lock = threading.Lock()

    def candles(rest, tk, series, o, cl, st):
        with lock:
            live["n"] += 1; live["max"] = max(live["max"], live["n"])
        _t.sleep(0.05)
        with lock:
            live["n"] -= 1
        if tk == "TSTH-F3":
            raise TimeoutError("network")
        return [{"end_ts": 60, **{k: 50.0 for k in ("yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close", "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
                                                    "price_open", "price_high", "price_low", "price_close", "price_mean")}, "volume": 1.0, "open_interest": 1.0}]
    monkeypatch.setattr(H, "pull_candles", candles); monkeypatch.setattr(H, "pull_trades", lambda *a: [])
    n = H.fill(object(), threads=4, limit=50)
    assert n == 11 and live["max"] > 1                                     # markets overlapped
    with transaction() as c:
        st = {r["ticker"]: (r["status"], r["last_error"]) for r in c.execute("SELECT ticker, status, last_error FROM kalshi_corpus WHERE ticker LIKE 'TSTH-F%'").fetchall()}
    assert st["TSTH-F3"][0] == "pending" and st["TSTH-F3"][1].startswith("retrying") and sum(1 for s, _ in st.values() if s == "done") == 11
    monkeypatch.setattr(H, "pull_candles", lambda *a: (_ for _ in ()).throw(TimeoutError("down")))
    assert H.fill(object(), threads=4, limit=50) == -1                     # everything left failed on the network
    _clean()


def test_markets_seeded_without_a_catalogue_get_their_events_and_series():
    _clean_walk()
    with transaction() as c:
        for tk, et in (("TSTH-C1", "TSTH-EVA"), ("TSTH-C2", "TSTH-EVA"), ("TSTH-C3", "TSTH-EVB")):
            c.execute("INSERT INTO kalshi_markets (ticker, event_ticker, status, source) VALUES (%s, %s, 'settled', 'dataset')", (tk, et))
            c.execute("INSERT INTO kalshi_corpus (ticker, status, close_time) VALUES (%s, 'built', now())", (tk,))
    rest = FakeRest([], [])
    assert H.catalogue_missing(rest, threads=3) == 2
    assert sorted(c[1] for c in rest.calls if c[0] == "event") == ["TSTH-EVA", "TSTH-EVB"]
    with transaction() as c:
        rows = {r["event_ticker"]: r["category"] for r in c.execute("SELECT event_ticker, category FROM kalshi_events WHERE event_ticker LIKE 'TSTH-EV%'").fetchall()}
        assert rows == {"TSTH-EVA": "Politics", "TSTH-EVB": "Politics"} and c.execute("SELECT count(*) AS n FROM kalshi_series WHERE ticker = 'TSTH'").fetchone()["n"] == 1
    assert H.catalogue_missing(rest) == 0                                   # nothing left to fetch
    _clean_walk()


def test_the_event_sample_is_blind_to_volume_and_outcome_and_keeps_whole_events(tmp_path, monkeypatch):
    """Membership depends on the event ticker and its category's rate only: two markets of one event are in or out together,
    whatever they traded or how they settled; combos (exotics) are never in."""
    rates = {"rates": {"crypto": 0.5, "sports": 1.0, "exotics": 0.0}}
    evs = [f"KXBTCD-26SEP{i:02d}17" for i in range(200)]
    kept = [e for e in evs if H.in_sample(e, "Crypto", rates)]
    assert 60 < len(kept) < 140 and kept == [e for e in evs if H.event_u(e) < 0.5]            # about half, fixed by the hash
    assert all(H.in_sample(e, "sports", rates) for e in evs[:20]) and not any(H.in_sample(e, "exotics", rates) for e in evs)
    assert H.event_u("KXBTCD-26SEP0117") == H.event_u("KXBTCD-26SEP0117")
    assert H.in_sample("ANY-EVENT", "a new category", rates)                                   # categories the dataset never had: kept whole


def test_seed_takes_every_market_of_a_sampled_event_and_nothing_else(tmp_path, monkeypatch):
    _clean()
    monkeypatch.setattr(config, "KALSHI_DATASET_DIR", str(tmp_path)); monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2025-01-01")
    monkeypatch.setattr(H.D, "TRADES_DIR", tmp_path / "trades_out")
    ins = next(e for e in (f"TSTH-IN{i}" for i in range(100)) if H.event_u(e) < 0.5); out = next(e for e in (f"TSTH-OUT{i}" for i in range(100)) if H.event_u(e) >= 0.5)
    monkeypatch.setattr(H, "sample_rates", lambda conn=None: {"rates": {"unknown": 0.5}})
    d = tmp_path / "data" / "kalshi"; (d / "markets").mkdir(parents=True); (d / "trades").mkdir()
    day = datetime(2026, 3, 1, 12, 0)
    rows = [(f"{ins}-A", ins, 5.0), (f"{ins}-B", ins, 0.0), (f"{out}-A", out, 90000.0)]       # volume is irrelevant either way
    pq.write_table(pa.table({"ticker": [r[0] for r in rows], "event_ticker": [r[1] for r in rows], "market_type": ["binary"] * 3, "title": ["t"] * 3, "yes_sub_title": [""] * 3,
                             "no_sub_title": [""] * 3, "status": ["settled"] * 3, "result": ["yes", "no", "yes"], "open_time": [day - timedelta(days=2)] * 3, "close_time": [day] * 3,
                             "volume": [r[2] for r in rows], "open_interest": [0.0] * 3}), d / "markets" / "markets_0.parquet")
    pq.write_table(pa.table({"trade_id": ["1"], "ticker": [f"{ins}-A"], "count": [5.0], "yes_price": [40], "no_price": [60], "taker_side": ["yes"],
                             "created_time": [day - timedelta(hours=5)]}), d / "trades" / "trades_0.parquet")
    out_ = H.seed_from_dataset()
    assert out_ == {"markets": 2, "trades": 1}
    with transaction() as c:
        got = {r["ticker"] for r in c.execute("SELECT ticker FROM kalshi_corpus WHERE ticker LIKE 'TSTH-%%'").fetchall()}
    assert got == {f"{ins}-A", f"{ins}-B"}
    assert H.seed_from_dataset() == {"skipped": "already seeded"}
    _clean()


def test_apply_sample_moves_rows_in_and_out_of_the_corpus(monkeypatch):
    _clean_walk()
    ins = next(e for e in (f"TSTH-IN{i}" for i in range(100)) if H.event_u(e) < 0.5); out = next(e for e in (f"TSTH-OUT{i}" for i in range(100)) if H.event_u(e) >= 0.5)
    monkeypatch.setattr(H, "sample_rates", lambda conn=None: {"rates": {"unknown": 0.5}})
    with transaction() as c:
        for tk, et, status, cp in ((f"{out}-1", out, "built", "/x.parquet"), (f"{out}-2", out, "pending", None), (f"{ins}-1", ins, "skipped", "/y.parquet"), (f"{ins}-2", ins, "skipped", None),
                                   (f"{ins}-3", ins, "built", "/z.parquet")):
            c.execute("INSERT INTO kalshi_markets (ticker, event_ticker, status, source) VALUES (%s, %s, 'settled', 'test')", (tk, et))
            c.execute("INSERT INTO kalshi_corpus (ticker, status, candle_path, close_time) VALUES (%s, %s, %s, now())", (tk, status, cp))
    assert H.apply_sample() == 4
    with transaction() as c:
        st = {r["ticker"]: (r["status"], r["last_error"]) for r in c.execute("SELECT ticker, status, last_error FROM kalshi_corpus WHERE ticker LIKE 'TSTH-%%'").fetchall()}
    assert st[f"{out}-1"] == ("skipped", H.NOT_SAMPLED) and st[f"{out}-2"][0] == "skipped"
    assert st[f"{ins}-1"][0] == "done" and st[f"{ins}-2"][0] == "pending" and st[f"{ins}-3"][0] == "built"
    assert H.apply_sample() == 0
    _clean_walk()
