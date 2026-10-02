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
        c.execute("DELETE FROM kalshi_universe WHERE ticker LIKE 'TSTH-%'")


def test_the_exchange_walk_resumes_from_its_saved_page_and_later_rounds_cover_only_new_settlements(monkeypatch):
    _clean_walk(); monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2026-01-01")
    t0 = datetime(2026, 9, 20, tzinfo=timezone.utc)
    live = [_mk(i, t0 - timedelta(hours=6 * i)) for i in range(10)]                     # 3 pages
    hist = [_mk(100 + i, datetime(2026, 7, 20, tzinfo=timezone.utc) - timedelta(days=i)) for i in range(6)]   # 2 pages
    rest = FakeRest(live, hist, fail_after=4)                                             # dies on the 5th page fetch (2nd of the archive)
    with pytest.raises(TimeoutError):
        H.refresh_markets(rest)
    st = H.walk_state()
    assert st["tier"] == "historical" and st["cursor"] == "c:4" and st.get("complete_through") is None and st["walk_started"]
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_universe WHERE ticker LIKE 'TSTH-W%'").fetchone()["n"] == 14        # 10 live + the archive's first page
    rest2 = FakeRest(live, hist)
    n = H.refresh_markets(rest2)
    assert n == 2 and rest2.calls == [("page", 4)]                                         # resumed at the archive's second page: no live page fetched again
    st = H.walk_state()
    assert st["tier"] is None and st["cursor"] is None and st["complete_through"] and st["walk_started"] is None
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_universe WHERE ticker LIKE 'TSTH-W%'").fetchone()["n"] == 16
    # the next round: only the live tier, only markets closing since the completed walk began (less the recovery margin)
    rest3 = FakeRest([_mk(50, datetime.now(timezone.utc).replace(microsecond=0))] + live, hist)     # settled after the walk began: newest first
    n = H.refresh_markets(rest3)
    assert n == 1
    assert [c for c in rest3.calls if c[0] == "page"] == [("page", 0)]                   # one live page, the archive untouched
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM kalshi_universe WHERE ticker = 'TSTH-W50'").fetchone()["n"] == 1
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


def test_select_events_takes_whole_events_in_hash_order_per_category_day():
    """Per (category, day), events in md5 order until the day holds the budget; blind to volume and outcome; combos never."""
    import pandas as pd
    day = pd.Timestamp("2026-05-01", tz="UTC")
    rows = []
    for i in range(40):                                        # 40 crypto events of 10 markets each on one day
        rows += [{"ticker": f"C{i}-{j}", "event_ticker": f"C{i}", "cat": "crypto", "day": day} for j in range(10)]
    rows += [{"ticker": f"W{i}", "event_ticker": f"W{i}", "cat": "climate and weather", "day": day} for i in range(5)]
    rows += [{"ticker": f"X{i}", "event_ticker": f"X{i}", "cat": "exotics", "day": day} for i in range(5)]
    sel = H.select_events(pd.DataFrame(rows), target=35)
    crypto = {t.split("-")[0] for t in sel if t.startswith("C")}
    order = sorted((f"C{i}" for i in range(40)), key=H.event_u)
    assert crypto == set(order[:4])                            # 0,10,20,30 before: four events (the fourth crosses 35, kept whole)
    assert all(f"{e}-{j}" in sel for e in crypto for j in range(10))
    assert {t for t in sel if t.startswith("W")} == {f"W{i}" for i in range(5)} and not any(t.startswith("X") for t in sel)
    assert H.select_events(pd.DataFrame(rows), target=35) == sel                      # deterministic


def test_select_corpus_puts_selected_markets_in_the_corpus_and_skips_the_rest(monkeypatch):
    _clean_walk()
    with transaction() as c:
        c.execute("DELETE FROM kalshi_universe WHERE ticker LIKE 'TSTH-%%'")
        c.execute("INSERT INTO kalshi_series (ticker, category) VALUES ('TSTHU', 'Crypto') ON CONFLICT (ticker) DO UPDATE SET category = 'Crypto'")
        close = datetime(2026, 5, 1, 12, tzinfo=timezone.utc)
        for i in range(6):
            for j in range(2):
                c.execute("INSERT INTO kalshi_universe (ticker, event_ticker, series_ticker, close_time, settled_ts, result, source) VALUES (%s,%s,'TSTHU',%s,%s,'yes','live')",
                          (f"TSTHU-E{i}-{j}", f"TSTHU-E{i}", close, close))
        c.execute("INSERT INTO kalshi_markets (ticker, event_ticker, status, source) VALUES ('TSTHU-E9-0', 'TSTHU-E9', 'settled', 'test')")
        c.execute("INSERT INTO kalshi_corpus (ticker, status, close_time, candle_path) VALUES ('TSTHU-E9-0', 'built', %s, '/x')", (close,))   # an old rule's pick
        c.execute("INSERT INTO kalshi_universe (ticker, event_ticker, series_ticker, close_time, result, source) VALUES ('TSTHU-E9-0','TSTHU-E9','TSTHU',%s,'no','live')", (close,))
    monkeypatch.setattr(config, "KALSHI_MARKETS_PER_DAY", 3)
    out = H.select_corpus()
    with transaction() as c:
        sel = {r["ticker"] for r in c.execute("SELECT ticker FROM kalshi_universe WHERE selected AND ticker LIKE 'TSTHU-%%'").fetchall()}
        corp = {r["ticker"]: r["status"] for r in c.execute("SELECT ticker, status FROM kalshi_corpus WHERE ticker LIKE 'TSTHU-%%'").fetchall()}
    evs = sorted({f"TSTHU-E{i}" for i in range(6)} | {"TSTHU-E9"}, key=H.event_u)[:2]          # two events of two markets reach the budget of three
    assert sel == {f"{e}-{j}" for e in evs for j in range(2)} - ({"TSTHU-E9-1"}) and out["selected"] == len(sel)
    assert all(corp[t] == ("built" if t == "TSTHU-E9-0" else "pending") for t in sel)
    if "TSTHU-E9-0" not in sel:
        assert corp["TSTHU-E9-0"] == "skipped"
    with transaction() as c:
        c.execute("DELETE FROM kalshi_universe WHERE ticker LIKE 'TSTHU-%%'"); c.execute("DELETE FROM kalshi_corpus WHERE ticker LIKE 'TSTHU-%%'")
        c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTHU-%%'"); c.execute("DELETE FROM kalshi_series WHERE ticker = 'TSTHU'")


def test_dataset_seed_loads_every_market_and_trades_go_only_to_selected_ones(tmp_path, monkeypatch):
    _clean_walk()
    monkeypatch.setattr(config, "KALSHI_DATASET_DIR", str(tmp_path)); monkeypatch.setattr(config, "KALSHI_HISTORY_START", "2025-01-01")
    monkeypatch.setattr(H.D, "TRADES_DIR", tmp_path / "trades_out")
    with transaction() as c:
        c.execute("DELETE FROM ui_settings WHERE key = %s", (H.SEED_KEY,)); c.execute("DELETE FROM kalshi_universe WHERE ticker LIKE 'TSTH-%%'")
    d = tmp_path / "data" / "kalshi"; (d / "markets").mkdir(parents=True); (d / "trades").mkdir()
    day = datetime(2025, 3, 1, 12, 0)
    rows = [("TSTH-D1-A", "TSTH-D1", 0.0, "yes"), ("TSTH-D1-B", "TSTH-D1", 50000.0, "no"), ("TSTH-D2-A", "TSTH-D2", 3.0, "yes"), ("TSTH-D3-A", "TSTH-D3", 1.0, "void")]
    pq.write_table(pa.table({"ticker": [r[0] for r in rows], "event_ticker": [r[1] for r in rows], "title": ["t"] * 4, "status": ["settled"] * 4, "result": [r[3] for r in rows],
                             "open_time": [day - timedelta(days=2)] * 4, "close_time": [day] * 4, "volume": [r[2] for r in rows]}), d / "markets" / "markets_0.parquet")
    pq.write_table(pa.table({"trade_id": ["1", "2"], "ticker": ["TSTH-D1-A", "TSTH-D2-A"], "count": [5.0, 1.0], "yes_price": [40, 30], "no_price": [60, 70], "taker_side": ["yes", "no"],
                             "created_time": [day - timedelta(hours=5)] * 2}), d / "trades" / "trades_0.parquet")
    assert H.seed_from_dataset() == {"universe": 3}                                     # every settled yes/no market, whatever its volume; void excluded
    assert H.seed_from_dataset() == {"skipped": "already seeded"}
    with transaction() as c:
        assert {r["ticker"] for r in c.execute("SELECT ticker FROM kalshi_universe WHERE ticker LIKE 'TSTH-D%%'").fetchall()} == {"TSTH-D1-A", "TSTH-D1-B", "TSTH-D2-A"}
        for tk in ("TSTH-D1-A", "TSTH-D1-B"):                                          # suppose the sample took event D1 only
            c.execute("UPDATE kalshi_universe SET selected = true WHERE ticker = %s", (tk,))
            c.execute("INSERT INTO kalshi_corpus (ticker, status, close_time) VALUES (%s, 'pending', %s)", (tk, day.replace(tzinfo=timezone.utc)))
    assert H.seed_dataset_trades() == 1
    with transaction() as c:
        got = {r["ticker"]: r["trades"] for r in c.execute("SELECT ticker, trades FROM kalshi_corpus WHERE ticker LIKE 'TSTH-D%%'").fetchall()}
    assert got == {"TSTH-D1-A": 1, "TSTH-D1-B": 0} and not (tmp_path / "trades_out" / "TSTH-D2-A.parquet").exists()   # an empty file, not an API call
    with transaction() as c:
        c.execute("DELETE FROM kalshi_universe WHERE ticker LIKE 'TSTH-D%%'"); c.execute("DELETE FROM kalshi_corpus WHERE ticker LIKE 'TSTH-D%%'"); c.execute("DELETE FROM ui_settings WHERE key = %s", (H.SEED_KEY,))
