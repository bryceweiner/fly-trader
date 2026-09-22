"""PumpAPI stream aggregation: filters, minute boundaries, idempotent flush."""
from datetime import datetime, timezone

from fly_trader.ingest import pumpstream

WSOL = "So11111111111111111111111111111111111111112"


def _ev(action, mint, ts_ms, price, sol, trader, pool="pump-amm", quote=WSOL, resq=80.0):
    return {"action": action, "pool": pool, "mint": mint, "timestamp": ts_ms, "price": price, "quoteMint": quote, "quoteInPool": resq,
            "txSigner": trader, "breakdown": [{"action": action, "trader": trader, "quoteAmount": sol, "tokenAmount": 1.0}], "poolId": "P1"}


def test_aggregation_and_flush(db_conn):
    agg = pumpstream.Aggregator()
    m = 1_800_000_000  # a minute boundary (seconds)
    agg.ingest(_ev("buy", "Apump", (m + 1) * 1000, 1.0, 2.0, "t1"))
    agg.ingest(_ev("sell", "Apump", (m + 30) * 1000, 0.9, 1.0, "t2"))
    agg.ingest(_ev("buy", "Apump", (m + 59) * 1000, 1.1, 3.0, "t1"))
    agg.ingest(_ev("buy", "Apump", (m + 61) * 1000, 1.2, 1.0, "t3"))                    # next minute
    agg.ingest(_ev("buy", "Bpump", (m + 5) * 1000, 1.0, 1.0, "t9", quote="USDC"))     # not SOL-quoted: ignored
    agg.ingest(_ev("buy", "Cother", (m + 5) * 1000, 1.0, 1.0, "t9"))                  # not pump.fun-origin (no "pump" suffix): ignored
    agg.ingest(_ev("buy", "Dpump", (m + 5) * 1000, 1.0, 1.0, "t9", pool="pump"))       # bonding curve: not a PumpSwap minute
    assert set(agg.minutes) == {m, m + 60} and list(agg.minutes[m]) == ["Apump"]
    r = agg.row(m, "Apump")
    assert (r.open, r.high, r.low, r.close) == (1.0, 1.1, 0.9, 1.1) and (r.buy, r.sell, r.nb, r.ns) == (5.0, 1.0, 2, 1) and len(r.traders) == 2
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM pump_minutes WHERE mint = 'Apump'")
    db_conn.commit()
    assert agg.flush(m) == 0                       # the current minute is still open
    assert agg.flush(m + 60) == 1                  # the minute that closed at m+60 is written
    assert agg.flushed_through == datetime.fromtimestamp(m, timezone.utc)
    row = db_conn.execute("SELECT open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol FROM pump_minutes WHERE mint = 'Apump'").fetchone()
    assert (row["open"], row["close"], row["buy_sol"], row["n_traders"], row["resq_sol"]) == (1.0, 1.1, 5.0, 2, 80.0)
    assert m + 60 in agg.minutes and m not in agg.minutes


def test_band_dominant_pool_and_watermark():
    agg = pumpstream.Aggregator()
    m = 1_800_000_000
    for k in range(5):
        agg.ingest(_ev("buy", "Epump", (m + 1 + k) * 1000, 1.0, 1.0, f"t{k}"))
    agg.ingest(_ev("buy", "Epump", (m + 10) * 1000, 100.0, 1.0, "tz"))          # 100x off the trailing median: dropped
    assert agg.stats["dropped_band"] == 1 and agg.row(m, "Epump").high == 1.0
    other = _ev("buy", "Epump", (m + 20) * 1000, 1.05, 9.0, "ty"); other["poolId"] = "P2"
    agg.ingest(other)                                                            # a second pool with fewer legs this minute
    assert agg.row(m, "Epump").pool_id == "P1" and agg.row(m, "Epump").nb == 5
    agg.ingest(_ev("buy", "Epump", (m + 1) * 1000, 1.0, 1.0, "t0", resq=500_000.0))   # reserve outside the band: ignored
    assert agg.row(m, "Epump").nb == 5
    # the minute is complete once an event stamped >= 2 s after its end arrives, or 15 s after it by wall clock
    assert agg.flush_bound(m + 30) == m            # minute m is still open: only minutes before m are complete
    agg.max_event_s = m + 62.5
    assert agg.flush_bound(m + 63) == m + 60       # an event 2.5 s past its end completes minute m
    agg.max_event_s = m + 20
    assert agg.flush_bound(m + 76) == m + 60       # quiet stream: 15 s of wall clock completes it


def test_midnight_late_legs_and_lagging_stream():
    agg = pumpstream.Aggregator(); M = 20834 * 86400     # a UTC midnight
    for k in range(5):
        agg.ingest(_ev("buy", "Fpump", (M + 1 + k) * 1000, 1.0, 1.0, f"t{k}"))
    agg.ingest(_ev("buy", "Fpump", (M - 1) * 1000, 1.0, 1.0, "late"))             # a late event from the previous day
    agg.ingest(_ev("buy", "Fpump", (M + 10) * 1000, 100.0, 1.0, "tz"))
    assert agg.stats["dropped_band"] == 1                                         # today's references survived the late event
    agg.flushed_through = datetime.fromtimestamp(M, timezone.utc)
    agg.ingest(_ev("buy", "Fpump", (M + 30) * 1000, 1.0, 1.0, "t9"))              # minute M is already written
    assert agg.stats["dropped_late"] == 1
    agg.max_event_s = M + 55; agg.last_event_wall = M + 70
    assert agg.flush_bound(M + 75) == M            # lagging but live: waits for the minute's events
    agg.last_event_wall = M + 55
    assert agg.flush_bound(M + 75) == M + 60       # quiet for 20 s: the wall clock completes it


def test_load_skill_falls_back_to_the_newest_table_on_disk(tmp_path, monkeypatch):
    """A distribution ships one table and no pipeline to make the next: the engine keeps using the newest it has rather
    than dropping the skill inputs (the selector fails closed without them, the fly reads them as zero)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fly_trader.train import mature
    monkeypatch.setattr(mature, "SKILL_DIR", tmp_path / "skill"); (tmp_path / "skill").mkdir()
    agg = pumpstream.Aggregator()
    pumpstream.load_skill(agg, datetime(2026, 9, 25, 12, tzinfo=timezone.utc).timestamp())
    assert agg.skill is None and agg.skill_day == "2026-09-25"                                   # nothing on disk: as before
    for d, w in (("2026-09-20", "OLD"), ("2026-09-23", "NEW")):
        pq.write_table(pa.table({"wallet": [w], "bucket": pa.array([7], pa.int8())}), tmp_path / "skill" / f"{d}.parquet")
    agg.skill = None
    pumpstream.load_skill(agg, datetime(2026, 9, 25, 12, tzinfo=timezone.utc).timestamp())
    assert agg.skill == {"NEW": 7} and agg.skill_day == "2026-09-25"                             # the newest, two days stale
    pumpstream.load_skill(agg, datetime(2026, 9, 23, 12, tzinfo=timezone.utc).timestamp())
    assert agg.skill == {"NEW": 7} and agg.skill_day == "2026-09-23"                             # the day's own table when it exists
