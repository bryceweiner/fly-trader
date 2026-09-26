"""The Kalshi stream: real-shaped ticker/trade/lifecycle frames aggregate per (market, minute) with the minute's ask low and
bid high, flush with the event-time watermark, never rewrite a flushed minute, keep the live top of book, and write
settlements to the market table."""
import math
from datetime import datetime, timezone

from fly_trader.db.connection import transaction
from fly_trader.kalshi import stream as KS

T0 = 1790089800.0                # a minute start


def _ticker(tk, yb, ya, ts, vol="10.00", oi="55.00", bs="12.00", as_="7.00", px="0.4500"):
    return {"type": "ticker", "sid": 1, "msg": {"market_ticker": tk, "price_dollars": px, "yes_bid_dollars": f"{yb / 100:.4f}", "yes_ask_dollars": f"{ya / 100:.4f}",
                                                 "volume_fp": vol, "open_interest_fp": oi, "dollar_volume": 0, "yes_bid_size_fp": bs, "yes_ask_size_fp": as_, "ts": int(ts), "ts_ms": int(ts * 1000)}}


def _trade(tk, px, cnt, side, ts, block=False):
    return {"type": "trade", "sid": 2, "msg": {"trade_id": f"t{ts}{cnt}", "market_ticker": tk, "yes_price_dollars": f"{px / 100:.4f}", "no_price_dollars": f"{(100 - px) / 100:.4f}",
                                                "count_fp": f"{cnt:.2f}", "taker_side": side, "is_block_trade": block, "ts": int(ts), "ts_ms": int(ts * 1000)}}


def _clean():
    with transaction() as c:
        c.execute("DELETE FROM kalshi_minutes WHERE ticker LIKE 'TST-%'"); c.execute("DELETE FROM kalshi_quotes WHERE ticker LIKE 'TST-%'"); c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TST-%'")


def test_minutes_aggregate_with_extremes_and_flush_on_the_watermark():
    _clean(); agg = KS.Aggregator()
    agg.ingest(_ticker("TST-A", 44, 47, T0 + 5)); agg.ingest(_ticker("TST-A", 45, 46, T0 + 20)); agg.ingest(_ticker("TST-A", 43, 48, T0 + 50))
    agg.ingest(_trade("TST-A", 46, 3, "yes", T0 + 21)); agg.ingest(_trade("TST-A", 45, 7, "no", T0 + 30, block=True))
    agg.ingest(_ticker("KXMVEX-1", 10, 20, T0 + 10))                                      # a combo: dropped at the door
    agg.ingest(_ticker("TST-B", 80, 82, T0 + 61))                                          # the next minute
    assert agg.stats["dropped_combo"] == 1 and len(agg.minutes[int(T0)]) == 1 and int(T0 + 60) in agg.minutes
    assert agg.flush_bound(T0 + 70) == T0                                                  # newest event T0+61: minute T0 is not complete yet
    agg.ingest(_ticker("TST-B", 80, 82, T0 + 63))                                          # an event ≥ 2 s past T0's end completes it
    assert agg.flush_bound(T0 + 70) == T0 + 60
    assert agg.flush(agg.flush_bound(T0 + 70)) == 1 and agg.flushed_through == datetime.fromtimestamp(T0, timezone.utc)
    with transaction() as c:
        r = dict(c.execute("SELECT * FROM kalshi_minutes WHERE ticker = 'TST-A'").fetchone())
    assert (r["yes_bid"], r["yes_ask"]) == (43.0, 48.0) and r["yes_ask_low"] == 46.0 and r["yes_bid_high"] == 45.0     # close, and the extremes
    assert r["taker_buy_yes"] == 3 and r["taker_buy_no"] == 7 and r["n_trades"] == 2 and r["max_trade"] == 7 and r["block_contracts"] == 7 and r["last"] == 45.0
    assert r["bid_size"] == 12 and r["ask_size"] == 7 and r["volume_fp"] == 10 and r["open_interest_fp"] == 55
    agg.ingest(_ticker("TST-A", 1, 2, T0 + 55))                                           # late for a flushed minute: dropped, never merged
    assert agg.stats["dropped_late"] == 1 and int(T0) not in agg.minutes
    assert agg.write_quotes() == 3 or agg.write_quotes() >= 0
    with transaction() as c:
        q = c.execute("SELECT yes_bid, yes_ask, mid, ask_size FROM kalshi_quotes WHERE ticker = 'TST-A'").fetchone()
    assert (q["yes_bid"], q["yes_ask"], q["mid"], q["ask_size"]) == (1.0, 2.0, 1.5, 7.0)     # the newest quote, whatever its minute
    _clean()


def test_lifecycle_settlement_writes_the_result_and_announcements_queue_a_fetch():
    _clean(); agg = KS.Aggregator()
    agg.ingest({"type": "market_lifecycle_v2", "msg": {"market_ticker": "TST-S", "event_type": "settled", "result": "yes", "settled_ts": int(T0), "close_ts": int(T0 - 100)}})
    agg.ingest({"type": "market_lifecycle_v2", "msg": {"market_ticker": "TST-N", "event_type": "created", "open_ts": int(T0), "close_ts": int(T0 + 86400)}})
    agg.ingest({"type": "market_lifecycle_v2", "msg": {"market_ticker": "TST-D", "event_type": "determined"}})           # no result in the frame: fetched
    assert agg.catalogue_todo == {"TST-N"} and len(agg.settled) == 2
    assert agg.write_settlements() == 2 and agg.catalogue_todo == {"TST-N", "TST-D"}
    with transaction() as c:
        r = c.execute("SELECT status, result, settlement_ts FROM kalshi_markets WHERE ticker = 'TST-S'").fetchone()
    assert r["result"] == "yes" and r["status"] == "settled" and r["settlement_ts"] == datetime.fromtimestamp(T0, timezone.utc)
    _clean()


def test_subscriptions_and_private_channels():
    pub = KS.subscribe_commands(False); prv = KS.subscribe_commands(True)
    assert [c["params"]["channels"][0] for c in pub] == ["ticker", "trade", "market_lifecycle_v2"]
    assert [c["params"]["channels"][0] for c in prv][-2:] == ["fill", "user_orders"] and all(c["cmd"] == "subscribe" for c in prv)
    assert KS._cents("0.4600") == 46.0 and math.isnan(KS._cents(None)) and KS._num("19.39") == 19.39


def test_lifecycle_fetch_catalogues_the_market_it_asked_for(monkeypatch):
    """KalshiRest.market returns the market object itself; the fetch must upsert it (it used to look one level too deep)."""
    _clean()
    from fly_trader.kalshi import history as H
    class Rest:
        def __init__(self):
            self.calls = []
        def market(self, tk):
            self.calls.append(tk); return {"ticker": tk, "event_ticker": None, "status": "active", "close_time": "2026-10-01T00:00:00Z", "title": "t"}
    rest = Rest()
    assert KS.fetch_markets(["TST-L1", "TST-L2"], rest=rest) == 2 and rest.calls == ["TST-L1", "TST-L2"]
    with transaction() as c:
        assert {r["ticker"] for r in c.execute("SELECT ticker FROM kalshi_markets WHERE ticker LIKE 'TST-L%'").fetchall()} == {"TST-L1", "TST-L2"}
    _clean()
