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
    r = agg.minutes[m]["Apump"]
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
