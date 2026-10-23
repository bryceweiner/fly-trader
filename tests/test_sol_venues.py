"""The archive keeps every venue's SOL legs (2026-10-23): the hour files and the refetch that rewrites older hours."""
from datetime import datetime, timedelta, timezone

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from fly_trader import config
from fly_trader.ingest import pumpstream, replay_pull

WSOL = pumpstream.WSOL
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _trade(pool, mint, quote, ts_ms, pid, action="buy", q=1.0, price=1e-5, by=None):
    e = {"action": action, "pool": pool, "poolId": pid, "mint": mint, "quoteMint": quote, "txSigner": "T", "tokenAmount": 10.0, "quoteAmount": q,
         "quoteInPool": 80.0, "tokensInPool": 1e6, "price": price, "timestamp": ts_ms, "block": 1,
         "breakdown": [{"action": action, "trader": "T", "tokenAmount": 10.0, "quoteAmount": q}]}
    if by is not None:
        e["poolCreatedBy"] = by
    return e


def test_hour_files_keep_every_venues_sol_legs_and_the_graduation_quote(tmp_path):
    t = 1_800_000_000_000
    evs = [_trade("pump-amm", "Apump", USDC, t, "P1"),                     # PumpSwap: kept in any quote, as before
           _trade("meteora-dlmm", "Apump", WSOL, t + 1, "D1"),             # another venue in SOL: kept
           _trade("meteora-dlmm", "Bother", USDC, t + 2, "D2"),            # another venue in another quote: dropped
           _trade("orca-whirlpool", "Cother", WSOL, t + 3, "O1"),
           {"action": "migrate", "pool": "pump-amm", "poolId": "P1", "mint": "Apump", "quoteMint": USDC, "quoteInPool": 7000.0, "timestamp": t + 4, "signature": "m"}]
    path = tmp_path / "h.zst"; path.write_bytes(zstandard.ZstdCompressor().compress(b"\n".join(orjson.dumps(e) for e in evs)))
    trades, events, n = replay_pull.parse_hour(path)
    assert n == 5 and [(r["pool"], r["mint"]) for r in trades.to_pylist()] == [("pump-amm", "Apump"), ("meteora-dlmm", "Apump"), ("orca-whirlpool", "Cother")]
    assert [(r["action"], r["quote_mint"]) for r in events.to_pylist()] == [("migrate", USDC)]
    h = datetime(2031, 2, 1, 5, tzinfo=timezone.utc); day = tmp_path / "replay" / "2031-02-01"
    replay_pull.write_hour(day, h, trades, events)
    assert sorted(p.name for p in day.iterdir()) == ["05_events.parquet", "05_trades.parquet"]          # no temporary file left
    assert replay_pull.trades_version(day / "05_trades.parquet") == replay_pull.TRADES_VERSION


def test_refetch_plans_older_unstamped_hours_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path); monkeypatch.setattr(config, "REFETCH_SKIP_RECENT_H", 6)
    now = datetime(2031, 2, 2, 12, tzinfo=timezone.utc)
    empty = pa.table({"ts": pa.array([], pa.timestamp("ms", tz="UTC"))})
    for h in (now - timedelta(hours=30), now - timedelta(hours=20), now - timedelta(hours=2)):     # unstamped (old writer)
        d = tmp_path / f"{h:%Y-%m-%d}"; d.mkdir(exist_ok=True); pq.write_table(empty, d / f"{h:%H}_trades.parquet")
    stamped = now - timedelta(hours=10)
    replay_pull.write_hour(tmp_path / f"{stamped:%Y-%m-%d}", stamped, empty, pa.table({"ts": pa.array([], pa.timestamp("ms", tz="UTC"))}))
    assert replay_pull.refetch_plan(now) == [now - timedelta(hours=20), now - timedelta(hours=30)]   # the newest 6 h are the console's
