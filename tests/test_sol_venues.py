"""Meteora DLMM pools of pump.fun coins make their minutes too (2026-10-23), and the archive keeps every venue's SOL legs:
the hour files, the refetch that rewrites older hours, the stream and a graduation's quote asset."""
import os
import time
from datetime import date, datetime, timedelta, timezone

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from fly_trader import config
from fly_trader.ingest import pumpstream, replay_pull
from fly_trader.train import corpus_meta, mature

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


def test_stream_keeps_pumpfun_dlmm_legs_and_learns_graduation_quotes():
    agg = pumpstream.Aggregator(); m = 1_800_000_000
    agg.ingest(_trade("meteora-dlmm", "Apump", WSOL, (m + 1) * 1000, "DLMM_A"))
    agg.ingest(_trade("meteora-dlmm", "Apump", WSOL, (m + 2) * 1000, "DLMM_A", "sell"))
    agg.ingest(_trade("pump-amm", "Apump", WSOL, (m + 3) * 1000, "SWAP_A"))
    agg.ingest(_trade("meteora-dlmm", "Zother", WSOL, (m + 4) * 1000, "DLMM_Z"))                # not a pump.fun coin
    agg.ingest(_trade("raydium-clmm", "Apump", WSOL, (m + 5) * 1000, "CLMM_A"))                # not a universe venue
    agg.ingest(_trade("pump-amm", "Bpump", USDC, (m + 6) * 1000, "SWAP_B"))                    # a graduate's own pool in USDC
    assert set(agg.minutes[m]) == {"Apump"} and set(agg.minutes[m]["Apump"]) == {"DLMM_A", "SWAP_A"}
    assert agg.row(m, "Apump").pool_id == "DLMM_A"                                             # the pool with the most legs
    assert agg.pending_quotes == [("Bpump", "SWAP_B", USDC)]
    agg.ingest(_trade("pump-amm", "Bpump", USDC, (m + 7) * 1000, "SWAP_B"))
    assert len(agg.pending_quotes) == 1                                                        # one per pool
    dl = _trade("meteora-dlmm", "Apump", WSOL, 0, "DLMM_A"); dl["signature"] = "s"
    assert agg.key(dl) == ("s", "DLMM_A") and agg.key({**dl, "pool": "pump"}) == ("s", "curve", "Apump")


def test_graduation_quote_turns_quote_unit_inputs_unknown(db_conn):
    db_conn.execute("INSERT INTO corpus_meta (mint, graduated_at, pool_id, rq0, dev_sol) VALUES ('GqUsdcpump', now(), 'GQ_P1', 7000, 300), "
                    "('GqSolpump', now(), 'GQ_P2', 67.41, 1.5) ON CONFLICT DO NOTHING"); db_conn.commit()
    try:
        assert corpus_meta.record_grad_quotes(db_conn, [("GqUsdcpump", "GQ_P1", USDC), ("GqSolpump", "OTHER_POOL", USDC)]) == 1   # its own pool only
        db_conn.commit()
        rows = {r["mint"]: r for r in db_conn.execute("SELECT mint, " + corpus_meta.sol_select(["rq0", "dev_sol", "ttg_min"]) + " FROM corpus_meta WHERE mint LIKE 'Gq%'").fetchall()}
        assert rows["GqUsdcpump"]["rq0"] is None and rows["GqUsdcpump"]["dev_sol"] is None          # USDC amounts are not SOL inputs
        assert rows["GqSolpump"]["rq0"] == 67.41 and rows["GqSolpump"]["dev_sol"] == 1.5           # an unknown quote reads as SOL, as before
    finally:
        db_conn.execute("DELETE FROM corpus_meta WHERE mint LIKE 'Gq%'"); db_conn.commit()


def test_aggregate_day_lets_a_dlmm_pool_make_the_minute(db_conn, tmp_path, monkeypatch):
    d = date(2031, 1, 4); t0 = datetime(2031, 1, 4, tzinfo=timezone.utc).timestamp()
    rows = [("Xpump", t0 + 1, "SWAP_X", "pump-amm", 1.0), ("Xpump", t0 + 2, "DLMM_X", "meteora-dlmm", 1.01), ("Xpump", t0 + 3, "DLMM_X", "meteora-dlmm", 1.02),
            ("Xpump", t0 + 61, "SWAP_X", "pump-amm", 1.0), ("Yother", t0 + 1, "DLMM_Y", "meteora-dlmm", 5.0), ("Xpump", t0 + 62, "ORCA_X", "orca-whirlpool", 9.0)]
    tab = pa.table({"ts": pa.array([datetime.fromtimestamp(r[1], timezone.utc) for r in rows], pa.timestamp("ms", tz="UTC")), "slot": pa.array(range(len(rows)), pa.int64()),
                    "pool": [r[3] for r in rows], "mint": [r[0] for r in rows], "trader": ["t"] * len(rows), "side": pa.array([1] * len(rows), pa.int8()),
                    "sol": [1.0] * len(rows), "price": [r[4] for r in rows], "quote_in_pool": [80.0] * len(rows), "quote_mint": [WSOL] * len(rows),
                    "pool_id": [r[2] for r in rows]})
    (tmp_path / "replay" / d.isoformat()).mkdir(parents=True)
    pq.write_table(tab, tmp_path / "replay" / d.isoformat() / "00_trades.parquet")
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    try:
        assert mature.aggregate_day(d) == 2
        out = pq.read_table(tmp_path / "mature" / f"{d.isoformat()}.parquet").to_pandas().sort_values("ts")
        assert set(out["mint"]) == {"Xpump"} and list(out["close"]) == [1.02, 1.0]                   # minute 0: DLMM (2 legs); minute 1: PumpSwap
        agg = tmp_path / "mature" / f"{d.isoformat()}.parquet"
        assert not mature.agg_stale(agg)
        later = time.time() + 5
        os.utime(tmp_path / "replay" / d.isoformat() / "00_trades.parquet", (later, later))           # the hour rewritten (refetch)
        assert mature.agg_stale(agg)
    finally:
        db_conn.execute("DELETE FROM mature_days WHERE day = %s", (d,)); db_conn.commit()
