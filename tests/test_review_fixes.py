"""Pins for the second review: shared creator history, discovery audit handling, parquet writer restore, config redaction."""
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.ingest import discovery
from fly_trader.ingest.parquet_writer import Writer
from fly_trader.train.corpus_meta import creator_history


def test_creator_history_point_in_time(db_conn):
    c = "CreatorTest1"; g0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE creator = %s", (c,)); cur.execute("DELETE FROM pump_events WHERE signer = %s", (c,))
        for k in range(3):   # three earlier creates
            cur.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, signer) VALUES (%s,%s,'create','pump',%s,%s)", (f"sigct{k}", g0 - timedelta(hours=5 - k), f"M{k}", c))
        # graduation A 3 h before B (rug, outcome known), graduation A2 30 min before B (outcome not yet known at B)
        cur.execute("INSERT INTO corpus_meta (mint, creator, graduated_at, own_dd60, own_max60) VALUES ('M0', %s, %s, -0.95, 0.1)", (c, g0 - timedelta(hours=3)))
        cur.execute("INSERT INTO corpus_meta (mint, creator, graduated_at, own_dd60, own_max60) VALUES ('M1', %s, %s, -0.10, 1.5)", (c, g0 - timedelta(minutes=30)))
    db_conn.commit()
    h = creator_history(db_conn, c, g0 - timedelta(hours=1), g0)
    assert h["prior_launches"] == 3 and h["prior_grads"] == 2
    assert h["prior_known"] == 1 and h["prior_rug_share"] == 1.0 and h["prior_moon_share"] == 0.0
    assert creator_history(db_conn, None, None, g0)["prior_grads"] is None
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE creator = %s", (c,)); cur.execute("DELETE FROM pump_events WHERE signer = %s", (c,))
    db_conn.commit()


def test_nul_bytes_never_reach_postgres_text():
    import numpy as np
    from fly_trader.ingest import pumpstream
    from fly_trader.train.corpus_meta import _pg
    assert _pg("a\x00b") == "ab" and _pg(np.float64("nan")) is None and _pg(np.int64(3)) == 3
    agg = pumpstream.Aggregator()
    agg.ingest({"action": "create", "pool": "pump", "mint": "Npump", "timestamp": 1, "txSigner": "c", "name": "x\x00y", "symbol": "S\x00", "uri": None, "signature": "s"})
    assert agg.creates["Npump"]["name"] == "xy" and agg.creates["Npump"]["symbol"] == "S"


def test_stream_trader_falls_back_to_signer_like_the_archive():
    from fly_trader.ingest import pumpstream
    agg = pumpstream.Aggregator(); m = 1_800_000_000
    agg.ingest({"action": "buy", "pool": "pump-amm", "mint": "Tpump", "timestamp": (m + 1) * 1000, "price": 1.0, "quoteMint": pumpstream.WSOL, "quoteInPool": 80.0,
                "txSigner": "signer1", "breakdown": [{"action": "buy", "quoteAmount": 1.0}], "poolId": "P1"})
    assert agg.row(m, "Tpump").traders == {"signer1"}


def test_replay_parses_both_archive_formats(tmp_path):
    import orjson
    import zstandard
    from fly_trader.ingest.replay_pull import parse_hour
    old = {"txType": "buy", "pool": "pump-amm", "poolId": "P", "mint": "Mpump", "quoteMint": "So11111111111111111111111111111111111111112", "txSigner": "T",
           "tokenAmount": 10.0, "solAmount": 0.5, "solInPool": 80.0, "tokensInPool": 1e6, "price": 1e-5, "timestamp": 1778414399974, "block": 1}
    new = {"action": "sell", "pool": "pump-amm", "poolId": "P", "mint": "Mpump", "quoteMint": "So11111111111111111111111111111111111111112", "txSigner": "U",
           "tokenAmount": 5.0, "quoteAmount": 0.2, "quoteInPool": 79.8, "tokensInPool": 1e6, "price": 1e-5, "timestamp": 1778414400974, "block": 2,
           "breakdown": [{"action": "sell", "trader": "U", "tokenAmount": 5.0, "quoteAmount": 0.2}]}
    mig = {"txType": "migrate", "pool": "pump-amm", "poolId": "P", "mint": "Mpump", "solAmount": 1.1, "solInPool": 85.0, "poolCreatedBy": "pump", "timestamp": 1778414401974, "signature": "s"}
    path = tmp_path / "h.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(b"\n".join(orjson.dumps(e) for e in (old, new, mig))))
    trades, events, n = parse_hour(path)
    t = trades.to_pylist()
    assert n == 3 and [(r["side"], r["sol"], r["quote_in_pool"]) for r in t] == [(1, 0.5, 80.0), (-1, 0.2, 79.8)]    # the pre-2026-05-21 fields are read
    e = events.to_pylist()
    assert len(e) == 1 and e[0]["action"] == "migrate" and e[0]["quote_in_pool"] == 85.0 and e[0]["pool_created_by"] == "pump"


def test_discovery_missing_audit_is_unknown(monkeypatch):
    monkeypatch.setattr(config, "LAUNCHPADS", ["pump.fun"])
    tok = {"id": "X", "launchpad": "pump.fun", "graduatedAt": "2026-09-01T00:00:00Z", "graduatedPool": "P"}
    assert discovery.evaluate(tok)[0] == "unknown"
    assert discovery.evaluate({**tok, "audit": {"mintAuthorityDisabled": False, "freezeAuthorityDisabled": True}})[0] == "excluded"
    assert discovery.evaluate({**tok, "audit": {"mintAuthorityDisabled": True, "freezeAuthorityDisabled": True}})[0] == "watch"


def test_parquet_writer_keeps_every_unwritten_group(tmp_path, monkeypatch):
    w = Writer(tmp_path, "t", flush_rows=10, flush_s=999)
    w.add({"ts": datetime(2026, 9, 1, 23, 59, tzinfo=timezone.utc), "v": 1})
    w.add({"ts": datetime(2026, 9, 2, 0, 1, tzinfo=timezone.utc), "v": 2})
    import fly_trader.ingest.parquet_writer as pw
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1
        raise OSError("disk full")
    monkeypatch.setattr(pw.pq, "write_table", boom)
    with pytest.raises(OSError):
        w.flush()
    assert sorted(r["v"] for r in w.buf) == [1, 2]


def test_summary_redacts_every_dsn_form(monkeypatch):
    for dsn in ("postgresql://bot:s3cret@host:5432", "postgresql://bot:s3cret@host:5432/fly", "dbname=fly user=u password=hunter2 host=db"):
        monkeypatch.setattr(config, "DATABASE_URL", dsn, raising=False)
        out = str(config.summary().get("DATABASE_URL"))
        assert "s3cret" not in out and "hunter2" not in out
