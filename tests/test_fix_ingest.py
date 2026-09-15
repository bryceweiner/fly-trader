"""Ingest/db review fixes (2026-09-14): DSN parsing, corpus/replay DB resilience, token stats for the selector's universe."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import httpx
import psycopg
import pytest

from fly_trader import config
from fly_trader.db import schema
from fly_trader.db.connection import database_name
from fly_trader.ingest import corpus_pull, discovery


def _tx_on(conn):
    @contextmanager
    def tx():
        yield conn
    return tx


# ---- 4: DSN parsing ----
@pytest.mark.parametrize("url,name", [("postgresql://fly_trader:pw@localhost/fly_trader", "fly_trader"),
                                      ("postgresql:///fly_trader_test", "fly_trader_test"),
                                      ("dbname=fly_trader_test user=fly_trader host=/tmp", "fly_trader_test"),
                                      ("host=localhost user=bob", "fly_trader")])
def test_database_name_reads_urls_and_key_value_dsns(url, name):
    assert database_name(url) == name


def test_ensure_database_admin_dsn_keeps_user_and_host(monkeypatch):
    seen = []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            class R:
                def fetchone(self):
                    return (1,)
            return R()

    monkeypatch.setattr(schema.psycopg, "connect", lambda conninfo, **kw: seen.append(conninfo) or Conn())
    assert schema.ensure_database("postgresql://fly_trader:pw@localhost/fly_trader") is False
    assert psycopg.conninfo.conninfo_to_dict(seen[0]) == {"user": "fly_trader", "password": "pw", "host": "localhost", "dbname": "postgres"}
    schema.ensure_database("dbname=fly_trader user=fly_trader host=/tmp")
    assert psycopg.conninfo.conninfo_to_dict(seen[1]) == {"user": "fly_trader", "host": "/tmp", "dbname": "postgres"}


# ---- 2 / 3: corpus + replay resilience ----
def test_db_retry_rides_out_postgres_errors(monkeypatch):
    monkeypatch.setattr(corpus_pull, "_sleep", lambda s, stop: None)
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        return x * 2

    def bug():
        raise ValueError("not a database error")

    assert corpus_pull._db_retry(None, "t", flaky, 21) == 42 and len(calls) == 3
    stopped = threading.Event()
    stopped.set()
    assert corpus_pull._db_retry(stopped, "t", flaky, 1) is None
    with pytest.raises(ValueError):
        corpus_pull._db_retry(None, "t", bug)


def test_swap_api_4xx_is_permanent():
    api = corpus_pull.SwapApi(corpus_pull.Pacer(0.0, 0.0), corpus_pull.Status(), None)
    api.c = httpx.Client(base_url="http://swap.test", transport=httpx.MockTransport(lambda req: httpx.Response(404, text="not found")))
    with pytest.raises(corpus_pull.Permanent):
        api.get("/v2/coins/M/candles", {})


def test_corpus_requeues_cooled_transient_errors_only(db_conn, monkeypatch):
    monkeypatch.setattr(corpus_pull, "transaction", _tx_on(db_conn))
    grad = datetime.fromisoformat(config.CORPUS_PULL_BEFORE).replace(tzinfo=timezone.utc) - timedelta(days=1)
    rows = {"TestFix3Pending": ("pending", None, 0.0),
            "TestFix3TransientOld": ("error", "RuntimeError: swap-api /v2/coins/M/candles: gave up after retries", corpus_pull.RETRY_ERROR_H + 1),
            "TestFix3TransientNew": ("error", "RuntimeError: swap-api /v2/coins/M/candles: gave up after retries", 0.5),
            "TestFix3Permanent": ("error", "permanent: Permanent: swap-api /v2/coins/M/candles HTTP 404: nf", corpus_pull.RETRY_ERROR_H + 1)}
    for m, (st, err, age_h) in rows.items():
        db_conn.execute("INSERT INTO corpus_tokens (mint, graduated_at, status, last_error, updated_at) VALUES (%s, %s, %s, %s, now() - %s * interval '1 hour')",
                        (m, grad, st, err, age_h))
    picked = {r["mint"] for r in corpus_pull._next_pending(limit=1_000_000)}
    assert {"TestFix3Pending", "TestFix3TransientOld"} <= picked
    assert not {"TestFix3TransientNew", "TestFix3Permanent"} & picked


# ---- stats for the selector's universe: token_stats only, no tokens/watch_pools side effects ----
def test_refresh_stats_covers_selector_universe_without_watching(db_conn):
    mint = "TestUniverseMint111111111111111111111111111"
    db_conn.execute("INSERT INTO pump_minutes (mint, ts, close, resq_sol) VALUES (%s, date_trunc('minute', now()) - interval '5 minutes', 1e-7, 30.0)", (mint,))
    asked = []

    class Client:
        def search_many(self, mints):
            asked.append(list(mints))
            return [{"id": mint, "organicScore": 42.0}] if mint in mints else []

    c = discovery.refresh_stats(db_conn, Client())
    assert c["universe"] >= 1 and c["refreshed"] == 1 and any(mint in a for a in asked)
    assert db_conn.execute("SELECT organic_score FROM token_stats WHERE mint = %s", (mint,)).fetchone()["organic_score"] == 42.0
    assert db_conn.execute("SELECT 1 FROM tokens WHERE mint = %s", (mint,)).fetchone() is None
    assert db_conn.execute("SELECT 1 FROM watch_pools WHERE mint = %s", (mint,)).fetchone() is None
