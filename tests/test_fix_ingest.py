"""Ingest/db/market review fixes (2026-09-14): discovery gate + dedupe, capture learning, corpus/replay DB resilience,
DSN parsing, context coverage, danger units."""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg
import pytest

from fly_trader import config
from fly_trader.db import schema
from fly_trader.db.connection import database_name
from fly_trader.ingest import corpus_pull, discovery
from fly_trader.ingest import swap_decoder as sd
from fly_trader.ingest.capture import Capture
from fly_trader.market.context_window import ContextWindow
from fly_trader.market.danger import _frac

FX = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FX / "manifest.json").read_text())
GRAD = {"launchpad": "pump.fun", "graduatedAt": "2026-09-01T00:00:00Z", "decimals": 6,
        "audit": {"mintAuthorityDisabled": True, "freezeAuthorityDisabled": True}}


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


# ---- 5: graduated payload without graduatedPool ----
def test_graduated_without_pool_is_unknown_and_keeps_watch(db_conn):
    mint = "TestFix5Mint1111111111111111111111111111111"
    tok = {**GRAD, "id": mint, "graduatedPool": "TestFix5Pool"}
    discovery.upsert_token(db_conn, tok, "watch")
    bare = {**tok, "graduatedPool": None}
    assert discovery.evaluate(bare) == ("unknown", "graduated without graduatedPool")
    c = discovery.process_tokens(db_conn, [bare], with_stats=False)
    assert c["unknown"] == 1 and c["excluded"] == 0 and c["deactivated"] == 0
    assert db_conn.execute("SELECT watch_status FROM tokens WHERE mint = %s", (mint,)).fetchone()["watch_status"] == "watch"


# ---- 8: /recent + categories dedupe ----
def test_dedupe_keeps_first_payload_per_mint():
    a1, b, a2 = {"id": "A", "v": 1}, {"id": "B"}, {"id": "A", "v": 2}
    assert discovery.dedupe([a1, b, {"symbol": "no id"}, a2]) == [a1, b]


def test_poll_fast_writes_one_stats_row_per_mint(db_conn, monkeypatch):
    mint = "TestFix8Mint1111111111111111111111111111111"
    tok = {"id": mint, "launchpad": "pump.fun", "symbol": "T8"}      # pre-graduation: no watch pool, no events

    class Client:
        def recent(self):
            return [dict(tok)]

        def category(self, cat, iv):
            return [dict(tok)]

    monkeypatch.setattr(discovery, "transaction", _tx_on(db_conn))
    d = discovery.Discoverer.__new__(discovery.Discoverer)
    d.client = Client()
    assert d.poll_fast()["seen"] == 1
    assert db_conn.execute("SELECT count(*) AS n FROM token_stats WHERE mint = %s", (mint,)).fetchone()["n"] == 1


# ---- 9: context coverage ----
def test_context_coverage_starts_at_first_activity():
    cw = ContextWindow(window_s=100.0, gap_s=50.0, ready_cov=0.9)
    for t in (0.0, 10.0, 20.0):
        cw.observe(t, had_activity=False)
    assert cw.start_ts is None and cw.coverage(20.0) == 0.0 and cw.feed_age_s(20.0) is None
    cw.observe(30.0, True)
    assert cw.coverage(80.0) == pytest.approx(0.5) and not cw.ready(110.0) and cw.ready(120.0)
    cw.observe(200.0, True)                                            # silence > gap_s: coverage restarts
    assert cw.gaps == 1 and cw.coverage(200.0) == 0.0


# ---- 10: danger units ----
def test_danger_audit_fields_are_percents():
    assert _frac(0.5) == pytest.approx(0.005) and _frac(40.0) == pytest.approx(0.4) and _frac(None) == 0.0


# ---- 1 / 6: capture learning ----
def _raydium_v4():
    fx = MANIFEST["fixtures"]["raydium_v4_buy"]
    return MANIFEST["pools"][fx["pool_key"]], json.loads((FX / fx["file"]).read_text())


def test_raydium_v4_graduation_labelled_pumpswap_still_learns():
    p, tx = _raydium_v4()
    cap = Capture.__new__(Capture)
    cap.learned, cap.vault_index = {}, {}
    cap.unlearned = {p["pool"]: {"pool": p["pool"], "mint": p["mint"], "quote_mint": p["quote_mint"],
                                 "program_label": "Pump.fun Amm", "program_id": None, "quote_decimals": p["quote_decimals"]}}

    async def noop(*a, **k):
        return None
    cap._db = noop
    cap.event = noop
    keys, _ = sd.account_keys(*sd.unwrap(tx)[:2])
    asyncio.run(cap._maybe_learn(tx, set(keys)))
    got = cap.learned[p["pool"]]
    assert (got.base_vault, got.quote_vault) == (p["base_vault"], p["quote_vault"]) and not cap.unlearned


def test_pool_reload_keeps_vaults_learned_in_memory():
    p, _ = _raydium_v4()
    pv = sd.PoolVaults(pool=p["pool"], mint=p["mint"], quote_mint=p["quote_mint"], base_vault=p["base_vault"],
                       quote_vault=p["quote_vault"], base_decimals=p["base_decimals"], quote_decimals=p["quote_decimals"])
    cap = Capture.__new__(Capture)
    cap.learned, cap.pool_addresses = {p["pool"]: pv}, [p["pool"]]
    stale = {"pool": p["pool"], "mint": p["mint"], "quote_mint": None, "program_label": "Pump.fun Amm", "program_id": None,
             "base_vault": None, "quote_vault": None, "base_decimals": 6, "quote_decimals": None}
    assert cap._apply_pools([stale]) is False
    assert cap.learned[p["pool"]] is pv and not cap.unlearned and cap.vault_index[p["base_vault"]] is pv


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
