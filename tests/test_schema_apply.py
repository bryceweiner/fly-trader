"""apply_schema never fights live workers for table locks: an unchanged DDL is not re-run (no locks at all), a changed
one is applied with every lock wait bounded by LOCK_TIMEOUT and retried after rolling back, so a worker holding a table
only delays it (2026-10-07: the RH stream's start-up apply deadlocked with the engine and crashed)."""
import threading
import time


from fly_trader.db import schema
from fly_trader.db.connection import connect, transaction


def _mark_stale():
    with transaction() as conn:
        conn.execute("UPDATE schema_version SET ddl_sha256 = 'stale' WHERE singleton")


def test_unchanged_ddl_is_not_rerun(monkeypatch):
    schema.apply_schema()
    called = []
    monkeypatch.setattr(schema, "_apply_all", lambda conn, fp: called.append(fp) or schema.SCHEMA_VERSION)
    assert schema.apply_schema() >= schema.SCHEMA_VERSION and called == []
    _mark_stale()
    schema.apply_schema()
    assert called == [schema.ddl_fingerprint()]                          # a changed fingerprint applies again


def test_a_held_table_delays_the_apply_instead_of_deadlocking_it(monkeypatch):
    schema.apply_schema(); _mark_stale()
    monkeypatch.setattr(schema, "LOCK_TIMEOUT", "200ms")
    sleeps = []
    real_sleep = time.sleep
    monkeypatch.setattr(schema.time, "sleep", lambda s: sleeps.append(s) or real_sleep(0.05))
    holder = connect()
    holder.execute("LOCK TABLE notional_ledger IN ACCESS EXCLUSIVE MODE")   # a live worker's transaction on a table RH_DDL alters
    t = threading.Timer(0.8, lambda: (holder.rollback(), holder.close()))
    t.start()
    try:
        v = schema.apply_schema()
    finally:
        t.join()
    assert v >= schema.SCHEMA_VERSION and len(sleeps) >= 1                 # waited, rolled back, retried, succeeded
    with transaction() as conn:
        assert conn.execute("SELECT ddl_sha256 FROM schema_version WHERE singleton").fetchone()["ddl_sha256"] == schema.ddl_fingerprint()


def test_no_transaction_is_left_open_holding_locks():
    schema.apply_schema()
    with transaction() as conn:
        n = conn.execute("SELECT count(*) AS n FROM pg_locks l JOIN pg_class c ON c.oid = l.relation WHERE c.relname = 'positions' "
                         "AND l.mode = 'AccessExclusiveLock'").fetchone()["n"]
    assert n == 0
