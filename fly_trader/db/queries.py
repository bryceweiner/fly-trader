"""Read-side helpers for the console and `status`. Everything counts rows; nothing reads status flags."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager

import psycopg

from .connection import connect

_tls = threading.local()


def reuse_connection_in_this_thread() -> None:
    """Keep one autocommit connection per thread for ``q``/``q1`` (the console's topic threads read every second; a fresh
    connection per query would churn ~30 connections a second). Other threads keep a connection per call."""
    _tls.reuse = True


def _reused():
    conn = getattr(_tls, "conn", None)
    if conn is None or conn.closed:
        conn = _tls.conn = connect(autocommit=True)
    return conn


@contextmanager
def read_connection():
    """A connection for reads that need one (functions taking ``conn``): this thread's reused one when it keeps one."""
    if getattr(_tls, "reuse", False):
        yield _reused()
        return
    with connect() as conn:
        yield conn


def q(sql: str, params=None) -> list[dict]:
    if getattr(_tls, "reuse", False):
        for attempt in (0, 1):
            try:
                return [dict(r) for r in _reused().execute(sql, params).fetchall()]
            except psycopg.OperationalError:
                conn = getattr(_tls, "conn", None)
                if attempt or (conn is not None and not conn.closed and not conn.broken):
                    raise                             # a cancelled query, a deadlock, a lock timeout: the connection is fine
                try:                                  # the server dropped the idle connection: open a new one once
                    conn.close()
                except Exception:
                    pass
                _tls.conn = None
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def q1(sql: str, params=None) -> dict | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def status() -> dict:
    out: dict = {}
    out["wealth"] = q("""SELECT DISTINCT ON (book) book, ts, wealth, sol_free, positions_value, exposure, n_open, peak, drawdown
                         FROM wealth_marks ORDER BY book, ts DESC""")
    out["positions"] = q("SELECT book, status, count(*) AS n, COALESCE(sum(realized_sol),0) AS realized FROM positions GROUP BY 1,2 ORDER BY 1,2")
    out["fills"] = q("SELECT book, count(*) AS n, min(ts) AS first, max(ts) AS last FROM fills GROUP BY 1 ORDER BY 1")
    out["beats_last_hour"] = (q1("SELECT count(*) AS n, max(ts) AS last FROM beats WHERE ts > now() - interval '1 hour'") or {})
    out["decisions_last_hour"] = q("SELECT kind, COALESCE(rail,'') AS rail, count(*) AS n FROM decisions WHERE ts > now() - interval '1 hour' GROUP BY 1,2 ORDER BY 3 DESC")
    out["tokens"] = q("SELECT watch_status, count(*) AS n FROM tokens GROUP BY 1 ORDER BY 1")
    out["circuit"] = q1("SELECT * FROM circuit_state WHERE id = 1")
    out["brain"] = q1("SELECT * FROM brain_state")
    out["processes"] = q("SELECT name, pid, started_at, stopped_at, exit_code FROM processes ORDER BY started_at DESC LIMIT 10")
    out["events"] = q("SELECT ts, level, source, message FROM events ORDER BY ts DESC LIMIT 10")
    return out


def print_status() -> None:
    s = status()
    print(json.dumps(s, indent=1, default=str))
