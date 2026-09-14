"""psycopg 3 connections to the fly_trader database.

Pattern from better_bot code/db_connection.py: a production database is refused while pytest is
running unless its name contains "test" (the autouse fixture in tests/conftest.py points
DATABASE_URL at fly_trader_test).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

from .. import config


class ProductionDatabaseUnderPytest(RuntimeError):
    pass


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or config.DATABASE_URL


def database_name(url: str | None = None) -> str:
    url = url or database_url()
    path = urlparse(url).path or ""
    return path.lstrip("/") or "fly_trader"


def _refuse_production_under_pytest(url: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ and "test" not in database_name(url):
        raise ProductionDatabaseUnderPytest(
            f"refusing to open non-test database {database_name(url)!r} under pytest"
        )


def connect(url: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    url = url or database_url()
    _refuse_production_under_pytest(url)
    return psycopg.connect(url, autocommit=autocommit, row_factory=dict_row, options="-c timezone=UTC")


@contextmanager
def transaction(url: str | None = None):
    """One transaction; commits on success, rolls back on exception."""
    conn = connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_all(sql: str, params=None, url: str | None = None) -> list[dict]:
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_one(sql: str, params=None, url: str | None = None) -> dict | None:
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def execute(sql: str, params=None, url: str | None = None) -> None:
    with transaction(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
