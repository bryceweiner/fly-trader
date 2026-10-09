"""Shim for better_bot's ``database.get_connection`` (SQLite, ``?`` placeholders): a Postgres connection whose
``execute`` rewrites ``?`` to ``%s``. The vendored code reaches it only from ``venue_kalshi._cached_half_spread``,
which reads ``venue_market_cache`` — a view over fly-trader's ``kalshi_quotes`` (db/schema.py)."""
from __future__ import annotations

from ...db.connection import connect


class _Conn:
    def __init__(self, conn):
        self._c = conn

    def execute(self, sql: str, params=()):
        return self._c.execute(sql.replace("?", "%s"), tuple(params))

    def commit(self) -> None:
        self._c.commit()

    def rollback(self) -> None:
        self._c.rollback()

    def close(self) -> None:
        self._c.close()


def get_connection(create: bool = False) -> _Conn:
    return _Conn(connect())
