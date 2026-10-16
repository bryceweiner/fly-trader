"""Operator commands (pause / resume / roll back) reach the running fly at its next minute; one queued while the engine was
down is dropped instead of landing whenever the engine next starts."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fly_trader.agent import fly_session as memecoin
from fly_trader.kalshi import fly_session as kalshi


class _Conn:
    def __init__(self, value):
        self.value, self.deleted = value, False

    def execute(self, sql, params=None):
        if sql.startswith("DELETE"):
            self.deleted = True
        return SimpleNamespace(fetchone=lambda: {"value": self.value} if self.value is not None and sql.startswith("SELECT") else None)


@pytest.mark.parametrize("mod", [memecoin, kalshi])
@pytest.mark.parametrize("age_s,applied", [(30, True), (memecoin.COMMAND_MAX_AGE_S + 60, False)])
def test_a_stale_command_is_dropped(mod, age_s, applied, monkeypatch):
    monkeypatch.setattr(mod, "record_event", lambda *a, **k: None)
    at = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
    conn = _Conn(json.dumps({"cmd": "pause", "at": at}))
    book = SimpleNamespace(learning_frozen=False)
    mod_cls = memecoin.FlyBook if mod is memecoin else kalshi.KalshiFlyBook
    mod_cls._commands(book, conn, 0.0)
    assert conn.deleted and book.learning_frozen is applied
