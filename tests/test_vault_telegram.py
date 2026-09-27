"""Telegram /panic and /status: only the operator's user in the operator's chat is obeyed; panic stops everything."""
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import state, telegram_cmd

from test_signer import vault_signer


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {"VAULT_ENABLED": True, "TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "100", "TELEGRAM_ADMIN_USER_ID": "7"}.items():
        monkeypatch.setattr(config, k, v)
    sent = []
    monkeypatch.setattr(telegram_cmd.alerts, "send", lambda text, **k: sent.append(text))
    with transaction() as c:
        c.execute("DELETE FROM vault_kv")
        c.execute("UPDATE circuit_state SET kill_switch = false, kill_reason = NULL WHERE id = 1")
    yield sent
    with transaction() as c:
        c.execute("UPDATE circuit_state SET kill_switch = false, kill_reason = NULL, entries_paused = false WHERE id = 1")
    state.resume()


def msg(uid, text, chat=100, user=7):
    return {"update_id": uid, "message": {"chat": {"id": chat}, "from": {"id": user}, "text": text}}


def test_only_the_operator_can_panic(monkeypatch, env):
    sg = vault_signer()
    monkeypatch.setattr(telegram_cmd, "_updates", lambda off: [m for m in [msg(1, "/panic", user=8), msg(2, "/panic", chat=999)] if m["update_id"] >= off])
    assert telegram_cmd.poll(sg) == 0
    assert not state.halted() and not sg.s.ledger.flag("panic")
    assert state.get("telegram_offset") == 3                                  # never looked at again


def test_panic_halts_signer_kill_switch_and_float(monkeypatch, env):
    sg = vault_signer()
    monkeypatch.setattr(telegram_cmd, "_updates", lambda off: [m for m in [msg(5, "/panic@flybot")] if m["update_id"] >= off])
    assert telegram_cmd.poll(sg) == 1
    assert state.halted() and state.get("panic") and sg.s.ledger.flag("panic")
    with transaction() as c:
        assert c.execute("SELECT kill_switch FROM circuit_state WHERE id = 1").fetchone()["kill_switch"] is True
    assert any("Resume only over SSH" in t for t in env)
    assert telegram_cmd.poll(sg) == 0                                         # handled once
    state.resume()
    assert not state.get("panic")


def test_status_and_help(monkeypatch, env):
    sg = vault_signer()
    monkeypatch.setattr(telegram_cmd, "_updates", lambda off: [m for m in [msg(9, "/status"), msg(10, "/sell everything")] if m["update_id"] >= off])
    assert telegram_cmd.poll(sg) == 1
    assert "running" in env[0] and "treasury" in env[0] and "signer: ok" in env[0]
    assert env[1] == telegram_cmd.HELP


def test_heartbeat_once_a_day(env):
    import calendar
    t = calendar.timegm((2026, 9, 27, 9, 30, 0))
    assert telegram_cmd.heartbeat(None, now=t) is True
    assert telegram_cmd.heartbeat(None, now=t + 3600) is False
    assert telegram_cmd.heartbeat(None, now=calendar.timegm((2026, 9, 28, 8, 0, 0))) is False      # before 09:00
    assert telegram_cmd.heartbeat(None, now=calendar.timegm((2026, 9, 28, 9, 0, 0))) is True
