"""One circuit per trading type: the Kalshi rails (circuit 2) trip, pause and reset without touching the memecoin circuit (1)."""
import json

from fly_trader import config
from fly_trader.agent import rails


def _state(conn, cid):
    return conn.execute("SELECT kill_switch, entries_paused, fail_count, tripped FROM circuit_state WHERE id = %s", (cid,)).fetchone()


def test_kalshi_circuit_is_independent(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_THRESHOLD", 2)
    db_conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false, fail_count = 0, tripped = false")
    assert rails.check_drawdown(db_conn, wealth=60.0, peak=100.0, circuit_id=rails.KALSHI_CIRCUIT, drawdown=0.3, unit="USD")
    assert _state(db_conn, 2)["kill_switch"] and not _state(db_conn, 1)["kill_switch"]
    assert not rails.check_drawdown(db_conn, wealth=0.9, peak=1.0)                       # the memecoin circuit at its own default limit
    rails.record_failure(db_conn, "x", rails.KALSHI_CIRCUIT); rails.record_failure(db_conn, "y", rails.KALSHI_CIRCUIT)
    assert _state(db_conn, 2)["tripped"] and _state(db_conn, 1)["fail_count"] == 0
    rails.record_success(db_conn, rails.KALSHI_CIRCUIT)
    assert _state(db_conn, 2)["fail_count"] == 0
    ev = db_conn.execute("SELECT detail FROM circuit_events ORDER BY id DESC LIMIT 1").fetchone()["detail"]
    assert (ev if isinstance(ev, dict) else json.loads(ev))["circuit"] == 2
    db_conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false, fail_count = 0, tripped = false")
