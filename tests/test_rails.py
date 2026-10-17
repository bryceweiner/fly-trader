from datetime import timedelta

from fly_trader import config
from fly_trader.agent import rails


def test_kill_switch_and_circuit(db_conn):
    db_conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, peak_wealth=NULL, entries_paused=false WHERE id=1")
    assert not rails.check_drawdown(db_conn, 4.0, 5.0)                                     # 20 % below the peak: trading on
    assert rails.check_drawdown(db_conn, 5.0 * (1 - config.KILL_SWITCH_DRAWDOWN) - 0.01, 5.0)
    assert rails.load_circuit(db_conn).kill_switch
    for _ in range(config.CIRCUIT_THRESHOLD):
        st = rails.record_failure(db_conn, "boom")
    assert st.tripped
    rails.record_success(db_conn)
    assert rails.load_circuit(db_conn).fail_count == 0


def test_a_tripped_memecoin_breaker_reopens_after_its_cooldown(db_conn, monkeypatch):
    """Paper never stops for the breaker, so a trip that waited for a hand reset kept live and paper apart: entries may try
    again CIRCUIT_COOLDOWN_S after the last failure, a success clears the trip and a failure trips it at once. Kalshi's
    breaker still waits for a reset."""
    monkeypatch.setattr(config, "CIRCUIT_COOLDOWN_S", 900.0)
    db_conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, last_failure_ts=NULL WHERE id IN (1, 2)")
    for _ in range(config.CIRCUIT_THRESHOLD):
        st = rails.record_failure(db_conn, "boom")
    assert st.tripped and rails.breaker_blocks(st) and st.last_failure_ts is not None
    assert not rails.breaker_blocks(st, now=st.last_failure_ts + timedelta(seconds=901))            # the cooldown passed: entries may try
    assert rails.breaker_blocks(rails.CircuitState(tripped=True))                                   # no failure on record: until reset
    db_conn.execute("UPDATE circuit_state SET last_failure_ts = now() - interval '20 minutes' WHERE id = 1")
    st = rails.record_failure(db_conn, "the retry failed")
    assert st.tripped and rails.breaker_blocks(st)                                                  # blocked for another cooldown
    assert db_conn.execute("SELECT kind FROM circuit_events ORDER BY id DESC LIMIT 1").fetchone()["kind"] == "retripped"
    rails.record_success(db_conn)
    st = rails.load_circuit(db_conn)
    assert not st.tripped and st.fail_count == 0 and not rails.breaker_blocks(st)
    assert db_conn.execute("SELECT kind FROM circuit_events ORDER BY id DESC LIMIT 1").fetchone()["kind"] == "rearmed"
    for _ in range(config.CIRCUIT_THRESHOLD):
        rails.record_failure(db_conn, "kalshi boom", rails.KALSHI_CIRCUIT)
    rails.record_success(db_conn, rails.KALSHI_CIRCUIT)
    k = rails.load_circuit(db_conn, rails.KALSHI_CIRCUIT)
    assert k.tripped and k.fail_count == 0
    db_conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false WHERE id IN (1, 2)")
