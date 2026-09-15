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


