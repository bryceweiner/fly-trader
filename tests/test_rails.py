from datetime import datetime, timedelta, timezone
from fly_trader import config
from fly_trader.agent import rails


def test_reflex_exits(monkeypatch):
    monkeypatch.setattr(config, "BRAIN_MODE", "lif")   # reflex exits belong to the LIF brain mode
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc).timestamp()
    base = {"entry_price": 1.0, "peak_price": 1.0, "satiety": 0.0, "opened_at": datetime.now(timezone.utc) - timedelta(minutes=5)}
    assert rails.forced_exit_kind(base, 0.49, now - 10, now) == "forced_stop"
    assert rails.forced_exit_kind(base, 1 - config.EXIT_LOSS - 0.001, now - 10, now) == "bitter"   # below the rejection line
    assert rails.forced_exit_kind(base, 1 - config.EXIT_LOSS + 0.005, now - 10, now) is None       # inside the band: hold
    turned = dict(base, peak_price=1.02)
    assert rails.forced_exit_kind(turned, 0.999, now - 10, now) == "turned"        # was +2%, now negative
    assert rails.forced_exit_kind(turned, 1.005, now - 10, now) is None
    sated = dict(base, peak_price=1.06, satiety=config.SATIETY_TARGET + 0.01)
    assert rails.forced_exit_kind(sated, 1.05, now - 10, now) == "satisfied"
    assert rails.forced_exit_kind(base, 0.999, now - config.DEAD_BAG_HOURS * 3600 - 1, now) == "forced_dead"


def test_kill_switch_and_circuit(db_conn):
    db_conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, peak_wealth=NULL, entries_paused=false WHERE id=1")
    peak, kill = rails.update_peak(db_conn, 5.0)
    assert peak == 5.0 and not kill
    peak, kill = rails.update_peak(db_conn, 5.0 * (1 - config.KILL_SWITCH_DRAWDOWN) - 0.01)
    assert kill
    blocks = rails.entry_blocks(db_conn, feed_age_s=1.0, sol_free=4.0, context_ready=True, brain_ok=True)
    assert "kill_switch" in blocks and "stale_feed" not in blocks
    for _ in range(config.CIRCUIT_THRESHOLD):
        st = rails.record_failure(db_conn, "boom")
    assert st.tripped
    rails.record_success(db_conn)
    assert rails.load_circuit(db_conn).fail_count == 0
    blocks = rails.entry_blocks(db_conn, feed_age_s=None, sol_free=0.1, context_ready=False, brain_ok=False)
    assert {"stale_feed", "context_not_ready", "brain_rates_out_of_range", "reserve"} <= set(blocks)


