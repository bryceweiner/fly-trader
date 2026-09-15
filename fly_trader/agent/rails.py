"""Risk rails outside the model. KillSwitch (−KILL_SWITCH_DRAWDOWN from peak wealth → no entries), CircuitBreaker
(CIRCUIT_THRESHOLD consecutive execution failures), the rolling 24 h notional ledger, and PauseEntries (console
toggle). State lives in circuit_state / circuit_events / notional_ledger; the trading engine reads it before entries.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)


@dataclass
class CircuitState:
    fail_count: int = 0
    tripped: bool = False
    kill_switch: bool = False
    kill_reason: str | None = None
    peak_wealth: float | None = None
    entries_paused: bool = False


def load_circuit(conn) -> CircuitState:
    r = conn.execute("SELECT * FROM circuit_state WHERE id = 1").fetchone()
    if not r:
        return CircuitState()
    return CircuitState(r["fail_count"], r["tripped"], r["kill_switch"], r["kill_reason"], r["peak_wealth"], r["entries_paused"])


def _event(conn, kind: str, detail: dict | None = None) -> None:
    conn.execute("INSERT INTO circuit_events (kind, detail) VALUES (%s, %s)", (kind, json.dumps(detail or {}, default=str)))


def record_failure(conn, reason: str) -> CircuitState:
    r = conn.execute(
        "UPDATE circuit_state SET fail_count = fail_count + 1, last_failure_ts = now(), updated_at = now() WHERE id = 1 RETURNING fail_count, tripped"
    ).fetchone()
    tripped = r["tripped"]
    if r["fail_count"] >= config.CIRCUIT_THRESHOLD and not tripped:
        conn.execute("UPDATE circuit_state SET tripped = true, updated_at = now() WHERE id = 1")
        _event(conn, "tripped", {"reason": reason, "fail_count": r["fail_count"]})
        record_event("error", "rails", "circuit breaker tripped", {"reason": reason})
    return load_circuit(conn)


def record_success(conn) -> None:
    conn.execute("UPDATE circuit_state SET fail_count = 0, updated_at = now() WHERE id = 1")


def check_drawdown(conn, wealth: float, peak: float) -> bool:
    """Trip the kill switch when ``wealth`` has fallen KILL_SWITCH_DRAWDOWN below ``peak`` (the book's own peak: the
    trading engine passes the highest wealth mark of the current run). Returns whether the kill switch is on."""
    st = load_circuit(conn)
    if st.kill_switch:
        return True
    if peak > 0 and wealth <= peak * (1.0 - config.KILL_SWITCH_DRAWDOWN):
        conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = %s, peak_wealth = %s, updated_at = now() WHERE id = 1",
                     (f"wealth {wealth:.4f} SOL <= {1 - config.KILL_SWITCH_DRAWDOWN:.2f} x peak {peak:.4f} SOL", peak))
        _event(conn, "kill_switch", {"wealth": wealth, "peak": peak})
        record_event("error", "rails", "kill switch tripped: entries blocked", {"wealth": wealth, "peak": peak})
        return True
    return False


def notional_24h(conn) -> float:
    r = conn.execute("SELECT COALESCE(sum(sol), 0) AS s FROM notional_ledger WHERE ts > now() - interval '24 hours'").fetchone()
    return float(r["s"])


def record_notional(conn, sol: float, kind: str = "entry") -> None:
    conn.execute("INSERT INTO notional_ledger (sol, kind) VALUES (%s, %s)", (sol, kind))


def reset_circuit(kill: bool = False) -> None:
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false, updated_at = now() WHERE id = 1")
        _event(conn, "reset", {"kill": kill})
        if kill:
            conn.execute("UPDATE circuit_state SET kill_switch = false, kill_reason = NULL, peak_wealth = NULL, updated_at = now() WHERE id = 1")
    record_event("info", "rails", "circuit reset" + (" incl. kill switch (peak re-based)" if kill else ""))
    print("circuit reset" + (" + kill switch cleared" if kill else ""))


def set_entries_paused(paused: bool) -> None:
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET entries_paused = %s, updated_at = now() WHERE id = 1", (paused,))
        _event(conn, "paused" if paused else "resumed", {})
    record_event("info", "rails", "entries paused" if paused else "entries resumed")
    print("entries paused" if paused else "entries resumed")
