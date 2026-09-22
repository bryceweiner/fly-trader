"""Risk rails outside the model. KillSwitch (−KILL_SWITCH_DRAWDOWN from peak wealth → no entries), CircuitBreaker
(CIRCUIT_THRESHOLD consecutive execution failures), the rolling 24 h notional ledger, and PauseEntries (console
toggle). State lives in circuit_state / circuit_events / notional_ledger; the trading engine reads it before entries.
One circuit per trading type: ``circuit_id`` 1 is the memecoin live book (the default everywhere), 2 the Kalshi live
books (``KALSHI_CIRCUIT``, fly_trader/kalshi); ``check_drawdown`` takes the drawdown limit of the caller's type.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)
SOLANA_CIRCUIT, KALSHI_CIRCUIT = 1, 2


@dataclass
class CircuitState:
    fail_count: int = 0
    tripped: bool = False
    kill_switch: bool = False
    kill_reason: str | None = None
    peak_wealth: float | None = None
    entries_paused: bool = False


def load_circuit(conn, circuit_id: int = SOLANA_CIRCUIT) -> CircuitState:
    r = conn.execute("SELECT * FROM circuit_state WHERE id = %s", (circuit_id,)).fetchone()
    if not r:
        return CircuitState()
    return CircuitState(r["fail_count"], r["tripped"], r["kill_switch"], r["kill_reason"], r["peak_wealth"], r["entries_paused"])


def _event(conn, kind: str, detail: dict | None = None, circuit_id: int = SOLANA_CIRCUIT) -> None:
    conn.execute("INSERT INTO circuit_events (kind, detail) VALUES (%s, %s)", (kind, json.dumps({**(detail or {}), "circuit": circuit_id}, default=str)))


def record_failure(conn, reason: str, circuit_id: int = SOLANA_CIRCUIT) -> CircuitState:
    r = conn.execute(
        "UPDATE circuit_state SET fail_count = fail_count + 1, last_failure_ts = now(), updated_at = now() WHERE id = %s RETURNING fail_count, tripped", (circuit_id,)
    ).fetchone()
    tripped = r["tripped"]
    if r["fail_count"] >= config.CIRCUIT_THRESHOLD and not tripped:
        conn.execute("UPDATE circuit_state SET tripped = true, updated_at = now() WHERE id = %s", (circuit_id,))
        _event(conn, "tripped", {"reason": reason, "fail_count": r["fail_count"]}, circuit_id)
        record_event("error", "rails", "circuit breaker tripped", {"reason": reason, "circuit": circuit_id})
    return load_circuit(conn, circuit_id)


def record_success(conn, circuit_id: int = SOLANA_CIRCUIT) -> None:
    conn.execute("UPDATE circuit_state SET fail_count = 0, updated_at = now() WHERE id = %s", (circuit_id,))


def check_drawdown(conn, wealth: float, peak: float, circuit_id: int = SOLANA_CIRCUIT, drawdown: float | None = None, unit: str = "SOL") -> bool:
    """Trip the kill switch when ``wealth`` has fallen ``drawdown`` (default KILL_SWITCH_DRAWDOWN) below ``peak`` (the book's
    own peak: the trading engine passes the highest wealth mark of the current run). Returns whether the kill switch is on."""
    st = load_circuit(conn, circuit_id)
    if st.kill_switch:
        return True
    dd = config.KILL_SWITCH_DRAWDOWN if drawdown is None else float(drawdown)
    if peak > 0 and wealth <= peak * (1.0 - dd):
        conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = %s, peak_wealth = %s, updated_at = now() WHERE id = %s",
                     (f"wealth {wealth:.4f} {unit} <= {1 - dd:.2f} x peak {peak:.4f} {unit}", peak, circuit_id))
        _event(conn, "kill_switch", {"wealth": wealth, "peak": peak}, circuit_id)
        record_event("error", "rails", "kill switch tripped: entries blocked", {"wealth": wealth, "peak": peak, "circuit": circuit_id})
        return True
    return False


def check_book_drawdown(conn, book: str, wealth: float, peak: float, drawdown: float | None = None, unit: str = "SOL") -> bool:
    """A paper race book's own kill switch: ``drawdown`` (default KILL_SWITCH_DRAWDOWN) below the book's peak halts that
    book's entries only, so one book's drawdown never stops the other during the race. Returns whether the book is halted."""
    if book_halted(conn, book):
        return True
    dd = config.KILL_SWITCH_DRAWDOWN if drawdown is None else float(drawdown)
    if peak > 0 and wealth <= peak * (1.0 - dd):
        reason = f"wealth {wealth:.4f} {unit} <= {1 - dd:.2f} x peak {peak:.4f} {unit}"
        conn.execute("INSERT INTO book_state (book, halted, reason, peak, updated_at) VALUES (%s, true, %s, %s, now()) "
                     "ON CONFLICT (book) DO UPDATE SET halted = true, reason = EXCLUDED.reason, peak = EXCLUDED.peak, updated_at = now()",
                     (book, reason, peak))
        _event(conn, "book_halt", {"book": book, "wealth": wealth, "peak": peak})
        record_event("error", "rails", f"{book}: drawdown halt, entries blocked", {"wealth": wealth, "peak": peak})
        return True
    return False


def book_halted(conn, book: str) -> bool:
    r = conn.execute("SELECT halted FROM book_state WHERE book = %s", (book,)).fetchone()
    return bool(r and r["halted"])


def clear_book_halt(book: str) -> None:
    with transaction() as conn:
        conn.execute("UPDATE book_state SET halted = false, reason = NULL, peak = NULL, updated_at = now() WHERE book = %s", (book,))
        _event(conn, "book_resumed", {"book": book})
    record_event("info", "rails", f"{book}: drawdown halt cleared")


def notional_24h(conn) -> float:
    r = conn.execute("SELECT COALESCE(sum(sol), 0) AS s FROM notional_ledger WHERE ts > now() - interval '24 hours'").fetchone()
    return float(r["s"])


def record_notional(conn, sol: float, kind: str = "entry") -> None:
    conn.execute("INSERT INTO notional_ledger (sol, kind) VALUES (%s, %s)", (sol, kind))


def reset_circuit(kill: bool = False, circuit_id: int = SOLANA_CIRCUIT) -> None:
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false, updated_at = now() WHERE id = %s", (circuit_id,))
        _event(conn, "reset", {"kill": kill}, circuit_id)
        if kill:
            conn.execute("UPDATE circuit_state SET kill_switch = false, kill_reason = NULL, peak_wealth = NULL, updated_at = now() WHERE id = %s", (circuit_id,))
    record_event("info", "rails", "circuit reset" + (" incl. kill switch (peak re-based)" if kill else ""), {"circuit": circuit_id})
    print("circuit reset" + (" + kill switch cleared" if kill else ""))


def set_entries_paused(paused: bool, circuit_id: int = SOLANA_CIRCUIT) -> None:
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET entries_paused = %s, updated_at = now() WHERE id = %s", (paused, circuit_id))
        _event(conn, "paused" if paused else "resumed", {}, circuit_id)
    record_event("info", "rails", "entries paused" if paused else "entries resumed", {"circuit": circuit_id})
    print("entries paused" if paused else "entries resumed")
