"""Risk rails outside the fly. Every block is recorded as a decisions row (kind='blocked', rail=...).

KillSwitch (−KILL_SWITCH_DRAWDOWN from peak live wealth → no entries), HardStop (−HARD_STOP_FRAC
from entry → forced exit), DeadBagSweep (no pool swap for DEAD_BAG_HOURS → forced exit; VOC
cross_sectional_env.py:227-236), CircuitBreaker (VOC safety.py:25-60; consecutive execution failures),
check_reserve (VOC safety.py:85-90), NotionalCap (VOC safety.py:63-82; rolling 24 h), StaleFeed,
PauseEntries (console toggle). State lives in circuit_state / circuit_events / notional_ledger.
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


def update_peak(conn, wealth: float) -> tuple[float, bool]:
    """Update the live peak; trip the kill switch on drawdown. Returns (peak, kill_switch)."""
    st = load_circuit(conn)
    peak = st.peak_wealth if st.peak_wealth is not None else wealth
    if wealth > peak:
        peak = wealth
    kill = st.kill_switch
    if not kill and peak > 0 and wealth <= peak * (1.0 - config.KILL_SWITCH_DRAWDOWN):
        kill = True
        conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = %s, updated_at = now() WHERE id = 1",
                     (f"wealth {wealth:.4f} <= {1 - config.KILL_SWITCH_DRAWDOWN:.2f} x peak {peak:.4f}",))
        _event(conn, "kill_switch", {"wealth": wealth, "peak": peak})
        record_event("error", "rails", "kill switch tripped: entries blocked", {"wealth": wealth, "peak": peak})
    conn.execute("UPDATE circuit_state SET peak_wealth = %s, updated_at = now() WHERE id = 1", (peak,))
    return peak, kill


def notional_24h(conn) -> float:
    r = conn.execute("SELECT COALESCE(sum(sol), 0) AS s FROM notional_ledger WHERE ts > now() - interval '24 hours'").fetchone()
    return float(r["s"])


def record_notional(conn, sol: float, kind: str = "entry") -> None:
    conn.execute("INSERT INTO notional_ledger (sol, kind) VALUES (%s, %s)", (sol, kind))


def entry_blocks(conn, *, feed_age_s: float | None, sol_free: float, context_ready: bool,
                 brain_ok: bool, edge: float | None = None) -> list[str]:
    """Names of rails that block NEW live entries right now (exits are never blocked here)."""
    st = load_circuit(conn)
    blocks = []
    if st.kill_switch:
        blocks.append("kill_switch")
    if st.tripped:
        blocks.append("circuit")
    if st.entries_paused:
        blocks.append("paused")
    if feed_age_s is None or feed_age_s > config.STALE_FEED_S:
        blocks.append("stale_feed")
    if not context_ready:
        blocks.append("context_not_ready")
    if not brain_ok:
        blocks.append("brain_rates_out_of_range")
    if edge is None or edge <= config.EDGE_MIN:
        blocks.append("no_edge")
    if sol_free <= config.GAS_RESERVE_SOL + config.SIZE_MIN_SOL:
        blocks.append("reserve")
    if notional_24h(conn) >= config.NOTIONAL_CAP_SOL_24H:
        blocks.append("notional_cap")
    return blocks


def forced_exit_kind(position: dict, mark_price: float | None, last_swap_ts: float | None, now: float,
                     rvol: float = 0.0) -> str | None:
    """Reflex exits (operator design): the fly chooses what to eat; these decide when it stops.
    bitter   — unrealized return <= -EXIT_LOSS (rejection; sharp, no noise scaling).
    turned   — the position was >= +TURN_MARGIN at its peak and is now <= 0 (a winner turning into a loser).
    satisfied — satiety (Σ max(u,0) × dt over the holding period) >= SATIETY_TARGET: the longer and larger the
               gain, the stronger the urge to realise it.
    forced_stop (-HARD_STOP_FRAC) and forced_dead (no swaps for DEAD_BAG_HOURS) remain as backstops."""
    entry = position.get("entry_price")
    if mark_price is not None and entry:
        u = mark_price / float(entry) - 1.0
        if u <= -config.HARD_STOP_FRAC:
            return "forced_stop"
        if config.BRAIN_MODE == "policy":      # the trained policy owns exits; only the backstops remain
            if last_swap_ts is not None and now - last_swap_ts > config.DEAD_BAG_HOURS * 3600:
                return "forced_dead"
            return None
        if u <= -config.EXIT_LOSS:
            return "bitter"
        peak = float(position.get("peak_price") or entry)
        if peak >= float(entry) * (1.0 + config.TURN_MARGIN) and u <= 0.0:
            return "turned"
        if float(position.get("satiety") or 0.0) >= config.SATIETY_TARGET and u > config.FEE_ROUND_TRIP:
            return "satisfied"   # only a gain that clears the round-trip cost is a gain
    if last_swap_ts is not None and now - last_swap_ts > config.DEAD_BAG_HOURS * 3600:
        return "forced_dead"
    if last_swap_ts is None:
        opened = position.get("opened_at")
        if opened is not None and now - opened.timestamp() > config.DEAD_BAG_HOURS * 3600:
            return "forced_dead"
    return None


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
