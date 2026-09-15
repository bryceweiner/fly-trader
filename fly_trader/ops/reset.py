"""Reset the training state so a fresh runner never carries over a bad run.

Wiped (after being archived as CSV under data/pg_archive/reset_<ts>/): beats, decisions, wealth_marks, runs (except
connectome build provenance), non-model brain_snapshots and positions/orders/fills of the paper book. brain_state keeps
only model pointers; the selector session status restarts. Runs on every runner start (``RESET_ON_START``). NEVER touched: the circuit (kill switch, trip, failure
count, peak wealth — only rails.reset_circuit re-arms it), live-book positions/orders/fills, wallet events, tokens,
pools, the swap tape, API logs, events, other ui_settings.

``reset_training_stats`` is the training-side counterpart, run at the start of every training run: the previous runs'
walk-forward / iteration events and the console's training status are archived and cleared (models are kept).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..db.apilog import record_event
from ..db.connection import connect, transaction

log = logging.getLogger(__name__)

_LIVE_DECISIONS = ("SELECT decision_id FROM orders WHERE book = 'live' AND decision_id IS NOT NULL "
                   "UNION SELECT entry_decision_id FROM positions WHERE book = 'live' AND entry_decision_id IS NOT NULL "
                   "UNION SELECT exit_decision_id FROM positions WHERE book = 'live' AND exit_decision_id IS NOT NULL")
WIPE_TABLES = ["beats", "wealth_marks"]
BOOK_TABLES = ["positions", "orders", "fills"]
TRAINING_EVENTS = {"selector": "(source = 'selector' AND (message LIKE 'walk-forward%' OR message LIKE 'selector saved%'))",   # per regimen: a fly
                   "fly_selector": "source = 'fly_selector'"}                                               # run keeps the selector's


def _archive(conn, table: str, where: str, out_dir: Path) -> int:
    n = conn.execute(f"SELECT count(*) AS n FROM {table} {where}").fetchone()["n"]
    if n:
        with open(out_dir / f"{table}.csv", "w") as f:
            with conn.cursor() as cur:
                with cur.copy(f"COPY (SELECT * FROM {table} {where}) TO STDOUT WITH CSV HEADER") as copy:
                    for chunk in copy:
                        f.write(bytes(chunk).decode("utf-8"))
    return int(n)


def reset_training_state(reason: str = "runner start", archive: bool = True) -> dict:
    ts = datetime.now(timezone.utc)
    tag = ts.strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.PG_ARCHIVE_DIR / f"reset_{tag}"
    counts: dict[str, int] = {}
    with transaction() as conn:
        if archive:
            out_dir.mkdir(parents=True, exist_ok=True)
            for t in WIPE_TABLES:
                counts[t] = _archive(conn, t, "", out_dir)
            counts["brain_snapshots"] = _archive(conn, "brain_snapshots", "WHERE kind NOT IN ('selector','fly_selector')", out_dir)
            for t in BOOK_TABLES:
                counts[t] = _archive(conn, t, "WHERE book <> 'live'", out_dir)
            counts["runs"] = _archive(conn, "runs", "WHERE kind NOT IN ('connectome_build','calibration')", out_dir)
            counts["decisions"] = _archive(conn, "decisions", "WHERE id NOT IN (" + _LIVE_DECISIONS + ")", out_dir)
        for t in WIPE_TABLES:
            conn.execute(f"TRUNCATE TABLE {t}")
        conn.execute("DELETE FROM brain_snapshots WHERE kind NOT IN ('selector','fly_selector')")   # trained models survive a reset
        conn.execute("DELETE FROM decisions WHERE id NOT IN (" + _LIVE_DECISIONS + ")")   # live-book references stay intact
        for t in BOOK_TABLES:
            conn.execute(f"DELETE FROM {t} WHERE book <> 'live'")
        conn.execute("DELETE FROM runs WHERE kind NOT IN ('connectome_build','calibration')")
        conn.execute("UPDATE brain_state SET live_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = live_snapshot_id) IN ('selector','fly_selector') THEN live_snapshot_id END, "
                     "pending_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = pending_snapshot_id) IN ('selector','fly_selector') THEN pending_snapshot_id END, updated_at = now() WHERE singleton")
        conn.execute("DELETE FROM ui_settings WHERE key = 'selector_status'")
        conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('training_reset', %s)", (f'{{"reason": "{reason}"}}',))
    record_event("info", "reset", f"training state reset ({reason})", {"archived_to": str(out_dir) if archive else None, "rows": counts})
    log.info("training state reset (%s): %s", reason, counts)
    return {"archived_to": str(out_dir) if archive else None, "rows": counts}


def reset_training_stats(regimen: str, reason: str = "training start", archive: bool = True) -> dict:
    """Clear the previous runs' statistics of this training ``regimen`` (selector walk-forward, fly comparisons; the console's training status) so a new run's numbers are never shown next to an old run's.
    Other regimens' results and all trained models (brain_snapshots) are kept."""
    where = TRAINING_EVENTS[regimen]
    out_dir = config.PG_ARCHIVE_DIR / f"training_{regimen}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with transaction() as conn:
        n = 0
        if archive:
            out_dir.mkdir(parents=True, exist_ok=True)
            n = _archive(conn, "events", "WHERE " + where, out_dir)
        conn.execute("DELETE FROM events WHERE " + where)
        conn.execute("DELETE FROM ui_settings WHERE key = 'training_status'")
    log.info("training stats reset (%s): %d events", reason, n)
    return {"archived_to": str(out_dir) if archive else None, "events": n}
