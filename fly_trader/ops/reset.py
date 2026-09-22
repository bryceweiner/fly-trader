"""Reset the training state so a fresh runner never carries over a bad run — without touching the race or the fly.

``reset_training_state`` (every runner start, ``RESET_ON_START``) wipes, after archiving them as CSV under
data/pg_archive/reset_<ts>/, what belongs to no book that must persist: positions/orders/fills and wealth marks of any
book but the live book and the two race books (``PRESERVED_BOOKS``), decisions and beats of runs that are not the race
sessions' (``RACE_RUN_KINDS``), those runs, and brain snapshots that are not models (``MODEL_KINDS``). NEVER touched:
the race books (paper_selector, paper_fly) and the live book with their runs, beats and decisions; the plastic fly's
tables (fly_*) and state file; the circuit (kill switch, trip, failure count, peak wealth — only rails.reset_circuit
re-arms it); wallet events, tokens, pools, market data, API logs, events, other ui_settings.

``reset_fly`` is the explicit console action: the plastic fly starts again from its bootstrap (its book, its learned
KC→MBON changes, scored minutes, calibrations, rollbacks, snapshots and its recorded network activity are archived and
cleared; the bootstrap is kept). ``reset_training_state`` never touches the activity files either.

``reset_training_stats`` is the training-side counterpart, run at the start of every training run: the previous runs'
walk-forward / iteration events and the console's training status are archived and cleared (models are kept).
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)

PRESERVED_BOOKS = ("live", "paper_selector", "paper_fly")
RACE_RUN_KINDS = ("selector", "fly")
MODEL_KINDS = ("selector", "fly_selector", "fly_plastic")
FLY_BOOK, FLY_RUN_KIND = "paper_fly", "fly"
FLY_TABLES = ["fly_scored", "fly_calibrations", "fly_updates", "fly_rollbacks"]
BOOK_TABLES = ["positions", "orders", "fills"]
TRAINING_EVENTS = {"selector": "(source = 'selector' AND (message LIKE 'walk-forward%' OR message LIKE 'selector saved%'))",   # per regimen: a fly
                   "fly_selector": "source = 'fly_selector'"}                                               # run keeps the selector's


def _sql_list(xs) -> str:
    return "(" + ",".join(f"'{x}'" for x in xs) + ")"


_BOOKS, _KINDS, _RACE = _sql_list(PRESERVED_BOOKS), _sql_list(MODEL_KINDS), _sql_list(RACE_RUN_KINDS)
_RACE_RUNS = f"SELECT run_id FROM runs WHERE kind IN {_RACE}"
_KEPT_DECISIONS = (f"SELECT decision_id FROM orders WHERE book IN {_BOOKS} AND decision_id IS NOT NULL "
                   f"UNION SELECT entry_decision_id FROM positions WHERE book IN {_BOOKS} AND entry_decision_id IS NOT NULL "
                   f"UNION SELECT exit_decision_id FROM positions WHERE book IN {_BOOKS} AND exit_decision_id IS NOT NULL "
                   f"UNION SELECT id FROM decisions WHERE run_id IN ({_RACE_RUNS})")
_LIVE_DECISIONS = ("SELECT decision_id FROM orders WHERE book = 'live' AND decision_id IS NOT NULL "
                   "UNION SELECT entry_decision_id FROM positions WHERE book = 'live' AND entry_decision_id IS NOT NULL "
                   "UNION SELECT exit_decision_id FROM positions WHERE book = 'live' AND exit_decision_id IS NOT NULL")


def _archive(conn, table: str, where: str, out_dir: Path) -> int:
    n = conn.execute(f"SELECT count(*) AS n FROM {table} {where}").fetchone()["n"]
    if n:
        with open(out_dir / f"{table}.csv", "w") as f:
            with conn.cursor() as cur:
                with cur.copy(f"COPY (SELECT * FROM {table} {where}) TO STDOUT WITH CSV HEADER") as copy:
                    for chunk in copy:
                        f.write(bytes(chunk).decode("utf-8"))
    return int(n)


def _wipe(conn, plan: list[tuple[str, str]], out_dir: Path | None) -> dict[str, int]:
    """Archive (when ``out_dir``) then delete each (table, WHERE clause); in order, so later clauses may read earlier tables."""
    counts: dict[str, int] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for t, where in plan:
            counts[t] = counts.get(t, 0) + _archive(conn, t, where, out_dir)
    for t, where in plan:
        conn.execute(f"DELETE FROM {t} {where}")
    return counts


def reset_training_state(reason: str = "runner start", archive: bool = True) -> dict:
    out_dir = config.PG_ARCHIVE_DIR / f"reset_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    plan = [("decisions", f"WHERE id NOT IN ({_KEPT_DECISIONS})"),
            ("beats", f"WHERE run_id IS NULL OR run_id NOT IN ({_RACE_RUNS})"),
            ("wealth_marks", f"WHERE book NOT IN {_BOOKS}"),
            ("brain_snapshots", f"WHERE kind NOT IN {_KINDS}"),
            *[(t, f"WHERE book NOT IN {_BOOKS}") for t in BOOK_TABLES],
            ("runs", f"WHERE kind NOT IN ('connectome_build','calibration') AND kind NOT IN {_RACE}")]
    with transaction() as conn:
        counts = _wipe(conn, plan, out_dir if archive else None)
        conn.execute(f"UPDATE brain_state SET live_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = live_snapshot_id) IN {_KINDS} THEN live_snapshot_id END, "
                     f"pending_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = pending_snapshot_id) IN {_KINDS} THEN pending_snapshot_id END, updated_at = now() WHERE singleton")
        conn.execute("DELETE FROM ui_settings WHERE key = 'selector_status'")
        conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('training_reset', %s)", (f'{{"reason": "{reason}"}}',))
    record_event("info", "reset", f"training state reset ({reason})", {"archived_to": str(out_dir) if archive else None, "rows": counts})
    log.info("training state reset (%s): %s", reason, counts)
    return {"archived_to": str(out_dir) if archive else None, "rows": counts}


def fly_state_dir() -> Path:
    return config.BRAIN_DIR / "plastic"


def activity_dir() -> Path:
    """The fly's per-minute network activity for the console's brain view (``brain/activity.py``)."""
    return config.BRAIN_DIR / "activity"


def reset_fly(reason: str = "operator", archive: bool = True) -> dict:
    """The plastic fly starts again from its bootstrap: its paper book, sessions, learned state and statistics are
    archived and cleared. The live book, the selector's book and the bootstrap snapshots are kept."""
    out_dir = config.PG_ARCHIVE_DIR / f"reset_fly_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    fly_runs = f"SELECT run_id FROM runs WHERE kind = '{FLY_RUN_KIND}'"
    plan = [("decisions", f"WHERE run_id IN ({fly_runs}) AND id NOT IN ({_LIVE_DECISIONS})"),
            ("beats", f"WHERE run_id IN ({fly_runs})"),
            ("wealth_marks", f"WHERE book = '{FLY_BOOK}'"),
            *[(t, f"WHERE book = '{FLY_BOOK}'") for t in BOOK_TABLES],
            ("brain_snapshots", "WHERE kind = 'fly_plastic'"),
            ("runs", f"WHERE kind = '{FLY_RUN_KIND}'"),
            *[(t, "") for t in FLY_TABLES]]
    with transaction() as conn:
        counts = _wipe(conn, plan, out_dir if archive else None)
        conn.execute("DELETE FROM book_state WHERE book = %s", (FLY_BOOK,))
        conn.execute("DELETE FROM ui_settings WHERE key = 'fly_status'")
    for d, name in ((fly_state_dir(), "plastic"), (activity_dir(), "activity")):     # its learned state and its recorded activity
        if d.exists():
            if archive:
                out_dir.mkdir(parents=True, exist_ok=True); shutil.move(str(d), str(out_dir / name))
            else:
                shutil.rmtree(d)
    record_event("warning", "reset", f"plastic fly reset to its bootstrap ({reason})", {"archived_to": str(out_dir) if archive else None, "rows": counts})
    log.info("plastic fly reset (%s): %s", reason, counts)
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
