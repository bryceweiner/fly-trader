"""Reset the training state so a fresh runner never carries over a bad run.

Wiped (after being archived as CSV under data/pg_archive/reset_<ts>/): beats, beat_slots, brain_activity,
decisions, rewards, wealth_marks, synapse_updates, slot_visits, brain_snapshots, runs (except connectome
build/calibration provenance), and positions/orders/fills of paper and replay books. brain_state and the
circuit (peak wealth, kill switch, failure count) are cleared; replay clocks restart. The plasticity journal
and snapshot files are moved aside. NEVER touched: live-book positions/orders/fills, wallet events, tokens,
pools, the swap tape, API logs, events, ui_settings other than replay clocks.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..db.apilog import record_event
from ..db.connection import connect, transaction

log = logging.getLogger(__name__)

WIPE_TABLES = ["beats", "beat_slots", "brain_activity", "decisions", "rewards", "wealth_marks", "synapse_updates",
               "slot_visits"]
BOOK_TABLES = ["positions", "orders", "fills"]


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
            counts["brain_snapshots"] = _archive(conn, "brain_snapshots", "WHERE kind <> 'policy'", out_dir)
            for t in BOOK_TABLES:
                counts[t] = _archive(conn, t, "WHERE book <> 'live'", out_dir)
            counts["runs"] = _archive(conn, "runs", "WHERE kind NOT IN ('connectome_build','calibration')", out_dir)
        for t in WIPE_TABLES:
            conn.execute(f"TRUNCATE TABLE {t}")
        conn.execute("DELETE FROM brain_snapshots WHERE kind <> 'policy'")   # trained policy checkpoints survive a reset
        for t in BOOK_TABLES:
            conn.execute(f"DELETE FROM {t} WHERE book <> 'live'")
        conn.execute("DELETE FROM runs WHERE kind NOT IN ('connectome_build','calibration')")
        conn.execute("UPDATE brain_state SET live_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = live_snapshot_id) = 'policy' THEN live_snapshot_id END, "
                     "pending_snapshot_id = CASE WHEN (SELECT kind FROM brain_snapshots WHERE id = pending_snapshot_id) = 'policy' THEN pending_snapshot_id END, updated_at = now() WHERE singleton")
        conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false, kill_switch = false, kill_reason = NULL, peak_wealth = NULL, updated_at = now() WHERE id = 1")
        conn.execute("DELETE FROM ui_settings WHERE key LIKE 'replay_clock:%'")
        conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('training_reset', %s)", (f'{{"reason": "{reason}"}}',))
    for sub in ("journal", "snapshots"):
        src = config.BRAIN_DIR / sub
        if src.exists() and any(src.iterdir()):
            dst = config.BRAIN_DIR / f"{sub}_archive" / tag
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        src.mkdir(parents=True, exist_ok=True)
    record_event("info", "reset", f"training state reset ({reason})", {"archived_to": str(out_dir) if archive else None, "rows": counts})
    log.info("training state reset (%s): %s", reason, counts)
    return {"archived_to": str(out_dir) if archive else None, "rows": counts}
