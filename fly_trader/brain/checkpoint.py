"""Brain snapshots (plastic weights + encoder statistics) and their promotion.

brain_state.live_snapshot_id = the snapshot the runner last saved or loaded (continuity of sanctioned
online learning across restarts). promote-checkpoint <id> only sets pending_snapshot_id; the runner
consumes it at its next start (or at the next beat if --hot was requested). Nothing is promoted
automatically. `snapshot --kind` writes an events row the runner honours at its next beat.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)


def save_snapshot(W_KM: np.ndarray, encoder_state: dict, *, kind: str, run_id: str | None,
                  beat_id: int | None, note: str | None = None, conn=None) -> tuple[int, Path, str]:
    root = config.BRAIN_DIR / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    path = root / f"brain_{ts.strftime('%Y%m%dT%H%M%SZ')}_{kind}.npz"
    np.savez_compressed(path, W_KM=W_KM.astype(np.float32), encoder_state=json.dumps(encoder_state),
                        beat_id=beat_id or -1, ts=ts.isoformat(), kind=kind)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def _insert(c):
        row = c.execute(
            "INSERT INTO brain_snapshots (run_id, beat_id, path, sha256, kind, note) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (run_id, beat_id, str(path), sha, kind, note),
        ).fetchone()
        return int(row["id"])

    if conn is not None:
        sid = _insert(conn)
    else:
        with transaction() as c:
            sid = _insert(c)
    return sid, path, sha


def load_snapshot(snapshot_id: int) -> tuple[np.ndarray, dict, dict]:
    with transaction() as conn:
        row = conn.execute("SELECT * FROM brain_snapshots WHERE id = %s", (snapshot_id,)).fetchone()
    if not row:
        raise KeyError(f"snapshot {snapshot_id} not found")
    z = np.load(row["path"], allow_pickle=False)
    enc = json.loads(str(z["encoder_state"]))
    return z["W_KM"], enc, dict(row)


def set_live(snapshot_id: int, conn=None) -> None:
    sql = "UPDATE brain_state SET live_snapshot_id = %s, updated_at = now() WHERE singleton"
    if conn is not None:
        conn.execute(sql, (snapshot_id,))
    else:
        with transaction() as c:
            c.execute(sql, (snapshot_id,))


def promote(snapshot_id: int, hot: bool = False) -> None:
    with transaction() as conn:
        row = conn.execute("SELECT id, path FROM brain_snapshots WHERE id = %s", (snapshot_id,)).fetchone()
        if not row:
            raise SystemExit(f"snapshot {snapshot_id} not found")
        conn.execute("UPDATE brain_state SET pending_snapshot_id = %s, updated_at = now() WHERE singleton", (snapshot_id,))
        conn.execute("UPDATE brain_snapshots SET promoted_at = now(), promoted_by = 'operator' WHERE id = %s", (snapshot_id,))
    record_event("info", "checkpoint", "snapshot promoted (pending)", {"snapshot_id": snapshot_id, "hot": hot})
    print(f"pending_snapshot_id={snapshot_id} hot={hot}; the runner applies it at its next {'beat' if hot else 'start'}")


def snapshot_from_live(kind: str = "manual") -> str:
    record_event("info", "checkpoint", "snapshot_request", {"kind": kind})
    return f"snapshot request ({kind}) queued; the runner writes it at its next beat"


def pending(conn) -> int | None:
    row = conn.execute("SELECT pending_snapshot_id FROM brain_state WHERE singleton").fetchone()
    return row["pending_snapshot_id"] if row else None


def consume_pending(conn) -> int | None:
    pid = pending(conn)
    if pid is not None:
        conn.execute("UPDATE brain_state SET live_snapshot_id = %s, pending_snapshot_id = NULL, updated_at = now() WHERE singleton", (pid,))
    return pid


def live_id(conn) -> int | None:
    row = conn.execute("SELECT live_snapshot_id FROM brain_state WHERE singleton").fetchone()
    return row["live_snapshot_id"] if row else None
