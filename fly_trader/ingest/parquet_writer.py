"""VOC-style rolling Parquet writer (pattern from VOC capture_meteora_live.py Writer, :125-162).

Parquet has no append, so each flush writes one immutable snappy chunk under
``root/<name>/<YYYY-MM-DD>/<launch_ts>-<seq>.parquet``; readers glob the tree. A flush fires at
``flush_rows`` buffered rows or after ``flush_s`` seconds (whichever first; the caller drives the
timer via ``maybe_flush()``). Writes are crash-safe: the chunk is written to ``<file>.tmp`` and
atomically renamed, so a reader never sees a partial file and a hard kill loses at most one
flush window of rows. ``root/_status.json`` is updated on every flush; several writers (and the
capture daemon) share that file, so updates are merged under the writer's ``name`` key.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_STATUS_LOCK = threading.Lock()


def write_status(root: Path, key: str, payload: dict) -> Path:
    """Merge ``payload`` under ``key`` into ``root/_status.json`` (atomic replace)."""
    root = Path(root)
    path = root / "_status.json"
    with _STATUS_LOCK:
        root.mkdir(parents=True, exist_ok=True)
        current: dict = {}
        try:
            current = json.loads(path.read_text())
            if not isinstance(current, dict):
                current = {}
        except (FileNotFoundError, ValueError):
            current = {}
        current[key] = payload
        current["updated"] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2, default=str))
        os.replace(tmp, path)
    return path


def _row_date(value) -> str:
    """UTC date string for a row's timestamp field (datetime, epoch seconds or epoch ms)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)):
        secs = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc).strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Writer:
    """Rolling, date-partitioned Parquet chunk writer.

    ``schema`` fixes the column set and types (rows are dicts; missing keys become nulls). Without a
    schema, columns are the union of keys seen in the flushed batch and types are inferred.
    """

    def __init__(self, root: Path, name: str, flush_rows: int = 5000, flush_s: float = 30.0,
                 schema: pa.Schema | None = None, ts_field: str = "ts"):
        self.root = Path(root)
        self.name = name
        self.flush_rows = int(flush_rows)
        self.flush_s = float(flush_s)
        self.schema = schema
        self.ts_field = ts_field
        self.buf: list[dict] = []
        self.seq = 0
        self.last_flush = time.time()
        self.launch_ts = int(time.time())
        self.files_written = 0
        self.rows_written = 0
        self.rows_added = 0
        self.last_file: Path | None = None
        self.last_flush_at: datetime | None = None
        (self.root / self.name).mkdir(parents=True, exist_ok=True)

    # ---- buffering ----
    def add(self, row: dict) -> None:
        self.buf.append(row)
        self.rows_added += 1

    def add_many(self, rows) -> None:
        for r in rows:
            self.add(r)

    def due(self) -> bool:
        if not self.buf:
            return False
        return len(self.buf) >= self.flush_rows or (time.time() - self.last_flush) >= self.flush_s

    def maybe_flush(self) -> list[Path]:
        return self.flush() if self.due() else []

    # ---- writing ----
    def _table(self, rows: list[dict]) -> pa.Table:
        if self.schema is not None:
            cols = {f.name: [r.get(f.name) for r in rows] for f in self.schema}
            return pa.table(cols, schema=self.schema)
        names: list[str] = []
        for r in rows:
            for k in r:
                if k not in names:
                    names.append(k)
        return pa.table({n: [r.get(n) for r in rows] for n in names})

    def _next_path(self, day: str) -> Path:
        d = self.root / self.name / day
        d.mkdir(parents=True, exist_ok=True)
        while True:
            p = d / f"{self.launch_ts}-{self.seq:06d}.parquet"
            self.seq += 1
            if not p.exists():
                return p

    def flush(self) -> list[Path]:
        """Write every buffered row (grouped by UTC date of ``ts_field``). Returns the files written."""
        if not self.buf:
            self.last_flush = time.time()
            return []
        rows, self.buf = self.buf, []
        self.last_flush = time.time()
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[_row_date(r.get(self.ts_field))].append(r)
        written: list[Path] = []
        for day, rs in sorted(groups.items()):
            path = self._next_path(day)
            tmp = path.with_name(path.name + ".tmp")
            pq.write_table(self._table(rs), tmp, compression="snappy")
            os.replace(tmp, path)
            written.append(path)
            self.files_written += 1
            self.rows_written += len(rs)
            self.last_file = path
        self.last_flush_at = datetime.now(timezone.utc)
        write_status(self.root, self.name, self.status())
        return written

    def close(self) -> list[Path]:
        return self.flush()

    def status(self) -> dict:
        return {
            "name": self.name,
            "launch_ts": self.launch_ts,
            "buffered": len(self.buf),
            "rows_added": self.rows_added,
            "rows_written": self.rows_written,
            "files_written": self.files_written,
            "last_file": str(self.last_file) if self.last_file else None,
            "last_flush": self.last_flush_at.isoformat() if self.last_flush_at else None,
        }
