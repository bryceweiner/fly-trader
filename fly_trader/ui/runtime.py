"""Which supervisor the console talks to. `fly-trader ui` hosts the workers (the process-wide Supervisor); `fly-trader ui
--observe` serves the same dashboard without hosting anything — for a preview next to a running console, where a second
Supervisor would wait on, and fight over, the first one's `processes` rows. The observer reads worker state from those
rows and refuses every control."""
from __future__ import annotations

import collections
from datetime import datetime, timezone

from .. import config

HOSTS_WORKERS = True


class ObserverMode(RuntimeError):
    pass


class Observer:
    """The Supervisor's read side, from the `processes` rows of whichever console hosts the workers."""
    autostarted = True

    def __init__(self):
        import os
        self.pid = os.getpid()

    def _rows(self) -> dict[str, dict]:
        from ..db.queries import q
        import psutil
        rows = q("SELECT DISTINCT ON (name) name, pid, started_at, stopped_at, exit_code, cmd FROM processes "
                 "WHERE cmd[1] = 'thread' ORDER BY name, started_at DESC")
        out = {}
        for r in rows:
            alive = r["stopped_at"] is None and psutil.pid_exists(int(r["pid"]))
            out[r["name"]] = {**r, "alive": alive}
        return out

    def status(self) -> dict[str, dict]:
        from ..ops.supervisor import WORKERS
        rows = self._rows()
        return {n: {"alive": bool(rows.get(n, {}).get("alive")), "started_at": rows.get(n, {}).get("started_at"),
                    "stopped_at": rows.get(n, {}).get("stopped_at"),
                    "error": (f"exit code {rows[n]['exit_code']}" if n in rows and rows[n].get("exit_code") not in (None, 0) else None),
                    "stopping": False} for n in WORKERS}

    def alive(self, name: str) -> bool:
        return bool(self.status().get(name, {}).get("alive"))

    def external(self) -> dict[str, dict]:
        """CLI-started worker processes, read only (procs.alive would also reconcile, i.e. write, the rows)."""
        from ..db.queries import q
        from ..ops.procs import _is_ours
        out = {}
        for r in q("SELECT * FROM processes WHERE stopped_at IS NULL AND cmd[1] <> 'thread' ORDER BY started_at"):
            if _is_ours(r["pid"], r["name"]):
                out[r["name"]] = r
        return out

    def log_tail(self, name: str, n: int = 60) -> str:
        return file_tail(name, n)

    def _refuse(self, *_a, **_k):
        raise ObserverMode("this console only observes (fly-trader ui --observe): start and stop workers from the console that hosts them")

    start = request_stop = stop = _refuse


_OBSERVER: Observer | None = None


def sup():
    """The Supervisor (hosting) or the Observer (preview)."""
    global _OBSERVER
    if HOSTS_WORKERS:
        from ..ops.supervisor import get_supervisor
        return get_supervisor()
    if _OBSERVER is None:
        _OBSERVER = Observer()
    return _OBSERVER


def file_tail(name: str, n: int = 200, max_bytes: int = 1 << 20) -> str:
    """The last ``n`` lines of logs/<name>.log (the rotating file every worker writes), read from the end."""
    p = config.LOG_DIR / f"{name}.log"
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", "replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]                     # the first line read mid-way is partial
    return "\n".join(collections.deque(lines, maxlen=n))


def now() -> datetime:
    return datetime.now(timezone.utc)
