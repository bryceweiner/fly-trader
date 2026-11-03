"""Logs: the rotating JSON-lines file every worker writes (logs/<name>.log, fly_trader/logging_setup.py), tailed live.

Only the workers' and the console's own logs can be read: the name is checked against that list, never joined into a
path from the browser. Lines are already scrubbed of secrets by the logger."""
from __future__ import annotations

from datetime import datetime, timezone

from ... import config
from .. import runtime
from ..hub import TopicError, topic

N_DEFAULT, N_MAX = 500, 5000


def names() -> list[str]:
    """Every log the page may read: each supervised worker, the console that hosts them, and an observer console."""
    from ...ops.supervisor import WORKERS
    return [*WORKERS, "console", "console-observer"]


def _name(params: dict) -> str:
    n = params.get("name")
    if not isinstance(n, str) or n not in names():
        raise TopicError(f"no log named {n!r}: pick one of {', '.join(names())}")
    return n


def _info(name: str) -> dict:
    p = config.LOG_DIR / f"{name}.log"
    try:
        st = p.stat()
    except OSError:
        return {"name": name, "exists": False, "size": 0, "mtime": None}
    return {"name": name, "exists": True, "size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)}


@topic("logs.files", every=10)
def files(params: dict) -> dict:
    return {"files": [_info(n) for n in names()], "dir": str(config.LOG_DIR)}


@topic("logs.tail", every=1)
def tail(params: dict) -> dict:
    """The newest ``n`` lines of one log, oldest first (raw JSON lines: the page parses and filters them)."""
    name = _name(params)
    try:
        n = int(params.get("n") or N_DEFAULT)
    except (TypeError, ValueError):
        raise TopicError("n must be a number of lines")
    n = max(10, min(N_MAX, n))
    text = runtime.file_tail(name, n, max_bytes=4 << 20)
    return {**_info(name), "n": n, "lines": text.splitlines() if text else []}
