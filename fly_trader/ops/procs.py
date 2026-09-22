"""Worker processes: started detached (their own session) so Streamlit reruns/restarts never kill them.
PIDs and commands are recorded in `processes`; liveness is checked with psutil, not trusted from the row."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .. import config
from ..db.apilog import record_event
from ..db.connection import connect, transaction

WORKERS: dict[str, list[str]] = {
    "discover": ["discover"],
    "runner": ["run"],
    "train": ["train-selector"],
    "replay": ["replay-pull"],
    "pumpstream": ["pumpstream"],
    "kalshi_stream": ["kalshi-stream"],
    "kalshi_runner": ["kalshi-run"],
    "kalshi_history": ["kalshi-history"],
    "kalshi_train": ["kalshi-train"],
}
PYTHON = str(config.REPO_ROOT / ".venv" / "bin" / "python")


def _cmd(name: str, extra: list[str] | None = None) -> list[str]:
    return [PYTHON, "-m", "fly_trader", *WORKERS[name], *(extra or [])]


def _is_ours(pid: int, name: str) -> bool:
    try:
        p = psutil.Process(pid)
        cl = " ".join(p.cmdline())
        return "fly_trader" in cl and WORKERS[name][0] in cl and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def reconcile(conn) -> None:
    """Mark dead EXTERNAL worker processes stopped. Thread rows (cmd[0] == 'thread') belong to the console's
    Supervisor and are left alone."""
    rows = conn.execute("SELECT id, name, pid, cmd FROM processes WHERE stopped_at IS NULL").fetchall()
    for r in rows:
        if r["cmd"] and r["cmd"][0] == "thread":
            continue
        if not _is_ours(r["pid"], r["name"]):
            conn.execute("UPDATE processes SET stopped_at = now(), exit_code = COALESCE(exit_code, -1) WHERE id = %s", (r["id"],))


def alive(name: str) -> dict | None:
    with transaction() as conn:
        reconcile(conn)
        r = conn.execute("SELECT * FROM processes WHERE name = %s AND stopped_at IS NULL AND cmd[1] <> 'thread' "
                         "ORDER BY started_at DESC LIMIT 1", (name,)).fetchone()
        return dict(r) if r else None


def start(name: str, extra: list[str] | None = None, started_by: str = "console") -> int:
    if name not in WORKERS:
        raise ValueError(f"unknown worker {name}")
    if (a := alive(name)) is not None:
        return int(a["pid"])
    with transaction() as conn:   # a console thread already running this worker: never launch a second copy
        t = conn.execute("SELECT pid FROM processes WHERE name = %s AND stopped_at IS NULL AND cmd[1] = 'thread'", (name,)).fetchone()
    if t:
        raise RuntimeError(f"{name} is already running as a console thread (pid {t['pid']}); stop it there first")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / f"{name}.out.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = _cmd(name, extra)
    with open(log_path, "ab") as out:
        proc = subprocess.Popen(cmd, cwd=str(config.REPO_ROOT), stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO processes (name, pid, cmd, log_path, started_by) VALUES (%s,%s,%s,%s,%s)",
                         (name, proc.pid, cmd, str(log_path), started_by))
    except Exception as e:        # e.g. processes_live_uniq raced by another starter: do not leave an unregistered worker running
        proc.terminate()
        raise RuntimeError(f"cannot register {name}: {type(e).__name__}: {e}") from e
    record_event("info", "procs", f"started {name}", {"pid": proc.pid, "cmd": cmd})
    return proc.pid


def stop(name: str, grace_s: float = 30.0) -> bool:
    a = alive(name)
    if a is None:
        return False
    pid = int(a["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s and _is_ours(pid, name):
        time.sleep(0.5)
    killed = False
    if _is_ours(pid, name):
        os.kill(pid, signal.SIGKILL)
        killed = True
    with transaction() as conn:
        conn.execute("UPDATE processes SET stopped_at = now(), exit_code = %s WHERE id = %s", (-9 if killed else 0, a["id"]))
    record_event("info", "procs", f"stopped {name}" + (" (SIGKILL)" if killed else ""), {"pid": pid})
    return True


def tail_log(name: str, lines: int = 60) -> str:
    p = config.LOG_DIR / f"{name}.out.log"
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 64000))
        data = f.read().decode("utf-8", "replace")
    return "\n".join(data.splitlines()[-lines:])


def run_streamlit(port: int = 8501) -> None:
    app = config.REPO_ROOT / "fly_trader" / "ui" / "app.py"
    cmd = [PYTHON, "-m", "streamlit", "run", str(app), "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    os.execv(PYTHON, cmd)


def worker_cli(action: str, name: str | None) -> None:
    if action == "status":
        for n in WORKERS:
            a = alive(n)
            print(f"{n:10s} {'alive pid=' + str(a['pid']) if a else 'stopped'}")
        return
    if not name:
        raise SystemExit("worker name required")
    if action == "start":
        print(f"{name} pid={start(name, started_by='cli')}")
    elif action == "stop":
        print(f"{name} stopped={stop(name)}")
