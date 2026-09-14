"""In-process supervisor: every worker is a thread inside the console process. Nothing is spawned.

The Supervisor is a process-wide singleton (get_supervisor()) shared by `fly-trader ui` (which
starts autostart workers before handing the main thread to Streamlit) and by the Streamlit script
runs (which import the same module). Each worker runs its ordinary entry point with a stop event;
stopping sets the event and joins. Start/stop are recorded in `processes` (pid = this process,
cmd = ['thread', name]) and `events`, so the audit trail is unchanged.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone

from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup, tail

log = logging.getLogger(__name__)


def _entry(name: str):
    if name == "discover":
        from ..ingest import discovery
        return discovery.run_forever
    if name == "capture":
        from ..ingest import capture
        return capture.main
    if name == "runner":
        from ..agent import runner
        return runner.main
    if name == "train":
        from ..train import ppo
        import json as _json
        def _train(stop_event=None):
            setup("train")   # log file + ring buffer keyed by this thread's name
            with transaction() as conn:
                r = conn.execute("SELECT value FROM ui_settings WHERE key = 'training_params'").fetchone()
            params = (r["value"] if (r and isinstance(r["value"], dict)) else _json.loads((r or {}).get("value") or "{}")) if r else {}
            ppo.main(iterations=params.get("iterations", 24), window=params.get("window", 400), eval_every=params.get("eval_every", 4),
                     imitate_epochs=params.get("imitate", 4), subgraph=params.get("subgraph"), init_from=params.get("init_from"), stop_event=stop_event)
        return _train
    if name == "corpus":
        from ..ingest import corpus_pull
        return corpus_pull.main
    if name == "replay":
        from ..ingest import replay_pull
        return replay_pull.main
    raise KeyError(name)


WORKERS = ("discover", "capture", "runner", "train", "corpus", "replay")


class Supervisor:
    def __init__(self):
        self.lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        self.stops: dict[str, threading.Event] = {}
        self.started_at: dict[str, datetime] = {}
        self.stopped_at: dict[str, datetime] = {}
        self.errors: dict[str, str] = {}
        self.row_ids: dict[str, int] = {}
        self.pid = os.getpid()
        setup("console")
        with transaction() as conn:  # thread rows left by a previous console process are stale
            live_pids = {p.pid for p in __import__("psutil").process_iter()}
            rows = conn.execute("SELECT id, pid FROM processes WHERE stopped_at IS NULL AND cmd[1] = 'thread'").fetchall()
            for r in rows:
                if int(r["pid"]) != self.pid and int(r["pid"]) not in live_pids:
                    conn.execute("UPDATE processes SET stopped_at = now(), exit_code = -1 WHERE id = %s", (r["id"],))

    # ---- queries ----
    def alive(self, name: str) -> bool:
        t = self.threads.get(name)
        return bool(t and t.is_alive())

    def status(self) -> dict[str, dict]:
        out = {}
        for n in WORKERS:
            out[n] = {"alive": self.alive(n), "started_at": self.started_at.get(n), "stopped_at": self.stopped_at.get(n),
                      "error": self.errors.get(n), "stopping": bool(self.stops.get(n) and self.stops[n].is_set() and self.alive(n))}
        return out

    def external(self) -> dict[str, dict]:
        """Worker processes running OUTSIDE this app (from the CLI); the console refuses to double-run them."""
        from . import procs
        out = {}
        for n in WORKERS:
            a = procs.alive(n)
            if a and int(a["pid"]) != self.pid:
                out[n] = a
        return out

    # ---- control ----
    def start(self, name: str, started_by: str = "console") -> bool:
        if name not in WORKERS:
            raise KeyError(name)
        with self.lock:
            if self.alive(name):
                return False
            if name in self.external():
                raise RuntimeError(f"{name} is already running as an external process; stop it first")
            ev = threading.Event()
            self.stops[name] = ev
            self.errors.pop(name, None)
            fn = _entry(name)
            t = threading.Thread(target=self._run, args=(name, fn, ev), name=name, daemon=True)
            self.threads[name] = t
            self.started_at[name] = datetime.now(timezone.utc)
            self.stopped_at.pop(name, None)
            with transaction() as conn:
                row = conn.execute("INSERT INTO processes (name, pid, cmd, log_path, started_by) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                                   (name, self.pid, ["thread", name], f"logs/{name}.log", started_by)).fetchone()
                self.row_ids[name] = int(row["id"])
            t.start()
        record_event("info", "console", f"started {name} thread", {"pid": self.pid, "by": started_by})
        return True

    def _run(self, name: str, fn, ev: threading.Event) -> None:
        code = 0
        try:
            fn(stop_event=ev)
        except Exception as e:
            code = 1
            self.errors[name] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
            log.exception("%s thread crashed", name)
            record_event("error", "console", f"{name} thread crashed: {type(e).__name__}: {e}")
        finally:
            self.stopped_at[name] = datetime.now(timezone.utc)
            try:
                with transaction() as conn:
                    conn.execute("UPDATE processes SET stopped_at = now(), exit_code = %s WHERE id = %s", (code, self.row_ids.get(name)))
            except Exception:
                pass

    def stop(self, name: str, grace_s: float = 90.0) -> bool:
        t = self.threads.get(name)
        ev = self.stops.get(name)
        if not t or not t.is_alive() or ev is None:
            return False
        ev.set()
        t.join(grace_s)
        still = t.is_alive()
        record_event("info", "console", f"stop requested for {name}" + (" (still finishing)" if still else " (stopped)"))
        return not still

    def stop_all(self) -> None:
        for n in WORKERS:
            if self.alive(n):
                self.stops[n].set()
        for n in WORKERS:
            t = self.threads.get(n)
            if t and t.is_alive():
                t.join(120)

    def autostart(self) -> list[str]:
        with transaction() as conn:
            r = conn.execute("SELECT value FROM ui_settings WHERE key = 'autostart'").fetchone()
        if not r:
            return []
        import json
        val = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")
        started = []
        for n in WORKERS:
            if val.get(n) and not self.alive(n):
                try:
                    if self.start(n, started_by="autostart"):
                        started.append(n)
                except RuntimeError as e:
                    log.warning("autostart %s: %s", n, e)
        return started

    def log_tail(self, name: str, n: int = 60) -> str:
        return tail(name, n)


_SUP: Supervisor | None = None
_SUP_LOCK = threading.Lock()


def get_supervisor() -> Supervisor:
    global _SUP
    with _SUP_LOCK:
        if _SUP is None:
            _SUP = Supervisor()
        return _SUP


def run_console(port: int = 8501) -> None:
    """`fly-trader ui`: start autostart workers in THIS process, then run Streamlit in the main thread."""
    from .. import config
    sup = get_supervisor()
    started = sup.autostart()
    log.info("console pid=%d autostarted=%s", os.getpid(), started)
    app = str(config.REPO_ROOT / "fly_trader" / "ui" / "app.py")
    from streamlit.web import bootstrap
    flag_options = {"server.port": port, "server.headless": True, "browser.gatherUsageStats": False,
                    "server.fileWatcherType": "none", "logger.level": "warning"}
    try:
        bootstrap.run(app, False, [], flag_options)
    finally:
        sup.stop_all()
