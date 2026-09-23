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
from ..logging_setup import scrub, setup, tail

log = logging.getLogger(__name__)


def _entry(name: str):
    if name == "discover":
        from ..ingest import discovery
        return discovery.run_forever
    if name == "runner":
        from ..agent import runner
        return runner.main
    if name == "train":                        # the training pipeline: selector, then the fly imitating it, every 7 days (train/pipeline.py)
        from ..train import pipeline
        return pipeline.main
    if name == "replay":
        from ..ingest import replay_pull
        return replay_pull.main
    if name == "pumpstream":
        from ..ingest import pumpstream
        return pumpstream.main
    if name == "kalshi_history":               # the Kalshi corpus: settled markets, candles, trades, feature rows (kalshi/history.py)
        from ..kalshi import history
        return history.main
    if name == "kalshi_stream":                # the Kalshi live feed (kalshi/stream.py)
        from ..kalshi import stream
        return stream.main
    if name == "kalshi_runner":                # the Kalshi trading engine (kalshi/engine.py)
        from ..kalshi import engine
        return engine.main
    if name == "kalshi_train":                 # the Kalshi training pipeline (kalshi/pipeline.py)
        from ..kalshi import pipeline
        return pipeline.main
    raise KeyError(name)


WORKERS = ("discover", "runner", "train", "replay", "pumpstream", "kalshi_stream", "kalshi_runner", "kalshi_history", "kalshi_train")
REPLACE_WAIT_S = 20.0        # how long a new console waits for the one it replaces to finish winding down (its workers flush on stop)


class Supervisor:
    def __init__(self):
        self.lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        self.stops: dict[str, threading.Event] = {}
        self.started_at: dict[str, datetime] = {}
        self.stopped_at: dict[str, datetime] = {}
        self.errors: dict[str, str] = {}
        self.row_ids: dict[str, int] = {}
        self.autostarted = False
        self.pid = os.getpid()
        setup("console")
        with transaction() as conn:  # thread rows left by a previous console process are stale
            psutil = __import__("psutil")
            rows = conn.execute("SELECT id, pid FROM processes WHERE stopped_at IS NULL AND cmd[1] = 'thread'").fetchall()
            others = {int(r["pid"]) for r in rows if int(r["pid"]) != self.pid}
            for _ in range(int(REPLACE_WAIT_S / 0.5)):     # a console being replaced is still winding its workers down: wait for it,
                if not others & {p.pid for p in psutil.process_iter()}:     # or its rows look live and the workers never start
                    break
                time.sleep(0.5)
            live_pids = {p.pid for p in psutil.process_iter()}
            for r in rows:
                # A row with our own pid is a previous console's too: this one has registered nothing yet. In a container
                # the console is always pid 1, so without this every restart refused to register its workers.
                if int(r["pid"]) == self.pid or int(r["pid"]) not in live_pids:
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
            try:
                fn = _entry(name)                        # imports first: a broken import must not leave a registered row
            except Exception as e:
                raise RuntimeError(f"cannot load {name}: {type(e).__name__}: {scrub(str(e))[:200]}") from e
            try:
                with transaction() as conn:
                    row = conn.execute("INSERT INTO processes (name, pid, cmd, log_path, started_by) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                                       (name, self.pid, ["thread", name], f"logs/{name}.log", started_by)).fetchone()
            except Exception as e:                       # e.g. processes_live_uniq: another console (or a stale row) owns this worker
                raise RuntimeError(f"cannot register {name}: {type(e).__name__}: {scrub(str(e))[:200]}") from e
            ev = threading.Event()
            t = threading.Thread(target=self._run, args=(name, fn, ev), name=name, daemon=True)
            self.row_ids[name] = int(row["id"]); self.stops[name] = ev; self.threads[name] = t
            try:
                t.start()
            except Exception as e:
                with transaction() as conn:
                    conn.execute("UPDATE processes SET stopped_at = now(), exit_code = -1 WHERE id = %s", (int(row["id"]),))
                raise RuntimeError(f"cannot start {name}: {type(e).__name__}: {e}") from e
            self.errors.pop(name, None)
            self.started_at[name] = datetime.now(timezone.utc)
            self.stopped_at.pop(name, None)
        record_event("info", "console", f"started {name} thread", {"pid": self.pid, "by": started_by})
        return True

    def _run(self, name: str, fn, ev: threading.Event) -> None:
        code = 0
        try:
            fn(stop_event=ev)
        except Exception as e:
            code = 1
            self.errors[name] = scrub(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            log.exception("%s thread crashed", name)
            record_event("error", "console", f"{name} thread crashed: {type(e).__name__}: {e}")
        finally:
            self.stopped_at[name] = datetime.now(timezone.utc)
            try:
                with transaction() as conn:
                    conn.execute("UPDATE processes SET stopped_at = now(), exit_code = %s WHERE id = %s", (code, self.row_ids.get(name)))
            except Exception:
                pass

    def request_stop(self, name: str, by: str = "console") -> bool:
        """Ask a worker to stop without waiting (the console's buttons); the audit row is written here."""
        ev = self.stops.get(name)
        if ev is None or not self.alive(name):
            return False
        ev.set()
        record_event("info", "console", f"stop requested for {name} thread", {"pid": self.pid, "by": by})
        return True

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
        self.autostarted = True          # process-level: the console runs autostart once, not once per browser tab
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
                except Exception as e:
                    log.warning("autostart %s: %s", n, scrub(str(e)))
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
    from ..db import schema
    schema.apply_schema()           # idempotent: tables added to the schema exist before any worker uses them
    sup = get_supervisor()
    try:
        started = sup.autostart()
    except Exception as e:
        started = []; log.exception("autostart failed: %s", scrub(str(e)))
    log.info("console pid=%d autostarted=%s", os.getpid(), started)
    app = str(config.REPO_ROOT / "fly_trader" / "ui" / "app.py")
    from streamlit.web import bootstrap
    flag_options = {"server.port": port, "server.headless": True, "browser.gatherUsageStats": False,
                    "server.fileWatcherType": "none", "logger.level": "warning",
                    "server.enableStaticServing": True}       # ui/static/brain: the 3D view's neuron geometry, fetched once

    bootstrap.load_config_options(flag_options)      # the CLI does this before run(); without it port/headless/watcher flags are ignored
    try:
        bootstrap.run(app, False, [], flag_options)
    finally:
        sup.stop_all()
