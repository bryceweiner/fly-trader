"""Processes: every worker thread with what it does, its state, start/stop, autostart and recent activity; the event log.

The workers are threads inside the console process (ops/supervisor.py). Every control here goes through ``runtime.sup()``,
so an observer console (``fly-trader ui --observe``) refuses them all; the guards the page shows by disabling a button
(already running, running outside the console, already stopping) are enforced again in the actions."""
from __future__ import annotations

import json
import threading
from contextlib import nullcontext

from ... import config
from ...db.apilog import record_event
from ...db.connection import transaction
from ...db.queries import q, q1
from ...logging_setup import scrub
from ...ops.supervisor import WORKERS
from .. import runtime
from ..hub import ActionError, TopicError, action, topic
from ..state import ROLE_LABEL, WORKER_INFO, WORKER_TYPE, jv, worker_order

GROUPS = ("memecoins", "prediction markets", "system")      # the menu's section order
LEVELS = ("info", "warning", "error")
ENGINES = ("runner", "kalshi_runner")                      # a Safety reset must never run under them (safety.no_reset)
STOP_NOTE = {   # what stopping costs, beyond "it finishes its current step first"
    "runner": "While it is stopped nothing is bought or sold: open positions stay open until it runs again.",
    "pumpstream": "Without it the Solana feed goes stale within 3 minutes and the trading engine holds: no Solana entries or exits until it runs again.",
    "rh_stream": "Without it the Robinhood Chain feed goes stale within 3 minutes and the engine's Robinhood books hold: no entries or exits there until it runs again.",
    "kalshi_runner": "No new Kalshi entries while it is stopped; open positions still settle when their markets close.",
    "kalshi_stream": "Without it the Kalshi feed goes stale and the Kalshi engine holds: no entries until it runs again.",
    "vault": "The vault's jobs (flows, claims, the trading float, NAV, weekly settlement, public stats) pause until it runs again.",
}
_AUTOSTART = threading.Lock()                              # the autostart map is read-modify-written: one writer at a time


def _title(name: str) -> str:
    return WORKER_INFO.get(name, (name,))[0]


def _state(s: dict, external: bool = False) -> str:
    """running / stopping / crashed / stopped for a console thread; ``external`` when only a CLI process runs the worker."""
    if s.get("alive"):
        return "stopping" if s.get("stopping") else "running"
    return "external" if external else "crashed" if s.get("error") else "stopped"


def _start_note(name: str) -> str | None:
    """What starting this worker sets off that the operator should know first (resets, real money)."""
    notes = []
    if name == "runner":
        if config.RESET_ON_START:
            notes.append("RESET_ON_START is on: it first archives and clears the books other than the race books and the live books.")
        if config.LIVE_ENABLED:
            notes.append("LIVE_ENABLED=1: once the fly holds the selector's seat it also trades the bot wallet with real SOL.")
        if config.RH_ENABLED and config.RH_LIVE_ENABLED:
            notes.append("RH_LIVE_ENABLED=1: once the RH fly holds the seat it also trades the Robinhood Chain wallet with real ETH.")
    elif name == "kalshi_runner" and config.KALSHI_LIVE_ENABLED and not config.kalshi_live_prerequisites_missing():
        notes.append(f"Kalshi live is on: both arms mirror onto subaccount {config.KALSHI_SUBACCOUNT} with real money (cap ${config.KALSHI_CAPITAL_USD:g}).")
    return " ".join(notes) or None


def _host_pid(sup) -> int | None:
    """The pid of the console that hosts the workers (an observer's own pid is not it)."""
    if not isinstance(sup, runtime.Observer):
        return sup.pid
    r = q1("SELECT pid FROM processes WHERE cmd[1] = 'thread' AND stopped_at IS NULL ORDER BY started_at DESC LIMIT 1")
    return int(r["pid"]) if r else None


def _hosting(sup) -> None:
    """Worker controls belong to the console that hosts the workers (procs.stop and the autostart row do not go through the
    Supervisor, so they would slip past the observer's refusal)."""
    if isinstance(sup, runtime.Observer):
        raise runtime.ObserverMode("this console only observes (fly-trader ui --observe): start and stop workers from the console that hosts them")


def _worker(body: dict, shown: bool = True) -> str:
    """The worker an action names: one of the cards on the page (``shown``), or any worker the Supervisor knows."""
    name = body.get("name")
    if not isinstance(name, str) or name not in (worker_order() if shown else WORKERS):
        raise ActionError(f"no worker {name!r} on this page", 400)
    return name


# ---------------------------------------------------------------- reads
@topic("processes.workers", every=3)
def workers(params: dict) -> dict:
    """One card per worker (grouped as the menu is), the external CLI processes, and which console hosts the threads."""
    sup = runtime.sup(); status = sup.status(); ext = sup.external()
    auto = jv((q1("SELECT value FROM ui_settings WHERE key = 'autostart'") or {}).get("value"))
    cards = []
    for name in worker_order():
        title, icon, desc, role = WORKER_INFO[name]; s = status.get(name) or {}
        err = (s.get("error") or "").splitlines()
        cards.append({"name": name, "title": title, "icon": icon, "desc": desc, "role": role, "role_label": ROLE_LABEL[role],
                      "group": WORKER_TYPE.get(name, "system"), "state": _state(s, name in ext), "alive": bool(s.get("alive")), "stopping": bool(s.get("stopping")),
                      "started_at": s.get("started_at"), "stopped_at": s.get("stopped_at"),
                      "error": err[0][:300] if err and not s.get("alive") else None, "external": name in ext, "autostart": bool(auto.get(name)),
                      "ext_pid": (ext.get(name) or {}).get("pid"), "ext_started_at": (ext.get(name) or {}).get("started_at"),
                      "start_note": _start_note(name), "stop_note": STOP_NOTE.get(name)})
    groups = [{"key": g, "title": g.capitalize(), "workers": [c for c in cards if c["group"] == g]} for g in GROUPS]
    counts = {k: sum(c["state"] == k for c in cards) for k in ("running", "stopping", "crashed", "stopped", "external")}
    return {"pid": sup.pid, "observer": isinstance(sup, runtime.Observer), "host_pid": _host_pid(sup),
            "external": [{"name": n, "title": _title(n), "pid": a.get("pid"), "started_at": a.get("started_at")} for n, a in ext.items()],
            "groups": [g for g in groups if g["workers"]],
            "counts": {**counts, "total": len(cards), "autostart": sum(c["autostart"] for c in cards),
                       "trade_needed": sum(c["role"] == "trade" for c in cards), "trade_up": sum(c["role"] == "trade" and c["state"] in ("running", "external") for c in cards)}}


@topic("processes.activity", every=3)
def activity(params: dict) -> dict:
    """A worker's last 12 log lines, newest first (the in-memory ring of this console, or logs/<name>.log for an observer)."""
    name = params.get("worker")
    if name not in WORKERS:
        raise TopicError(f"no worker {name!r}")
    rows = []
    for ln in [ln for ln in runtime.sup().log_tail(name, 200).splitlines() if ln.strip()][-12:]:
        try:
            d = json.loads(ln)
        except ValueError:
            d = None
        if isinstance(d, dict):
            rows.append({"time": str(d.get("ts", ""))[11:19], "level": str(d.get("level", "")), "what": scrub(str(d.get("msg", "")))[:300]})
        else:
            rows.append({"time": "", "level": "", "what": scrub(ln)[:300]})
    return {"name": name, "rows": rows[::-1]}


@topic("processes.events", every=10)
def events(params: dict) -> dict:
    """The last 200 events at the chosen levels (none chosen: errors)."""
    lv = params.get("levels")
    levels = [x for x in LEVELS if isinstance(lv, list) and x in lv] or ["error"]
    rows = q("SELECT ts, level, source, message FROM events WHERE level = ANY(%s) ORDER BY id DESC LIMIT 200", (levels,))
    return {"levels": levels, "rows": rows}


@topic("processes.event_stats", every=30)
def event_stats(params: dict) -> dict:
    """The last 24 hours of events: how many per level, and warnings and errors per hour (24 buckets, oldest first)."""
    by_level = {r["level"]: int(r["n"]) for r in q("SELECT level, count(*) AS n FROM events WHERE ts > now() - interval '24 hours' GROUP BY level")}
    rows = q("""SELECT h.h, COALESCE(sum(CASE WHEN e.level = 'warning' THEN 1 END), 0) AS warning, COALESCE(sum(CASE WHEN e.level = 'error' THEN 1 END), 0) AS error
                FROM generate_series(date_trunc('hour', now()) - interval '23 hours', date_trunc('hour', now()), interval '1 hour') AS h(h)
                LEFT JOIN events e ON e.ts >= h.h AND e.ts < h.h + interval '1 hour' AND e.level IN ('warning', 'error')
                GROUP BY h.h ORDER BY h.h""")
    return {"levels": {lv: by_level.get(lv, 0) for lv in LEVELS} | {k: v for k, v in by_level.items() if k not in LEVELS},
            "hours": [r["h"] for r in rows], "warning": [int(r["warning"]) for r in rows], "error": [int(r["error"]) for r in rows]}


# ---------------------------------------------------------------- controls
@action("processes.start")
def start(body: dict) -> dict:
    name = _worker(body); sup = runtime.sup(); _hosting(sup); title = _title(name)
    if sup.alive(name):
        raise ActionError(f"{title} is already running.")
    ext = sup.external()
    if name in ext:
        raise ActionError(f"{title} is running outside the console (pid {ext[name].get('pid')}): stop the external process first.")
    from . import safety
    with safety.no_reset() if name in ENGINES else nullcontext():       # never start an engine while a reset clears its books
        try:
            started = sup.start(name, started_by="console")
        except RuntimeError as e:
            raise ActionError(str(e)) from None
    if not started:
        raise ActionError(f"{title} is already running.")
    return {"message": f"Started {title}"}


@action("processes.stop")
def stop(body: dict) -> dict:
    name = _worker(body, shown=False); sup = runtime.sup(); _hosting(sup); title = _title(name)
    s = sup.status().get(name) or {}
    if not s.get("alive"):
        raise ActionError(f"{title} is not running.")
    if s.get("stopping"):
        raise ActionError(f"{title} is already stopping: it finishes its current step first.")
    if not sup.request_stop(name, by="console"):
        raise ActionError(f"{title} is not running.")
    return {"message": f"Stop requested: {title} finishes its current step, then stops."}


@action("processes.stop_external")
def stop_external(body: dict) -> dict:
    """A worker started from the command line (`fly-trader worker start`): SIGTERM, 30 s, then SIGKILL (ops/procs.stop)."""
    name = _worker(body, shown=False); sup = runtime.sup(); _hosting(sup)
    a = sup.external().get(name)
    if not a:
        raise ActionError(f"No external {name} process is running.")
    from ...ops import procs
    if not procs.stop(name):
        return {"message": f"The external {name} process had already exited"}
    return {"message": f"Stopped the external {name} process (pid {a.get('pid')})"}


@action("processes.autostart")
def autostart(body: dict) -> dict:
    """Whether a worker starts with the console: the whole ``autostart`` map is read, changed and written back."""
    name = _worker(body); on = body.get("on")
    if not isinstance(on, bool):
        raise ActionError("on must be true or false", 400)
    _hosting(runtime.sup())
    with _AUTOSTART:
        r = q1("SELECT value FROM ui_settings WHERE key = 'autostart'"); val = jv(r["value"]) if r else {}
        val[name] = on
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('autostart', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps(val),))
    record_event("info", "console", "autostart changed", val)
    return {"message": f"Autostart {'on' if on else 'off'} for {_title(name)}"}
