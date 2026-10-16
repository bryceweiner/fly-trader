"""fly-trader operator console: one process that hosts the worker threads (ops/supervisor.py) and serves the React
dashboard (console/, built into ui/static/console), its JSON API and the WebSocket that keeps every page live.

It never runs trading logic itself: every control writes a Postgres row or signals a worker thread. Routes:
  /ws                          the WebSocket (hub.py)
  /api/meta                    what this install has turned on (markets, Kalshi, the vault) — the menu reads it
  /api/topic/<name>?params=    one read, the same function the WebSocket pushes
  /api/action/<name>           one write (POST, JSON body)
  /api/brain/...               the brain view's binary files (activity per minute, a neuron's history)
  /static/brain/...            the neuron geometry (generated into ui/static/brain)
  everything else              the dashboard (index.html; React routes the path)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles

from .. import config
from ..logging_setup import scrub
from . import runtime
from .hub import ActionError, Hub, run_action, run_topic
from .jsonio import dumps, loads
from .security import Guard

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
CONSOLE_DIR = STATIC_DIR / "console"
GEOMETRY_DIR = STATIC_DIR / "brain"
NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>fly-trader</title>
<body style="background:#16141a;color:#eee;font:16px system-ui;padding:3rem">
<h1>The console is not built</h1><p>Build it once: <code>cd console &amp;&amp; npm ci &amp;&amp; npm run build</code>
(it writes fly_trader/ui/static/console). The workers are running regardless.</p>"""


class JSONBytes(Response):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return dumps(content)


def create_app(*, hosts_workers: bool = True) -> FastAPI:
    runtime.HOSTS_WORKERS = hosts_workers
    from . import api  # noqa: F401  (registers every topic and action)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.hub = Hub()
        try:
            yield
        finally:
            app.state.hub.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None, default_response_class=JSONBytes)
    app.add_middleware(Guard)

    @app.get("/api/meta")
    async def meta(request: Request):
        return await _in_thread(request, api.meta.meta, {})

    @app.get("/api/topic/{name}")
    async def get_topic(name: str, request: Request, params: str | None = None):
        try:
            p = loads(params) if params else {}
        except ValueError:
            return JSONResponse({"ok": False, "error": "params must be JSON"}, status_code=400)
        return await _in_thread(request, run_topic, name, p if isinstance(p, dict) else {})

    @app.post("/api/action/{name}")
    async def post_action(name: str, request: Request):
        try:
            body = loads(await request.body() or b"{}")
        except ValueError:
            return JSONResponse({"ok": False, "error": "body must be JSON"}, status_code=400)
        res = await _in_thread(request, run_action, name, body if isinstance(body, dict) else {})
        if isinstance(res, Response):
            return res
        request.app.state.hub.poke()
        return {"ok": True, **res}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.app.state.hub.serve(websocket)

    api.brain.routes(app)
    app.mount("/static/brain", StaticFiles(directory=GEOMETRY_DIR, check_dir=False), name="brain")
    app.mount("/assets", StaticFiles(directory=CONSOLE_DIR / "assets", check_dir=False), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith(("api/", "ws", "static/")):
            return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
        f = CONSOLE_DIR / path
        if path and ".." not in path and f.is_file() and f.resolve().is_relative_to(CONSOLE_DIR.resolve()):
            return FileResponse(f)                 # favicon, fonts and images at the build's root
        index = CONSOLE_DIR / "index.html"
        if not index.exists():
            return HTMLResponse(NOT_BUILT)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


async def _in_thread(request: Request, fn, *args):
    """Run a read or a write in the hub's threads (psycopg is synchronous); refusals and errors become JSON with a status."""
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(request.app.state.hub.executor, fn, *args)
    except runtime.ObserverMode as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=409)
    except ActionError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=e.status)
    except Exception as e:
        log.exception("console request failed")
        return JSONResponse({"ok": False, "error": scrub(f"{type(e).__name__}: {e}")[:600]}, status_code=500)


def default_host() -> str:
    """CONSOLE_HOST, else 127.0.0.1 — except inside a container, where the console must listen on the container's own
    interface for the published port to reach it (compose publishes it on the host's 127.0.0.1 only). Decided here, not in
    the image: a self-updated install keeps its old image and only receives this code."""
    h = config.env_str("CONSOLE_HOST")
    if h:
        return h
    return "0.0.0.0" if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists() else "127.0.0.1"


def run(port: int = 8501, host: str | None = None, observe: bool = False) -> None:
    """`fly-trader ui`: start the autostart workers in THIS process, then serve the console in the main thread until
    SIGINT/SIGTERM, then stop every worker. ``observe`` serves the dashboard without hosting workers (a preview)."""
    import uvicorn
    host = host or default_host()
    sup = None
    if not observe:
        from ..db import schema
        from ..ops.supervisor import get_supervisor
        schema.apply_schema()           # idempotent: tables added to the schema exist before any worker uses them
        sup = get_supervisor()
        try:
            started = sup.autostart()
        except Exception as e:
            started = []; log.exception("autostart failed: %s", scrub(str(e)))
        log.info("console pid=%d autostarted=%s", os.getpid(), started)
    else:
        from ..logging_setup import setup
        setup("console-observer")
    app = create_app(hosts_workers=not observe)
    try:
        uvicorn.run(app, host=host, port=port, log_config=None, access_log=False, timeout_graceful_shutdown=5, ws_ping_interval=20)
    finally:
        if sup is not None:
            sup.stop_all()
