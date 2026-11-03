"""Topics, actions and the WebSocket that pushes them.

A *topic* is a read: ``fn(params) -> JSON-able`` registered with a cadence. While at least one browser tab subscribes to a
(topic, params) pair, ONE poller computes it every ``every`` seconds in a worker thread and pushes the result to every
subscriber, only when it changed. Ten open tabs cost the database what one does. An *action* is a write
(``fn(body) -> dict``) behind ``POST /api/action/<name>``; after one succeeds every poller recomputes at once, so the page
shows its effect without waiting for the next tick.

Wire format (JSON text frames):
  client → server  {"op": "sub", "id": "<client's id>", "topic": "overview.summary", "params": {"chain": "sol"}}
                   {"op": "unsub", "id": "..."}     {"op": "ping"}
  server → client  {"op": "data", "id": "...", "data": ...}   {"op": "error", "id": "...", "error": "..."}   {"op": "pong"}
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..db.queries import reuse_connection_in_this_thread
from ..logging_setup import scrub
from . import runtime
from .jsonio import dumps, loads

log = logging.getLogger(__name__)


@dataclass
class Topic:
    name: str
    fn: Callable[[dict], Any]
    every: float


@dataclass
class Action:
    name: str
    fn: Callable[[dict], dict | None]


class ActionError(Exception):
    """A refused action: the message is shown to the operator as is."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class TopicError(ActionError):
    """A topic called with parameters it does not accept."""

    def __init__(self, message: str):
        super().__init__(message, 400)


TOPICS: dict[str, Topic] = {}
ACTIONS: dict[str, Action] = {}


def topic(name: str, every: float = 5.0):
    def deco(fn):
        if name in TOPICS:
            raise ValueError(f"topic {name} registered twice")
        TOPICS[name] = Topic(name, fn, float(every))
        return fn
    return deco


def action(name: str):
    def deco(fn):
        if name in ACTIONS:
            raise ValueError(f"action {name} registered twice")
        ACTIONS[name] = Action(name, fn)
        return fn
    return deco


def run_topic(name: str, params: dict | None) -> Any:
    t = TOPICS.get(name)
    if t is None:
        raise TopicError(f"no topic {name}")
    return t.fn(dict(params or {}))


def run_action(name: str, body: dict | None) -> dict:
    a = ACTIONS.get(name)
    if a is None:
        raise ActionError(f"no action {name}", 404)
    if isinstance(runtime.sup(), runtime.Observer):   # a preview changes nothing: every write belongs to the hosting console
        raise runtime.ObserverMode("this console only observes (fly-trader ui --observe): act from the console that hosts the workers")
    return a.fn(dict(body or {})) or {}


# ---------------------------------------------------------------- pushing
class Client:
    """One WebSocket. Outgoing frames go through a bounded queue drained by its own task, so a slow tab never holds up a
    poller; a tab that falls 256 frames behind is disconnected (it reconnects and resubscribes)."""

    def __init__(self, ws):
        self.ws = ws
        self.q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        self.subs: dict[str, str] = {}            # client sub id → poller key
        self.dead = False

    def send(self, frame: str) -> None:
        if self.dead:
            return
        try:
            self.q.put_nowait(frame)
        except asyncio.QueueFull:
            self.dead = True
            while not self.q.empty():
                self.q.get_nowait()
            self.q.put_nowait(None)               # the pump closes the socket

    async def pump(self) -> None:
        while True:
            frame = await self.q.get()
            if frame is None:
                await self.ws.close(code=1013)
                return
            await self.ws.send_text(frame)


def _frame(op: str, sid: str, body: bytes) -> str:
    key = "data" if op == "data" else "error"
    return (b'{"op":"' + op.encode() + b'","id":' + dumps(sid) + b',"' + key.encode() + b'":' + body + b"}").decode()


@dataclass
class Poller:
    hub: "Hub"
    topic: Topic
    params: dict
    key: str
    subs: dict = field(default_factory=dict)      # Client → set of its sub ids
    last: bytes | None = None
    last_err: str | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def add(self, client: Client, sid: str) -> None:
        self.subs.setdefault(client, set()).add(sid)
        if self.last is not None:
            client.send(_frame("data", sid, self.last))
        elif self.last_err is not None:
            client.send(_frame("error", sid, dumps(self.last_err)))

    def remove(self, client: Client, sid: str | None = None) -> None:
        s = self.subs.get(client)
        if s is None:
            return
        if sid is None:
            s.clear()
        else:
            s.discard(sid)
        if not s:
            self.subs.pop(client, None)
        if not self.subs:
            self.wake.set()                        # let the loop see it is unwatched and end

    def _broadcast(self, op: str, body: bytes) -> None:
        for c, sids in list(self.subs.items()):
            for sid in sids:
                c.send(_frame(op, sid, body))

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self.subs:
                t0 = time.monotonic()
                try:
                    data = await loop.run_in_executor(self.hub.executor, self.topic.fn, dict(self.params))
                    body, err = dumps(data), None
                except Exception as e:             # one broken panel must never take the dashboard down
                    body, err = None, scrub(f"{type(e).__name__}: {e}")[:600]
                    if err != self.last_err:
                        log.warning("topic %s %s: %s", self.topic.name, self.params, err)
                if body is not None and (body != self.last or self.last_err is not None):   # a recovery clears the error banner
                    self.last, self.last_err = body, None
                    self._broadcast("data", body)
                elif err is not None and err != self.last_err:
                    self.last_err = err
                    self._broadcast("error", dumps(err))
                if not self.subs:
                    break
                wait = max(0.05, self.topic.every - (time.monotonic() - t0))
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=wait)
                except TimeoutError:
                    pass
                self.wake.clear()
        finally:
            if self.hub.pollers.get(self.key) is self:
                self.hub.pollers.pop(self.key, None)


class Hub:
    def __init__(self, workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="console-hub", initializer=reuse_connection_in_this_thread)
        self.pollers: dict[str, Poller] = {}

    @staticmethod
    def key(name: str, params: dict) -> str:
        return name + "?" + dumps(dict(sorted(params.items()))).decode()

    def subscribe(self, client: Client, sid: str, name: str, params: dict) -> None:
        t = TOPICS.get(name)
        if t is None:
            client.send(_frame("error", sid, dumps(f"no topic {name}")))
            return
        self.unsubscribe(client, sid)
        k = self.key(name, params)
        p = self.pollers.get(k)
        if p is None or p.task is None or p.task.done():
            p = self.pollers[k] = Poller(self, t, params, k)
            p.add(client, sid)
            p.task = asyncio.get_running_loop().create_task(p.run(), name=f"topic {k}")
        else:
            p.add(client, sid)
        client.subs[sid] = k

    def unsubscribe(self, client: Client, sid: str) -> None:
        k = client.subs.pop(sid, None)
        if k and (p := self.pollers.get(k)):
            p.remove(client, sid)

    def drop(self, client: Client) -> None:
        for sid in list(client.subs):
            self.unsubscribe(client, sid)

    def poke(self) -> None:
        """Recompute every watched topic now (after an action)."""
        for p in list(self.pollers.values()):
            p.wake.set()

    async def serve(self, ws) -> None:
        client = Client(ws)
        pump = asyncio.get_running_loop().create_task(client.pump())
        try:
            while not client.dead:
                recv = asyncio.ensure_future(ws.receive_text())
                done, _ = await asyncio.wait({recv, pump}, return_when=asyncio.FIRST_COMPLETED)
                if pump in done:
                    recv.cancel()
                    break
                try:
                    msg = loads(recv.result())
                except ValueError:
                    continue
                op = msg.get("op")
                if op == "sub" and isinstance(msg.get("id"), str) and isinstance(msg.get("topic"), str):
                    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
                    self.subscribe(client, msg["id"], msg["topic"], params)
                elif op == "unsub" and isinstance(msg.get("id"), str):
                    self.unsubscribe(client, msg["id"])
                elif op == "ping":
                    client.send('{"op":"pong"}')
        except Exception:                           # the tab went away (WebSocketDisconnect and friends)
            pass
        finally:
            self.drop(client)
            pump.cancel()

    def close(self) -> None:
        for p in list(self.pollers.values()):
            if p.task:
                p.task.cancel()
        self.pollers.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)
