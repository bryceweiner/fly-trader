"""The console has no login: it trusts whoever can reach its port, which is 127.0.0.1 (or a port published on 127.0.0.1).
That alone does not stop a web page open in the operator's browser from talking to it, so every request is checked:

- **Host** must be a loopback name (or one listed in CONSOLE_ALLOWED_HOSTS): a DNS-rebinding page arrives with its own
  domain in Host and is refused.
- **State-changing requests** (anything but GET/HEAD/OPTIONS) need the ``X-Fly-Console: 1`` header — a cross-site page
  cannot send it without a CORS preflight this server never approves — and, when the browser says where they come from
  (Origin, Sec-Fetch-Site), they must come from this origin.
- **WebSockets** are not covered by CORS at all, so their Origin must be this origin.
- **Reads of the API** from another site are refused as well (Sec-Fetch-Site / Origin): it could not read them, but it
  could keep the console busy computing them.
"""
from __future__ import annotations

import json
from urllib.parse import urlsplit

from .. import config

LOOPBACK = {"localhost", "127.0.0.1", "::1"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
HEADER = "x-fly-console"


def _hostname(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):                          # [::1]:8501
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def allowed_hosts() -> set[str]:
    return LOOPBACK | {h.strip().lower() for h in (config.env_str("CONSOLE_ALLOWED_HOSTS") or "").split(",") if h.strip()}


def same_origin(origin: str | None, host: str) -> bool:
    if not origin:
        return True                                    # not a browser (curl, tests): nothing to forge
    if origin == "null":
        return False
    return urlsplit(origin).netloc.lower() == host.strip().lower()


def refusal(scope) -> str | None:
    """Why this request is refused, or None."""
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
    host = headers.get("host", "")
    if _hostname(host) not in allowed_hosts():
        return f"host {host!r} is not allowed (set CONSOLE_ALLOWED_HOSTS to reach the console by that name)"
    origin = headers.get("origin")
    if scope["type"] == "websocket":
        return None if same_origin(origin, host) else "cross-origin WebSocket refused"
    if scope.get("method", "GET") in SAFE_METHODS:
        # a page elsewhere cannot read the answers, but it could still make the console compute them (the expensive topics
        # would tie up the threads every control needs): the API answers only this origin's own requests
        if scope.get("path", "").startswith("/api/") and (headers.get("sec-fetch-site") in ("cross-site", "same-site") or not same_origin(origin, host)):
            return "cross-site read of the API refused"
        return None
    if headers.get(HEADER) != "1":
        return f"missing {HEADER} header"
    if not same_origin(origin, host):
        return "cross-origin request refused"
    if headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return "cross-site request refused"
    return None


class Guard:
    """ASGI middleware applying ``refusal`` to every HTTP request and WebSocket."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        why = refusal(scope)
        if why is None:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": why[:120]})
            return
        body = json.dumps({"ok": False, "error": why}).encode()
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
