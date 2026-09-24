"""The fly's side of the relay (docs/vault/SPEC.md §3): HMAC-signed, outbound-only HTTPS to the site's /api."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from urllib.parse import parse_qsl, quote, urlsplit

import httpx

from .. import config
from ..db.apilog import record_api_call

log = logging.getLogger(__name__)
PREFIX = "FLY-RELAY-1"


class RelayError(RuntimeError):
    pass


def canonical(method: str, path: str, query: str, ts: str, nonce: str, body: bytes) -> str:
    q = "&".join(f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in sorted(parse_qsl(query, keep_blank_values=True)))
    return "\n".join([PREFIX, method.upper(), path, q, ts, nonce, hashlib.sha256(body).hexdigest()])


def sign_headers(method: str, url: str, body: bytes, key_id: str, secret: str, ts: int | None = None, nonce: str | None = None) -> dict:
    parts = urlsplit(url)
    ts_s = str(int(time.time() if ts is None else ts))
    nonce = nonce or secrets.token_hex(16)
    mac = hmac.new(secret.encode("utf-8"), canonical(method, parts.path, parts.query, ts_s, nonce, body).encode(), hashlib.sha256)
    return {"X-Fly-Key": key_id, "X-Fly-Ts": ts_s, "X-Fly-Nonce": nonce, "X-Fly-Sig": mac.hexdigest()}


class RelayClient:
    def __init__(self, base: str | None = None, key_id: str | None = None, secret: str | None = None, timeout: float = 20.0):
        self.base = (base or config.RELAY_URL or "").rstrip("/")
        self.key_id = key_id or config.RELAY_KEY_ID
        self.secret = secret or config.RELAY_SECRET
        if not self.base or not self.secret:
            raise RelayError("RELAY_URL and RELAY_SECRET must be set")
        self._client = httpx.Client(timeout=timeout)

    def _req(self, method: str, path: str, params: dict | None = None, payload=None):
        url = self.base + path
        if params:
            url += "?" + "&".join(f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in params.items())
        body = b"" if payload is None else json.dumps(payload, separators=(",", ":"), default=str).encode()
        headers = sign_headers(method, url, body, self.key_id, self.secret)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        t0, status, ok = time.monotonic(), None, False
        try:
            r = self._client.request(method, url, content=body or None, headers=headers)
            status = r.status_code
            if status >= 400:
                raise RelayError(f"relay {method} {path}: HTTP {status} {r.text[:200]}")
            ok = True
            return r.json()
        except httpx.HTTPError as e:
            raise RelayError(f"relay {method} {path}: {type(e).__name__}") from None
        finally:
            record_api_call("relay", path, method, status, int((time.monotonic() - t0) * 1000), ok)

    def push(self, stats: dict | None = None, history: dict | None = None, accounts: list | None = None) -> dict:
        body = {k: v for k, v in (("stats", stats), ("history", history), ("accounts", accounts)) if v}
        return self._req("POST", "/api/fly/push", payload=body)

    def pending_claims(self, after: int = 0, limit: int = 50) -> list[dict]:
        return self._req("GET", "/api/fly/claims", {"after": int(after), "limit": int(limit)}).get("claims") or []

    def report(self, results: list[dict]) -> dict:
        return self._req("POST", "/api/fly/claims/result", payload={"results": results})

    def probe(self) -> dict:
        return self._req("GET", "/api/fly/probe")
