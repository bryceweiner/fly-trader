"""HMAC authentication for the fly's requests (SPEC §3).

canonical = "FLY-RELAY-1\\n{METHOD}\\n{PATH}\\n{QUERY}\\n{TS}\\n{NONCE}\\n{sha256hex(body)}"
sig       = hex HMAC-SHA256(key = the configured secret string's UTF-8 bytes, canonical UTF-8)

``sign()`` is the client half; the fly can copy it verbatim. Nonce memory lives in store.py.

CLI (operator checks, e.g. the probe):
    FLY_RELAY_SECRET=<hex> python hmacauth.py https://fly-trader.app/api/fly/probe k1
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from urllib.parse import parse_qsl, quote

PREFIX = "FLY-RELAY-1"
WINDOW_S = 300
NONCE_TTL_S = 900
_TS = re.compile(r"[0-9]{1,15}")
_NONCE = re.compile(r"[0-9a-fA-F]{32}")
_SIG = re.compile(r"[0-9a-fA-F]{64}")


class AuthError(Exception):
    """Always answered with 401."""


def canonical_query(query) -> str:
    """``query`` is a raw query string, a mapping, or a sequence of (k, v) pairs."""
    if query is None:
        pairs = []
    elif isinstance(query, str):
        pairs = parse_qsl(query, keep_blank_values=True)
    elif hasattr(query, "items"):
        pairs = [(str(k), str(v)) for k, v in query.items()]
    else:
        pairs = [(str(k), str(v)) for k, v in query]
    return "&".join("%s=%s" % (quote(k, safe=""), quote(v, safe="")) for k, v in sorted(pairs))


def canonical(method: str, path: str, query, ts: str, nonce: str, body: bytes) -> str:
    return "\n".join([PREFIX, method.upper(), path, canonical_query(query), str(ts), nonce,
                      hashlib.sha256(body or b"").hexdigest()])


def signature(secret: str, text: str) -> str:
    return hmac.new(secret.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()


def sign(method, path, query, body, key_id, secret, ts=None, nonce=None) -> dict:
    """Headers for one request. ``body`` is bytes, str (UTF-8) or None; send exactly those bytes."""
    if isinstance(body, str):
        body = body.encode("utf-8")
    ts = str(int(time.time()) if ts is None else int(ts))
    nonce = nonce or secrets.token_hex(16)
    sig = signature(secret, canonical(method, path, query, ts, nonce, body or b""))
    return {"X-Fly-Key": key_id, "X-Fly-Ts": ts, "X-Fly-Nonce": nonce, "X-Fly-Sig": sig}


class Parsed:
    __slots__ = ("key_id", "secret", "ts", "nonce", "sig")

    def __init__(self, key_id, secret, ts, nonce, sig):
        self.key_id, self.secret, self.ts, self.nonce, self.sig = key_id, secret, ts, nonce, sig


def parse_headers(environ: dict, keys: dict, now: float) -> Parsed:
    """Cheap checks before the body is read. Raises AuthError."""
    key_id = environ.get("HTTP_X_FLY_KEY", "")
    ts = environ.get("HTTP_X_FLY_TS", "")
    nonce = environ.get("HTTP_X_FLY_NONCE", "")
    sig = environ.get("HTTP_X_FLY_SIG", "")
    if not (key_id and ts and nonce and sig):
        raise AuthError("missing auth headers")
    secret = keys.get(key_id)
    if secret is None:
        raise AuthError("unknown key")
    if not _TS.fullmatch(ts):
        raise AuthError("bad timestamp")
    if abs(int(ts) - now) > WINDOW_S:
        # The server time helps diagnose clock skew; it is public anyway.
        raise AuthError("timestamp outside +-%d s of server time %d" % (WINDOW_S, int(now)))
    if not _NONCE.fullmatch(nonce):
        raise AuthError("bad nonce")
    if not _SIG.fullmatch(sig):
        raise AuthError("bad signature")
    return Parsed(key_id, secret, ts, nonce, sig.lower())


def verify_signature(environ: dict, body: bytes, p: Parsed) -> None:
    path = environ.get("SCRIPT_NAME", "") + environ.get("PATH_INFO", "")
    text = canonical(environ.get("REQUEST_METHOD", "GET"), path, environ.get("QUERY_STRING", ""),
                     p.ts, p.nonce, body)
    if not hmac.compare_digest(signature(p.secret, text), p.sig):
        raise AuthError("bad signature")


if __name__ == "__main__":  # pragma: no cover - operator tool
    import os
    import sys
    import urllib.request
    from urllib.parse import urlsplit

    if len(sys.argv) < 3:
        sys.exit("usage: FLY_RELAY_SECRET=<hex> python hmacauth.py <url> <key_id> [POST body-file]")
    url, kid = sys.argv[1], sys.argv[2]
    data = open(sys.argv[4], "rb").read() if len(sys.argv) > 4 else None
    method = "POST" if data is not None else "GET"
    u = urlsplit(url)
    hdrs = sign(method, u.path, u.query, data, kid, os.environ["FLY_RELAY_SECRET"])
    hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(r.status, dict(r.headers))
            print(r.read().decode())
    except urllib.error.HTTPError as e:
        print(e.code, dict(e.headers))
        print(e.read().decode())
