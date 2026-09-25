"""WSGI app for /api/* (SPEC §3): routing, config, JSON helpers and error mapping.

Every response is JSON with Cache-Control: no-store. web/wsgi.py adds the security headers.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import sys
import time
import traceback
from http import HTTPStatus
from urllib.parse import parse_qs

from . import hmacauth, ratelimit, store, validate

PUSH_LIMIT = 2 * 1024 * 1024
PUBLIC_LIMIT = 4 * 1024
REQUEST_ID = "fly-vault-claim-v1"
HISTORY_KINDS = ("nav", "trades", "flows", "settlements")
# Which field orders each history kind, newest first; the first integer found wins.
HISTORY_TS = {"nav": ("ts",), "trades": ("closed_at", "opened_at"), "flows": ("ts",),
              "settlements": ("period_end", "period_start")}
PURGE_EVERY_S = 600
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "relay.json")

clock = time.time                                  # tests replace these two
new_nonce = lambda: secrets.token_hex(16)          # noqa: E731


class HttpError(Exception):
    def __init__(self, status: int, message: str, headers=()):
        Exception.__init__(self, message)
        self.status, self.message, self.headers = status, message, list(headers)


def log(msg: str) -> None:
    sys.stderr.write("[relay] %s\n" % msg)
    sys.stderr.flush()


# ---- config ----
class Config:
    def __init__(self, d: dict, base: str):
        keys = d.get("keys")
        if not isinstance(keys, dict) or not keys:
            raise ValueError("keys must be a non-empty object")
        for k, v in keys.items():
            if not isinstance(v, str) or len(v) < 32:
                raise ValueError("key %r: secret must be a string of at least 32 characters" % k)
        self.keys = dict(keys)
        db = d.get("db_path")
        if not isinstance(db, str) or not db:
            raise ValueError("db_path is required")
        self.db_path = os.path.normpath(os.path.join(base, os.path.expanduser(db)))
        self.domain = _need(d, "domain", str)
        self.uri = _need(d, "uri", str)
        self.chain_id = _need(d, "chain_id", int)
        self.sol_chain = _need(d, "sol_chain", str)
        if self.sol_chain not in ("mainnet", "devnet"):
            raise ValueError("sol_chain must be mainnet or devnet")
        self.claim_ttl_s = _need(d, "claim_ttl_s", int, 900)
        self.min_lamports = _need(d, "min_lamports", int, 2000000)
        h = d.get("client_ip_header")
        if h is not None and not isinstance(h, str):
            raise ValueError("client_ip_header must be a string or null")
        if h:  # accept "X-Forwarded-For" as well as the environ key "HTTP_X_FORWARDED_FOR"
            h = h.strip().upper().replace("-", "_")
            h = h if h.startswith("HTTP_") or h == "REMOTE_ADDR" else "HTTP_" + h
        self.client_ip_header = h or None


def _need(d, name, typ, default=None):
    v = d.get(name, default)
    if not isinstance(v, typ) or isinstance(v, bool) or v is None:
        raise ValueError("%s must be a %s" % (name, typ.__name__))
    return v


_cfg_cache = {"key": None, "cfg": None}


def config() -> "Config | None":
    """Re-read when relay.json changes, so a key rotation needs no restart."""
    path = os.environ.get("FLY_RELAY_CONFIG") or DEFAULT_CONFIG
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    if _cfg_cache["key"] == key:
        return _cfg_cache["cfg"]
    cfg = None
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level must be an object")
        cfg = Config(data, os.path.dirname(os.path.abspath(path)))
    except (OSError, ValueError) as e:  # never print the file: it holds the secrets
        log("relay config %s is invalid: %s" % (path, e))
    _cfg_cache.update(key=key, cfg=cfg)
    return cfg


# ---- request helpers ----
def read_body(environ: dict, limit: int) -> bytes:
    raw = (environ.get("CONTENT_LENGTH") or "").strip()
    if raw:
        try:
            length = int(raw)
        except ValueError:
            raise HttpError(400, "bad content length")
        if length < 0:
            raise HttpError(400, "bad content length")
        if length > limit:
            raise HttpError(413, "body too large")
        want = length
    elif environ.get("wsgi.input_terminated"):
        want = limit + 1
    else:
        return b""
    stream, chunks, got = environ["wsgi.input"], [], 0
    while got < want:
        chunk = stream.read(min(65536, want - got))
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    body = b"".join(chunks)
    if len(body) > limit:
        raise HttpError(413, "body too large")
    return body


def parse_json(body: bytes):
    try:
        # NaN/Infinity are not JSON; browsers would choke on them, so they become null.
        return json.loads(body.decode("utf-8"), parse_constant=lambda _: None)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise HttpError(400, "invalid JSON")


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def client_ip(environ: dict, cfg: Config) -> str:
    ip = ""
    if cfg.client_ip_header:
        # The RIGHTMOST entry: proxies append the address they saw, so every entry left of it is client-controlled
        # (taking the first one would let anyone pick their own rate-limit bucket).
        ip = (environ.get(cfg.client_ip_header) or "").split(",")[-1].strip()
    return ratelimit.ip_key(ip or environ.get("REMOTE_ADDR", ""))


def query(environ: dict) -> dict:
    q = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
    return {k: v[0] for k, v in q.items()}


def int_param(q: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = q.get(name)
    if raw is None or raw == "":
        return default
    if not re.fullmatch(r"[0-9]{1,18}", raw) or not lo <= int(raw) <= hi:
        raise HttpError(400, "%s must be an integer in %d..%d" % (name, lo, hi))
    return int(raw)


def rfc3339(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def limit(conn, name: str, who: str, now: float) -> None:
    with store.write(conn):
        wait = ratelimit.take(conn, name, who, now)
    if wait:
        raise HttpError(429, "rate limited", [("Retry-After", str(int(wait) + 1))])


_last_purge = [0.0]


def maybe_purge(conn, now: float) -> None:
    """Housekeeping inside a write transaction the caller already holds, at most every 10 min per process."""
    if now - _last_purge[0] >= PURGE_EVERY_S:
        _last_purge[0] = now
        store.purge(conn, now)


# ---- public handlers ----
class Req:
    def __init__(self, environ, cfg, conn, now):
        self.environ, self.cfg, self.conn, self.now = environ, cfg, conn, now
        self.q = query(environ)
        self.ip = client_ip(environ, cfg)


def get_stats(r: Req):
    limit(r.conn, "read", r.ip, r.now)
    body = store.get_stats(r.conn)
    if body is None:
        raise HttpError(503, "no stats yet")
    return 200, body


def get_history(r: Req):
    limit(r.conn, "read", r.ip, r.now)
    kind = r.q.get("kind", "")
    if kind not in HISTORY_KINDS:
        raise HttpError(400, "kind must be one of " + "|".join(HISTORY_KINDS))
    n = int_param(r.q, "limit", 100, 1, 500)
    before = None
    if r.q.get("before"):
        before = store.decode_cursor(r.q["before"])
        if before is None:
            raise HttpError(400, "bad cursor")
    bodies, nxt = store.history_page(r.conn, kind, before, n)
    return 200, '{"kind":%s,"items":[%s],"next":%s}' % (dumps(kind), ",".join(bodies), dumps(nxt))


def get_account(r: Req):
    limit(r.conn, "read", r.ip, r.now)
    evm = validate.evm_address(r.q.get("evm"))
    if evm is None:
        raise HttpError(400, "bad evm address")
    row = store.get_account(r.conn, evm)
    if row:
        return 200, row[1]
    return 200, {"evm": evm, "allocated": 0, "claimed": 0, "in_flight": 0, "owed": 0, "allocations": [],
                 "claims": []}


def get_claim(r: Req, claim_id: int):
    limit(r.conn, "read", r.ip, r.now)
    c = store.get_claim(r.conn, claim_id)
    if c is None:
        raise HttpError(404, "claim not found")
    return 200, {k: c[k] for k in ("id", "status", "reason", "lamports", "tx", "created_at", "updated_at")}


def get_challenge(r: Req):
    evm = validate.evm_address(r.q.get("evm"))
    if evm is None:
        raise HttpError(400, "bad evm address")
    sol = r.q.get("sol")
    if not validate.sol_address(sol):
        raise HttpError(400, "bad solana address")
    limit(r.conn, "challenge", r.ip, r.now)
    cfg, now = r.cfg, int(r.now)
    ch = {"nonce": new_nonce(), "evm": evm, "sol": sol, "issued_at": now, "expires_at": now + cfg.claim_ttl_s,
          "domain": cfg.domain, "uri": cfg.uri, "chain_id": cfg.chain_id, "sol_chain": cfg.sol_chain}
    with store.write(r.conn):
        maybe_purge(r.conn, r.now)
        store.add_challenge(r.conn, ch)
    return 200, {"nonce": ch["nonce"], "issued_at": rfc3339(ch["issued_at"]), "expires_at": rfc3339(ch["expires_at"]),
                 "domain": cfg.domain, "uri": cfg.uri, "chain_id": cfg.chain_id, "sol_chain": cfg.sol_chain,
                 "request_id": REQUEST_ID, "min_lamports": cfg.min_lamports}


def post_claim(r: Req):
    body = read_body(r.environ, PUBLIC_LIMIT)
    d = parse_json(body)
    if not isinstance(d, dict):
        raise HttpError(400, "body must be a JSON object")
    nonce = d.get("nonce")
    if not validate.nonce(nonce):
        raise HttpError(400, "bad nonce")
    evm = validate.evm_address(d.get("evm"))
    if evm is None:
        raise HttpError(400, "bad evm address")
    sol = d.get("sol")
    if not validate.sol_address(sol):
        raise HttpError(400, "bad solana address")
    if not validate.evm_signature(d.get("evm_sig")):
        raise HttpError(400, "bad evm signature")
    if not validate.sol_signature(d.get("sol_sig")):
        raise HttpError(400, "bad solana signature")
    limit(r.conn, "claim_ip", r.ip, r.now)
    now, conn = int(r.now), r.conn
    # One IMMEDIATE transaction: two submits of the same nonce, or two claims for one address, serialize here.
    with store.write(conn):
        ch = store.get_challenge(conn, nonce)
        if ch is None:
            raise HttpError(400, "unknown nonce")
        if ch["used_at"] is not None:
            raise HttpError(409, "nonce already used")
        if now > ch["expires_at"]:
            raise HttpError(400, "nonce expired")
        if ch["evm"] != evm or ch["sol"] != sol:
            raise HttpError(400, "nonce was issued for another address")
        acct = store.get_account(conn, evm)
        owed = acct[0] if acct else 0
        if owed <= 0:
            raise HttpError(400, "nothing to claim")
        if owed < r.cfg.min_lamports:
            raise HttpError(400, "owed is below the claim minimum")
        if store.claim_in_flight(conn, evm):
            raise HttpError(409, "a claim for this address is in flight")
        # Per address AND client: the relay cannot tell a stranger's junk signatures from real ones, so a bucket
        # keyed on the address alone would let strangers drain the holder's.
        wait = ratelimit.take(conn, "claim_evm", "%s|%s" % (evm, r.ip), r.now)
        if wait:
            raise HttpError(429, "rate limited", [("Retry-After", str(int(wait) + 1))])
        if not store.use_challenge(conn, nonce, now):
            raise HttpError(409, "nonce already used")
        claim_id = store.add_claim(conn, ch, d["evm_sig"], d["sol_sig"], now)
    return 202, {"id": claim_id, "status": "received"}


# ---- fly handlers (HMAC) ----
def authenticate(r: Req) -> bytes:
    try:
        parsed = hmacauth.parse_headers(r.environ, r.cfg.keys, r.now)
        body = read_body(r.environ, PUSH_LIMIT)
        hmacauth.verify_signature(r.environ, body, parsed)
    except hmacauth.AuthError as e:
        raise HttpError(401, str(e))
    with store.write(r.conn):
        maybe_purge(r.conn, r.now)
        fresh = store.remember_nonce(r.conn, parsed.nonce, int(r.now))
    if not fresh:
        raise HttpError(401, "replayed nonce")
    return body


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _history_key(kind: str, item: dict) -> "tuple[str, int]":
    ts = next((item[f] for f in HISTORY_TS[kind] if _is_int(item.get(f))), None)
    if ts is None:
        raise HttpError(400, "%s item needs an integer %s" % (kind, " or ".join(HISTORY_TS[kind])))
    if kind == "nav":
        return "%020d" % ts, ts
    i = item.get("id")
    if _is_int(i) and i >= 0:
        return "%020d" % i, ts   # zero-padded so ties on ts order numerically
    if isinstance(i, str) and 0 < len(i) <= 200:
        return i, ts
    raise HttpError(400, "%s item needs an id (integer >= 0 or string)" % kind)


def post_push(r: Req):
    d = parse_json(authenticate(r))
    if not isinstance(d, dict):
        raise HttpError(400, "body must be a JSON object")
    extra = set(d) - {"stats", "history", "accounts"}
    if extra:
        raise HttpError(400, "unknown push field: %s" % ", ".join(sorted(extra)))
    stats, history, accounts = d.get("stats"), d.get("history") or {}, d.get("accounts") or []
    if stats is not None and not isinstance(stats, dict):
        raise HttpError(400, "stats must be an object")
    if not isinstance(history, dict):
        raise HttpError(400, "history must be an object")
    docs = []
    for kind, items in history.items():
        if kind not in HISTORY_KINDS:
            raise HttpError(400, "unknown history kind: %s" % kind)
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise HttpError(400, "history.%s must be a list of objects" % kind)
        docs.extend((kind,) + _history_key(kind, i) + (dumps(i),) for i in items)
    if not isinstance(accounts, list):
        raise HttpError(400, "accounts must be a list")
    accts = []
    for a in accounts:
        evm = validate.evm_address(a.get("evm")) if isinstance(a, dict) else None
        if evm is None:
            raise HttpError(400, "account needs a valid evm address")
        owed = a.get("owed", 0)
        if not _is_int(owed):
            raise HttpError(400, "account owed must be an integer")
        accts.append((evm, owed, dumps(a)))
    now = int(r.now)
    with store.write(r.conn):
        if stats is not None:
            store.put_stats(r.conn, dumps(stats), now)
        for kind, key, ts, body in docs:
            store.put_doc(r.conn, kind, key, ts, body)
        for evm, owed, body in accts:
            store.put_account(r.conn, evm, owed, body, now)
    return 200, {"ok": True}


def get_fly_claims(r: Req):
    authenticate(r)
    after = int_param(r.q, "after", 0, 0, 10 ** 18)
    n = int_param(r.q, "limit", 50, 1, 50)
    out = []
    for c in store.received_claims(r.conn, after, n):
        item = {k: c[k] for k in ("id", "nonce", "evm", "sol", "evm_sig", "sol_sig", "domain", "uri", "chain_id",
                                  "sol_chain", "created_at")}
        item["issued_at"], item["expires_at"] = rfc3339(c["issued_at"]), rfc3339(c["expires_at"])
        out.append(item)
    return 200, {"claims": out}


def post_results(r: Req):
    d = parse_json(authenticate(r))
    results = d.get("results") if isinstance(d, dict) else None
    if not isinstance(results, list):
        raise HttpError(400, "results must be a list")
    updates = []
    for res in results:
        if not isinstance(res, dict) or not _is_int(res.get("id")) or res["id"] <= 0:
            raise HttpError(400, "each result needs an integer id")
        if res.get("status") not in store.STATUSES:
            raise HttpError(400, "status must be one of " + "|".join(store.STATUSES))
        fields = {}
        if "reason" in res:
            if res["reason"] is not None and not isinstance(res["reason"], str):
                raise HttpError(400, "reason must be a string or null")
            fields["reason"] = None if res["reason"] is None else res["reason"][:500]
        if "lamports" in res:
            if res["lamports"] is not None and not (_is_int(res["lamports"]) and res["lamports"] >= 0):
                raise HttpError(400, "lamports must be a non-negative integer or null")
            fields["lamports"] = res["lamports"]
        if "tx" in res:
            if res["tx"] is not None and not (isinstance(res["tx"], str) and len(res["tx"]) <= 200):
                raise HttpError(400, "tx must be a string of at most 200 characters or null")
            fields["tx"] = res["tx"]
        updates.append((res["id"], res["status"], fields))
    now = int(r.now)
    with store.write(r.conn):
        unknown = [cid for cid, status, fields in updates if not store.set_result(r.conn, cid, status, fields, now)]
    if unknown:
        log("results for unknown claim ids ignored: %s" % unknown[:20])
    return 200, {"ok": True}


def get_probe(r: Req):
    authenticate(r)
    writable, wal = store.probe(r.conn, r.cfg.db_path, int(r.now))
    headers = {k: v for k, v in r.environ.items() if k.startswith("HTTP_") and k != "HTTP_COOKIE"}
    return 200, {"python": sys.version, "db_path": r.cfg.db_path, "writable": writable, "wal": wal,
                 "pid": os.getpid(), "remote_addr": r.environ.get("REMOTE_ADDR"), "headers": headers,
                 # beyond the spec, for the deployment checklist:
                 "sqlite": sqlite3.sqlite_version, "time": r.now, "client_ip": r.ip,
                 "multiprocess": bool(r.environ.get("wsgi.multiprocess")),
                 "multithread": bool(r.environ.get("wsgi.multithread"))}


ROUTES = {
    ("GET", "/api/stats"): get_stats,
    ("GET", "/api/history"): get_history,
    ("GET", "/api/account"): get_account,
    ("GET", "/api/claim/challenge"): get_challenge,
    ("POST", "/api/claim"): post_claim,
    ("POST", "/api/fly/push"): post_push,
    ("GET", "/api/fly/claims"): get_fly_claims,
    ("POST", "/api/fly/claims/result"): post_results,
    ("GET", "/api/fly/probe"): get_probe,
}
_CLAIM_ID = re.compile(r"/api/claim/([1-9][0-9]{0,17})")


def _route(method: str, path: str):
    """(handler, args) or raises 404/405."""
    h = ROUTES.get((method, path))
    if h:
        return h, ()
    m = _CLAIM_ID.fullmatch(path)
    if m:
        if method != "GET":
            raise HttpError(405, "method not allowed", [("Allow", "GET")])
        return get_claim, (int(m.group(1)),)
    allowed = sorted(meth for meth, p in ROUTES if p == path)
    if allowed:
        raise HttpError(405, "method not allowed", [("Allow", ", ".join(allowed))])
    raise HttpError(404, "not found")


def _respond(start_response, status: int, payload, headers=()):
    if isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = dumps(payload).encode("utf-8")
    start_response("%d %s" % (status, HTTPStatus(status).phrase),
                   [("Content-Type", "application/json"), ("Cache-Control", "no-store"),
                    ("Content-Length", str(len(body)))] + list(headers))
    return [body]


def application(environ, start_response):
    conn = None
    try:
        handler, args = _route(environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", ""))
        cfg = config()
        if cfg is None:
            raise HttpError(503, "relay not configured")
        conn = store.connect(cfg.db_path)
        status, payload = handler(Req(environ, cfg, conn, clock()), *args)
        return _respond(start_response, status, payload)
    except HttpError as e:
        return _respond(start_response, e.status, {"error": e.message}, e.headers)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if isinstance(e, sqlite3.DatabaseError) and not isinstance(e, sqlite3.IntegrityError):
            return _respond(start_response, 503, {"error": "storage unavailable"}, [("Retry-After", "5")])
        return _respond(start_response, 500, {"error": "internal error"})
    finally:
        if conn is not None:
            conn.close()
