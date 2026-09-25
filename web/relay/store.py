"""SQLite storage for the relay. One connection per request; WAL + busy_timeout make several uWSGI processes safe.

Only SQL that SQLite 3.8 era builds understand (no UPSERT, no RETURNING, no row values): Gandi's Python may link an
old library. Read-then-write sequences run under BEGIN IMMEDIATE so they are serialized across processes.
"""
from __future__ import annotations

import base64
import os
import sqlite3
import threading
from contextlib import contextmanager

from . import hmacauth, ratelimit

BUSY_MS = 5000
IN_FLIGHT = ("received", "verified", "waiting_liquidity", "sending")
STATUSES = IN_FLIGHT + ("paid", "rejected", "failed")
CHALLENGE_KEEP_S = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, body TEXT NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS docs (kind TEXT NOT NULL, id TEXT NOT NULL, ts INTEGER NOT NULL, body TEXT NOT NULL,
                                 PRIMARY KEY (kind, id));
CREATE INDEX IF NOT EXISTS docs_order ON docs (kind, ts, id);
CREATE TABLE IF NOT EXISTS accounts (evm TEXT PRIMARY KEY, owed INTEGER NOT NULL, body TEXT NOT NULL,
                                     updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS challenges (nonce TEXT PRIMARY KEY, evm TEXT NOT NULL, sol TEXT NOT NULL,
    issued_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, domain TEXT NOT NULL, uri TEXT NOT NULL,
    chain_id INTEGER NOT NULL, sol_chain TEXT NOT NULL, used_at INTEGER);
CREATE INDEX IF NOT EXISTS challenges_expiry ON challenges (expires_at);
CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY AUTOINCREMENT, nonce TEXT NOT NULL UNIQUE,
    evm TEXT NOT NULL, sol TEXT NOT NULL, evm_sig TEXT NOT NULL, sol_sig TEXT NOT NULL, domain TEXT NOT NULL,
    uri TEXT NOT NULL, chain_id INTEGER NOT NULL, sol_chain TEXT NOT NULL, issued_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, lamports INTEGER, tx TEXT,
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS claims_status ON claims (status, id);
CREATE INDEX IF NOT EXISTS claims_evm ON claims (evm, status);
CREATE TABLE IF NOT EXISTS hmac_nonces (nonce TEXT PRIMARY KEY, seen_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS hmac_nonces_seen ON hmac_nonces (seen_at);
CREATE TABLE IF NOT EXISTS buckets (key TEXT PRIMARY KEY, tokens REAL NOT NULL, updated REAL NOT NULL);
"""

_ready = set()
_ready_lock = threading.Lock()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=BUSY_MS / 1000.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = %d" % BUSY_MS)
    conn.execute("PRAGMA synchronous = NORMAL")
    if db_path not in _ready:
        with _ready_lock:
            if db_path not in _ready:
                _init(conn)
                _ready.add(db_path)
    return conn


def _init(conn) -> None:
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass  # another process holds the file mid-switch; it will leave it in WAL
    with write(conn):
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                conn.execute(stmt)


@contextmanager
def write(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# ---- stats / history / accounts (pushed by the fly) ----
def put_stats(conn, body: str, now: int) -> None:
    conn.execute("INSERT OR REPLACE INTO kv(key, body, updated_at) VALUES ('stats', ?, ?)", (body, now))


def get_stats(conn) -> "str | None":
    row = conn.execute("SELECT body FROM kv WHERE key = 'stats'").fetchone()
    return row[0] if row else None


def put_doc(conn, kind: str, key: str, ts: int, body: str) -> None:
    conn.execute("INSERT OR REPLACE INTO docs(kind, id, ts, body) VALUES (?, ?, ?, ?)", (kind, key, ts, body))


def _cursor(ts: int, key: str) -> str:
    return base64.urlsafe_b64encode(("%d:%s" % (ts, key)).encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> "tuple[int, str] | None":
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("utf-8")
        ts, key = raw.split(":", 1)
        return int(ts), key
    except (ValueError, UnicodeDecodeError, TypeError):
        return None


def history_page(conn, kind: str, before: "tuple[int, str] | None", limit: int):
    """Newest first by (ts, id). Returns (bodies, next_cursor or None)."""
    if before is None:
        rows = conn.execute("SELECT ts, id, body FROM docs WHERE kind = ? ORDER BY ts DESC, id DESC LIMIT ?",
                            (kind, limit + 1)).fetchall()
    else:
        ts, key = before
        rows = conn.execute("SELECT ts, id, body FROM docs WHERE kind = ? AND ts <= ? AND (ts < ? OR id < ?) "
                            "ORDER BY ts DESC, id DESC LIMIT ?", (kind, ts, ts, key, limit + 1)).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    return [r[2] for r in rows], (_cursor(rows[-1][0], rows[-1][1]) if more else None)


def put_account(conn, evm: str, owed: int, body: str, now: int) -> None:
    conn.execute("INSERT OR REPLACE INTO accounts(evm, owed, body, updated_at) VALUES (?, ?, ?, ?)",
                 (evm, owed, body, now))


def get_account(conn, evm: str):
    """(owed, body) or None."""
    return conn.execute("SELECT owed, body FROM accounts WHERE evm = ?", (evm,)).fetchone()


# ---- challenges and claims ----
def add_challenge(conn, c: dict) -> None:
    conn.execute("INSERT INTO challenges(nonce, evm, sol, issued_at, expires_at, domain, uri, chain_id, sol_chain) "
                 "VALUES (:nonce, :evm, :sol, :issued_at, :expires_at, :domain, :uri, :chain_id, :sol_chain)", c)


def get_challenge(conn, nonce: str) -> "dict | None":
    cur = conn.execute("SELECT * FROM challenges WHERE nonce = ?", (nonce,))
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row)) if row else None


def use_challenge(conn, nonce: str, now: int) -> bool:
    cur = conn.execute("UPDATE challenges SET used_at = ? WHERE nonce = ? AND used_at IS NULL", (now, nonce))
    return cur.rowcount == 1


# In flight AND verified by the fly. A 'received' claim has only passed syntax checks (the relay checks no
# signatures), so letting it block the address would let a stranger with no keys hold anyone's claim slot; the fly
# rejects junk and pays each address at most once, whatever the relay accepts.
VERIFIED_IN_FLIGHT = ("verified", "waiting_liquidity", "sending")


def claim_in_flight(conn, evm: str) -> bool:
    q = "SELECT 1 FROM claims WHERE evm = ? AND status IN (%s) LIMIT 1" % ",".join("?" * len(VERIFIED_IN_FLIGHT))
    return conn.execute(q, (evm,) + VERIFIED_IN_FLIGHT).fetchone() is not None


def add_claim(conn, ch: dict, evm_sig: str, sol_sig: str, now: int) -> int:
    cur = conn.execute(
        "INSERT INTO claims(nonce, evm, sol, evm_sig, sol_sig, domain, uri, chain_id, sol_chain, issued_at, "
        "expires_at, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'received', ?, ?)",
        (ch["nonce"], ch["evm"], ch["sol"], evm_sig, sol_sig, ch["domain"], ch["uri"], ch["chain_id"],
         ch["sol_chain"], ch["issued_at"], ch["expires_at"], now, now))
    return cur.lastrowid


_CLAIM_COLS = ("id", "nonce", "evm", "sol", "evm_sig", "sol_sig", "domain", "uri", "chain_id", "sol_chain",
               "issued_at", "expires_at", "status", "reason", "lamports", "tx", "created_at", "updated_at")


def get_claim(conn, claim_id: int) -> "dict | None":
    row = conn.execute("SELECT %s FROM claims WHERE id = ?" % ",".join(_CLAIM_COLS), (claim_id,)).fetchone()
    return dict(zip(_CLAIM_COLS, row)) if row else None


def received_claims(conn, after: int, limit: int) -> list:
    rows = conn.execute("SELECT %s FROM claims WHERE status = 'received' AND id > ? ORDER BY id LIMIT ?"
                        % ",".join(_CLAIM_COLS), (after, limit)).fetchall()
    return [dict(zip(_CLAIM_COLS, r)) for r in rows]


def set_result(conn, claim_id: int, status: str, fields: dict, now: int) -> bool:
    """``fields`` holds only the keys the fly sent (reason/lamports/tx); omitted ones keep their value."""
    sets = ["status = ?", "updated_at = ?"]
    args = [status, now]
    for k in ("reason", "lamports", "tx"):
        if k in fields:
            sets.append("%s = ?" % k)
            args.append(fields[k])
    cur = conn.execute("UPDATE claims SET %s WHERE id = ?" % ", ".join(sets), args + [claim_id])
    return cur.rowcount == 1


# ---- HMAC nonces, housekeeping, probe ----
def remember_nonce(conn, nonce: str, now: int) -> bool:
    """False when the nonce was already seen (a replay)."""
    try:
        conn.execute("INSERT INTO hmac_nonces(nonce, seen_at) VALUES (?, ?)", (nonce.lower(), now))
    except sqlite3.IntegrityError:
        return False
    return True


def purge(conn, now: float) -> None:
    conn.execute("DELETE FROM hmac_nonces WHERE seen_at < ?", (int(now) - hmacauth.NONCE_TTL_S,))
    conn.execute("DELETE FROM challenges WHERE expires_at < ?", (int(now) - CHALLENGE_KEEP_S,))
    ratelimit.purge(conn, now)


def probe(conn, db_path: str, now: int) -> "tuple[bool, bool]":
    """(writable, wal): a real write, and the journal mode actually in force."""
    wal = (conn.execute("PRAGMA journal_mode").fetchone()[0] or "").lower() == "wal"
    try:
        with write(conn):
            conn.execute("INSERT OR REPLACE INTO kv(key, body, updated_at) VALUES ('probe', '{}', ?)", (now,))
        writable = os.access(os.path.dirname(os.path.abspath(db_path)), os.W_OK)
    except sqlite3.DatabaseError:
        writable = False
    return writable, wal
