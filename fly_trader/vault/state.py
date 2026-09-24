"""Small persistent vault state in ``vault_kv``: the halt flag and named values (relay cursors, the vault start).

A halt (unknown transaction signed by our key, lamports leaving without our signature, a settlement that cannot be
trusted) stops claims and new entries until the operator clears it with ``fly-trader vault resume``.
"""
from __future__ import annotations

import logging

from psycopg.types.json import Jsonb

from ..db.connection import transaction

log = logging.getLogger(__name__)


def get(key: str, default=None, conn=None):
    def _q(c):
        r = c.execute("SELECT value FROM vault_kv WHERE key = %s", (key,)).fetchone()
        return default if r is None or r["value"] is None else r["value"]
    if conn is not None:
        return _q(conn)
    with transaction() as c:
        return _q(c)


def put(key: str, value, conn=None) -> None:
    def _q(c):
        c.execute("INSERT INTO vault_kv (key, value, updated_at) VALUES (%s, %s, now()) "
                  "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (key, Jsonb(value)))
    if conn is not None:
        _q(conn)
    else:
        with transaction() as c:
            _q(c)


def halt(reason: str, detail: dict | None = None) -> None:
    """Stop claims and entries. Idempotent: the first reason is kept, later ones are appended."""
    cur = halted() or {}
    reasons = list(cur.get("reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    put("halt", {"reasons": reasons, "detail": detail or cur.get("detail")})
    try:
        from ..agent import rails
        rails.set_entries_paused(True)
    except Exception:
        log.exception("vault halt: could not pause entries")
    try:
        from . import alerts
        alerts.send(f"VAULT HALTED: {reason}. Claims and entries are stopped until `fly-trader vault resume`.")
    except Exception:
        log.exception("vault halt: alert failed")


def halted() -> dict | None:
    v = get("halt")
    return v if v and v.get("reasons") else None


def resume() -> None:
    put("halt", None)
