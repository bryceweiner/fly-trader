"""Read-side helpers for the console and `status`. Everything counts rows; nothing reads status flags."""
from __future__ import annotations

import json

from .connection import connect


def q(sql: str, params=None) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def q1(sql: str, params=None) -> dict | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def status() -> dict:
    out: dict = {}
    out["wealth"] = q("""SELECT DISTINCT ON (book) book, ts, wealth, sol_free, positions_value, exposure, n_open, peak, drawdown
                         FROM wealth_marks ORDER BY book, ts DESC""")
    out["positions"] = q("SELECT book, status, count(*) AS n, COALESCE(sum(realized_sol),0) AS realized FROM positions GROUP BY 1,2 ORDER BY 1,2")
    out["fills"] = q("SELECT book, count(*) AS n, min(ts) AS first, max(ts) AS last FROM fills GROUP BY 1 ORDER BY 1")
    out["beats_last_hour"] = (q1("SELECT count(*) AS n, max(ts) AS last FROM beats WHERE ts > now() - interval '1 hour'") or {})
    out["decisions_last_hour"] = q("SELECT kind, COALESCE(rail,'') AS rail, count(*) AS n FROM decisions WHERE ts > now() - interval '1 hour' GROUP BY 1,2 ORDER BY 3 DESC")
    out["capture"] = q1("SELECT * FROM capture_status")
    out["tape"] = q1("SELECT count(*) AS n, max(ts) AS last FROM swap_tape WHERE ts > now() - interval '1 hour'")
    out["watch_pools"] = q1("SELECT count(*) FILTER (WHERE active) AS active, count(*) AS total FROM watch_pools")
    out["tokens"] = q("SELECT watch_status, count(*) AS n FROM tokens GROUP BY 1 ORDER BY 1")
    out["circuit"] = q1("SELECT * FROM circuit_state WHERE id = 1")
    out["brain"] = q1("SELECT * FROM brain_state")
    out["synapse"] = q1("SELECT count(*) AS n, max(ts) AS last, avg(frob) AS avg_frob FROM synapse_updates WHERE ts > now() - interval '1 hour'")
    out["processes"] = q("SELECT name, pid, started_at, stopped_at, exit_code FROM processes ORDER BY started_at DESC LIMIT 10")
    out["events"] = q("SELECT ts, level, source, message FROM events ORDER BY ts DESC LIMIT 10")
    return out


def print_status() -> None:
    s = status()
    print(json.dumps(s, indent=1, default=str))
