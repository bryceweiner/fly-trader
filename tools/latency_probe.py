"""Measure the minute pipeline's latency from outside the running console, read-only (no profiler can attach on macOS
without root): every 0.25 s for ``seconds``, what the console's database connections are executing
(pg_stat_activity), when each stream and book status lands (ui_settings), the console's CPU and every live order.

    .venv/bin/python tools/latency_probe.py [seconds=1200]
    .venv/bin/python tools/latency_report.py logs/latency/probe_<UTC stamp>.jsonl

Writes logs/latency/probe_<UTC stamp>.jsonl. Run it before and after a change to the minute pipeline: the 2026-10-19
baseline of the old code is in logs/latency_baseline_20261019/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

REPO = Path(__file__).resolve().parents[1]
KEYS = ("pumpstream_status", "rh_stream_status", "fly_status", "selector_status", "fly_status_rh", "selector_status_rh")
STEP_S = 0.25


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0
    out = REPO / "logs" / "latency"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}.jsonl"
    found = subprocess.run(["pgrep", "-f", "fly_trader ui --port 8501"], capture_output=True, text=True).stdout.split()
    console = int(found[0]) if found else None
    conn = psycopg.connect(os.environ.get("DATABASE_URL", "postgresql:///fly_trader"), autocommit=True, row_factory=dict_row)
    me = conn.execute("SELECT pg_backend_pid() AS p").fetchone()["p"]
    t_end = time.time() + seconds
    with open(path, "w") as f:
        while time.time() < t_end:
            t = time.time()
            act = conn.execute("""SELECT pid, state, extract(epoch from query_start) AS qs, extract(epoch from xact_start) AS xs, wait_event_type, wait_event,
                                         left(regexp_replace(query, '\\s+', ' ', 'g'), 160) AS q
                                  FROM pg_stat_activity WHERE datname = current_database() AND pid <> %s AND state IS NOT NULL AND state <> 'idle'""",
                               (me,)).fetchall()
            st = {r["key"]: {"at": r["at"], "minute": r["minute"], "ft": r["ft"], "lat": r["lat"]} for r in conn.execute(
                """SELECT key, extract(epoch from updated_at) AS at, value->>'minute' AS minute, value->>'flushed_through' AS ft, value->>'latency_s' AS lat
                   FROM ui_settings WHERE key = ANY(%s)""", (list(KEYS),)).fetchall()}
            orders = conn.execute("SELECT id, extract(epoch from ts) AS t, side, status FROM orders WHERE book = 'live' AND ts > to_timestamp(%s) - interval '2 minutes' ORDER BY id",
                                  (t,)).fetchall()
            cpu = None
            if console is not None:
                try:
                    cpu = float(subprocess.run(["ps", "-o", "%cpu=", "-p", str(console)], capture_output=True, text=True, timeout=2).stdout.strip() or 0)
                except Exception:
                    pass
            f.write(json.dumps({"t": t, "act": [{k: (float(v) if k in ("qs", "xs") and v is not None else v) for k, v in r.items()} for r in act],
                                "st": {k: {kk: (float(vv) if kk == "at" and vv is not None else vv) for kk, vv in v.items()} for k, v in st.items()},
                                "orders": [{k: (float(v) if k == "t" else v) for k, v in o.items()} for o in orders], "cpu": cpu}, default=str) + "\n")
            f.flush()
            time.sleep(max(0.0, STEP_S - (time.time() - t)))
    print(path)


if __name__ == "__main__":
    main()
