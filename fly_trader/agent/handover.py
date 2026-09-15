"""The handover: the plastic fly takes the selector's seat (agent/fly_session.py decides; ui_settings['handover']).

After it the selector stops entering, exits to flat and is dropped while a fly trades; the weekly training schedule
stops (train/pipeline.due runs only on request). If no fly can trade (replaced definitions, a failed re-bootstrap), the
selector trades again until one can.
"""
from __future__ import annotations

import json

KEY = "handover"


def state(conn) -> dict | None:
    r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (KEY,)).fetchone()
    if not r or r["value"] is None:
        return None
    v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])
    return v or None


def record(conn, detail: dict) -> None:
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                 (KEY, json.dumps(detail, default=str)))
