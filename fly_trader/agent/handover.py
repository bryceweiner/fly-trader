"""The handover: the plastic fly takes the selector's seat (agent/fly_session.py decides; ui_settings['handover']).

After it the selector stops entering, exits to flat and is dropped while a fly trades; the weekly training schedule
stops (train/pipeline.due runs only on request). If no fly can trade (replaced definitions, a failed re-bootstrap), the
selector trades again until one can.
"""
from __future__ import annotations

import json

KEY = "handover"


def _key(chain: str | None) -> str:
    """Each chain's fly takes its own chain's seat (fly_trader/markets: 'handover', 'handover_rh')."""
    if chain in (None, "sol"):
        return KEY
    from ..markets import for_chain
    return for_chain(chain).handover_key


def state(conn, chain: str | None = "sol") -> dict | None:
    r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (_key(chain),)).fetchone()
    if not r or r["value"] is None:
        return None
    v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])
    return v or None


def clear(conn, chain: str | None = "sol") -> None:
    """The chain's seat goes back to the selector: its fly has not won its own race (agent/fly_session.FlyBook._new_race)."""
    conn.execute("DELETE FROM ui_settings WHERE key = %s", (_key(chain),))


def record(conn, detail: dict, chain: str | None = "sol") -> None:
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                 (_key(chain), json.dumps(detail, default=str)))
