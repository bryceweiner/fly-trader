"""The status strip on every page: one row of badges per market (ui/state.py status_rows)."""
from __future__ import annotations

from ..hub import topic
from ..state import status_rows


@topic("status", every=1)
def status(params: dict) -> dict:
    return {"rows": status_rows()}
