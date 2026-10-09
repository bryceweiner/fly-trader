"""Shim for better_bot's ``prediction_utils``: only ``utc_now`` is imported by the vendored modules."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
