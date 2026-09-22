"""Shim for better_bot's ``db_connection``: ``kalshi_stream.flush``/``top_structures`` are never called by fly-trader
(its own stream handlers live in fly_trader/kalshi/stream.py), so the connection is the same adapter as ``database``."""
from .database import get_connection  # noqa: F401
