"""JSON for the console's API and WebSocket: datetimes as ISO strings, Decimals and numpy scalars as numbers, NaN as null."""
from __future__ import annotations

import math
from datetime import timedelta
from decimal import Decimal

import orjson

_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS


def _default(o):
    if isinstance(o, Decimal):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, timedelta):
        return o.total_seconds()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, (bytes, bytearray, memoryview)):
        return bytes(o).hex()
    if hasattr(o, "item"):                      # numpy scalars orjson leaves alone (e.g. float16)
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj) -> bytes:
    """orjson writes NaN and infinities as null, datetimes as RFC 3339 (with their offset), dict keys of any type as strings."""
    return orjson.dumps(obj, default=_default, option=_OPTS)


def loads(b: bytes | str):
    return orjson.loads(b)
