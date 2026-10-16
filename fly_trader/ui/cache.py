"""A small thread-safe TTL cache for the console's slow reads (parquet globs, chain balances, the brain's files)."""
from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict


def ttl_cache(ttl: float | None = None, maxsize: int = 128):
    """Memoize by positional and keyword arguments for ``ttl`` seconds (None: forever, like a resource). Concurrent misses
    on one key may compute twice; the last result wins."""
    def deco(fn):
        store: OrderedDict = OrderedDict(); lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit is not None and (ttl is None or now - hit[0] < ttl):
                    store.move_to_end(key)
                    return hit[1]
            val = fn(*args, **kwargs)
            with lock:
                store[key] = (now, val); store.move_to_end(key)
                while len(store) > maxsize:
                    store.popitem(last=False)
            return val

        def cache_clear():
            with lock:
                store.clear()
        wrapper.cache_clear = cache_clear
        return wrapper
    return deco
