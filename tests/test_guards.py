"""Small guards found by the review: finite JSON for status/events, worker-thread log routing."""
import logging
import math
import threading

from fly_trader.db import apilog
from fly_trader.logging_setup import _ThreadFilter
from fly_trader.train import progress


def test_finite_json_guards():
    x = {"a": float("inf"), "b": [1.0, float("nan"), {"c": -math.inf}], "d": 2.5}
    assert progress._finite(x) == {"a": None, "b": [1.0, None, {"c": None}], "d": 2.5}
    assert apilog._finite(x) == progress._finite(x)


def test_thread_filter_routes_helper_threads():
    got = []
    def _in_worker():
        f = _ThreadFilter("replay")
        for tn in ("replay", "replay-dl-0", "replay-assemble", "corpus", "MainThread"):
            rec = logging.LogRecord("x", logging.INFO, "", 0, "m", (), None); rec.threadName = tn
            got.append((tn, f.filter(rec)))
    t = threading.Thread(target=_in_worker, name="replay"); t.start(); t.join()
    assert dict(got) == {"replay": True, "replay-dl-0": True, "replay-assemble": True, "corpus": False, "MainThread": False}
    f = _ThreadFilter("capture")   # created on the main thread (CLI use): MainThread records pass
    rec = logging.LogRecord("x", logging.INFO, "", 0, "m", (), None); rec.threadName = "MainThread"
    assert f.filter(rec) is True
