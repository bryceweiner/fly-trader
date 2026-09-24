"""JSON-lines logging with a secret scrubber, safe for several workers in ONE process.

Pattern from VOC dexlp/logging_setup.py. setup(name) is idempotent: the root logger gets one stderr
handler; each worker name gets a file handler (logs/<name>.log) and an in-memory ring buffer (for
the console) that only accept records emitted from the thread named <name> (or the main thread, for
the CLI). The scrubber removes API keys and any 87-88-char base58 string (a 64-byte secret key)."""
from __future__ import annotations

import collections
import json
import logging
import logging.handlers
import re
import sys
import threading
from pathlib import Path

from . import config

_B58_SECRET = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{87,88}\b")
_API_KEY_QS = re.compile(r"(api[-_]?key=)[^&\s\"']+", re.IGNORECASE)
_API_KEY_HDR = re.compile(r"(['\"]?x-api-key['\"]?\s*[:=]\s*['\"]?)[^'\",\s}]+", re.IGNORECASE)
RING: dict[str, collections.deque] = {}
_configured: set[str] = set()
_lock = threading.Lock()


def scrub(text: str) -> str:
    text = _B58_SECRET.sub("<secret>", text)
    text = _API_KEY_QS.sub(r"\1<redacted>", text)
    text = _API_KEY_HDR.sub(r"\1<redacted>", text)
    for name in config.SECRET_ENV_NAMES:
        val = config.env_str(name)
        if val and len(val) >= 8 and val in text:
            text = text.replace(val, "<redacted>")
    return text


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        record.msg = scrub(msg)
        record.args = ()
        return True


class _ThreadFilter(logging.Filter):
    """Keep records from the worker thread and its helper threads (``<name>-*``); accept MainThread only when the worker itself
    runs in the main thread (CLI use), never the console's Streamlit server thread."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        import threading as _t
        self.main_ok = _t.current_thread() is _t.main_thread()

    def filter(self, record: logging.LogRecord) -> bool:
        tn = record.threadName or ""
        return tn == self.name or tn.startswith(self.name + "-") or (self.main_ok and tn == "MainThread")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "logger": record.name, "thread": record.threadName, "msg": record.getMessage()}
        if record.exc_info:
            payload["exc"] = scrub(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


class _RingHandler(logging.Handler):
    def __init__(self, name: str):
        super().__init__()
        RING.setdefault(name, collections.deque(maxlen=500))
        self.buf = RING[name]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append(self.format(record))
        except Exception:
            pass


def setup(name: str, level: int = logging.INFO, to_file: bool = True) -> logging.Logger:
    with _lock:
        root = logging.getLogger()
        root.setLevel(level)
        fmt = _JsonFormatter()
        if "_root" not in _configured:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            sh.addFilter(_ScrubFilter())
            root.addHandler(sh)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("websockets").setLevel(logging.WARNING)
            logging.getLogger("watchdog").setLevel(logging.WARNING)
            _configured.add("_root")
        if name not in _configured:
            tf = _ThreadFilter(name)
            if to_file:
                config.LOG_DIR.mkdir(parents=True, exist_ok=True)
                fh = logging.handlers.RotatingFileHandler(Path(config.LOG_DIR) / f"{name}.log", maxBytes=50_000_000, backupCount=5)
                fh.setFormatter(fmt)
                fh.addFilter(_ScrubFilter())
                fh.addFilter(tf)
                root.addHandler(fh)
            rh = _RingHandler(name)
            rh.setFormatter(fmt)
            rh.addFilter(_ScrubFilter())
            rh.addFilter(tf)
            root.addHandler(rh)
            _configured.add(name)
    return logging.getLogger(name)


def tail(name: str, n: int = 60) -> str:
    return "\n".join(list(RING.get(name, []))[-n:])
