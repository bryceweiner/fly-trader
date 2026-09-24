"""Telegram alerts for the unattended vault fly: halts, kill switch, claims that fail or stick, settlements, releases,
relay outages, low gas. No token configured = log only. Repeats of the same key are suppressed for ``cooldown_s``."""
from __future__ import annotations

import logging
import threading
import time

import httpx

from .. import config
from ..logging_setup import scrub

log = logging.getLogger(__name__)
_last: dict[str, float] = {}
_lock = threading.Lock()


def send(text: str, key: str | None = None, cooldown_s: float = 1800.0) -> bool:
    """Best effort; never raises. ``key`` groups repeats (e.g. 'relay_down')."""
    if key:
        with _lock:
            now = time.time()
            if now - _last.get(key, 0.0) < cooldown_s:
                return False
            _last[key] = now
    msg = f"[fly vault{'' if config.VAULT_CLUSTER == 'mainnet-beta' else ' ' + config.VAULT_CLUSTER}] {text}"
    log.warning("alert: %s", scrub(msg))
    token, chat = config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
    if not token or not chat:
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": msg[:4000], "disable_web_page_preview": True}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.warning("telegram alert failed: %s", type(e).__name__)
        return False
