"""Persist every external API call to ``api_calls``. Never stores keys; never breaks the caller."""
from __future__ import annotations

import json
import logging

from .connection import transaction
from ..logging_setup import scrub

log = logging.getLogger(__name__)


def record_api_call(service: str, endpoint: str, method: str = "GET", status: int | None = None,
                    latency_ms: int | None = None, ok: bool = True, error: str | None = None,
                    request: dict | None = None, response_bytes: int | None = None) -> None:
    try:
        req_json = None
        if request is not None:
            req_json = json.loads(scrub(json.dumps(request, default=str)))
        with transaction() as conn:
            conn.execute(
                "INSERT INTO api_calls (service, endpoint, method, status, latency_ms, ok, error, request, response_bytes) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (service, endpoint, method, status, latency_ms, ok, scrub(error) if error else None,
                 json.dumps(req_json) if req_json is not None else None, response_bytes),
            )
    except Exception as e:  # a DB hiccup must never break an API call
        log.warning("api_calls insert failed: %s", type(e).__name__)


def record_event(level: str, source: str, message: str, detail: dict | None = None) -> None:
    try:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO events (level, source, message, detail) VALUES (%s,%s,%s,%s)",
                (level, source, scrub(message), json.dumps(detail, default=str) if detail is not None else None),
            )
    except Exception as e:
        log.warning("events insert failed: %s", type(e).__name__)
