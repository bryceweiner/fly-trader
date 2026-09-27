"""The signer's Unix socket: one JSON request per line, one JSON reply per line, one request at a time (so the caps
and the claim ledger see every signature in order). The socket file is group-writable for the brain's group only; the
directory it lives in is the only thing the two containers share."""
from __future__ import annotations

import json
import logging
import os
import socketserver
from pathlib import Path

from .core import SERVED, Signer
from .policy import PolicyError

log = logging.getLogger("fly.signer")
MAX_REQUEST = 256 * 1024


def handle(signer: Signer, line: bytes) -> dict:
    try:
        req = json.loads(line)
        method, params = req.get("method"), req.get("params") or {}
        if method not in SERVED or not isinstance(params, dict):
            return {"ok": False, "code": "policy", "error": f"unknown method {method!r}"}
        out = getattr(signer, method)(**params)
        log.info("signed %s", method)
        return {"ok": True, "result": out}
    except PolicyError as e:
        log.warning("refused %s: %s", locals().get("method"), e)
        return {"ok": False, "code": e.code, "error": str(e)}
    except TypeError as e:
        return {"ok": False, "code": "policy", "error": f"bad parameters: {e}"}
    except Exception as e:                                  # an RPC failure etc.: the caller retries later
        log.exception("error in %s", locals().get("method"))
        return {"ok": False, "code": "error", "error": f"{type(e).__name__}: {e}"}


def serve(signer: Signer, path: str, group_mode: int = 0o660) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()

    class H(socketserver.StreamRequestHandler):
        timeout = 60

        def handle(self):
            line = self.rfile.readline(MAX_REQUEST + 1)
            if not line or len(line) > MAX_REQUEST:
                return
            self.wfile.write(json.dumps(handle(signer, line)).encode() + b"\n")

    old = os.umask(0o117)
    try:
        srv = socketserver.UnixStreamServer(str(p), H)
    finally:
        os.umask(old)
    os.chmod(p, group_mode)
    log.info("signer listening on %s for %s", p, signer.pubkeys())
    srv.serve_forever()
