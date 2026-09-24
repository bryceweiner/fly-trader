"""Drives web/wsgi.py in-process (through wsgiref's validator) with a temp relay.json and DB.

Must stay importable on Python 3.8 and must not import fly_trader.
"""
from __future__ import annotations

import io
import json
import os
import sys
import warnings
from urllib.parse import urlencode
from wsgiref.util import setup_testing_defaults
from wsgiref.validate import validator

import pytest

WEB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(WEB)
if WEB not in sys.path:
    sys.path.insert(0, WEB)

import wsgi  # noqa: E402
from relay import app as relay_app, hmacauth, ratelimit  # noqa: E402

KEY_ID = "k1"
SECRET = "5f" * 32
T0 = 1790553600  # 2026-09-28T00:00:00Z, the mainnet vector's issued_at


def relay_config(db: str, **over) -> dict:
    cfg = {"keys": {KEY_ID: SECRET, "k2": "a1" * 32}, "db_path": db, "domain": "fly-trader.app",
           "uri": "https://fly-trader.app/vault.html", "chain_id": 4663, "sol_chain": "mainnet",
           "claim_ttl_s": 900, "min_lamports": 2000000, "client_ip_header": None}
    cfg.update(over)
    return cfg


@pytest.fixture(scope="session")
def vectors():
    with open(os.path.join(REPO, "tests", "vectors", "claim_v1.json")) as f:
        return {v["name"]: v for v in json.load(f)["vectors"]}


class Response:
    def __init__(self, status, headers, body):
        self.status = int(status.split()[0])
        self.status_line = status
        self.headers = headers
        self.body = body

    def header(self, name):
        vals = [v for n, v in self.headers if n.lower() == name.lower()]
        assert len(vals) <= 1, "duplicate header %s" % name
        return vals[0] if vals else None

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class Client:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.now = float(T0)
        self.config_path = str(tmp_path / "relay.json")
        self.db_path = str(tmp_path / "relay.sqlite3")
        monkeypatch.setenv("FLY_RELAY_CONFIG", self.config_path)
        monkeypatch.setattr(relay_app, "clock", lambda: self.now)
        monkeypatch.setattr(relay_app, "_last_purge", [0.0])
        monkeypatch.setattr(ratelimit, "LIMITS", dict(ratelimit.LIMITS))
        self.site = tmp_path / "site"
        self.site.mkdir()
        monkeypatch.setattr(wsgi, "ROOT", str(self.site))
        monkeypatch.setattr(wsgi, "HEADERS_PATH", str(tmp_path / "site-headers.json"))
        self.write_config()

    def write_config(self, **over):
        with open(self.config_path, "w") as f:
            json.dump(relay_config(self.db_path, **over), f)
        relay_app._cfg_cache.update(key=None, cfg=None)

    def request(self, method, path, query="", body=b"", headers=None, ip="203.0.113.7", content_length=None):
        if isinstance(query, dict):
            query = urlencode(query)
        env = {"REQUEST_METHOD": method, "SCRIPT_NAME": "", "PATH_INFO": path, "QUERY_STRING": query, "REMOTE_ADDR": ip,
               "wsgi.input": io.BytesIO(body),
               "CONTENT_LENGTH": str(len(body) if content_length is None else content_length)}
        for k, v in (headers or {}).items():
            env["HTTP_" + k.upper().replace("-", "_")] = v
        setup_testing_defaults(env)
        captured = {}

        def start_response(status, hdrs, exc_info=None):
            captured["status"], captured["headers"] = status, hdrs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validator(wsgi.application)(env, start_response)
            try:
                data = b"".join(result)
            finally:
                result.close()
        return Response(captured["status"], captured["headers"], data)

    def get(self, path, query="", **kw):
        return self.request("GET", path, query, **kw)

    def post(self, path, obj, **kw):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        return self.request("POST", path, body=body, **kw)

    def fly(self, method, path, query="", obj=None, key_id=KEY_ID, secret=SECRET, ts=None, nonce=None, headers=None,
            **kw):
        if isinstance(query, dict):
            query = urlencode(query)
        body = b"" if obj is None else (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
        h = hmacauth.sign(method, path, query, body, key_id, secret, ts=self.now if ts is None else ts, nonce=nonce)
        h.update(headers or {})
        return self.request(method, path, query, body=body, headers=h, **kw)

    def push(self, **payload):
        r = self.fly("POST", "/api/fly/push", obj=payload)
        assert r.status == 200, r.body
        return r


@pytest.fixture
def client(tmp_path, monkeypatch):
    return Client(tmp_path, monkeypatch)
