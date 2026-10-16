"""Every test runs against fly_trader_test. DATABASE_URL is forced before any fly_trader import so
the refuse-production guard in fly_trader.db.connection can never see the real database."""
import os

os.environ["DATABASE_URL"] = "postgresql:///fly_trader_test"
for _name in ("RH_RPC_URL_TX", "QUICKNODE_RH_API_URL", "QUICKNODE_RH_API_KEY"):   # no test reaches the real transaction endpoint
    os.environ[_name] = ""
for _name in ("LIVE_ENABLED", "RH_LIVE_ENABLED", "KALSHI_LIVE_ENABLED", "LIVE_WITHOUT_HANDOVER"):   # nor runs with the operator's
    os.environ[_name] = "0"                                                                   # live switches from .env
os.environ["RH_CAPITAL_ETH"] = ""      # empty = the code's pinned default, not the operator's wallet balance (load_dotenv never overrides)

import pytest  # noqa: E402

TEST_URL = "postgresql:///fly_trader_test"


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    from fly_trader.db import schema
    schema.ensure_database(TEST_URL)
    schema.apply_schema(TEST_URL)
    yield


@pytest.fixture
def db_conn():
    from fly_trader.db.connection import connect
    conn = connect(TEST_URL)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


class FakeSupervisor:
    """The console's view of the Supervisor in tests: worker state you set, and every control call recorded."""

    def __init__(self):
        from fly_trader.ops.supervisor import WORKERS
        self.alive_set: set[str] = set()
        self.calls: list[tuple] = []
        self.pid = os.getpid()
        self.autostarted = True
        self.workers = WORKERS

    def status(self) -> dict:
        return {n: {"alive": n in self.alive_set, "started_at": None, "stopped_at": None, "error": None, "stopping": False} for n in self.workers}

    def alive(self, name: str) -> bool:
        return name in self.alive_set

    def external(self) -> dict:
        return {}

    def start(self, name: str, started_by: str = "console") -> bool:
        self.calls.append(("start", name, started_by)); self.alive_set.add(name); return True

    def request_stop(self, name: str, by: str = "console") -> bool:
        self.calls.append(("request_stop", name)); return name in self.alive_set

    def stop(self, name: str, grace_s: float = 90.0) -> bool:
        self.calls.append(("stop", name)); self.alive_set.discard(name); return True

    def log_tail(self, name: str, n: int = 60) -> str:
        return ""


@pytest.fixture
def fake_sup(monkeypatch):
    """Point the console (fly_trader.ui.runtime.sup) at a FakeSupervisor."""
    from fly_trader.ui import runtime
    s = FakeSupervisor()
    monkeypatch.setattr(runtime, "sup", lambda: s)
    return s


@pytest.fixture
def console(fake_sup):
    """A TestClient on the console app (no workers hosted). ``console.act(name, body)`` POSTs an action with the
    console's CSRF header; ``console.topic(name, **params)`` reads a topic."""
    from fastapi.testclient import TestClient
    from fly_trader.ui.server import create_app
    import json as _json
    from fly_trader.ui import runtime
    hosted = runtime.HOSTS_WORKERS
    app = create_app(hosts_workers=False)
    with TestClient(app, base_url="http://localhost") as c:
        c.act = lambda name, body=None: c.post(f"/api/action/{name}", json=body or {}, headers={"X-Fly-Console": "1"})
        c.topic = lambda topic_name, **params: c.get(f"/api/topic/{topic_name}", params={"params": _json.dumps(params)} if params else None)
        c.sup = fake_sup
        yield c
    runtime.HOSTS_WORKERS = hosted