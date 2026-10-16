"""The console's core (fly_trader/ui/{server,hub,security,runtime}.py): the request guard a login-less localhost console
relies on, the WebSocket that pushes topics, actions and their refusals, and the dashboard fallback."""
import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from fly_trader.ui import hub, runtime, security
from fly_trader.ui.hub import ActionError


def _scope(kind="http", method="GET", **headers):
    return {"type": kind, "method": method, "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()]}


@pytest.mark.parametrize("host,ok", [("localhost:8501", True), ("127.0.0.1:8501", True), ("[::1]:8501", True), ("localhost", True),
                                     ("evil.example:8501", False), ("fly.attacker.test", False), ("", False)])
def test_only_loopback_hosts_are_served(host, ok):
    assert (security.refusal(_scope(host=host)) is None) is ok


def test_allowed_hosts_extend_by_env(monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_HOSTS", "fly-vault, 100.64.0.7")
    assert security.refusal(_scope(host="fly-vault:8501")) is None
    assert security.refusal(_scope(host="100.64.0.7:8501")) is None
    assert security.refusal(_scope(host="other:8501")) is not None


def test_writes_need_the_console_header_and_this_origin():
    h = {"host": "localhost:8501"}
    assert "x-fly-console" in security.refusal(_scope(method="POST", **h))
    assert security.refusal(_scope(method="POST", x_fly_console="1", **h)) is None                                 # curl, tests
    assert security.refusal(_scope(method="POST", x_fly_console="1", origin="http://localhost:8501", **h)) is None
    assert security.refusal(_scope(method="POST", x_fly_console="1", origin="https://evil.example", **h)) is not None
    assert security.refusal(_scope(method="POST", x_fly_console="1", origin="null", **h)) is not None
    assert security.refusal(_scope(method="POST", x_fly_console="1", sec_fetch_site="cross-site", **h)) is not None
    assert security.refusal(_scope(method="GET", origin="https://evil.example", **h)) is None                      # reads are same-origin-policy protected


def test_websockets_must_come_from_this_origin():
    assert security.refusal(_scope("websocket", host="localhost:8501", origin="http://localhost:8501")) is None
    assert security.refusal(_scope("websocket", host="localhost:8501", origin="https://evil.example")) is not None


def test_guard_answers_403_over_http(console):
    assert console.post("/api/action/anything", json={}).status_code == 403                         # no header
    r = console.post("/api/action/anything", json={}, headers={"X-Fly-Console": "1", "Origin": "https://evil.example"})
    assert r.status_code == 403 and "cross-origin" in r.json()["error"]
    assert console.get("/api/meta", headers={"host": "evil.example"}).status_code == 403


@pytest.fixture
def temp_registry():
    """Topics and actions registered by a test, removed afterwards."""
    added_t, added_a = [], []

    def t(name, fn, every=0.05):
        hub.TOPICS[name] = hub.Topic(name, fn, every); added_t.append(name)

    def a(name, fn):
        hub.ACTIONS[name] = hub.Action(name, fn); added_a.append(name)
    yield t, a
    for n in added_t:
        hub.TOPICS.pop(n, None)
    for n in added_a:
        hub.ACTIONS.pop(n, None)


def test_actions_return_ok_or_the_refusal(console, temp_registry):
    _, a = temp_registry
    a("t.ok", lambda body: {"message": f"did {body.get('x')}"})
    a("t.refused", lambda body: (_ for _ in ()).throw(ActionError("Stop the trading engine first.")))
    a("t.observer", lambda body: runtime.Observer().start("runner"))
    a("t.boom", lambda body: 1 / 0)
    assert console.act("t.ok", {"x": 3}).json() == {"ok": True, "message": "did 3"}
    r = console.act("t.refused"); assert r.status_code == 409 and r.json()["error"] == "Stop the trading engine first."
    r = console.act("t.observer"); assert r.status_code == 409 and "only observes" in r.json()["error"]
    r = console.act("t.boom"); assert r.status_code == 500 and "ZeroDivisionError" in r.json()["error"]
    assert console.act("t.missing").status_code == 404
    assert console.post("/api/action/t.ok", content=b"not json", headers={"X-Fly-Console": "1"}).status_code == 400


def test_topics_over_rest(console, temp_registry):
    t, _ = temp_registry
    t("t.echo", lambda p: {"got": p})
    assert console.topic("t.echo", a=1).json() == {"got": {"a": 1}}
    assert console.get("/api/topic/t.missing").status_code == 400
    assert console.get("/api/topic/t.echo", params={"params": "{bad"}).status_code == 400


def test_websocket_pushes_changes_errors_and_stops_on_unsub(console, temp_registry):
    t, _ = temp_registry
    n = {"i": 0}; lock = threading.Lock()

    def counter(p):
        with lock:
            n["i"] += 1
            return {"i": min(n["i"], 3), "p": p}             # changes three times, then holds still
    t("t.count", counter)
    t("t.fail", lambda p: (_ for _ in ()).throw(RuntimeError("panel broke")))
    with console.websocket_connect("ws://localhost/ws", headers={"origin": "http://localhost"}) as ws:
        ws.send_json({"op": "sub", "id": "a", "topic": "t.count", "params": {"chain": "sol"}})
        seen = [ws.receive_json() for _ in range(3)]
        assert [m["data"]["i"] for m in seen] == [1, 2, 3] and all(m["op"] == "data" and m["id"] == "a" for m in seen)
        assert seen[0]["data"]["p"] == {"chain": "sol"}
        ws.send_json({"op": "sub", "id": "b", "topic": "t.fail"})
        m = ws.receive_json(); assert m == {"op": "error", "id": "b", "error": "RuntimeError: panel broke"}
        ws.send_json({"op": "sub", "id": "c", "topic": "t.nope"})
        m = ws.receive_json(); assert m["op"] == "error" and "no topic" in m["error"]
        ws.send_json({"op": "unsub", "id": "a"}); ws.send_json({"op": "unsub", "id": "b"})
        ws.send_json({"op": "ping"})
        assert ws.receive_json() == {"op": "pong"}         # nothing else arrives once unsubscribed


def test_two_subscribers_share_one_poller(console, temp_registry):
    t, _ = temp_registry
    calls = []
    t("t.shared", lambda p: calls.append(1) or {"v": 1}, every=60)
    app_hub = console.app.state.hub
    with console.websocket_connect("ws://localhost/ws", headers={"origin": "http://localhost"}) as w1, \
            console.websocket_connect("ws://localhost/ws", headers={"origin": "http://localhost"}) as w2:
        w1.send_json({"op": "sub", "id": "x", "topic": "t.shared", "params": {"k": 1}})
        assert w1.receive_json()["data"] == {"v": 1}
        w2.send_json({"op": "sub", "id": "y", "topic": "t.shared", "params": {"k": 1}})
        assert w2.receive_json() == {"op": "data", "id": "y", "data": {"v": 1}}    # served from the poller's last value
        assert len([k for k in app_hub.pollers if k.startswith("t.shared?")]) == 1 and len(calls) == 1


def test_cross_origin_websocket_is_refused(console):
    with pytest.raises(WebSocketDisconnect):
        with console.websocket_connect("ws://localhost/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


def test_dashboard_routes_fall_back_to_the_app(console):
    for path in ("/", "/vault", "/kalshi-trades", "/no/such/page"):
        r = console.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert console.get("/api/nothing").status_code == 404
    r = console.get("/static/brain/../../server.py")          # normalised to /server.py: the app, never the source
    assert "def create_app" not in r.text


def test_meta_says_what_is_turned_on(console, monkeypatch):
    from fly_trader import config
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", None)
    m = console.get("/api/meta").json()
    assert m["kalshi"] is False and m["hosts_workers"] is False and m["markets"][0]["chain"] == "sol"


def test_status_strip_hides_dormant_kalshi(console, monkeypatch):
    from fly_trader import config
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", None)
    rows = console.topic("status").json()["rows"]
    assert [r["key"] for r in rows][0] == "sol" and "kalshi" not in [r["key"] for r in rows]


def test_reused_connection_survives_a_dropped_session():
    from fly_trader.db import queries
    out = {}

    def run():
        queries.reuse_connection_in_this_thread()
        out["a"] = queries.q1("SELECT 1 AS one")["one"]
        queries._tls.conn.close()                           # the server dropped it
        out["b"] = queries.q1("SELECT 2 AS two")["two"]
        with queries.read_connection() as c:
            out["c"] = c.execute("SELECT 3 AS three").fetchone()["three"]
    th = threading.Thread(target=run); th.start(); th.join()
    assert out == {"a": 1, "b": 2, "c": 3}


def test_binds_loopback_except_in_a_container(monkeypatch):
    from fly_trader.ui import server
    monkeypatch.delenv("CONSOLE_HOST", raising=False)
    inside = {"on": False}
    monkeypatch.setattr(server.Path, "exists", lambda self: inside["on"] and str(self) == "/.dockerenv")
    assert server.default_host() == "127.0.0.1"
    inside["on"] = True
    assert server.default_host() == "0.0.0.0"              # a self-updated container keeps its old image: decided in code
    monkeypatch.setenv("CONSOLE_HOST", "100.64.0.7")
    assert server.default_host() == "100.64.0.7"


def test_an_observer_refuses_every_action(monkeypatch):
    """Every registered write is refused by a console that only observes, before it runs (hub.run_action)."""
    from fastapi.testclient import TestClient
    from fly_trader.ui.server import create_app
    hosted = runtime.HOSTS_WORKERS
    try:
        app = create_app(hosts_workers=False)                       # no fake supervisor: runtime.sup() is the Observer
        ran = []
        monkeypatch.setattr(hub, "ACTIONS", {n: hub.Action(n, lambda body, n=n: ran.append(n) or {}) for n in hub.ACTIONS})
        with TestClient(app, base_url="http://localhost") as c:
            for name in hub.ACTIONS:
                r = c.post(f"/api/action/{name}", json={}, headers={"X-Fly-Console": "1"})
                assert r.status_code == 409 and "only observes" in r.json()["error"], name
        assert ran == [] and len(hub.ACTIONS) > 20
    finally:
        runtime.HOSTS_WORKERS = hosted


def test_cross_site_reads_of_the_api_are_refused(console):
    assert console.get("/api/meta", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert console.get("/api/meta", headers={"Origin": "https://evil.example"}).status_code == 403
    assert console.get("/api/meta", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert console.get("/vault", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200      # a link to the dashboard still opens it


def test_a_recovered_topic_clears_its_error(console, temp_registry):
    t, _ = temp_registry
    state = {"fail": False}

    def flaky(p):
        if state["fail"]:
            raise RuntimeError("db blip")
        return {"v": 1}
    t("t.flaky", flaky)
    with console.websocket_connect("ws://localhost/ws", headers={"origin": "http://localhost"}) as ws:
        ws.send_json({"op": "sub", "id": "f", "topic": "t.flaky"})
        assert ws.receive_json()["data"] == {"v": 1}
        state["fail"] = True
        assert ws.receive_json()["op"] == "error"
        state["fail"] = False
        assert ws.receive_json() == {"op": "data", "id": "f", "data": {"v": 1}}    # same data as before the error, sent again


def test_runs_elsewhere_sees_another_consoles_thread_and_cli_processes(db_conn, monkeypatch):
    import os
    from fly_trader.ops import procs
    from fly_trader.ui import state
    from fly_trader.db.connection import transaction
    with transaction() as conn:
        conn.execute("UPDATE processes SET stopped_at = now() WHERE name = 'replay' AND stopped_at IS NULL")
        rid = conn.execute("INSERT INTO processes (name, pid, cmd, started_by) VALUES ('replay', %s, %s, 'test') RETURNING id",
                           (os.getppid(), ["thread", "replay"])).fetchone()["id"]                 # a live process that is not this one
    try:
        assert "another console" in state.runs_elsewhere("replay")
        with transaction() as conn:
            conn.execute("UPDATE processes SET cmd = %s WHERE id = %s", (["python", "-m", "fly_trader"], rid))
        monkeypatch.setattr(procs, "_is_ours", lambda pid, name: True)
        assert "outside the console" in state.runs_elsewhere("replay")
        with transaction() as conn:
            conn.execute("UPDATE processes SET pid = %s WHERE id = %s", (os.getpid(), rid))
        assert state.runs_elsewhere("replay") is None                                          # this console's own row
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM processes WHERE id = %s", (rid,))
