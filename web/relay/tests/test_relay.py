"""Relay API (docs/vault/SPEC.md §3) driven through web/wsgi.py."""
from __future__ import annotations

import json
import threading

import pytest

from conftest import KEY_ID, SECRET, T0
from relay import app as relay_app, hmacauth, ratelimit, validate

EVM = "0xc85382f3028a0d2a96b13768048da55f27d02332"
SOL = "HdkTVFM1vaZYZeFc9YsDzPT8fmFgq1CL7z8g2m1ujHDo"
OWED = 5000000


def account(evm=EVM, owed=OWED):
    return {"evm": evm, "allocated": owed, "claimed": 0, "in_flight": 0, "owed": owed,
            "allocations": [{"period_end": T0, "lamports": owed, "weight": "1", "share": 1.0}], "claims": []}


def challenge(client, evm=EVM, sol=SOL, ip="198.51.100.1"):
    r = client.get("/api/claim/challenge", {"evm": evm, "sol": sol}, ip=ip)
    assert r.status == 200, r.body
    return r.json()


def claim_body(vec, issued_nonce=None, **over):
    f = vec["fields"]
    b = {"nonce": issued_nonce or f["nonce"], "evm": vec["evm_checksummed"], "sol": f["sol"],
         "evm_sig": vec["evm_sig"], "sol_sig": vec["sol_sig"]}
    b.update(over)
    return b


def submit(client, vec, ip="198.51.100.1", **over):
    """Challenge + claim for the vector's addresses; returns the claim response."""
    ch = challenge(client, vec["fields"]["evm"], vec["fields"]["sol"], ip=ip)
    return client.post("/api/claim", claim_body(vec, ch["nonce"], **over), ip=ip)


def result(client, *results):
    r = client.fly("POST", "/api/fly/claims/result", obj={"results": list(results)})
    assert r.status == 200 and r.json() == {"ok": True}, r.body


# ---- config / availability ----
def test_unconfigured_is_503_and_static_still_served(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FLY_RELAY_CONFIG", str(tmp_path / "missing.json"))
    (client.site / "index.html").write_text("<p>hi</p>")
    r = client.get("/api/stats")
    assert r.status == 503 and r.json() == {"error": "relay not configured"}
    assert r.header("Cache-Control") == "no-store" and r.header("Content-Type") == "application/json"
    assert client.get("/").status == 200


def test_invalid_config_is_503_without_leaking_secrets(client, capsys):
    with open(client.config_path, "w") as f:
        json.dump({"keys": {"k1": "short-secret"}, "db_path": client.db_path}, f)
    relay_app._cfg_cache.update(key=None, cfg=None)
    assert client.get("/api/stats").status == 503
    assert "short-secret" not in capsys.readouterr().err


def test_stats_503_before_first_push(client):
    r = client.get("/api/stats")
    assert r.status == 503 and "error" in r.json()


def test_unknown_route_and_method(client):
    assert client.get("/api/nope").status == 404
    r = client.request("DELETE", "/api/stats")
    assert r.status == 405 and r.header("Allow") == "GET"
    assert client.request("POST", "/api/claim/3").status == 405


# ---- HMAC ----
def test_hmac_ok_and_canonical_string(client):
    assert client.fly("GET", "/api/fly/claims").status == 200
    text = hmacauth.canonical("get", "/api/fly/claims", "limit=5&after=0", "100", "ab" * 16, b"")
    assert text == ("FLY-RELAY-1\nGET\n/api/fly/claims\nafter=0&limit=5\n100\n" + "ab" * 16 + "\n"
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_hmac_canonical_query_ordering_and_quoting(client):
    assert hmacauth.canonical_query("b=2&a=1&a=0&c=") == "a=0&a=1&b=2&c="
    assert hmacauth.canonical_query("x=a b&y=%2F") == "x=a%20b&y=%2F"
    assert hmacauth.canonical_query({"limit": 5, "after": 0}) == "after=0&limit=5"
    # The server canonicalizes whatever order the query arrives in.
    h = hmacauth.sign("GET", "/api/fly/claims", {"limit": "5", "after": "0"}, b"", KEY_ID, SECRET, ts=client.now)
    assert client.request("GET", "/api/fly/claims", "limit=5&after=0", headers=h).status == 200


def test_hmac_bad_signature(client):
    h = hmacauth.sign("GET", "/api/fly/claims", "", b"", KEY_ID, "00" * 32, ts=client.now)
    r = client.request("GET", "/api/fly/claims", headers=h)
    assert r.status == 401 and r.json() == {"error": "bad signature"}


def test_hmac_body_is_signed(client):
    h = hmacauth.sign("POST", "/api/fly/push", "", b'{"stats":{"v":1}}', KEY_ID, SECRET, ts=client.now)
    r = client.request("POST", "/api/fly/push", body=b'{"stats":{"v":2}}', headers=h)
    assert r.status == 401


def test_hmac_query_is_signed(client):
    h = hmacauth.sign("GET", "/api/fly/claims", "after=0", b"", KEY_ID, SECRET, ts=client.now)
    assert client.request("GET", "/api/fly/claims", "after=5", headers=h).status == 401


@pytest.mark.parametrize("skew,ok", [(-300, True), (300, True), (-301, False), (301, False)])
def test_hmac_timestamp_window(client, skew, ok):
    r = client.fly("GET", "/api/fly/claims", ts=client.now + skew)
    assert (r.status == 200) is ok
    if not ok:
        assert r.status == 401 and "server time" in r.json()["error"]


def test_hmac_replayed_nonce(client):
    nonce = "cd" * 16
    assert client.fly("GET", "/api/fly/claims", nonce=nonce).status == 200
    r = client.fly("GET", "/api/fly/claims", nonce=nonce)
    assert r.status == 401 and r.json() == {"error": "replayed nonce"}
    # Forgotten after 15 min; by then the timestamp alone would be refused anyway.
    client.now += 16 * 60
    assert client.fly("GET", "/api/fly/claims", nonce=nonce).status == 200


def test_hmac_wrong_key_id_and_rotation(client):
    r = client.fly("GET", "/api/fly/claims", key_id="k9")
    assert r.status == 401 and r.json() == {"error": "unknown key"}
    assert client.fly("GET", "/api/fly/claims", key_id="k2", secret="a1" * 32).status == 200
    assert client.fly("GET", "/api/fly/claims", key_id="k2", secret=SECRET).status == 401


def test_hmac_missing_headers(client):
    assert client.get("/api/fly/claims").status == 401
    assert client.request("POST", "/api/fly/push", body=b"{}").status == 401


def test_push_body_limit(client):
    h = hmacauth.sign("POST", "/api/fly/push", "", b"{}", KEY_ID, SECRET, ts=client.now)
    r = client.request("POST", "/api/fly/push", body=b"{}", headers=h, content_length=2 * 1024 * 1024 + 1)
    assert r.status == 413


# ---- push and public reads ----
def test_push_then_stats_account(client):
    stats = {"v": 1, "ts": T0, "cluster": "mainnet", "prices": {"sol_usd": float("nan"), "fly_usd": 0.1, "ts": T0}}
    client.push(stats=stats, accounts=[account(evm=EVM.upper().replace("0X", "0x"))])
    s = client.get("/api/stats")
    assert s.status == 200 and s.header("Cache-Control") == "no-store"
    assert s.json()["prices"]["sol_usd"] is None and s.json()["ts"] == T0   # NaN is not JSON
    a = client.get("/api/account", {"evm": "0xC85382F3028A0D2A96B13768048DA55F27D02332"}).json()
    assert a["owed"] == OWED and a["allocations"][0]["lamports"] == OWED
    z = client.get("/api/account", {"evm": "0x" + "1" * 40}).json()
    assert z == {"evm": "0x" + "1" * 40, "allocated": 0, "claimed": 0, "in_flight": 0, "owed": 0,
                 "allocations": [], "claims": []}
    assert client.get("/api/account", {"evm": "0x123"}).status == 400
    # Upsert: a later push replaces.
    client.push(stats=dict(stats, ts=T0 + 60), accounts=[account(owed=0)])
    assert client.get("/api/stats").json()["ts"] == T0 + 60
    assert client.get("/api/account", {"evm": EVM}).json()["owed"] == 0


def test_push_rejects_malformed(client):
    for bad in ({"stat": {}}, {"history": {"trade": []}}, {"history": {"nav": [{"nav": 1}]}},
                {"history": {"trades": [{"id": 1}]}}, {"accounts": [{"evm": "nope"}]},
                {"accounts": [{"evm": EVM, "owed": "5"}]}, {"stats": []}):
        r = client.fly("POST", "/api/fly/push", obj=bad)
        assert r.status == 400, bad
    assert client.fly("POST", "/api/fly/push", obj=b"{not json").status == 400


def test_history_pagination_newest_first(client):
    trades = [{"id": i, "mint": "m%d" % i, "symbol": "S", "opened_at": T0 + i, "closed_at": T0 + 100 + i // 2,
               "cost": 1, "proceeds": 2, "realized": 1, "exit_kind": "x"} for i in range(1, 12)]
    client.push(history={"trades": trades})
    seen, cursor = [], None
    while True:
        q = {"kind": "trades", "limit": 4}
        if cursor:
            q["before"] = cursor
        page = client.get("/api/history", q).json()
        assert page["kind"] == "trades" and len(page["items"]) <= 4
        seen += [t["id"] for t in page["items"]]
        cursor = page["next"]
        if cursor is None:
            break
    # newest closed_at first; ties broken by id, numerically
    assert seen == sorted(range(1, 12), key=lambda i: (T0 + 100 + i // 2, i), reverse=True)
    assert client.get("/api/history", {"kind": "trades"}).json()["next"] is None   # default limit 100


def test_history_limits_and_kinds(client):
    for q in ({"kind": "trades", "limit": 0}, {"kind": "trades", "limit": 501}, {"kind": "trades", "limit": "x"},
              {"kind": "bogus"}, {}, {"kind": "nav", "before": "!!!"}):
        assert client.get("/api/history", q).status == 400, q
    assert client.get("/api/history", {"kind": "nav", "limit": 500}).json() == {"kind": "nav", "items": [],
                                                                                "next": None}


def test_history_nav_keyed_by_ts(client):
    client.push(history={"nav": [{"ts": T0, "nav": 1, "index": 1.0, "sol_usd": 150.0},
                                 {"ts": T0 + 60, "nav": 2, "index": 1.1, "sol_usd": 151.0}]})
    client.push(history={"nav": [{"ts": T0, "nav": 9, "index": 1.0, "sol_usd": 150.0}]})
    items = client.get("/api/history", {"kind": "nav"}).json()["items"]
    assert [(i["ts"], i["nav"]) for i in items] == [(T0 + 60, 2), (T0, 9)]


def test_history_flows_and_settlements_upsert_by_id(client):
    client.push(history={"flows": [{"id": "sig1", "ts": T0, "signature": "sig1", "direction": "in",
                                    "kind": "deposit", "counterparty": "x", "lamports": 5}],
                         "settlements": [{"id": 1, "period_start": T0, "period_end": T0 + 604800, "realized": 1,
                                          "pot": 1, "allocated": 1, "carried": 0, "earners": 1,
                                          "total_weight": "1", "status": "open"}]})
    client.push(history={"settlements": [{"id": 1, "period_start": T0, "period_end": T0 + 604800, "realized": 1,
                                          "pot": 1, "allocated": 1, "carried": 0, "earners": 1,
                                          "total_weight": "1", "status": "final"}]})
    s = client.get("/api/history", {"kind": "settlements"}).json()["items"]
    assert len(s) == 1 and s[0]["status"] == "final"
    assert client.get("/api/history", {"kind": "flows"}).json()["items"][0]["id"] == "sig1"


def test_read_rate_limit(client):
    ratelimit.LIMITS["read"] = (3, 3600)
    assert [client.get("/api/account", {"evm": EVM}).status for _ in range(4)] == [200, 200, 200, 429]
    r = client.get("/api/stats")
    assert r.status == 429 and int(r.header("Retry-After")) > 0
    assert client.get("/api/stats", ip="192.0.2.99").status == 503   # other IP, own bucket
    client.now += 1200                                                # refills 1 token / 20 min
    assert client.get("/api/account", {"evm": EVM}).status == 200


def test_client_ip_header(client):
    ratelimit.LIMITS["read"] = (1, 3600)
    client.write_config(client_ip_header="X-Forwarded-For")
    hdr = {"X-Forwarded-For": "192.0.2.10, 10.0.0.1"}
    assert client.get("/api/account", {"evm": EVM}, headers=hdr, ip="10.0.0.1").status == 200
    assert client.get("/api/account", {"evm": EVM}, headers=hdr, ip="10.0.0.2").status == 429
    other = {"X-Forwarded-For": "192.0.2.11"}
    assert client.get("/api/account", {"evm": EVM}, headers=other, ip="10.0.0.1").status == 200


def test_ipv6_bucketed_by_64():
    assert ratelimit.ip_key("2001:db8:1:2:3::1") == ratelimit.ip_key("2001:db8:1:2:ffff::9") == "2001:db8:1:2::/64"
    assert ratelimit.ip_key("::ffff:192.0.2.1") == "192.0.2.1"
    assert ratelimit.ip_key("") == "unknown"


# ---- validation ----
def test_validate_syntax(vectors):
    v = vectors["mainnet"]
    assert validate.sol_address(v["fields"]["sol"]) and validate.sol_signature(v["sol_sig"])
    assert not validate.sol_address(v["sol_sig"]) and not validate.sol_signature(v["fields"]["sol"])
    assert not validate.sol_address("0OIl" + v["fields"]["sol"][4:])
    assert validate.b58decode("1" * 32, 44) == b"\0" * 32
    assert validate.evm_signature(v["evm_sig"]) and validate.evm_signature("0x" + "ab" * 1024)
    assert not validate.evm_signature("0x" + "ab" * 64) and not validate.evm_signature("0x" + "ab" * 1025)
    assert not validate.evm_signature(v["evm_sig"] + "\n") and not validate.evm_signature(v["evm_sig"] + "0")
    assert validate.nonce("0123456789abcdef" * 2) and not validate.nonce("0123456789ABCDEF" * 2)
    assert validate.evm_address(v["evm_checksummed"]) == v["fields"]["evm"]
    assert validate.evm_address(v["fields"]["evm"] + "\n") is None


# ---- challenge ----
@pytest.mark.parametrize("name,config", [
    ("mainnet", {}),
    ("rehearsal", {"domain": "localhost:5173", "uri": "http://localhost:5173/vault.html", "chain_id": 46630,
                   "sol_chain": "devnet"}),
])
def test_challenge_reproduces_vector_fields(client, vectors, monkeypatch, name, config):
    v = vectors[name]
    f = v["fields"]
    client.write_config(**config)
    monkeypatch.setattr(relay_app, "new_nonce", lambda: f["nonce"])
    client.now = float(1790553600 if name == "mainnet" else 1790557200)
    ch = challenge(client, v["evm_checksummed"], f["sol"])
    assert ch == {"nonce": f["nonce"], "issued_at": f["issued_at"], "expires_at": f["expires_at"],
                  "domain": f["domain"], "uri": f["uri"], "chain_id": f["chain_id"], "sol_chain": f["sol_chain"],
                  "request_id": "fly-vault-claim-v1", "min_lamports": 2000000}
    client.push(accounts=[account(evm=f["evm"])])
    r = client.post("/api/claim", claim_body(v))
    assert r.status == 202 and r.json() == {"id": 1, "status": "received"}
    pulled = client.fly("GET", "/api/fly/claims").json()["claims"]
    assert pulled == [{"id": 1, "nonce": f["nonce"], "evm": f["evm"], "sol": f["sol"], "evm_sig": v["evm_sig"],
                       "sol_sig": v["sol_sig"], "domain": f["domain"], "uri": f["uri"], "chain_id": f["chain_id"],
                       "sol_chain": f["sol_chain"], "issued_at": f["issued_at"], "expires_at": f["expires_at"],
                       "created_at": int(client.now)}]


def test_challenge_syntax(client, vectors):
    sol = vectors["mainnet"]["fields"]["sol"]
    for q in ({"evm": "0x123", "sol": sol}, {"evm": EVM, "sol": "not-base58!"}, {"evm": EVM},
              {"sol": sol}, {"evm": EVM, "sol": vectors["mainnet"]["sol_sig"]}):
        r = client.get("/api/claim/challenge", q)
        assert r.status == 400, q


def test_challenge_rate_limit_30_per_hour_per_ip(client):
    codes = [client.get("/api/claim/challenge", {"evm": EVM, "sol": SOL}).status for _ in range(31)]
    assert codes == [200] * 30 + [429]
    assert client.get("/api/claim/challenge", {"evm": EVM, "sol": SOL}, ip="192.0.2.50").status == 200
    client.now += 120   # 30/h refills one token every 2 min
    assert client.get("/api/claim/challenge", {"evm": EVM, "sol": SOL}).status == 200


# ---- claim submit ----
def test_claim_happy_path_and_status(client, vectors):
    v = vectors["mainnet"]
    client.push(accounts=[account()])
    r = submit(client, v)
    assert r.status == 202
    cid = r.json()["id"]
    s = client.get("/api/claim/%d" % cid).json()
    assert s == {"id": cid, "status": "received", "reason": None, "lamports": None, "tx": None,
                 "created_at": T0, "updated_at": T0}
    assert client.get("/api/claim/999").status == 404


def test_claim_unknown_nonce(client, vectors):
    client.push(accounts=[account()])
    r = client.post("/api/claim", claim_body(vectors["mainnet"], "ee" * 16))
    assert r.status == 400 and r.json() == {"error": "unknown nonce"}


def test_claim_expired_nonce(client, vectors):
    client.push(accounts=[account()])
    ch = challenge(client)
    client.now += 901
    r = client.post("/api/claim", claim_body(vectors["mainnet"], ch["nonce"]), ip="198.51.100.1")
    assert r.status == 400 and r.json() == {"error": "nonce expired"}


def test_claim_used_nonce_is_409(client, vectors):
    client.push(accounts=[account()])
    ch = challenge(client)
    body = claim_body(vectors["mainnet"], ch["nonce"])
    assert client.post("/api/claim", body).status == 202
    r = client.post("/api/claim", body)
    assert r.status == 409 and r.json() == {"error": "nonce already used"}


def test_claim_nonce_for_another_address(client, vectors):
    v = vectors["mainnet"]
    client.push(accounts=[account()])
    other = vectors["rehearsal"]["fields"]
    ch = challenge(client, evm=other["evm"])
    r = client.post("/api/claim", claim_body(v, ch["nonce"]))
    assert r.status == 400 and r.json() == {"error": "nonce was issued for another address"}
    ch = challenge(client, sol=other["sol"])
    assert client.post("/api/claim", claim_body(v, ch["nonce"])).status == 400


def test_claim_bad_signature_syntax(client, vectors):
    v = vectors["mainnet"]
    client.push(accounts=[account()])
    ch = challenge(client)
    for over in ({"evm_sig": v["evm_sig"][:-2]}, {"evm_sig": v["evm_sig"][2:]}, {"sol_sig": v["fields"]["sol"]},
                 {"sol_sig": v["sol_sig"] + "1"}, {"evm": "0x12"}, {"nonce": "ABC"}, {"sol": "0" * 44}):
        r = client.post("/api/claim", claim_body(v, ch["nonce"], **over))
        assert r.status == 400, over
    assert client.post("/api/claim", b"[1,2]").status == 400
    assert client.post("/api/claim", b"\xff\xfe").status == 400
    # The challenge was not consumed by any of those.
    assert client.post("/api/claim", claim_body(v, ch["nonce"])).status == 202


def test_claim_owed_checks_use_pushed_account(client, vectors):
    v = vectors["mainnet"]
    r = submit(client, v)                                   # unknown account: owed 0
    assert r.status == 400 and r.json() == {"error": "nothing to claim"}
    client.push(accounts=[account(owed=1999999)])
    r = submit(client, v)
    assert r.status == 400 and "minimum" in r.json()["error"]
    client.push(accounts=[account(owed=2000000)])
    assert submit(client, v).status == 202


def test_claim_in_flight_409_then_allowed_after_terminal(client, vectors):
    v = vectors["mainnet"]
    client.push(accounts=[account()])
    first = submit(client, v).json()["id"]
    # an unverified ('received') claim does not block: the relay checks no signatures, so it may be a stranger's junk
    second = submit(client, v)
    assert second.status == 202
    result(client, {"id": second.json()["id"], "status": "rejected", "reason": "duplicate"})
    for status in ("verified", "waiting_liquidity", "sending"):
        result(client, {"id": first, "status": status})
        r = submit(client, v)
        assert r.status == 409 and "in flight" in r.json()["error"], status
    for terminal in ("paid", "rejected", "failed"):
        result(client, {"id": first, "status": terminal})
        r = submit(client, v, ip="192.0.2.%d" % len(terminal))
        assert r.status == 202, terminal
        first = r.json()["id"]


def test_claim_body_over_4kb_is_413(client, vectors):
    body = json.dumps(claim_body(vectors["mainnet"], pad="x" * 4096)).encode()
    assert client.post("/api/claim", body).status == 413
    # A 1024-byte contract-wallet signature still fits.
    client.push(accounts=[account()])
    assert submit(client, vectors["mainnet"], evm_sig="0x" + "ab" * 1024).status == 202


def test_claim_per_ip_limit(client, vectors):
    v = vectors["mainnet"]
    codes = [client.post("/api/claim", claim_body(v, "%032x" % i)).status for i in range(11)]
    assert codes == [400] * 10 + [429]
    assert client.post("/api/claim", claim_body(v, "%032x" % 99), ip="192.0.2.1").status == 400


def test_claim_per_evm_limit(client, vectors):
    v = vectors["mainnet"]
    client.push(accounts=[account()])
    for i in range(5):
        r = submit(client, v, ip="192.0.2.1")
        assert r.status == 202
        result(client, {"id": r.json()["id"], "status": "rejected", "reason": "bad signature"})
    r = submit(client, v, ip="192.0.2.1")
    assert r.status == 429 and r.header("Retry-After")
    # the bucket is per address AND client: junk from one client cannot lock the holder out
    assert submit(client, v, ip="192.0.2.77").status == 202


def test_concurrent_submits_of_one_nonce_accept_exactly_one(client, vectors):
    client.push(accounts=[account()])
    ch = challenge(client)
    body = claim_body(vectors["mainnet"], ch["nonce"])
    codes = []
    threads = [threading.Thread(target=lambda i=i: codes.append(
        client.post("/api/claim", body, ip="192.0.2.%d" % i).status)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [202] + [409] * 7


# ---- fly pull and results ----
def test_fly_pull_only_received_in_id_order(client, vectors):
    client.push(accounts=[account(), account(evm=vectors["rehearsal"]["fields"]["evm"])])
    a = submit(client, vectors["mainnet"]).json()["id"]
    b = submit(client, vectors["rehearsal"], ip="192.0.2.2").json()["id"]
    result(client, {"id": a, "status": "verified", "lamports": OWED})
    c = submit(client, vectors["mainnet"], ip="192.0.2.3")
    assert c.status == 409
    ids = [x["id"] for x in client.fly("GET", "/api/fly/claims", {"after": 0, "limit": 50}).json()["claims"]]
    assert ids == [b]
    result(client, {"id": a, "status": "rejected"})
    c = submit(client, vectors["mainnet"], ip="192.0.2.3").json()["id"]
    pulled = client.fly("GET", "/api/fly/claims", {"after": 0}).json()["claims"]
    assert [x["id"] for x in pulled] == [b, c]
    assert [x["id"] for x in client.fly("GET", "/api/fly/claims", {"after": b}).json()["claims"]] == [c]
    assert [x["id"] for x in client.fly("GET", "/api/fly/claims", {"limit": 1}).json()["claims"]] == [b]
    for q in ({"limit": 0}, {"limit": 51}, {"after": -1}):
        assert client.fly("GET", "/api/fly/claims", q).status == 400, q


def test_results_update_status_and_fields(client, vectors):
    client.push(accounts=[account()])
    cid = submit(client, vectors["mainnet"]).json()["id"]
    client.now += 30
    result(client, {"id": cid, "status": "verified", "lamports": OWED})
    result(client, {"id": cid, "status": "waiting_liquidity", "reason": "waiting for SOL"})
    client.now += 30
    result(client, {"id": cid, "status": "paid", "reason": None, "tx": "5" * 88}, {"id": 12345, "status": "paid"})
    s = client.get("/api/claim/%d" % cid).json()
    assert s == {"id": cid, "status": "paid", "reason": None, "lamports": OWED, "tx": "5" * 88,
                 "created_at": T0, "updated_at": T0 + 60}
    for bad in ({"id": cid, "status": "done"}, {"id": "1", "status": "paid"}, {"id": cid, "status": "paid",
                                                                                 "lamports": -1}):
        assert client.fly("POST", "/api/fly/claims/result", obj={"results": [bad]}).status == 400, bad
    assert client.fly("POST", "/api/fly/claims/result", obj={"result": []}).status == 400


# ---- probe ----
def test_probe(client):
    client.write_config(client_ip_header="HTTP_X_REAL_IP")
    r = client.fly("GET", "/api/fly/probe", headers={"X-Real-IP": "192.0.2.44"})
    assert r.status == 200 and r.header("Cache-Control") == "no-store"
    p = r.json()
    assert {"python", "db_path", "writable", "wal", "pid", "remote_addr", "headers"} <= set(p)
    assert p["writable"] is True and p["wal"] is True and p["db_path"] == client.db_path
    assert p["remote_addr"] == "203.0.113.7" and p["client_ip"] == "192.0.2.44"
    assert p["headers"]["HTTP_X_REAL_IP"] == "192.0.2.44"
    assert client.get("/api/fly/probe").status == 401


def test_relative_db_path_is_next_to_config(client, tmp_path):
    client.write_config(db_path="data.sqlite3")
    assert client.fly("GET", "/api/fly/probe").json()["db_path"] == str(tmp_path / "data.sqlite3")


def test_api_responses_carry_security_headers(client):
    r = client.get("/api/stats")
    assert r.header("X-Content-Type-Options") == "nosniff" and r.header("X-Frame-Options") == "DENY"
    assert r.header("Strict-Transport-Security") == "max-age=31536000"
    assert r.header("Content-Security-Policy") is None
