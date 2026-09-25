"""Red team of the off-chain profit-redemption path (fly_trader/vault + web/relay), 2026-09-25.

Tests named ``test_refused_*`` document an attack that is correctly refused and pass. Tests named ``test_vuln_*``
assert the SAFE behaviour for a real weakness and FAIL until it is fixed.
"""
import io
import json
from pathlib import Path

import pytest
from solders.keypair import Keypair

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.execution.broker_live import await_confirmation as REAL_AWAIT
from fly_trader.vault import claim_message as cm, claims, settle, sigs, state

V = json.loads((Path(__file__).parent / "vectors" / "claim_v1.json").read_text())["vectors"][1]   # rehearsal: devnet/46630
V_MAIN = json.loads((Path(__file__).parent / "vectors" / "claim_v1.json").read_text())["vectors"][0]
PAYER = Keypair.from_seed(bytes([9]) * 32)
VICTIM_EVM_KEY = bytes.fromhex(V["test_keys"]["evm_private_key"][2:])
VICTIM_SOL = Keypair.from_seed(bytes.fromhex(V["test_keys"]["sol_seed"]))
ATTACKER_EVM_KEY = bytes([7]) * 32
ATTACKER_SOL = Keypair.from_seed(bytes([7]) * 32)
OWED = 30_000_000


# ============================================================ fly side (Postgres test DB)
@pytest.fixture
def fly(monkeypatch):
    for k, v in {"VAULT_ENABLED": True, "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": "http://devnet.invalid",
                 "VAULT_SITE_DOMAIN": V["fields"]["domain"], "VAULT_SITE_URI": V["fields"]["uri"], "RH_CHAIN_ID": 46630,
                 "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": False, "GAS_RESERVE_SOL": 0.01, "TELEGRAM_BOT_TOKEN": None}.items():
        monkeypatch.setattr(config, k, v)
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": 123}))
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv"):
            c.execute(f"DELETE FROM {t}")
        sid = c.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) "
                        "VALUES (now() - interval '7 days', now(), 'allocated', 50000000, %s) RETURNING id", (OWED,)).fetchone()["id"]
        c.execute("INSERT INTO vault_allocations VALUES (%s, %s, 1, %s)", (sid, V["evm_checksummed"], OWED))
    yield


class Relay:
    def __init__(self, claims_):
        self.claims, self.reports = claims_, []

    def pending_claims(self, after=0, limit=50):
        return [c for c in self.claims if int(c["id"]) > after]

    def report(self, results):
        self.reports.extend(results)


class Rpc:
    def __init__(self, balance=5_000_000_000):
        self.balance, self.sent = balance, []

    def get_balance(self, pk): return self.balance
    def get_latest_blockhash(self): return {"blockhash": "11111111111111111111111111111111", "last_valid_block_height": 100}
    def send_transaction(self, b64): self.sent.append(b64); return "sig"
    def get_signature_statuses(self, sigs_, search_history=False): return [None]
    def get_block_height(self): return 50


def _now():
    return cm.parse_rfc3339(V["fields"]["issued_at"]) + 60


def _signed(rid, evm_key=VICTIM_EVM_KEY, sol_kp=VICTIM_SOL, sign_fields=None, **over):
    """A relay claim row. ``sign_fields`` are the fields the keys actually signed; ``over`` is what the row says."""
    f = dict(V["fields"], sol=str(sol_kp.pubkey()))
    f.update(sign_fields or {})
    cf = cm.ClaimFields(**f)
    row = {"id": rid, **f, "evm_sig": sigs.sign_evm(cm.evm_text(cf), evm_key),
           "sol_sig": str(sol_kp.sign_message(cm.sol_text(cf).encode()))}
    row.update(over)
    return row


def _paid_total():
    with transaction() as c:
        return int(c.execute("SELECT COALESCE(sum(lamports), 0) AS s FROM vault_flows WHERE kind = 'claim'").fetchone()["s"])


def _statuses():
    with transaction() as c:
        return [(r["id"], r["status"], r["reason"]) for r in c.execute("SELECT id, status, reason FROM vault_claims ORDER BY id")]


def test_refused_swap_solana_destination_after_signing(fly):
    # relay (or MITM) rewrites the destination on a genuinely signed claim
    tampered = _signed(1, sol=str(ATTACKER_SOL.pubkey()))
    # attacker re-signs the Solana half with his own key but keeps the victim's EVM signature (bound to the victim's SOL)
    half = _signed(2, nonce="1" * 32, sign_fields={"nonce": "1" * 32})
    atk = _signed(99, sol_kp=ATTACKER_SOL, evm_key=ATTACKER_EVM_KEY, sign_fields={"nonce": "1" * 32, "sol": str(ATTACKER_SOL.pubkey())})
    half.update(sol=str(ATTACKER_SOL.pubkey()), sol_sig=atk["sol_sig"])
    rpc = Rpc()
    out = claims.process(Relay([tampered, half]), rpc, PAYER, now=_now())
    assert out["paid"] == 0 and out["rejected"] == 2 and not rpc.sent


def test_refused_claim_for_someone_elses_evm_address(fly):
    # attacker signs both texts himself but names the victim's EVM address as the account to be paid out
    row = _signed(1, evm_key=ATTACKER_EVM_KEY, sol_kp=ATTACKER_SOL)
    rpc = Rpc()
    out = claims.process(Relay([row]), rpc, PAYER, now=_now())
    assert out["rejected"] == 1 and not rpc.sent and "another address" in _statuses()[0][2]


def test_refused_relay_supplied_text_and_extra_fields_are_ignored(fly):
    row = _signed(1, evm_text="anything", sol_text="anything", lamports=10 ** 15, status="verified")
    rpc = Rpc()
    assert claims.process(Relay([row]), rpc, PAYER, now=_now())["paid"] == 1
    assert _paid_total() == OWED                                   # the relay cannot choose the amount


@pytest.mark.parametrize("over,why", [
    ({"chain_id": 4663}, "network"), ({"sol_chain": "mainnet"}, "network"),
    ({"domain": "evil.example"}, "site"), ({"uri": "https://evil.example/vault.html"}, "site"),
])
def test_refused_cross_chain_or_cross_site_replay(fly, over, why):
    row = _signed(1, sign_fields=over)                             # a valid signature, for another deployment
    rpc = Rpc()
    out = claims.process(Relay([row]), rpc, PAYER, now=_now())
    assert out["rejected"] == 1 and not rpc.sent and why in _statuses()[0][2]


def test_refused_mainnet_vector_replayed_to_rehearsal_fly(fly):
    f = V_MAIN["fields"]
    row = {"id": 1, **f, "evm_sig": V_MAIN["evm_sig"], "sol_sig": V_MAIN["sol_sig"]}
    assert claims.process(Relay([row]), Rpc(), PAYER, now=_now())["rejected"] == 1


def test_refused_same_nonce_replayed_under_new_relay_id(fly):
    rpc = Rpc()
    first = _signed(1)
    assert claims.process(Relay([first]), rpc, PAYER, now=_now())["paid"] == 1
    replay = dict(first, id=2)
    out = claims.process(Relay([first, replay]), rpc, PAYER, now=_now())
    assert out["ingested"] == 0 and out["paid"] == 0 and len(rpc.sent) == 1
    # a fresh, genuinely signed claim after payment: nothing owed
    fresh = _signed(3, sign_fields={"nonce": "2" * 32})
    out = claims.process(Relay([fresh]), rpc, PAYER, now=_now())
    assert out["paid"] == 0 and out["rejected"] == 1 and _paid_total() == OWED


def test_refused_two_valid_claims_in_one_round_pay_once(fly):
    other_sol = Keypair.from_seed(bytes([5]) * 32)
    a = _signed(1)
    b = _signed(2, sol_kp=other_sol, sign_fields={"nonce": "3" * 32})
    rpc = Rpc()
    out = claims.process(Relay([a, b]), rpc, PAYER, now=_now())
    assert out["paid"] == 1 and len(rpc.sent) == 1 and _paid_total() == OWED


def test_refused_inflight_unique_index_blocks_a_second_verified_claim(fly):
    with transaction() as c:
        c.execute("INSERT INTO vault_claims (evm, sol, nonce, status) VALUES (%s, 'x', 'n1', 'sending')", (V["evm_checksummed"],))
        with pytest.raises(Exception):
            with c.transaction():
                c.execute("INSERT INTO vault_claims (evm, sol, nonce, status) VALUES (%s, 'x', 'n2', 'verified')", (V["evm_checksummed"],))


@pytest.mark.parametrize("issued_ago,ok", [(3 * 3600, False), (60, True), (-2 * 3600, False)])
def test_refused_expired_or_future_dated_claims(fly, issued_ago, ok):
    """Staleness is judged by the fly's own ingest time (created_at = DB clock), not by anything the relay says."""
    import time
    now = int(time.time())
    iss = now - issued_ago
    row = _signed(1, sign_fields={"issued_at": cm.rfc3339(iss), "expires_at": cm.rfc3339(iss + cm.TTL_S)})
    out = claims.process(Relay([row]), Rpc(), PAYER, now=now)
    assert (out["paid"] == 1) is ok


def test_fixed_one_malformed_relay_row_poisons_the_claim_queue(fly):
    """claims.process() ingests outside its per-claim try (claims.py:161-166): a relay row that violates a column
    constraint raises out of process(), the relay cursor never advances, and every later claim is stuck forever."""
    bad = _signed(1, sol=None)                                      # NOT NULL violation on vault_claims.sol
    good = _signed(2, sign_fields={"nonce": "4" * 32})
    rpc = Rpc()
    out = claims.process(Relay([bad, good]), rpc, PAYER, now=_now())
    assert out["paid"] == 1


class HistoryRpc(Rpc):
    """Real Solana semantics: without searchTransactionHistory only the recent status cache (~150 blocks) answers."""
    def __init__(self):
        Rpc.__init__(self)
        self.landed, self.height = set(), 50

    def send_transaction(self, b64):
        self.sent.append(b64)
        return "sig"

    def get_signature_statuses(self, sigs_, search_history=False):
        return [{"err": None, "confirmationStatus": "finalized", "slot": 7} if (s in self.landed and search_history) else None
                for s in sigs_]

    def get_block_height(self):
        return self.height


def test_fixed_resume_after_crash_pays_a_landed_claim_twice(fly, monkeypatch):
    """claims.py:109-122 + broker_live.await_confirmation. The fly broadcasts, then dies before recording the result;
    the payment lands and finalizes. On restart (> ~1 min later) pay() finds the signature WITH history (finalized,
    err None) and calls await_confirmation, which polls WITHOUT history: the status has aged out of the recent cache and
    the block height is past last_valid_block_height, so it returns "expired". Neither resume branch records it, and
    pay() falls through to re-sign and send the whole amount again."""
    import fly_trader.execution.broker_live as bl
    rpc = HistoryRpc()
    monkeypatch.setattr(bl, "await_confirmation", lambda *a: ("expired", None))   # 1st run: we crash before it confirms
    claims.process(Relay([_signed(1)]), rpc, PAYER, now=_now())
    with transaction() as c:
        row = c.execute("SELECT status, tx_signature FROM vault_claims").fetchone()
    assert row["status"] == "sending" and len(rpc.sent) == 1
    rpc.landed.add(row["tx_signature"])                   # ...but it did land and finalize
    rpc.height = 10_000                                   # restart well after, status cache long rolled over
    monkeypatch.setattr(bl, "await_confirmation", REAL_AWAIT)       # the production poller
    claims.process(Relay([]), rpc, PAYER, now=_now())
    assert len(rpc.sent) == 1, "the holder was paid twice"
    assert _statuses()[0][1] == "paid"


# ============================================================ signatures (pure)
def test_refused_malleable_and_malformed_evm_signatures():
    f = cm.ClaimFields(**V["fields"])
    text, addr = cm.evm_text(f), V["evm_checksummed"]
    sig = bytes.fromhex(V["evm_sig"][2:])
    r, s, v = sig[:32], int.from_bytes(sig[32:64], "big"), sig[64]
    # v=0/1 and v=27/28 are the same signature (same signer): accepted, harmless (the nonce makes the claim unique)
    assert sigs.verify_evm(text, "0x" + (sig[:64] + bytes([v - 27])).hex(), addr) == "eoa"
    bad = [
        "0x" + (r + (sigs.SECP256K1_N - s).to_bytes(32, "big") + bytes([v ^ 1])).hex(),  # high-s twin
        "0x" + sig[:64].hex(),                                                           # 64-byte (EIP-2098 compact)
        "0x" + (sig[:64] + bytes([29])).hex(),                                           # v out of range
        "0x" + (bytes(32) + sig[32:]).hex(),                                             # r = 0
        "0x" + (sigs.SECP256K1_N.to_bytes(32, "big") + sig[32:]).hex(),                  # r = n
        "0x" + sig.hex() + "00",                                                         # trailing byte
        "zz",
    ]
    for b in bad:
        with pytest.raises(sigs.SignatureInvalid):
            sigs.verify_evm(text, b, addr)


def test_refused_ed25519_cross_text_and_wrong_key():
    f = cm.ClaimFields(**V["fields"])
    good = str(VICTIM_SOL.sign_message(cm.sol_text(f).encode()))
    sigs.verify_sol(cm.sol_text(f), good, f.sol)
    for text, sig, pk in [(cm.sol_text(f), str(VICTIM_SOL.sign_message(cm.evm_text(f).encode())), f.sol),  # other text
                          (cm.sol_text(f), str(ATTACKER_SOL.sign_message(cm.sol_text(f).encode())), f.sol),  # wrong key
                          (cm.sol_text(f), good, "11111111111111111111111111111111")]:                       # off-curve
        with pytest.raises(sigs.SignatureInvalid):
            sigs.verify_sol(text, sig, pk)
    # small-order public keys with R small-order / S = 0 (the classic "verifies for any message" forgery) are refused
    import base58
    for pk_hex in ("01" + "00" * 31, "00" * 32, "ec" + "ff" * 30 + "7f", "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a"):
        for r_hex in ("01" + "00" * 31, "00" * 32, pk_hex):
            forged = base58.b58encode(bytes.fromhex(r_hex) + bytes(32)).decode()
            with pytest.raises(sigs.SignatureInvalid):
                sigs.verify_sol(cm.sol_text(f), forged, base58.b58encode(bytes.fromhex(pk_hex)).decode())


def test_refused_the_two_texts_are_not_interchangeable():
    f = cm.ClaimFields(**V["fields"])
    e, s = cm.evm_text(f), cm.sol_text(f)
    assert e != s and f.sol in e and cm.evm.to_checksum(f.evm) in s
    for t in (e, s):
        assert f.nonce in t and f.expires_at in t and f.domain in t


def test_refused_eip1271_only_for_addresses_with_code():
    f = cm.ClaimFields(**V["fields"])
    atk = sigs.sign_evm(cm.evm_text(f), ATTACKER_EVM_KEY)

    class EOA:
        def code(self, a): return "0x"
        def eth_call(self, *a, **k): raise AssertionError("must not call an EOA")
    with pytest.raises(sigs.SignatureInvalid):
        sigs.verify_evm(cm.evm_text(f), atk, V["evm_checksummed"], EOA())


def test_fixed_eip1271_accepts_a_contract_that_echoes_calldata():
    """sigs.py:89 checks only the first 4 bytes of the return data. A locker contract whose fallback returns its
    calldata (forwarders, some proxies, `return(0, calldatasize())`) "returns" 0x1626ba7e..., so ANYONE can claim its
    allocation to their own Solana wallet. The return must be exactly one ABI word equal to the magic value."""
    f = cm.ClaimFields(**V["fields"])
    atk = sigs.sign_evm(cm.evm_text(f), ATTACKER_EVM_KEY)

    class Echo:
        def code(self, a): return "0x3660006000376000363d f3".replace(" ", "")
        def eth_call(self, to, data, gas=None): return data           # the fallback echoes msg.data
    with pytest.raises(sigs.SignatureInvalid):
        sigs.verify_evm(cm.evm_text(f), atk, "0x" + "12" * 20, Echo())


# ============================================================ settlement math (pure)
T1 = 345600 + 2921 * 604800
T0 = T1 - 604800
A, B = "0xA", "0xB"


def test_refused_flash_lock_before_boundary_earns_only_its_seconds():
    w = settle.weights({A: 100}, [(T1 - 1, B, 10 ** 9)], T0, T1)        # a whale locks 1 s before the boundary
    assert w[B] == 10 ** 9 * 1 and w[A] == 100 * 604800              # token-seconds, nothing more
    alloc = settle.allocate(1_000_000_000, w)
    assert alloc[B] == 1_000_000_000 * 10 ** 9 // (10 ** 9 + 100 * 604800) and sum(alloc.values()) <= 1_000_000_000
    # the same capital held all week earns 604800x more than the 1-second flash lock
    w2 = settle.weights({A: 10 ** 9}, [(T1 - 1, B, 10 ** 9)], T0, T1)
    assert w2[A] == 604800 * w2[B]


def test_refused_lock_and_request_in_same_block_earns_nothing():
    w = settle.weights({}, [(T0 + 10, B, 10 ** 24), (T0 + 10, B, 0)], T0, T1)
    assert w == {}


def test_refused_cancel_restores_earning_from_the_cancel_time():
    # lock 100, request 100 (earning 0), cancel (earning 100 again) halfway through
    w = settle.weights({A: 100}, [(T0 + 100, A, 0), (T0 + 302400, A, 100)], T0, T1)
    assert w[A] == 100 * 100 + 100 * 302400


def test_refused_allocation_rounding_never_exceeds_the_pot():
    for amount in (0, 1, 7, 10 ** 9 + 3):
        w = {"a": 1, "b": 1, "c": 1, "d": 10 ** 30}
        al = settle.allocate(amount, w)
        assert sum(al.values()) <= amount and all(v > 0 for v in al.values())
    assert settle.allocate(-5, {"a": 1}) == {} and settle.allocate(5, {}) == {} and settle.allocate(5, {"a": 0}) == {}
    p = settle.plan(-10 ** 9, 5, 0, 10 ** 12, 0)                        # a loss: nothing goes out
    assert p["pot"] == 0 and p["amount"] == 0
    p = settle.plan(10 ** 9, 0, 0, 10, 10 ** 6)                         # illiquid: nothing, never negative
    assert p["amount"] == 0


@pytest.fixture
def settle_db(monkeypatch):
    monkeypatch.setattr(config, "VAULT_PERIOD_S", 604800)
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_kv"):
            c.execute(f"DELETE FROM {t}")
    state.put("vault_started_at", T0 - 100)
    yield


def test_fixed_a_missed_week_is_never_settled(settle_db):
    """settle.due() used to return only the NEWEST ended period. If the settle job could not finish a snapshot for a
    whole week (worker down, flow scan or finality timeout), week 1 was skipped for good and its profit paid out by
    week 2's weights. Fixed 2026-09-25: after a settlement, the next window starts where it ended, so missed weeks are
    settled as one window weighted over all of them (profit is cumulative, so that is exact)."""
    assert settle.due(T1 + 60) == (T0, T1)
    with transaction() as c:
        c.execute("INSERT INTO vault_settlements (period_start, period_end, status, snapshot_slot, snapshot_at, native, token_acct, "
                  "open_cost, deposits, withdrawals, payouts, realized) VALUES (to_timestamp(%s), to_timestamp(%s), 'allocated', "
                  "0, now(), 0, 0, 0, 0, 0, 0, 0)", (T0, T1))
    assert settle.due(T1 + 60) is None
    assert settle.due(T1 + 604800 + 60) == (T1, T1 + 604800)              # the normal next week
    assert settle.due(T1 + 3 * 604800 + 60) == (T1, T1 + 3 * 604800)      # three missed weeks: one window


# ============================================================ relay (in-process WSGI, temp SQLite)
WEB = Path(__file__).parents[1] / "web"


class RelayHttp:
    def __init__(self, tmp_path, monkeypatch, **cfg_over):
        monkeypatch.syspath_prepend(str(WEB))
        from relay import app as relay_app, ratelimit
        self.app, self.now = relay_app, 1790557200.0
        cfg = {"keys": {"k1": "c3" * 32}, "db_path": str(tmp_path / "r.sqlite3"), "domain": V["fields"]["domain"],
               "uri": V["fields"]["uri"], "chain_id": 46630, "sol_chain": "devnet", "claim_ttl_s": 900,
               "min_lamports": 2_000_000, "client_ip_header": None, **cfg_over}
        (tmp_path / "relay.json").write_text(json.dumps(cfg))
        monkeypatch.setenv("FLY_RELAY_CONFIG", str(tmp_path / "relay.json"))
        monkeypatch.setattr(relay_app, "clock", lambda: self.now)
        monkeypatch.setattr(relay_app, "_last_purge", [0.0])
        monkeypatch.setattr(ratelimit, "LIMITS", dict(ratelimit.LIMITS))
        relay_app._cfg_cache.update(key=None, cfg=None)

    def req(self, method, path, query="", body=b"", ip="203.0.113.7", headers=None):
        env = {"REQUEST_METHOD": method, "SCRIPT_NAME": "", "PATH_INFO": path, "QUERY_STRING": query, "REMOTE_ADDR": ip,
               "wsgi.input": io.BytesIO(body), "CONTENT_LENGTH": str(len(body))}
        for k, v in (headers or {}).items():
            env["HTTP_" + k.upper().replace("-", "_")] = v
        out = {}
        data = b"".join(self.app.application(env, lambda s, h, e=None: out.update(status=int(s.split()[0]))))
        return out["status"], json.loads(data)

    def fly(self, method, path, query="", obj=None):
        from relay import hmacauth
        body = b"" if obj is None else json.dumps(obj).encode()
        h = hmacauth.sign(method, path, query, body, "k1", "c3" * 32, ts=self.now)
        return self.req(method, path, query, body, headers=h)

    def challenge(self, evm, sol, ip="203.0.113.7", headers=None):
        from urllib.parse import urlencode
        return self.req("GET", "/api/claim/challenge", urlencode({"evm": evm, "sol": sol}), ip=ip, headers=headers)

    def claim(self, nonce, evm, sol, ip="203.0.113.7", evm_sig=None, sol_sig=None):
        body = {"nonce": nonce, "evm": evm, "sol": sol, "evm_sig": evm_sig or "0x" + "11" * 65,
                "sol_sig": sol_sig or str(ATTACKER_SOL.sign_message(b"junk"))}
        return self.req("POST", "/api/claim", body=json.dumps(body).encode(), ip=ip)


@pytest.fixture
def relay(tmp_path, monkeypatch):
    r = RelayHttp(tmp_path, monkeypatch)
    assert r.fly("POST", "/api/fly/push", obj={"accounts": [{"evm": V["fields"]["evm"], "owed": OWED}]})[0] == 200
    yield r            # modules stay imported: web/relay/tests patch the same module objects (see test_vault_relay_e2e)


VICTIM = V["fields"]["evm"]
VICTIM_SOL_ADDR = V["fields"]["sol"]
ATTACKER_SOL_ADDR = str(ATTACKER_SOL.pubkey())


def _junk(relay, ip):
    """A stranger's claim for the victim's EVM address, to the stranger's Solana wallet, with garbage signatures."""
    st, ch = relay.challenge(VICTIM, ATTACKER_SOL_ADDR, ip=ip)
    assert st == 200
    return relay.claim(ch["nonce"], VICTIM, ATTACKER_SOL_ADDR, ip=ip)


def test_fixed_stranger_holds_the_victims_claim_slot_with_junk_signatures(relay):
    """app.py:263-310 never checks signatures, yet treats 'received' as in flight (store.py:17). A stranger with no key
    parks a junk claim on the victim's address; the victim's real claim gets 409 until the fly rejects the junk, and
    the stranger re-parks right after each rejection (worker loop is 10 s)."""
    assert _junk(relay, "198.51.100.1")[0] == 202
    st, ch = relay.challenge(VICTIM, VICTIM_SOL_ADDR, ip="192.0.2.9")
    assert relay.claim(ch["nonce"], VICTIM, VICTIM_SOL_ADDR, ip="192.0.2.9")[0] == 202


def test_fixed_stranger_drains_the_victims_per_address_bucket(relay):
    """The comment at app.py:302 says strangers cannot drain the per-address bucket, but junk signatures pass every
    relay check, so 5 junk claims an hour (3 IPs, or one IPv6 /48) lock a holder out of claiming indefinitely."""
    for i in range(5):
        st, body = _junk(relay, "198.51.100.%d" % (i // 2))
        assert st == 202
        assert relay.fly("POST", "/api/fly/claims/result", obj={"results": [{"id": body["id"], "status": "rejected",
                                                                             "reason": "evm signature is from another address"}]})[0] == 200
    st, ch = relay.challenge(VICTIM, VICTIM_SOL_ADDR, ip="192.0.2.9")
    assert relay.claim(ch["nonce"], VICTIM, VICTIM_SOL_ADDR, ip="192.0.2.9")[0] == 202


def test_fixed_x_forwarded_for_leftmost_entry_is_client_controlled(tmp_path, monkeypatch):
    """app.py:154 takes the FIRST X-Forwarded-For entry. Behind a proxy that appends (nginx
    $proxy_add_x_forwarded_for, most CDNs), that entry is whatever the client sent: every IP rate limit is bypassed.
    With client_ip_header null instead, all clients share the proxy's bucket (10 claims/hour for everyone)."""
    r = RelayHttp(tmp_path, monkeypatch, client_ip_header="X-Forwarded-For")
    codes = [r.challenge(VICTIM, VICTIM_SOL_ADDR, ip="10.0.0.1",
                         headers={"X-Forwarded-For": "1.2.3.%d, 198.51.100.200" % i})[0] for i in range(31)]
    assert codes[-1] == 429


def test_refused_relay_input_confusion(relay):
    bad = [b"[]", b'{"nonce": ["a"]}', b'{"nonce": "' + b"a" * 32 + b'", "evm": 1}', b"\xff", b"NaN",
           json.dumps({"nonce": "a" * 32, "evm": VICTIM + "\u0000", "sol": VICTIM_SOL_ADDR}).encode(),
           json.dumps({"nonce": "ａ" * 32, "evm": VICTIM, "sol": VICTIM_SOL_ADDR}).encode()]
    for b in bad:
        assert relay.req("POST", "/api/claim", body=b)[0] == 400, b
    # SQL metacharacters only ever reach bound parameters or strict validators
    assert relay.req("GET", "/api/account", "evm=0x' OR 1=1 --")[0] == 400
    assert relay.req("GET", "/api/history", "kind=nav&before=' OR 1=1 --")[0] in (200, 400)
    assert relay.req("GET", "/api/history", "kind=nav' --")[0] == 400
    assert relay.req("GET", "/api/claim/1 OR 1=1")[0] == 404
    assert relay.req("GET", "/api/claim/../fly/probe")[0] == 404
    # push: booleans and floats are not integers
    for owed in (True, 1.5, "1", None):
        assert relay.fly("POST", "/api/fly/push", obj={"accounts": [{"evm": VICTIM, "owed": owed}]})[0] == 400


def test_refused_oversized_bodies(relay):
    st, _ = relay.req("POST", "/api/claim", body=b"{" + b" " * 5000 + b"}")
    assert st == 413
    env_body = b"x" * 5000
    env = {"REQUEST_METHOD": "POST", "PATH_INFO": "/api/claim", "QUERY_STRING": "", "REMOTE_ADDR": "1.1.1.1",
           "wsgi.input": io.BytesIO(env_body), "wsgi.input_terminated": True}           # chunked, no Content-Length
    out = {}
    b"".join(relay.app.application(env, lambda s, h, e=None: out.update(status=int(s.split()[0]))))
    assert out["status"] == 413
    assert relay.req("POST", "/api/claim", body=b"{}", headers={})[0] == 400


def test_refused_hmac_forgeries(relay):
    from relay import hmacauth
    body = json.dumps({"accounts": [{"evm": VICTIM, "owed": 10 ** 15}]}).encode()
    h = hmacauth.sign("POST", "/api/fly/push", "", b'{"accounts":[]}', "k1", "c3" * 32, ts=relay.now)
    assert relay.req("POST", "/api/fly/push", body=body, headers=h)[0] == 401                   # body swapped
    h = hmacauth.sign("GET", "/api/fly/claims", "after=0", b"", "k1", "c3" * 32, ts=relay.now)
    assert relay.req("GET", "/api/fly/claims", "after=0", headers=h)[0] == 200
    assert relay.req("GET", "/api/fly/claims", "after=0", headers=h)[0] == 401                  # replayed nonce
    h = hmacauth.sign("GET", "/api/fly/claims", "after=0", b"", "k1", "c3" * 32, ts=relay.now)
    assert relay.req("GET", "/api/fly/claims", "after=0&after=5", headers=h)[0] == 401          # query param added
    h = hmacauth.sign("GET", "/api/fly/probe", "", b"", "k1", "c3" * 32, ts=relay.now)
    assert relay.req("GET", "/api/fly/claims", "", headers=h)[0] == 401                         # path swapped
    h = hmacauth.sign("GET", "/api/fly/claims", "", b"", "k1", "c3" * 32, ts=relay.now - 301)
    assert relay.req("GET", "/api/fly/claims", "", headers=h)[0] == 401                         # stale
    h = hmacauth.sign("GET", "/api/fly/claims", "", b"", "k1", "wrong" * 8, ts=relay.now)
    assert relay.req("GET", "/api/fly/claims", "", headers=h)[0] == 401                         # wrong key
    h = hmacauth.sign("GET", "/api/fly/claims", "", b"", "k1", "c3" * 32, ts=relay.now)
    h["X-Fly-Key"] = "k1 "
    assert relay.req("GET", "/api/fly/claims", "", headers=h)[0] == 401                         # key-id games
    # the public cannot push balances or results
    assert relay.req("POST", "/api/fly/push", body=body)[0] == 401
    assert relay.req("POST", "/api/fly/claims/result", body=b'{"results":[{"id":1,"status":"paid"}]}')[0] == 401


def test_refused_relay_claim_binding(relay):
    # a challenge is bound to (evm, sol): the stranger cannot reuse the victim's nonce for his own Solana wallet
    st, ch = relay.challenge(VICTIM, VICTIM_SOL_ADDR)
    assert relay.claim(ch["nonce"], VICTIM, ATTACKER_SOL_ADDR)[0] == 400
    assert relay.claim(ch["nonce"], VICTIM.upper().replace("0X", "0x"), VICTIM_SOL_ADDR)[0] == 202  # case-insensitive
    assert relay.claim(ch["nonce"], VICTIM, VICTIM_SOL_ADDR, ip="192.0.2.3")[0] == 409             # used
