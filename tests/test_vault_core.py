"""Vault accounting core: claim texts and signatures, settlement math, flow classification, event decoding."""
import json
from pathlib import Path

import pytest
from solders.keypair import Keypair

from fly_trader.vault import claim_message as cm, evm, flows, rh_index, settle, sigs

VECTORS = json.loads((Path(__file__).parent / "vectors" / "claim_v1.json").read_text())["vectors"]
WALLET = str(Keypair.from_seed(bytes(32)).pubkey())
FUNDER = str(Keypair.from_seed(bytes([1]) * 32).pubkey())
STRANGER = str(Keypair.from_seed(bytes([2]) * 32).pubkey())


# ---------------------------------------------------------------- claim texts + signatures
@pytest.mark.parametrize("v", VECTORS, ids=lambda v: v["name"])
def test_vectors_render_and_verify(v):
    f = cm.ClaimFields(**v["fields"])
    f.check()
    assert cm.evm_text(f) == v["evm_text"] and cm.sol_text(f) == v["sol_text"]
    assert sigs.verify_evm(v["evm_text"], v["evm_sig"], v["evm_checksummed"]) == "eoa"
    sigs.verify_sol(v["sol_text"], v["sol_sig"], v["fields"]["sol"])


def test_signature_rejections():
    v = VECTORS[0]
    other = sigs.evm_address(evm.keccak256(b"someone else"))
    with pytest.raises(sigs.SignatureInvalid):
        sigs.verify_evm(v["evm_text"], v["evm_sig"], other)
    with pytest.raises(sigs.SignatureInvalid):          # tampered text
        sigs.verify_evm(v["evm_text"].replace("mainnet", "devnet") + " ", v["evm_sig"], v["evm_checksummed"])
    raw = bytes.fromhex(v["evm_sig"][2:])               # high-s malleated twin must be refused
    s = int.from_bytes(raw[32:64], "big")
    hi = raw[:32] + (sigs.SECP256K1_N - s).to_bytes(32, "big") + bytes([raw[64] ^ 1])
    with pytest.raises(sigs.SignatureInvalid):
        sigs.recover(v["evm_text"], "0x" + hi.hex())
    with pytest.raises(sigs.SignatureInvalid):
        sigs.verify_sol(v["sol_text"] + "x", v["sol_sig"], v["fields"]["sol"])
    with pytest.raises(sigs.SignatureInvalid):          # the Solana signature bound to the other text
        sigs.verify_sol(v["evm_text"], v["sol_sig"], v["fields"]["sol"])


def test_eip1271_path():
    v = VECTORS[0]
    safe = "0x" + "11" * 20

    class Rpc:
        def __init__(self, ok): self.ok = ok
        def code(self, a): return "0x6000"
        def eth_call(self, to, data, gas=None):
            assert data.startswith(sigs.EIP1271_SELECTOR)
            return "0x1626ba7e" + "0" * 56 if self.ok else "0x" + "0" * 64
    assert sigs.verify_evm(v["evm_text"], v["evm_sig"], safe, Rpc(True)) == "eip1271"
    with pytest.raises(sigs.SignatureInvalid):
        sigs.verify_evm(v["evm_text"], v["evm_sig"], safe, Rpc(False))


def test_claim_fields_check():
    f = cm.ClaimFields(**VECTORS[0]["fields"])
    for bad in ({"nonce": "XYZ"}, {"sol_chain": "testnet"}, {"expires_at": f.issued_at}, {"domain": "a b"}, {"uri": "ftp://x"}):
        with pytest.raises(ValueError):
            cm.ClaimFields(**{**f.as_dict(), **bad}).check()


def test_checksum():
    assert evm.to_checksum("0x2fc7f9e2911f20b2c4660d2aef808aa91bddb3d3") == "0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3"


# ---------------------------------------------------------------- settlement math
def test_realized_formula_counts_everything_once():
    # 5 SOL deposited, bought 1 SOL (still open), closed a trade +0.3, paid 0.1 claim, withdrew 0.5 (+fee)
    native = 5_000_000_000 - 1_000_000_000 + 300_000_000 - 100_000_000 - 500_005_000
    r = settle.realized(native, 2_039_280, 1_000_000_000, 5_000_000_000, 500_005_000, 100_000_000)
    assert r == 300_000_000 + 2_039_280


def test_period_boundaries_are_mondays():
    import datetime as dt
    t = int(dt.datetime(2026, 9, 24, 13, 0, tzinfo=dt.timezone.utc).timestamp())    # a Thursday
    end = settle.period_end_at_or_before(t, 7 * 86400, 345600)
    d = dt.datetime.fromtimestamp(end, dt.timezone.utc)
    assert d.weekday() == 0 and (d.hour, d.minute) == (0, 0) and d.date() == dt.date(2026, 9, 21)


def test_weights_time_weighted_and_request_stops_earning():
    t0, t1 = 1000, 2000
    w = settle.weights({"A": 10}, [(1500, "B", 10), (1500, "A", 0), (1800, "A", 5)], t0, t1)
    assert w == {"A": 10 * 500 + 5 * 200, "B": 10 * 500}


def test_weights_late_locker_gets_share_of_time_only():
    w = settle.weights({"A": 100}, [(1990, "B", 100)], 1000, 2000)
    assert w["B"] * 100 == w["A"]


def test_allocate_floors_and_keeps_dust():
    out = settle.allocate(100, {"a": 1, "b": 1, "c": 1})
    assert out == {"a": 33, "b": 33, "c": 33}
    assert settle.allocate(100, {}) == {} and settle.allocate(0, {"a": 1}) == {}


def test_plan_carries_losses_and_caps_liquidity():
    assert settle.plan(-50, 0, 0, 10_000, 100)["amount"] == 0                    # a loss pays nothing
    p = settle.plan(1_000, 800, 300, 900, 100)                                   # 200 new profit; 500 still owed
    assert p["pot"] == 200 and p["reserved"] == 500 and p["amount"] == 200
    p = settle.plan(1_000, 800, 300, 650, 100)                                   # only 50 liquid after the reserve
    assert p["amount"] == 50 and p["pot"] == 200
    assert settle.plan(700, 800, 0, 10_000, 0)["pot"] == 0                       # below the high-water mark


def test_token_account_lamports():
    accts = [{"mint": "So11111111111111111111111111111111111111112", "is_native": True, "amount": 5, "lamports": 2_039_285},
             {"mint": "EMPTY", "amount": 0, "lamports": 2_039_280},
             {"mint": "OPEN", "amount": 0, "lamports": 2_039_280},
             {"mint": "BAG", "amount": 999, "lamports": 2_039_280}]
    assert settle.token_account_lamports(accts, {"OPEN"}) == 2_039_285 + 2_039_280


# ---------------------------------------------------------------- flow classification
def _tx(keys, pre, post, fee=5000, err=None, sig="S" * 88, tok_pre=None, tok_post=None):
    return {"slot": 7, "blockTime": 1_790_000_000,
            "transaction": {"signatures": [sig], "message": {"accountKeys": [{"pubkey": k, "signer": s} for k, s in keys]}},
            "meta": {"err": err, "fee": fee, "preBalances": pre, "postBalances": post,
                     "preTokenBalances": tok_pre or [], "postTokenBalances": tok_post or []}}


def test_deposit_vs_gift():
    tx = _tx([(FUNDER, True), (WALLET, False)], [10_000_000_000, 0], [4_999_995_000, 5_000_000_000])
    f = flows.classify(tx, WALLET, False, {FUNDER})
    assert f.kind == "deposit" and f.lamports == 5_000_000_000 and f.counterparty == FUNDER
    assert flows.classify(tx, WALLET, False, set()).kind == "profit"


def test_our_known_tx_is_internal_and_unknown_halts():
    tx = _tx([(WALLET, True), (STRANGER, False)], [1_000, 0], [0, 995])
    assert flows.classify(tx, WALLET, True, set()) is None
    assert flows.classify(tx, WALLET, False, set()).kind == "unknown_outbound"


def test_failed_foreign_and_wsol_gift():
    assert flows.classify(_tx([(STRANGER, True), (WALLET, False)], [5, 0], [5, 0], err={"x": 1}), WALLET, False, set()) is None
    w = "So11111111111111111111111111111111111111112"
    tx = _tx([(STRANGER, True), (WALLET, False)], [5, 0], [5, 0],
             tok_pre=[{"owner": WALLET, "mint": w, "uiTokenAmount": {"amount": "0"}}],
             tok_post=[{"owner": WALLET, "mint": w, "uiTokenAmount": {"amount": "7000"}}])
    f = flows.classify(tx, WALLET, False, set())
    assert f.kind == "profit" and f.lamports == 7000


def test_lamports_leaving_without_our_signature_is_anomaly():
    f = flows.classify(_tx([(STRANGER, True), (WALLET, False)], [0, 10], [5, 5]), WALLET, False, set())
    assert f.kind == "anomaly"


# ---------------------------------------------------------------- event decoding
def _word(n):
    return format(n, "064x")


def test_decode_events():
    user = "0x" + "ab" * 20
    lg = {"topics": [rh_index.T_REQUESTED, "0x" + "0" * 24 + "ab" * 20, "0x" + _word(3)],
          "data": "0x" + _word(40) + _word(1_800_000_000) + _word(60)}
    d = rh_index.decode(lg)
    assert d == {"event": "WithdrawRequested", "account": evm.to_checksum(user), "request_id": 3, "amount": 40,
                 "ready_at": 1_800_000_000, "locked_after": 60}
    d = rh_index.decode({"topics": [rh_index.T_LOCKED, "0x" + "0" * 24 + "ab" * 20], "data": "0x" + _word(10) + _word(100)})
    assert d["event"] == "Locked" and d["locked_after"] == 100
    assert rh_index.decode({"topics": ["0x" + "1" * 64], "data": "0x"}) is None


# ---------------------------------------------------------------- performance index
def test_index_ignores_deposits_and_payouts():
    from fly_trader.vault import nav
    marks = [(0, 100), (60, 110), (120, 210), (180, 160), (240, 176)]
    flows_ = [(90, 100), (150, -60)]              # deposit 100 before t=120, claim 60 before t=180
    s = nav.index_series(marks, flows_)
    assert [round(i, 6) for _, i in s] == [1.0, 1.1, 1.1, round(1.1 * 220 / 210, 6), round(1.1 * 220 / 210 * 1.1, 6)]


def test_peak_skips_fresh_marks():
    from fly_trader.vault import nav
    s = [(0, 1.0), (1000, 1.2), (1500, 1.5)]
    d = nav.drawdown(s, now=1600)                 # the 1.5 mark is 100 s old: not yet a peak
    assert d["peak"] == 1.2 and d["value"] == 1.5 and d["drawdown"] < 0


def test_relay_hmac_matches_the_relay_implementation():
    import importlib.util
    from fly_trader.vault import relay_client
    src = Path(__file__).parents[1] / "web" / "relay" / "hmacauth.py"
    if not src.exists():
        pytest.skip("the relay lives on master (web/relay)")
    spec = importlib.util.spec_from_file_location("relay_hmac", src)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    secret = "ab" * 32
    ours = relay_client.sign_headers("GET", "https://x.app/api/fly/claims?limit=50&after=3", b"", "k1", secret, ts=1790000000, nonce="0" * 32)
    theirs = mod.sign("GET", "/api/fly/claims", "limit=50&after=3", b"", "k1", secret, ts=1790000000, nonce="0" * 32)
    assert ours["X-Fly-Sig"] == theirs["X-Fly-Sig"]


def test_timelock_ops_read_the_delay_word_and_watch_both_targets(monkeypatch):
    """CallScheduled's delay is the 5th data word (after target, value, the offset of `data`, predecessor); reading the
    dynamic tail instead once made every upgrade look executable ~100 s after scheduling."""
    from fly_trader import config
    vault_addr, tl = "0x" + "ab" * 20, "0x" + "cd" * 20
    monkeypatch.setattr(config, "VAULT_ADDRESS", vault_addr)
    monkeypatch.setattr(config, "VAULT_TIMELOCK", tl)
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", None)

    def sched(target, delay, opid):
        data = "0x" + _word(int(target, 16)) + _word(0) + _word(160) + _word(0) + _word(delay) + _word(4) + "4f1ef286" + "0" * 56
        return {"topics": [rh_index.T_SCHEDULED, opid, "0x" + _word(0)], "data": data, "blockNumber": hex(7), "blockTimestamp": hex(1_000)}

    class Rpc:
        def __init__(self, logs):
            self.logs = logs

        def get_logs(self, *a, **k):
            return self.logs

        def block(self, n):
            return {"timestamp": hex(1_000)}

    a, b, c = "0x" + "11" * 32, "0x" + "22" * 32, "0x" + "33" * 32
    logs = [sched(vault_addr, 691_200, a), sched(tl, 691_200, b), sched("0x" + "ee" * 20, 5, c)]
    ops = rh_index._timelock_ops(Rpc(logs), 1, 10, {}, {})
    assert ops[a] == {"eta": 1_000 + 691_200, "id": a, "target": vault_addr}
    assert ops[b]["eta"] == 1_000 + 691_200 and ops[b]["target"] == tl
    assert c not in ops                                              # a call to some other contract is not ours
    done = {"topics": [rh_index.T_EXECUTED, a, "0x" + _word(0)], "data": "0x", "blockNumber": hex(8), "blockTimestamp": hex(1_001)}
    gone = {"topics": [rh_index.T_TL_CANCELLED, b], "data": "0x", "blockNumber": hex(9), "blockTimestamp": hex(1_002)}
    assert rh_index._timelock_ops(Rpc([done, gone]), 11, 12, dict(ops), {}) == {}
