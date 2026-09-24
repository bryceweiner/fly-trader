"""Claim path end to end against the test database with a fake relay and a fake Solana RPC."""
import json
from pathlib import Path

import pytest
from solders.keypair import Keypair

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import claims, state

V = json.loads((Path(__file__).parent / "vectors" / "claim_v1.json").read_text())["vectors"][1]   # rehearsal: devnet/46630
KP = Keypair.from_seed(bytes([9]) * 32)


@pytest.fixture(autouse=True)
def vault_env(monkeypatch):
    for k, v in {"VAULT_ENABLED": True, "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": "http://devnet.invalid",
                 "VAULT_SITE_DOMAIN": V["fields"]["domain"], "VAULT_SITE_URI": V["fields"]["uri"], "RH_CHAIN_ID": 46630,
                 "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": False, "GAS_RESERVE_SOL": 0.01, "TELEGRAM_BOT_TOKEN": None}.items():
        monkeypatch.setattr(config, k, v)
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv"):
            c.execute(f"DELETE FROM {t}")
        sid = c.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) "
                        "VALUES (now() - interval '7 days', now(), 'allocated', 50000000, 30000000) RETURNING id").fetchone()["id"]
        c.execute("INSERT INTO vault_allocations VALUES (%s, %s, 1, 30000000)", (sid, V["evm_checksummed"]))
    yield


class Relay:
    def __init__(self, claims_):
        self.claims, self.reports = claims_, []

    def pending_claims(self, after=0, limit=50):
        return [c for c in self.claims if c["id"] > after]

    def report(self, results):
        self.reports.extend(results)


class Rpc:
    def __init__(self, balance=5_000_000_000):
        self.balance, self.sent = balance, []

    def get_balance(self, pk): return self.balance
    def get_latest_blockhash(self): return {"blockhash": "11111111111111111111111111111111", "last_valid_block_height": 100}
    def send_transaction(self, b64): self.sent.append(b64); return "sig"
    def get_signature_statuses(self, sigs, search_history=False): return [None]
    def get_block_height(self): return 50


def _claim(rid=1, **over):
    f = V["fields"]
    return {"id": rid, "nonce": f["nonce"], "evm": f["evm"], "sol": f["sol"], "evm_sig": V["evm_sig"], "sol_sig": V["sol_sig"],
            "domain": f["domain"], "uri": f["uri"], "chain_id": f["chain_id"], "sol_chain": f["sol_chain"],
            "issued_at": f["issued_at"], "expires_at": f["expires_at"], **over}


def _confirm(monkeypatch, status="confirmed"):
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: (status, {"slot": 123}))


def _now():
    from fly_trader.vault.claim_message import parse_rfc3339
    return parse_rfc3339(V["fields"]["issued_at"]) + 60


def test_valid_claim_pays_all_owed_once(monkeypatch):
    _confirm(monkeypatch)
    relay, rpc = Relay([_claim()]), Rpc()
    out = claims.process(relay, rpc, KP, now=_now())
    assert out["paid"] == 1 and len(rpc.sent) == 1
    with transaction() as c:
        acct = c.execute("SELECT * FROM vault_accounts WHERE evm = %s", (V["evm_checksummed"],)).fetchone()
        flow = c.execute("SELECT * FROM vault_flows WHERE kind = 'claim'").fetchone()
    assert acct["owed"] == 0 and acct["claimed"] == 30_000_000 and flow["lamports"] == 30_000_000
    assert relay.reports[-1]["status"] == "paid"
    out = claims.process(Relay([_claim(), _claim(2)]), rpc, KP, now=_now())       # replayed nonce and nothing owed
    assert out["paid"] == 0 and len(rpc.sent) == 1


def test_tampered_or_foreign_claims_are_rejected(monkeypatch):
    _confirm(monkeypatch)
    bad = [_claim(1, sol_sig=V["sol_sig"][:-2] + "11"), _claim(2, nonce="a" * 32, sol="11111111111111111111111111111112"), _claim(3, nonce="b" * 32, domain="evil.app")]
    relay, rpc = Relay(bad), Rpc()
    out = claims.process(relay, rpc, KP, now=_now())
    assert out["rejected"] == 3 and not rpc.sent
    assert {r["status"] for r in relay.reports} == {"rejected"}


def test_gas_reserve_makes_claim_wait_and_halt_stops_everything(monkeypatch):
    _confirm(monkeypatch)
    rpc = Rpc(balance=15_000_000)                                               # 0.015 SOL, reserve 0.01, owed 0.03
    claims.process(Relay([_claim()]), rpc, KP, now=_now())
    with transaction() as c:
        assert c.execute("SELECT status FROM vault_claims").fetchone()["status"] == "waiting_liquidity"
    state.put("halt", {"reasons": ["test"]})
    rpc.balance = 5_000_000_000
    assert claims.process(Relay([]), rpc, KP, now=_now()).get("halted")
    assert not rpc.sent
    state.resume()
    assert claims.process(Relay([]), rpc, KP, now=_now())["paid"] == 1


def test_expired_unlanded_payment_is_resigned_not_doubled(monkeypatch):
    _confirm(monkeypatch, "expired")
    rpc = Rpc()
    claims.process(Relay([_claim()]), rpc, KP, now=_now())
    assert len(rpc.sent) == 1
    with transaction() as c:
        assert c.execute("SELECT status FROM vault_claims").fetchone()["status"] == "sending"
    rpc.get_block_height = lambda: 500                                          # blockhash now dead
    _confirm(monkeypatch)
    assert claims.process(Relay([]), rpc, KP, now=_now())["paid"] == 1
    assert len(rpc.sent) == 2
    with transaction() as c:
        assert c.execute("SELECT count(*) AS n FROM vault_flows WHERE kind = 'claim'").fetchone()["n"] == 1
