"""The fly and the real relay (web/wsgi.py + web/relay) over HTTP on localhost: push stats, a browser-style challenge
and signed claim, the fly pulling, verifying, paying (fake Solana RPC) and reporting back."""
import importlib
import json
import sys
import threading
import urllib.request
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest
from solders.keypair import Keypair

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.vault import claim_message as cm, claims, publish, sigs
from fly_trader.vault.relay_client import RelayClient

WEB = Path(__file__).parents[1] / "web"
V = json.loads((Path(__file__).parent / "vectors" / "claim_v1.json").read_text())["vectors"][1]
SECRET = "c3" * 32


class _Quiet(WSGIRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture()
def relay(tmp_path, monkeypatch):
    cfg = {"keys": {"k1": SECRET}, "db_path": str(tmp_path / "relay.sqlite3"), "domain": V["fields"]["domain"],
           "uri": V["fields"]["uri"], "chain_id": 46630, "sol_chain": "devnet", "claim_ttl_s": 900, "min_lamports": 2_000_000,
           "client_ip_header": None}
    (tmp_path / "relay.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("FLY_RELAY_CONFIG", str(tmp_path / "relay.json"))
    monkeypatch.syspath_prepend(str(WEB))
    for m in [m for m in sys.modules if m == "wsgi" or m.startswith("relay")]:
        sys.modules.pop(m)
    wsgi = importlib.import_module("wsgi")
    srv = make_server("127.0.0.1", 0, wsgi.application, handler_class=_Quiet)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    for k, v in {"VAULT_ENABLED": True, "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": "http://devnet.invalid",
                 "VAULT_SITE_DOMAIN": V["fields"]["domain"], "VAULT_SITE_URI": V["fields"]["uri"], "RH_CHAIN_ID": 46630,
                 "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": False, "GAS_RESERVE_SOL": 0.01, "TELEGRAM_BOT_TOKEN": None}.items():
        monkeypatch.setattr(config, k, v)
    with transaction() as c:
        for t in ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv", "vault_nav"):
            c.execute(f"DELETE FROM {t}")
        sid = c.execute("INSERT INTO vault_settlements (period_start, period_end, status, realized, allocated) "
                        "VALUES (now() - interval '7 days', now(), 'allocated', 40000000, 25000000) RETURNING id").fetchone()["id"]
        c.execute("INSERT INTO vault_allocations VALUES (%s, %s, 1, 25000000)", (sid, V["evm_checksummed"]))
    yield base, RelayClient(base, "k1", SECRET)
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


class Rpc:
    sent = []
    def get_balance(self, pk): return 3_000_000_000
    def get_latest_blockhash(self): return {"blockhash": "11111111111111111111111111111111", "last_valid_block_height": 9}
    def send_transaction(self, b64): self.sent.append(b64)
    def get_signature_statuses(self, s, search_history=False): return [None]
    def get_block_height(self): return 1


def test_push_claim_pay_report_roundtrip(relay, monkeypatch):
    import fly_trader.execution.broker_live as bl
    monkeypatch.setattr(bl, "await_confirmation", lambda rpc, sig, lvbh: ("confirmed", {"slot": 77}))
    base, client = relay
    publish.push(client, "WALLETPUBKEY")
    st = _get(base + "/api/stats")
    assert st["fly"]["wallet"] == "WALLETPUBKEY" and st["ledger"]["allocated"] == 25_000_000
    acct = _get(base + "/api/account?evm=" + V["fields"]["evm"])
    assert acct["owed"] == 25_000_000

    evm_key = bytes.fromhex(V["test_keys"]["evm_private_key"][2:])
    kp = Keypair.from_seed(bytes.fromhex(V["test_keys"]["sol_seed"]))
    ch = _get(f"{base}/api/claim/challenge?evm={V['evm_checksummed']}&sol={kp.pubkey()}")
    f = cm.ClaimFields(domain=ch["domain"], uri=ch["uri"], chain_id=ch["chain_id"], sol_chain=ch["sol_chain"], evm=V["evm_checksummed"],
                       sol=str(kp.pubkey()), nonce=ch["nonce"], issued_at=ch["issued_at"], expires_at=ch["expires_at"])
    status, body = _post(base + "/api/claim", {"nonce": ch["nonce"], "evm": V["evm_checksummed"], "sol": str(kp.pubkey()),
                                                "evm_sig": sigs.sign_evm(cm.evm_text(f), evm_key),
                                                "sol_sig": str(kp.sign_message(cm.sol_text(f).encode()))})
    assert status == 202
    out = claims.process(client, Rpc(), Keypair.from_seed(bytes([7]) * 32))
    assert out["paid"] == 1, out
    got = _get(f"{base}/api/claim/{body['id']}")
    assert got["status"] == "paid" and got["lamports"] == 25_000_000
    publish.push(client, "WALLETPUBKEY")
    assert _get(base + "/api/account?evm=" + V["fields"]["evm"])["owed"] == 0
