"""The testnet checklist on a local anvil chain with the RH testnet's id (46630): real node semantics, no funds needed.

- the guard signs only on the chain this install expects (RH_TESTNET);
- a transaction stuck in the mempool when the process died is cancelled at its nonce by recovery (the node's
  replacement rule accepts the 25 % fee bump) and the original is marked replaced;
- an ETH claim payout reaches the claim's EVM address once, and the reconciler agrees with the books.
Skipped when anvil is not installed.
"""
import hashlib
import shutil
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh import accounting as A, guard
from fly_trader.rh.exec import RhExecutor
from fly_trader.rh.rpc import RhRpc
from fly_trader.rh.tx import address_of
from fly_trader.rh.wallet import RhWallet

pytestmark = pytest.mark.skipif(not shutil.which("anvil"), reason="needs anvil (Foundry)")
KEY = hashlib.sha256(b"fly-trader rh anvil rehearsal").digest()
ADDR = address_of(KEY)
WEI = 10 ** 18


@pytest.fixture
def chain(monkeypatch):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen(["anvil", "--chain-id", "46630", "--port", str(port), "--no-mining", "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    rpc = RhRpc(url, 46630)
    for _ in range(50):
        try:
            rpc.chain(); break
        except Exception:
            time.sleep(0.2)
    rpc.call("anvil_setBalance", [ADDR, hex(5 * WEI)])
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 0); monkeypatch.setattr("fly_trader.rh.wallet.RECEIPT_POLL_S", 0.05)
    real_wait = RhWallet.wait
    monkeypatch.setattr(RhWallet, "wait", lambda self, h, timeout_s=120.0, confirmations=None: real_wait(self, h, min(timeout_s, 1.0), confirmations))
    monkeypatch.setattr("fly_trader.vault.alerts.send", lambda *a, **k: None)

    def clean():
        with transaction() as conn:
            for t in ("rh_legs", "rh_base_lots", "rh_intents", "rh_wallet_marks", "rh_wallet_flows", "vault_eth_allocations", "vault_eth_settlements", "vault_claims"):
                conn.execute(f"DELETE FROM {t}")
            conn.execute("DELETE FROM vault_settlements WHERE period_end < '2020-01-01'")
            conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,))
    clean()
    yield rpc
    clean(); proc.terminate(); proc.wait(timeout=10)


def test_guard_signs_only_on_the_expected_chain(chain, monkeypatch):
    monkeypatch.setenv("RH_BOT_PRIVATE_KEY", KEY.hex()); monkeypatch.setattr(config, "RH_BOT_ADDRESS", ADDR); monkeypatch.setattr(config, "RH_LIVE_ENABLED", True)
    monkeypatch.setattr(config, "RH_EXPECTED_CHAIN_ID", 46630)
    assert guard.check(chain)[1] == ADDR                                         # a testnet install signs on the testnet
    monkeypatch.setattr(config, "RH_EXPECTED_CHAIN_ID", 4663)
    with pytest.raises(guard.SigningRefused, match="46630"):                     # a mainnet install never signs here
        guard.check(chain)


def test_recovery_records_a_transaction_that_landed_while_down_and_cancels_one_that_never_will(chain):
    w = RhWallet(chain, KEY, ADDR); dest = "0x" + "33" * 20
    s1 = w.send(to=dest, data=b"", value=WEI // 100, kind="transfer", gas=21_000)
    assert chain.receipt(s1["hash"]) is None                                      # in the mempool; the process "dies" here
    chain.call("evm_mine", [])                                                   # ...and it lands while nobody watches
    s2 = w.send(to=dest, data=b"", value=WEI // 100, kind="transfer", gas=21_000)
    chain.call("anvil_dropTransaction", [s2["hash"]])                             # this one the node forgets
    with transaction() as conn:
        conn.execute("UPDATE rh_txs SET created_at = now() - interval '10 minutes' WHERE from_addr = %s", (ADDR,))
    chain.call("anvil_setAutomine", [True])
    RhExecutor(RhWallet(chain, KEY, ADDR), start=False)                          # a new start recovers
    with transaction() as conn:
        rows = sorted((r["nonce"], r["kind"], r["status"]) for r in conn.execute("SELECT nonce, kind, status FROM rh_txs WHERE from_addr = %s", (ADDR,)).fetchall())
    assert rows == [(0, "transfer", "mined_ok"), (1, "cancel", "mined_ok"), (1, "transfer", "replaced")]
    assert chain.get_balance(dest) == WEI // 100 and chain.tx_count(ADDR, "latest") == 2         # paid once; the forgotten one never lands


def test_eth_claim_payout_reaches_the_evm_address_once(chain):
    from fly_trader.vault import claims_eth
    chain.call("anvil_setAutomine", [True])
    w = RhWallet(chain, KEY, ADDR); dest = "0x" + "44" * 20
    with transaction() as conn:
        A.reconcile(conn, w, datetime.now(timezone.utc) - timedelta(seconds=5))
        sid = conn.execute("INSERT INTO vault_settlements (period_start, period_end, status) VALUES ('2019-01-01', '2019-01-08', 'allocated') RETURNING id").fetchone()["id"]
        eid = conn.execute("INSERT INTO vault_eth_settlements (settlement_id, period_start, period_end, status, book, allocated) "
                           "VALUES (%s, '2019-01-01', '2019-01-08', 'allocated', 'live_rh', %s) RETURNING id", (sid, 2 * 10 ** 16)).fetchone()["id"]
        conn.execute("INSERT INTO vault_eth_allocations (eth_settlement_id, evm, weight, wei) VALUES (%s, %s, 1, %s)", (eid, dest, 2 * 10 ** 16))
        cid = conn.execute("INSERT INTO vault_claims (evm, sol, nonce, status) VALUES (%s, 'SoL', 'n1', 'paid') RETURNING id", (dest,)).fetchone()["id"]
        assert claims_eth.fix_amount(conn, cid, dest)
    assert claims_eth.process(w)["paid"] == 1 and claims_eth.process(w)["paid"] == 0
    assert chain.get_balance(dest) == 2 * 10 ** 16
    with transaction() as conn:
        assert A.reconcile(conn, w)["ok"]
