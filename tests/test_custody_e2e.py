"""The signer against the REAL Squads v4 program (local validator; skipped unless CUSTODY_E2E points at the JSON that
web/src/lib/squads.int.test.ts writes with SQUADS_IT_OUT). Proves on chain what the unit tests assume:
the trading key refills itself through L1, cannot send L1 anywhere else even when it builds the transaction itself,
and the payout key pays a holder through L2 up to the weekly cap."""
import base64
import json
import os
import time

import base58
import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from fly_trader.signer import core, squads
from fly_trader.signer.ledger import Ledger
from fly_trader.signer.policy import PolicyError
from fly_trader.signer.rpc import Rpc

CFG = os.environ.get("CUSTODY_E2E")
pytestmark = pytest.mark.skipif(not CFG, reason="needs a local validator set up by web/src/lib/squads.int.test.ts (CUSTODY_E2E)")
SOL = 10**9


@pytest.fixture(scope="module")
def chain():
    c = json.load(open(CFG))
    kp = lambda s: Keypair.from_bytes(base58.b58decode(s))           # noqa: E731
    rpc = Rpc(c["rpc"])
    s = core.Signer(kp(c["trading"]), kp(c["payout"]), rpc,
                    core.SignerConfig(multisig=c["multisig"], limit_trading=c["l1"], limit_payout=c["l2"]), Ledger())
    return c, rpc, s


def land(rpc: Rpc, tx_b64: str) -> dict:
    sig = rpc.call("sendTransaction", [tx_b64, {"encoding": "base64", "skipPreflight": True}])
    for _ in range(120):
        st = rpc.signature_status(sig)
        if st and st.get("confirmationStatus") in ("confirmed", "finalized"):
            return st
        time.sleep(0.25)
    raise TimeoutError(sig)


def test_topup_through_l1(chain):
    c, rpc, s = chain
    me = s.trading.pubkey()
    before, tb = rpc.balance(me), rpc.balance(Pubkey.from_string(c["treasury"]))
    st = land(rpc, s.topup(SOL)["tx"])
    assert st["err"] is None
    assert rpc.balance(Pubkey.from_string(c["treasury"])) == tb - SOL
    assert rpc.balance(me) == before + SOL - 5000


def test_a_stolen_trading_key_cannot_send_l1_elsewhere(chain):
    c, rpc, s = chain
    thief = Keypair().pubkey()
    ix = squads.spending_limit_use_sol(Pubkey.from_string(c["multisig"]), s.trading.pubkey(), Pubkey.from_string(c["l1"]),
                                       Pubkey.from_string(c["treasury"]), thief, SOL // 10)
    bh, _ = rpc.blockhash()
    tx = VersionedTransaction(MessageV0.try_compile(s.trading.pubkey(), [ix], [], Hash.from_string(bh)), [s.trading])
    st = land(rpc, base64.b64encode(bytes(tx)).decode())
    assert st["err"] is not None                                         # InvalidDestination, on chain
    assert rpc.balance(thief) == 0


def test_a_stolen_payout_key_cannot_use_l1(chain):
    c, rpc, s = chain
    ix = squads.spending_limit_use_sol(Pubkey.from_string(c["multisig"]), s.payout.pubkey(), Pubkey.from_string(c["l1"]),
                                       Pubkey.from_string(c["treasury"]), s.trading.pubkey(), SOL // 10)
    bh, _ = rpc.blockhash()
    tx = VersionedTransaction(MessageV0.try_compile(s.trading.pubkey(), [ix], [], Hash.from_string(bh)), [s.trading, s.payout])
    assert land(rpc, base64.b64encode(bytes(tx)).decode())["err"] is not None     # Unauthorized: not L1's member


def test_claim_through_l2_and_the_weekly_cap(chain):
    c, rpc, s = chain
    holder = Keypair().pubkey()
    st = land(rpc, s.pay_claim("e2e:1", str(holder), SOL // 2)["tx"])
    assert st["err"] is None and rpc.balance(holder) == SOL // 2
    with pytest.raises(PolicyError) as e:                                # L2 is 1 SOL a week: 0.5 left
        s.pay_claim("e2e:2", str(holder), SOL)
    assert e.value.code == "cap"
    with pytest.raises(PolicyError) as e:                                # the same claim again: already paid
        s.pay_claim("e2e:1", str(holder), SOL // 2)
    assert e.value.code == "paid"
