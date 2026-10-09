"""Rehearsal on a local anvil fork of Robinhood Chain mainnet (opt-in: RH_FORK=1; needs anvil and the network).

Real Pons pools, real KyberSwap routes (built for the anvil account, executed on the fork), the Uniswap v4
UniversalRouter fallback with Permit2 exact approvals, and the executor + accounting on the real signing path: every
round trip is booked from measured balances, and the reconciler finds the chain and the books equal.
"""
import hashlib
import os
import shutil
import socket
import subprocess
import time

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh import accounting as A, kyber as K, router
from fly_trader.rh.exec import RhExecutor, RhRequest
from fly_trader.rh.rpc import RhRpc
from fly_trader.rh.tx import address_of
from fly_trader.rh.wallet import RhWallet

pytestmark = pytest.mark.skipif(not os.environ.get("RH_FORK") or not shutil.which("anvil"), reason="fork rehearsal: set RH_FORK=1 (needs anvil)")

FORK_URL = os.environ.get("RH_FORK_URL", "https://robinhood-rpc.publicnode.com")
# a fresh key funded on the fork (KyberSwap refuses to build for anvil's well-known accounts: "wallets invalid")
KEY = hashlib.sha256(b"fly-trader rh fork rehearsal").digest()
ADDR = address_of(KEY)
HOOK = "0xe5e702641ea86f4ae6cc3cdaed2b886f976be044"
ETH_POOL = {"pool_id": "0x8052650014fbe824973a674946e08182a6a116cd9b72b50a97444bbe8a2d8b86", "token": "0x9ca1cc0c90d97b4f36c5e2232d4fbd705a73c65d",
            "currency0": "0x0000000000000000000000000000000000000000", "currency1": "0x9ca1cc0c90d97b4f36c5e2232d4fbd705a73c65d",
            "fee": 0, "tick_spacing": 200, "hooks": HOOK, "quote_asset": "0x0000000000000000000000000000000000000000"}
STOCK_POOL = {"pool_id": "0x0c208afd549aed429b34d8b3a64eeb063ab51552a2bd345dd342d35704e023d2", "token": "0xedf530ac305923314302a729464e942a0fedc342",
              "currency0": "0x117cc2133c37b721f49de2a7a74833232b3b4c0c", "currency1": "0xedf530ac305923314302a729464e942a0fedc342",
              "fee": 0, "tick_spacing": 200, "hooks": HOOK, "quote_asset": "0x117cc2133c37b721f49de2a7a74833232b3b4c0c"}          # SPY-quoted
WEI = 10 ** 18
ATTEMPTS = 3                                  # fresh forks tried when one ages out of the endpoint's state window


def _port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _start_fork():
    port = _port()
    proc = subprocess.Popen(["anvil", "--fork-url", FORK_URL, "--port", str(port), "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if RhRpc(url, 4663).chain() == 4663:
                break
        except Exception:
            time.sleep(1)
    RhRpc(url, 4663).call("anvil_setBalance", [ADDR, hex(10 * WEI)])
    return proc, url


def _clean():
    with transaction() as conn:
        for t in ("rh_legs", "rh_base_lots", "rh_intents", "rh_wallet_marks", "rh_wallet_flows"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("DELETE FROM rh_txs WHERE from_addr = %s", (ADDR,)); conn.execute("DELETE FROM positions WHERE book = 'live_rh'")


@pytest.fixture
def on_fork(monkeypatch):
    """Runs ``fn(wallet)`` on a fresh fork, again on a new fork (up to 3 times) when the fork aged out of the endpoint's
    state window. The public endpoints keep no archive state and the chain makes 10–20 blocks a second, so a fork a few
    seconds old can no longer fetch accounts it has not touched yet; that is the endpoint, not the code under test."""
    monkeypatch.setattr(config, "RH_CONFIRMATIONS", 0); monkeypatch.setattr(config, "RH_MAX_FEE_GWEI", 5.0)
    monkeypatch.setattr(config, "SLIPPAGE_STEP_BPS", 200)

    def run(fn, prepare=None):
        for attempt in range(ATTEMPTS):
            _clean()
            pre = prepare() if prepare else None                              # network work that needs no fork: before it starts
            proc, url = _start_fork()
            try:
                w = RhWallet(RhRpc(url, 4663), KEY, ADDR)
                with transaction() as conn:
                    A.reconcile(conn, w)                                      # the opening balance
                return fn(w, pre) if prepare else fn(w)
            except AssertionError as e:
                if not any(m in str(e) for m in ("Archive requests", "failed to get", "403")):
                    raise
                if attempt == ATTEMPTS - 1:
                    pytest.skip(f"the fork aged out of {FORK_URL}'s state window {ATTEMPTS} times: set RH_FORK_URL to an archive endpoint")
            finally:
                proc.terminate(); proc.wait(timeout=10)
    yield run
    _clean()


class _NoKyber:
    def get_route(self, *a, **k):
        raise K.KyberError("forced: the v4 fallback")


def _round_trip(w, pool, planner=None, amount=WEI // 50):
    e = RhExecutor(w, planner=planner, start=False, recover=False)
    native0 = w.balance()
    b = e.execute(RhRequest(kind="buy", token=pool["token"], quote_asset=pool["quote_asset"], amount_in=amount, slippage_bps=300, max_slippage_bps=800,
                            pool=pool, hold_s=60.0))
    assert b.ok, b.error
    with transaction() as conn:
        p = dict(conn.execute("SELECT * FROM positions WHERE id = %s", (b.position_id,)).fetchone())
    assert int(p["qty"]) == w.balance_of(pool["token"]) and p["cost_sol"] == pytest.approx((native0 - w.balance()) / WEI, rel=1e-9)
    s = e.execute(RhRequest(kind="sell", token=pool["token"], quote_asset=pool["quote_asset"], amount_in=int(p["qty"]), slippage_bps=300, max_slippage_bps=800,
                            pool=pool, position_id=b.position_id))
    assert s.ok, s.error
    with transaction() as conn:
        p = dict(conn.execute("SELECT * FROM positions WHERE id = %s", (b.position_id,)).fetchone())
        rec = A.reconcile(conn, w)
    assert p["status"] == "closed" and w.balance_of(pool["token"]) == 0
    assert p["realized_sol"] == pytest.approx((w.balance() - native0) / WEI, abs=1e-12)          # the books' P&L is the wallet's, to the wei
    assert rec["ok"], rec
    return b, s, p


def test_kyber_round_trip_on_an_eth_pool(on_fork):
    b, s, p = on_fork(lambda w: _round_trip(w, ETH_POOL))
    assert b.route == "atomic_kyber"
    with transaction() as conn:
        legs = conn.execute("SELECT eth_in, eth_out FROM rh_legs WHERE position_id = %s ORDER BY id", (b.position_id,)).fetchall()
    swap_only = legs[1]["eth_out"] / legs[0]["eth_in"] - 1                                          # the round trip before gas
    assert -0.06 < swap_only < -0.015                                                               # 2 × the hook fee (0.99 % on this launch) + impact


def test_v4_fallback_round_trip_with_permit2(on_fork):
    planner = lambda account, tin, tout, amount, slip, rpc=None, pool=None: router.plan_swap(account, tin, tout, amount, slip, kyber=_NoKyber(), rpc=rpc, pool=pool)  # noqa: E731
    b, s, p = on_fork(lambda w: _round_trip(w, ETH_POOL, planner))
    assert b.route == "atomic_ur" and s.route == "atomic_ur"


def test_stock_quoted_coin_round_trip_returns_to_eth(on_fork):
    amount = WEI // 50

    def prepare():                                               # the buy's Kyber route, built before the fork starts
        return router.plan_swap(ADDR, K.NATIVE, STOCK_POOL["token"], amount, 300)

    def go(w, buy_plan):
        def planner(account, tin, tout, amt, slip, rpc=None, pool=None):
            if tin == K.NATIVE and amt == amount and buy_plan is not None:
                return buy_plan
            return router.plan_swap(account, tin, tout, amt, slip, rpc=rpc, pool=pool)
        b, s, p = _round_trip(w, STOCK_POOL, planner, amount)
        return b, w.balance_of(STOCK_POOL["quote_asset"])
    b, base_left = on_fork(go, prepare)
    assert b.route in ("atomic_kyber", "two_leg") and (base_left == 0 or b.route == "two_leg")      # atomic: no base left behind


def test_the_smoke_command_on_the_fork(on_fork, monkeypatch):
    """``fly-trader rh-smoke`` end to end: the guard, the round trips, the reconciler — on the fork instead of mainnet. Only the
    ETH pool is registered: a third trip would start ~20 s into the fork, past the endpoint's state window (the stock
    path has its own fresh fork above)."""
    from fly_trader.rh import smoke
    with transaction() as conn:
        conn.execute("DELETE FROM rh_pools WHERE pool_id IN (%s, %s)", (ETH_POOL["pool_id"], STOCK_POOL["pool_id"]))
        conn.execute("DELETE FROM rh_tokens WHERE token IN (%s, %s)", (ETH_POOL["token"], STOCK_POOL["token"]))
        for pl in (ETH_POOL,):
            conn.execute("INSERT INTO rh_tokens (token, status) VALUES (%s, 'graduated') ON CONFLICT (token) DO UPDATE SET status = 'graduated'", (pl["token"],))
            conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, fee, tick_spacing, hooks, quote_asset, is_pons) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,true) "
                         "ON CONFLICT (pool_id) DO NOTHING", (pl["pool_id"], pl["token"], pl["currency0"], pl["currency1"], pl["fee"], pl["tick_spacing"], pl["hooks"], pl["quote_asset"]))

    def go(w):
        monkeypatch.setattr(config, "RH_RPC_URL", w.rpc.url); monkeypatch.setattr(config, "RH_LIVE_ENABLED", True)
        monkeypatch.setattr(config, "RH_BOT_ADDRESS", ADDR); monkeypatch.setenv("RH_BOT_PRIVATE_KEY", KEY.hex())
        out = smoke.run(0.002, 0.01)
        assert out["passed"], out
        return out
    out = on_fork(go)
    assert [t["trip"] for t in out["trips"]] == ["kyber, ETH pool", "v4 fallback, ETH pool"]
    assert all(t["position"]["status"] == "closed" and t["txs"] for t in out["trips"])
    with transaction() as conn:
        conn.execute("DELETE FROM rh_pools WHERE pool_id IN (%s, %s)", (ETH_POOL["pool_id"], STOCK_POOL["pool_id"]))
        conn.execute("DELETE FROM rh_tokens WHERE token IN (%s, %s)", (ETH_POOL["token"], STOCK_POOL["token"]))
