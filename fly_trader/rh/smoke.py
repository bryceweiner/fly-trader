"""The RH bot wallet's key, and the tiny mainnet smoke test (plan P11).

``new_key``: a fresh secp256k1 key written to the project .env as RH_BOT_PRIVATE_KEY with its address pinned as
RH_BOT_ADDRESS (the guard refuses a key that does not derive to it). Refuses when a key is already there.

``run``: three real round trips of ``size`` ETH through the live executor and accounting (book live_rh, strategy
'smoke'): KyberSwap on an ETH-quoted Pons pool, the Uniswap v4 fallback on the same pool, and a stock- or USDG-quoted
coin (ETH → base → coin in one transaction). Each is booked from measured balances; afterwards the reconciler must find
the chain and the books equal. Refuses unless signing is allowed (RH_LIVE_ENABLED=1, key, pinned address, chain 4663)
and the whole test fits under ``cap`` ETH of spend.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .. import config
from ..db.connection import transaction
from .tx import address_of

WEI = 10 ** 18
SMOKE = "smoke"


def new_key(env_path: Path | None = None) -> str:
    from coincurve import PrivateKey
    path = Path(env_path or config.REPO_ROOT / ".env")
    text = path.read_text() if path.exists() else ""
    if re.search(r"^RH_BOT_PRIVATE_KEY=\S", text, re.M) or config.env_str("RH_BOT_PRIVATE_KEY"):
        raise SystemExit("RH_BOT_PRIVATE_KEY is already set: refusing to replace a key that may hold funds")
    while True:
        key = os.urandom(32)
        try:
            PrivateKey(key); break
        except ValueError:
            continue
    addr = address_of(key)
    text = re.sub(r"^RH_BOT_ADDRESS=.*\n?", "", text, flags=re.M)
    if text and not text.endswith("\n"):
        text += "\n"
    text += f"# the Robinhood Chain bot wallet (fly-trader rh-wallet new); fund it with ETH on Robinhood Chain\nRH_BOT_PRIVATE_KEY=0x{key.hex()}\nRH_BOT_ADDRESS={addr}\n"
    path.write_text(text); os.chmod(path, 0o600)
    return addr


def _pools(conn) -> dict:
    """The most traded graduated ETH-quoted pool and non-ETH-quoted pool the indexer knows."""
    q = ("SELECT p.pool_id, p.token, p.currency0, p.currency1, p.fee, p.tick_spacing, p.hooks, p.quote_asset FROM rh_pools p JOIN rh_tokens t ON t.token = p.token "
         "WHERE t.status = 'graduated' AND p.is_pons AND (p.quote_asset = %s) = %s ORDER BY (SELECT count(*) FROM rh_swaps s WHERE s.pool_id = p.pool_id) DESC LIMIT 1")
    zero = "0x" + "00" * 20
    return {"eth": conn.execute(q, (zero, True)).fetchone(), "base": conn.execute(q, (zero, False)).fetchone()}


def run(size_eth: float = 0.002, cap_eth: float = 0.01) -> dict:
    from . import accounting as A, kyber as K, router
    from .exec import RhExecutor, RhRequest
    from .rpc import RhRpc
    from .wallet import RhWallet
    if 3 * size_eth > cap_eth:
        raise SystemExit(f"three round trips of {size_eth} ETH exceed the {cap_eth} ETH cap")
    w = RhWallet(RhRpc())                                        # the guard: RH_LIVE_ENABLED, key, pinned address, chain id
    native = w.balance()
    if native < int((size_eth + config.RH_GAS_RESERVE_ETH) * WEI):
        raise SystemExit(f"the RH wallet {w.address} holds {native / WEI:.6f} ETH: fund it with at least {size_eth + config.RH_GAS_RESERVE_ETH:.4f} ETH")
    with transaction() as conn:
        pools = {k: dict(v) for k, v in _pools(conn).items() if v}
        A.reconcile(conn, w)
    if "eth" not in pools:
        raise SystemExit("no graduated ETH-quoted Pons pool indexed yet (run rh-backfill first)")

    class _NoKyber:
        def get_route(self, *a, **k):
            raise K.KyberError("smoke: the v4 fallback")

    fallback = lambda account, tin, tout, amount, slip, rpc=None, pool=None: router.plan_swap(account, tin, tout, amount, slip, kyber=_NoKyber(), rpc=rpc, pool=pool)  # noqa: E731
    trips = [("kyber, ETH pool", pools["eth"], None), ("v4 fallback, ETH pool", pools["eth"], fallback)]
    if "base" in pools:
        trips.append(("kyber, non-ETH quote", pools["base"], None))
    out = []
    for name, pool, planner in trips:
        ex = RhExecutor(w, planner=planner, start=False)
        b = ex.execute(RhRequest(kind="buy", token=pool["token"], quote_asset=pool["quote_asset"], amount_in=int(size_eth * WEI), slippage_bps=300,
                                 max_slippage_bps=800, pool=pool, hold_s=60.0, strategy=SMOKE))
        trip = {"trip": name, "token": pool["token"], "buy": {"ok": b.ok, "route": b.route, "error": b.error}}
        if b.ok:
            with transaction() as conn:
                qty = int(conn.execute("SELECT qty FROM positions WHERE id = %s", (b.position_id,)).fetchone()["qty"])
            s = ex.execute(RhRequest(kind="sell", token=pool["token"], quote_asset=pool["quote_asset"], amount_in=qty, slippage_bps=300, max_slippage_bps=800,
                                     pool=pool, position_id=b.position_id, strategy=SMOKE))
            with transaction() as conn:
                p = conn.execute("SELECT cost_sol, realized_sol, gas_q, status FROM positions WHERE id = %s", (b.position_id,)).fetchone()
                txs = [r["hash"] for r in conn.execute("SELECT t.hash FROM rh_txs t JOIN rh_intents i ON i.id = t.intent_id WHERE i.position_id = %s ORDER BY t.id",
                                                       (b.position_id,)).fetchall()]
            trip.update(sell={"ok": s.ok, "route": s.route, "error": s.error}, position=dict(p), txs=txs)
        out.append(trip)
    with transaction() as conn:
        rec = A.reconcile(conn, w)
    return {"wallet": w.address, "start_eth": native / WEI, "end_eth": w.balance() / WEI, "trips": out, "reconcile": rec,
            "passed": all(t["buy"]["ok"] and t.get("sell", {}).get("ok") for t in out) and rec["ok"]}
