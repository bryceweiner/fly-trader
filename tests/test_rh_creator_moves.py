"""A plain v4 coin whose creator can move holders' tokens without an allowance (ANT, 2026-11-03): the contract check's
simulated transferFrom refuses it at registration, and the RH engine never buys a v4 coin refused after it went live."""
import pytest

from fly_trader import config, markets
from fly_trader.agent import minute_engine as me
from fly_trader.rh import templates as TP, token_check as T, v4_universe as U
from fly_trader.chain.evm import EvmRpcError

TOKEN, CREATOR = "0x36c2861a1f94c6f277a388338a4f8a686e0d9b30", "0x1edc8ec9eef1d4eb09058c5f2bdb5ad1ad3e3d2b"


class _Rpc:
    def __init__(self, moves=None, error=None):
        self.moves, self.error, self.seen = moves or set(), error, []

    def call(self, method, params):
        q = params[0]; self.seen.append(q)
        assert method == "eth_call" and q["data"].startswith("0x" + T.TRANSFER_FROM) and q["data"][10 + 24:10 + 64] == config.V4_POOL_MANAGER.lower()[2:]
        if self.error:
            raise self.error
        if q["from"] in self.moves:
            return "0x" + "0" * 63 + "1"
        raise EvmRpcError("eth_call", {"code": 3, "message": "execution reverted"})

    def tx_by_hash(self, h):
        return {"from": CREATOR}


def test_the_simulation_tells_a_backdoor_from_a_standard_token():
    assert T.creator_can_move(_Rpc({CREATOR}), TOKEN, CREATOR, config.V4_POOL_MANAGER)
    assert not T.creator_can_move(_Rpc(), TOKEN, CREATOR, config.V4_POOL_MANAGER)                 # no allowance: reverts
    with pytest.raises(EvmRpcError):                                                             # unreadable: the caller retries later
        T.creator_can_move(_Rpc(error=EvmRpcError("eth_call", {"code": -32000, "message": "header not found"})), TOKEN, CREATOR, config.V4_POOL_MANAGER)


def _pending(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_v4_tokens WHERE token = %s", (TOKEN,)); cur.execute("DELETE FROM rh_v4_keys WHERE pool_id = 'POOLANT'")
        cur.execute("INSERT INTO rh_v4_keys (pool_id, currency0, currency1, init_block, init_tx) VALUES ('POOLANT', '0x0', %s, 1, '0xinit')", (TOKEN,))
        cur.execute("INSERT INTO rh_v4_tokens (token, first_pool, status) VALUES (%s, 'POOLANT', 'pending')", (TOKEN,))
    conn.commit()


@pytest.mark.parametrize("moves, status, reason", [({CREATOR}, "refused", U.CREATOR_MOVES), (set(), "ok", "no owner or admin powers")])
def test_registration_refuses_a_coin_whose_creator_moves_holders_tokens(db_conn, monkeypatch, moves, status, reason):
    monkeypatch.setattr(T, "check", lambda rpc, t: {"ok": True, "reason": "no owner or admin powers", "symbol": "ANT", "name": "Anthropic", "decimals": 18,
                                                    "supply_raw": 10 ** 27, "powers": [], "template": next(iter(TP.KNOWN_GOOD))})
    _pending(db_conn)
    try:
        out = U.run_checks(db_conn, _Rpc(moves), limit=5); db_conn.commit()
        row = db_conn.execute("SELECT status, reason, creator FROM rh_v4_tokens WHERE token = %s", (TOKEN,)).fetchone()
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM rh_v4_tokens WHERE token = %s", (TOKEN,)); cur.execute("DELETE FROM rh_v4_keys WHERE pool_id = 'POOLANT'")
        db_conn.commit()
    assert out[status] == 1 and row["status"] == status and row["reason"] == reason and row["creator"] == CREATOR


def test_the_rh_engine_never_buys_a_refused_v4_coin_and_still_prices_it(db_conn, monkeypatch):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM rh_v4_tokens WHERE token = %s", (TOKEN,)); cur.execute("INSERT INTO rh_v4_tokens (token, status, reason) VALUES (%s, 'refused', %s)", (TOKEN, U.CREATOR_MOVES))
    db_conn.commit()
    e = me.MinuteEngine(market=markets.RH); sol = me.MinuteEngine(); t = 1_800_000_000.0
    try:
        e._refresh_refused(db_conn, t); sol._refresh_refused(db_conn, t)
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM rh_v4_tokens WHERE token = %s", (TOKEN,))
        db_conn.commit()
    assert TOKEN in e.refused and sol.refused == set()
    import numpy as np
    info = {"open": 1.0, "price": 1.0, "ec": 0.01, "broken": False, "resq": 100.0, "logvol_15m": 100.0, "source": "v4", "seen_h": 10.0}
    monkeypatch.setattr(e, "_features", lambda conn, mint, a, t_end: (np.zeros(len(me.X_COLS), np.float32), dict(info)))
    monkeypatch.setattr(me, "in_universe", lambda X, cols: np.array([True]))
    X, infos, mints, bars = e._minute_rows(db_conn, {TOKEN: {"buy": 1.0, "sell": 0.0}, "0xother": {"buy": 1.0, "sell": 0.0}}, t)
    assert mints == ["0xother"] and TOKEN in bars
