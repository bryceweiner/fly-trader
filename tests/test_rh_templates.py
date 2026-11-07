"""Known-good contract templates (rh/templates.py, Bryce 2026-11-05): a plain v4 coin whose code matches none is refused at
registration, and coins accepted before the check are read again and refused when they deviate."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader.rh import templates as TP, token_check as T, v4_universe as U


def _code(addr: int, jump: int, meta: bytes = b"\x01" * 34, tail: bytes = b"") -> str:
    """PUSH20 addr, PUSH2 jump, JUMP, STOP (+ tail), then a CBOR metadata trailer."""
    body = bytes([0x73]) + addr.to_bytes(20, "big") + bytes([0x61]) + jump.to_bytes(2, "big") + bytes([0x56, 0x00]) + tail
    trailer = bytes([0xa2, 0x64]) + b"ipfs" + bytes([0x58, 0x22]) + meta + bytes([0x64]) + b"solc" + bytes([0x43, 0, 8, 0x1d])
    return "0x" + (body + trailer + len(trailer).to_bytes(2, "big")).hex()


GOOD, BAD = _code(0xaa, 0x0102), _code(0xaa, 0x0103)
IMPL = "0x" + "ee" * 20


def test_a_template_ignores_addresses_constants_and_metadata_not_code():
    t = TP.template_id(GOOD)
    assert TP.template_id(_code(0xbb, 0x0102, meta=b"\x02" * 34)) == t          # another address, another build's metadata
    assert TP.template_id(BAD) != t                                            # another jump target: other code
    assert TP.template_id(_code(0xaa, 0x0102, tail=b"\x01")) != t              # one more opcode
    assert TP.template_id("0x") is None and TP.template_id(None) is None and not TP.known(None)


def test_the_check_reads_a_clones_implementation_template():
    clone = "0x" + T.CLONE_PREFIXES[0] + IMPL[2:] + "5af43d82803e903d91602b57fd5bf3"

    class _Rpc:
        def batch(self, calls):
            if calls[0][1][0] == IMPL:
                return [GOOD]
            word = "0x" + "00" * 32
            return [clone, word, word, "0x" + format(18, "064x"), None, None, "0x" + format(10 ** 27, "064x"), None]
    assert T.check(_Rpc(), "0x" + "cc" * 20)["template"] == TP.template_id(GOOD)


TOKEN, CREATOR = "0xtpl0000000000000000000000000000000000001", "0xtplcreator"


def _pending(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_v4_tokens WHERE token LIKE '0xtpl%%'"); cur.execute("DELETE FROM rh_v4_keys WHERE pool_id = 'POOLTPL'")
        cur.execute("INSERT INTO rh_v4_keys (pool_id, currency0, currency1, init_block, init_tx) VALUES ('POOLTPL', '0x0', %s, 1, '0xinit')", (TOKEN,))
        cur.execute("INSERT INTO rh_v4_tokens (token, first_pool, status) VALUES (%s, 'POOLTPL', 'pending')", (TOKEN,))
    conn.commit()


def _clear(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_v4_tokens WHERE token LIKE '0xtpl%%'"); cur.execute("DELETE FROM rh_v4_keys WHERE pool_id = 'POOLTPL'")
    conn.commit()


class _ProbeRpc:
    def __init__(self):
        self.probed = 0

    def tx_by_hash(self, h):
        return {"from": CREATOR}

    def call(self, method, params):
        from fly_trader.chain.evm import EvmRpcError
        self.probed += 1
        raise EvmRpcError("eth_call", {"code": 3, "message": "execution reverted"})


@pytest.mark.parametrize("code, status, reason, probed", [(GOOD, "ok", "no owner or admin powers", 1), (BAD, "refused", TP.DEVIATES, 0)])
def test_registration_refuses_a_coin_off_every_known_good_template(db_conn, monkeypatch, code, status, reason, probed):
    monkeypatch.setattr(TP, "KNOWN_GOOD", {TP.template_id(GOOD): (5, 5, 5)})
    monkeypatch.setattr(T, "check", lambda rpc, t: {"ok": True, "reason": "no owner or admin powers", "symbol": "X", "name": "x", "decimals": 18,
                                                    "supply_raw": 10 ** 27, "powers": [], "template": TP.template_id(code)})
    _pending(db_conn); rpc = _ProbeRpc()
    try:
        out = U.run_checks(db_conn, rpc, limit=5); db_conn.commit()
        row = db_conn.execute("SELECT status, reason, detail FROM rh_v4_tokens WHERE token = %s", (TOKEN,)).fetchone()
    finally:
        _clear(db_conn)
    assert out[status] == 1 and row["status"] == status and row["reason"] == reason and row["detail"]["template"] == TP.template_id(code)
    assert rpc.probed == probed                                                 # a deviating coin costs no creator probe


def test_coins_accepted_before_the_check_are_read_again(db_conn, monkeypatch):
    monkeypatch.setattr(TP, "KNOWN_GOOD", {TP.template_id(GOOD): (5, 5, 5)}); monkeypatch.setitem(U._RETEMPLATED, "done", False)
    clone = "0x" + T.CLONE_PREFIXES[0] + IMPL[2:] + "5af43d82803e903d91602b57fd5bf3"
    toks = {"0xtpl_good": GOOD, "0xtpl_bad": BAD, "0xtpl_clone": clone, "0xtpl_unread": None, "0xtpl_done": BAD}
    codes = {**{k: v for k, v in toks.items() if v}, IMPL: GOOD}

    class _Rpc:
        batches = 0

        def batch(self, calls):
            _Rpc.batches += 1
            return [codes.get(a) for _, (a, _) in calls]                        # None: an unreadable answer
    future = datetime.now(timezone.utc) + timedelta(days=3650)                 # newest first: ahead of any other row
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM rh_v4_tokens WHERE token LIKE '0xtpl%%'")
        for i, t in enumerate(toks):
            detail = {"reason": "no owner or admin powers", **({"template": "old"} if t == "0xtpl_done" else {})}
            cur.execute("INSERT INTO rh_v4_tokens (token, status, reason, detail, first_ts) VALUES (%s, 'ok', 'no owner or admin powers', %s, %s)",
                        (t, json.dumps(detail), future + timedelta(minutes=i)))
    db_conn.commit()
    try:
        out = U.retemplate(db_conn, _Rpc(), limit=4); db_conn.commit()
        rows = {r["token"]: r for r in db_conn.execute("SELECT token, status, reason, detail FROM rh_v4_tokens WHERE token LIKE '0xtpl%%'").fetchall()}
        again = U.retemplate(db_conn, _Rpc(), limit=4); db_conn.commit()
    finally:
        _clear(db_conn)
    assert out == {"kept": 2, "refused": 1, "unread": 1} and _Rpc.batches >= 2
    assert rows["0xtpl_good"]["status"] == "ok" and rows["0xtpl_good"]["detail"]["template"] == TP.template_id(GOOD)
    assert rows["0xtpl_clone"]["status"] == "ok" and rows["0xtpl_clone"]["detail"]["template"] == TP.template_id(GOOD)
    assert rows["0xtpl_bad"]["status"] == "refused" and rows["0xtpl_bad"]["reason"] == TP.DEVIATES
    assert rows["0xtpl_unread"]["status"] == "ok" and "template" not in rows["0xtpl_unread"]["detail"]    # read again next slot
    assert rows["0xtpl_done"]["status"] == "ok"                                 # checked under the template already: left alone
    assert again["unread"] >= 1 and again["refused"] == 0


def test_the_re_check_stops_once_every_coin_is_read(db_conn, monkeypatch):
    monkeypatch.setitem(U._RETEMPLATED, "done", False)

    class _Rpc:
        def batch(self, calls):
            raise AssertionError("nothing left to read")
    with db_conn.cursor() as cur:
        cur.execute("""UPDATE rh_v4_tokens SET detail = coalesce(detail, '{}'::jsonb) || '{"template": "t"}' WHERE status = 'ok'""")
    try:
        assert U.retemplate(db_conn, _Rpc()) == {"kept": 0, "refused": 0, "unread": 0} and U._RETEMPLATED["done"]
        assert U.retemplate(db_conn, _Rpc()) == {}
    finally:
        db_conn.rollback()


def test_the_shipped_templates_are_full_hashes():
    assert len(TP.KNOWN_GOOD) == 9 and all(len(t) == 64 and int(t, 16) >= 0 for t in TP.KNOWN_GOOD)
    assert all(c >= 5 and s >= 5 for _, c, s in TP.KNOWN_GOOD.values())
