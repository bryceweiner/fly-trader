"""A graduate's curve back-fill runs off the live stream (rh/curves.py): the live index step only notes where the curve ends,
the back-fill walks it one chunk per call with its progress saved, and the coin's meta row waits until it is done."""
from fly_trader.rh import curves, index, meta, scan

TOK, CURVE = "0xc0ffee0000000000000000000000000000000001", "0xc0ffee00000000000000000000000000000000c1"


def _token(conn, launch, until, **kw):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_curve_trades WHERE token LIKE '0xc0ffee%%'"); cur.execute("DELETE FROM rh_tokens WHERE token LIKE '0xc0ffee%%'")
        cur.execute("INSERT INTO rh_tokens (token, curve, launch_block, curve_until, status, graduated_at) VALUES (%s, %s, %s, %s, 'graduated', now())",
                    (TOK, CURVE, launch, until))
        for k, v in kw.items():
            cur.execute(f"UPDATE rh_tokens SET {k} = %s WHERE token = %s", (v, TOK))
    conn.commit()


def _clear(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_curve_trades WHERE token LIKE '0xc0ffee%%'"); cur.execute("DELETE FROM rh_tokens WHERE token LIKE '0xc0ffee%%'")
    conn.commit()


def test_the_back_fill_walks_the_curve_one_chunk_at_a_time_and_resumes(db_conn, monkeypatch):
    asked = []
    monkeypatch.setattr(index, "_get_logs", lambda rpc, addrs, lo, hi, topics: asked.append((addrs[0], lo, hi)) or [])
    span = 2 * scan.MAX_RANGE + 10
    _token(db_conn, 1000, 1000 + span)
    try:
        steps = []
        while True:
            n, done = index.index_curve_chunk(db_conn, None, TOK); db_conn.commit(); steps.append(done)
            if done:
                break
        row = db_conn.execute("SELECT curve_through, curve_indexed FROM rh_tokens WHERE token = %s", (TOK,)).fetchone()
        assert steps == [False, False, True] and row["curve_indexed"] and row["curve_through"] == 1000 + span
        assert asked == [(CURVE, 1000, 1000 + scan.MAX_RANGE - 1), (CURVE, 1000 + scan.MAX_RANGE, 1000 + 2 * scan.MAX_RANGE - 1), (CURVE, 1000 + 2 * scan.MAX_RANGE, 1000 + span)]
        assert index.index_curve_chunk(db_conn, None, TOK) == (0, True) and len(asked) == 3          # done: nothing more is read
    finally:
        _clear(db_conn)


def test_the_back_fill_writes_the_curve_trades(db_conn, monkeypatch):
    log = {"blockNumber": hex(1005), "logIndex": "0x1"}
    monkeypatch.setattr(index, "_get_logs", lambda rpc, addrs, lo, hi, topics: [log] if lo <= 1005 <= hi else [])
    monkeypatch.setattr(index.pons, "decode", lambda x: {"block": 1005, "log_index": 1, "tx_hash": "0xt", "ts": 1_800_000_000.0, "side": 1, "trader": "0xw",
                                                         "recipient": "0xw", "quote_raw": 10, "tokens_raw": 20, "fee_raw": 0, "tax_raw": 0})
    _token(db_conn, 1000, 1010)
    try:
        assert index.index_curve_chunk(db_conn, None, TOK) == (1, True); db_conn.commit()
        assert db_conn.execute("SELECT count(*) AS n FROM rh_curve_trades WHERE token = %s", (TOK,)).fetchone()["n"] == 1
    finally:
        _clear(db_conn)


def test_the_live_index_step_only_notes_where_the_curve_ends(db_conn, monkeypatch):
    monkeypatch.setattr(index, "_get_logs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no curve read in the live step")))
    _token(db_conn, 1000, None)
    try:
        index.apply_pons(db_conn, None, [{"event": "PoolRegistered", "quote_asset": "0x0000000000000000000000000000000000000000", "token": TOK,
                                          "pool_id": "0xpoolc0ffee", "block": 5000, "ts": 1_800_000_000.0}])
        db_conn.commit()
        row = db_conn.execute("SELECT curve_until, curve_indexed FROM rh_tokens WHERE token = %s", (TOK,)).fetchone()
        assert row["curve_until"] == 5000 and not row["curve_indexed"]
        assert curves.next_token(db_conn) == TOK
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM rh_pools WHERE pool_id = '0xpoolc0ffee'")
        db_conn.commit(); _clear(db_conn)


def test_meta_waits_for_the_back_fill(db_conn, monkeypatch):
    seen = []
    monkeypatch.setattr(meta, "build", lambda conn, token: seen.append(token) or None)
    _token(db_conn, 1000, 9000)
    try:
        meta.run_once(); assert TOK not in seen
        with db_conn.cursor() as cur:
            cur.execute("UPDATE rh_tokens SET curve_indexed = true WHERE token = %s", (TOK,))
        db_conn.commit()
        meta.run_once(); assert TOK in seen
    finally:
        _clear(db_conn)
