"""The Robinhood Chain tables apply idempotently beside the ladder, circuit 3 exists, and Solana rows keep their meaning."""
from fly_trader.db import schema


def test_rh_ddl_applies_twice_and_adds_circuit_3(db_conn):
    schema.apply_schema(); schema.apply_schema()                      # idempotent
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM circuit_state ORDER BY id")
        assert {r["id"] for r in cur.fetchall()} >= {1, 2, 3}
        cur.execute("SELECT to_regclass(t) AS r FROM unnest(ARRAY['rh_scan','rh_assets','rh_tokens','rh_pools','rh_curve_trades','rh_swaps',"
                    "'rh_liquidity','rh_base_prices','rh_minutes','rh_meta','rh_insiders','rh_intents','rh_txs','rh_legs','rh_base_lots',"
                    "'rh_wallet_flows','rh_wallet_marks']) AS t")
        assert all(r["r"] is not None for r in cur.fetchall())
        cur.execute("SELECT column_name, column_default FROM information_schema.columns WHERE table_name = 'positions' AND column_name IN ('chain', 'gas_q')")
        cols = {r["column_name"]: r["column_default"] for r in cur.fetchall()}
        assert "'sol'" in cols["chain"] and cols["gas_q"].startswith("0")   # a Solana position needs no new value


def test_a_uint256_fits(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("INSERT INTO rh_legs (intent_id, leg_no, asset_in, asset_out, amount_in_raw, amount_out_raw, verified_by) "
                    "VALUES (-1, 1, 'a', 'b', %s, 0, 'model') RETURNING amount_in_raw", (2 ** 256 - 1,))
        assert int(cur.fetchone()["amount_in_raw"]) == 2 ** 256 - 1
    db_conn.rollback()
