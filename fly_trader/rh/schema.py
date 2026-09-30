"""Robinhood Chain tables: the Pons/Uniswap-v4 index, RH minutes and meta, and the live RH book's on-chain ledger.

Idempotent DDL outside the version ladder in db/schema.py (applied after VAULT_DDL), like the vault's: no migration
number is taken, so master's concurrent Kalshi/vault migrations never collide. Only ever ADD here.

Units are in the names: ``*_raw`` = integer token units (numeric(78,0): a uint256 fits), ``*_wei`` = native ETH wei,
``*_eth`` = ETH as a float, ``*_q`` = the pool's quote asset in whole units. Addresses are lowercase hex; pool ids are
the v4 PoolId (bytes32 hex). The existing ledger tables (positions, decisions, beats, wealth_marks) carry the RH books
too: their ``*_sol`` columns hold the book's own unit (ETH for paper_rh_* / live_rh, see fly_trader/markets.py).
"""
from __future__ import annotations

RH_DDL: list[str] = [
    # ---- indexer cursors: one per log source, each with its adaptive getLogs range
    """CREATE TABLE IF NOT EXISTS rh_scan (
        name text PRIMARY KEY, block bigint NOT NULL DEFAULT 0, range_blocks bigint NOT NULL DEFAULT 50000,
        detail jsonb, updated_at timestamptz NOT NULL DEFAULT now())""",
    # ---- quote assets a Pons launch may pair with (factory-approved): the base legs and their ETH marks
    """CREATE TABLE IF NOT EXISTS rh_assets (
        asset text PRIMARY KEY, symbol text, class text NOT NULL CHECK (class IN ('eth', 'stable', 'btc', 'stock')),
        decimals int NOT NULL, approved boolean NOT NULL DEFAULT true, graduation_threshold_raw numeric(78,0),
        ref_pool text, ref_kind text, ref_quote text, ref_fee_bps int, trading_hours jsonb, note text,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    # ---- Pons launches (the curve phase feeds features; trading starts at graduation)
    """CREATE TABLE IF NOT EXISTS rh_tokens (
        token text PRIMARY KEY, launch_block bigint, launch_ts timestamptz, launch_tx text, creator text, curve text,
        quote_asset text, quote_class text, decimals int NOT NULL DEFAULT 18, name text, symbol text, uri text,
        supply_raw numeric(78,0), dev_quote_in_raw numeric(78,0), dev_tokens_raw numeric(78,0), creator_tax_bps int, curve_fee_bps int,
        graduated_at timestamptz, grad_block bigint, pool_id text, grad_position_id numeric(78,0), grad_quote_raw numeric(78,0),
        grad_token_raw numeric(78,0), status text NOT NULL DEFAULT 'curve' CHECK (status IN ('curve', 'graduated')),
        updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS rh_tokens_creator_idx ON rh_tokens (creator, launch_ts)",
    "CREATE INDEX IF NOT EXISTS rh_tokens_curve_idx ON rh_tokens (curve)",
    """CREATE TABLE IF NOT EXISTS rh_pools (
        pool_id text PRIMARY KEY, token text, currency0 text NOT NULL, currency1 text NOT NULL, token_is_0 boolean,
        fee int, tick_spacing int, hooks text, quote_asset text, init_block bigint, init_ts timestamptz, is_pons boolean NOT NULL DEFAULT false,
        sqrt_price_x96 numeric(78,0), liquidity numeric(78,0), tick int, updated_block bigint)""",
    "CREATE INDEX IF NOT EXISTS rh_pools_token_idx ON rh_pools (token)",
    """CREATE TABLE IF NOT EXISTS rh_curve_trades (
        block bigint NOT NULL, log_index int NOT NULL, tx_hash text NOT NULL, ts timestamptz NOT NULL, token text NOT NULL,
        side smallint NOT NULL, trader text, recipient text, quote_raw numeric(78,0), tokens_raw numeric(78,0), fee_raw numeric(78,0),
        tax_raw numeric(78,0), snipe_tax_raw numeric(78,0), quote_eth double precision, PRIMARY KEY (block, log_index))""",
    "CREATE INDEX IF NOT EXISTS rh_curve_trades_token_idx ON rh_curve_trades (token, ts)",
    # ---- graduated-pool swaps (Uniswap v4 PoolManager Swap on Pons pools), priced in the quote and in ETH
    """CREATE TABLE IF NOT EXISTS rh_swaps (
        block bigint NOT NULL, log_index int NOT NULL, tx_hash text NOT NULL, block_hash text, ts timestamptz NOT NULL,
        pool_id text NOT NULL, token text NOT NULL, trader text, side smallint NOT NULL, token_raw numeric(78,0), quote_raw numeric(78,0),
        hook_fee_raw numeric(78,0), hook_tax_raw numeric(78,0), sqrt_price_x96 numeric(78,0), liquidity numeric(78,0), tick int,
        price_q double precision, base_eth double precision, price_eth double precision, quote_eth double precision,
        resq_q double precision, resq_eth double precision, PRIMARY KEY (block, log_index))""",
    "CREATE INDEX IF NOT EXISTS rh_swaps_token_ts_idx ON rh_swaps (token, ts)",
    "CREATE INDEX IF NOT EXISTS rh_swaps_ts_idx ON rh_swaps (ts)",
    """CREATE TABLE IF NOT EXISTS rh_liquidity (
        block bigint NOT NULL, log_index int NOT NULL, tx_hash text, ts timestamptz, pool_id text NOT NULL, sender text,
        tick_lower int, tick_upper int, liquidity_delta numeric(78,0), salt text, PRIMARY KEY (block, log_index))""",
    # ---- each quote asset's ETH (and USD) value per minute: the base leg's mark
    """CREATE TABLE IF NOT EXISTS rh_base_prices (
        asset text NOT NULL, ts timestamptz NOT NULL, price_eth double precision, price_usd double precision, n_obs int NOT NULL DEFAULT 0,
        stale boolean NOT NULL DEFAULT false, PRIMARY KEY (asset, ts))""",
    # ---- RH minutes: pump_minutes' twin, amounts in ETH (non-ETH-quoted pools converted at the minute's base/ETH)
    """CREATE TABLE IF NOT EXISTS rh_minutes (
        mint text NOT NULL, ts timestamptz NOT NULL, pool_id text, quote_asset text, quote_class text,
        open double precision, high double precision, low double precision, close double precision, close_q double precision, base_eth double precision,
        buy_eth double precision, sell_eth double precision, n_buys int, n_sells int, n_traders int, resq_eth double precision, resq_q double precision,
        fee_rate double precision, n_buyers int, wash_eth double precision, wash_buy_eth double precision, top_sell_eth double precision,
        insider_sell_eth double precision, skill_buy real[], revised boolean NOT NULL DEFAULT false, PRIMARY KEY (mint, ts))""",
    "CREATE INDEX IF NOT EXISTS rh_minutes_ts_idx ON rh_minutes (ts)",
    # ---- per-launch facts in corpus_meta's column names (values in ETH) so both chains feed the same model inputs
    """CREATE TABLE IF NOT EXISTS rh_meta (
        mint text PRIMARY KEY, graduated_at timestamptz, create_ts timestamptz, creator text, quote_asset text, quote_class text,
        supply double precision, ttg_min double precision, dev_sol double precision, dev_tokens double precision, dev_share double precision,
        mayhem boolean NOT NULL DEFAULT false, rq0 double precision, pool_id text,
        prior_launches int, prior_grads int, prior_known int, prior_rug_share real, prior_moon_share real,
        own_dd60 real, own_max60 real, own_alive6h boolean,
        bundle_share real, dev_hold_share real, dev_sold_frac real, grad_hhi real, curve_known real,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS rh_meta_creator_idx ON rh_meta (creator, graduated_at)",
    """CREATE TABLE IF NOT EXISTS rh_insiders (
        mint text NOT NULL, wallet text NOT NULL, kind text NOT NULL, PRIMARY KEY (mint, wallet))""",
    # ---- the live RH book's on-chain ledger (fly_trader/rh/exec.py, rh/accounting.py)
    """CREATE TABLE IF NOT EXISTS rh_intents (
        id bigserial PRIMARY KEY, book text NOT NULL, decision_id bigint, position_id bigint,
        kind text NOT NULL CHECK (kind IN ('buy', 'sell', 'base_out', 'sweep', 'liquidate', 'payout')),
        token text, quote_asset text, route text CHECK (route IN ('atomic_kyber', 'atomic_ur', 'two_leg', 'transfer')),
        amount_in_raw numeric(78,0), min_out_raw numeric(78,0),
        state text NOT NULL DEFAULT 'new' CHECK (state IN ('new', 'leg1', 'leg2', 'done', 'failed', 'aborted')),
        attempts int NOT NULL DEFAULT 0, committed_wei numeric(78,0) NOT NULL DEFAULT 0, plan jsonb, error text,
        created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE UNIQUE INDEX IF NOT EXISTS rh_intents_decision_uniq ON rh_intents (decision_id, kind) WHERE decision_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rh_intents_open_idx ON rh_intents (state) WHERE state NOT IN ('done', 'failed', 'aborted')",
    """CREATE TABLE IF NOT EXISTS rh_txs (
        id bigserial PRIMARY KEY, intent_id bigint, position_id bigint,
        kind text NOT NULL CHECK (kind IN ('approve', 'swap', 'cancel', 'transfer', 'payout', 'wrap')),
        from_addr text NOT NULL, nonce bigint NOT NULL, hash text NOT NULL UNIQUE, raw text NOT NULL, to_addr text,
        value_wei numeric(78,0) NOT NULL DEFAULT 0, data_sha256 text, gas_limit bigint, max_fee_wei numeric(78,0), max_prio_wei numeric(78,0),
        status text NOT NULL DEFAULT 'signed' CHECK (status IN ('signed', 'sent', 'mined_ok', 'reverted', 'dropped', 'replaced')),
        block bigint, block_hash text, gas_used bigint, eff_gas_price_wei numeric(78,0), fee_wei numeric(78,0),
        router text, quote jsonb, build jsonb, applied_at timestamptz, error text,
        created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS rh_txs_nonce_idx ON rh_txs (from_addr, nonce)",
    "CREATE INDEX IF NOT EXISTS rh_txs_open_idx ON rh_txs (status) WHERE applied_at IS NULL",
    """CREATE TABLE IF NOT EXISTS rh_legs (
        id bigserial PRIMARY KEY, intent_id bigint NOT NULL, tx_id bigint, position_id bigint, leg_no int NOT NULL,
        asset_in text NOT NULL, asset_out text NOT NULL, amount_in_raw numeric(78,0) NOT NULL, amount_out_raw numeric(78,0) NOT NULL,
        eth_in double precision, eth_out double precision,
        verified_by text NOT NULL CHECK (verified_by IN ('balance_delta', 'receipt_logs', 'model')), ts timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS rh_legs_position_idx ON rh_legs (position_id)",
    """CREATE TABLE IF NOT EXISTS rh_base_lots (
        id bigserial PRIMARY KEY, asset text NOT NULL, qty_raw numeric(78,0) NOT NULL, basis_eth double precision NOT NULL,
        position_id bigint, intent_id bigint, source text NOT NULL CHECK (source IN ('entry_leg', 'exit_leg', 'stray')),
        status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'consumed', 'liquidated', 'dust')),
        proceeds_eth double precision, fee_eth double precision, opened_at timestamptz NOT NULL DEFAULT now(), closed_at timestamptz)""",
    "CREATE INDEX IF NOT EXISTS rh_base_lots_open_idx ON rh_base_lots (status) WHERE status = 'open'",
    """CREATE TABLE IF NOT EXISTS rh_wallet_flows (
        id bigserial PRIMARY KEY, block bigint, ts timestamptz NOT NULL DEFAULT now(), direction text NOT NULL CHECK (direction IN ('in', 'out')),
        kind text NOT NULL CHECK (kind IN ('deposit', 'withdrawal', 'payout', 'unexplained_in', 'unknown_outbound')),
        wei numeric(78,0) NOT NULL, note text)""",
    """CREATE TABLE IF NOT EXISTS rh_wallet_marks (
        ts timestamptz PRIMARY KEY, native_wei numeric(78,0), positions_eth double precision, lots_eth double precision,
        exit_cost_eth double precision, reserved_wei numeric(78,0), wealth_eth double precision, consistent boolean NOT NULL DEFAULT true)""",
    # ---- additive columns on shared tables (defaults keep every Solana row as it was)
    # the hook's fee + tax as a fraction of the swap's gross amount (Pons takes it in whichever currency comes in)
    "ALTER TABLE rh_swaps ADD COLUMN IF NOT EXISTS fee_frac double precision",
    # when a receipt's fee was recorded: the reconciler counts gas by it (updated_at moves on later status changes)
    "ALTER TABLE rh_txs ADD COLUMN IF NOT EXISTS fee_at timestamptz",
    # a pool's hook fee + tax is fixed at launch (measured: 1,665 of 1,732 pools vary < 0.05 points): read from its first
    # FEE_OBS swaps' HookFeeCollected logs, then applied to its later swaps without fetching their fee logs (rh/index.py)
    "ALTER TABLE rh_pools ADD COLUMN IF NOT EXISTS hook_fee double precision",
    "ALTER TABLE rh_pools ADD COLUMN IF NOT EXISTS fee_obs int NOT NULL DEFAULT 0",
    "ALTER TABLE rh_tokens ADD COLUMN IF NOT EXISTS curve_indexed boolean NOT NULL DEFAULT false",
    "ALTER TABLE rh_assets ADD COLUMN IF NOT EXISTS ref_path jsonb",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS chain text NOT NULL DEFAULT 'sol'",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS quote_asset text",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS gas_q double precision NOT NULL DEFAULT 0",
    "ALTER TABLE fly_scored ADD COLUMN IF NOT EXISTS chain text NOT NULL DEFAULT 'sol'",
    "ALTER TABLE notional_ledger ADD COLUMN IF NOT EXISTS market text NOT NULL DEFAULT 'sol'",
    # rails circuit 3 (fly_trader/markets.RH_CIRCUIT); the id CHECK only allowed 1 before the Kalshi migration dropped it,
    # and an install still on the distribution's older ladder may carry it
    "ALTER TABLE circuit_state DROP CONSTRAINT IF EXISTS circuit_state_id_check",
    "INSERT INTO circuit_state (id) VALUES (3) ON CONFLICT DO NOTHING",
]
