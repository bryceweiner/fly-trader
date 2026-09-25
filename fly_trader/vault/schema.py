"""Vault tables (docs/vault/SPEC.md; plan "Core accounting").

Idempotent DDL outside the version ladder in db/schema.py: master (schema 9) and the distribution branch (schema 6)
apply the same statements, so the vault code ports without renumbering migrations. Only ever ADD here (a code rollback
must keep working against a newer database).
"""
from __future__ import annotations

VAULT_DDL: list[str] = [
    # SOL moving in or out of the trading wallet that is not trading: deposits and gifts in, operator withdrawals and
    # claim payouts out, and anything our key signed that we cannot account for (a possible key leak).
    """CREATE TABLE IF NOT EXISTS vault_flows (
        id bigserial PRIMARY KEY,
        signature text NOT NULL,
        slot bigint NOT NULL,
        block_time timestamptz,
        direction text NOT NULL CHECK (direction IN ('in', 'out')),
        kind text NOT NULL CHECK (kind IN ('deposit', 'profit', 'withdrawal', 'claim', 'unknown_outbound', 'anomaly')),
        counterparty text,
        lamports bigint NOT NULL CHECK (lamports >= 0),
        fee_lamports bigint NOT NULL DEFAULT 0,
        classified_by text NOT NULL DEFAULT 'auto',
        settlement_id bigint,
        note text,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (signature, direction, kind))""",
    "CREATE INDEX IF NOT EXISTS vault_flows_slot ON vault_flows (slot)",
    # 'sweep': owed SOL moved from the trading wallet to the payout wallet (internal: counted in neither direction)
    "ALTER TABLE vault_flows DROP CONSTRAINT IF EXISTS vault_flows_kind_check",
    "ALTER TABLE vault_flows ADD CONSTRAINT vault_flows_kind_check CHECK (kind IN "
    "('deposit', 'profit', 'withdrawal', 'claim', 'sweep', 'unknown_outbound', 'anomaly'))",
    """CREATE TABLE IF NOT EXISTS vault_scan (
        name text PRIMARY KEY,
        cursor text,
        slot bigint,
        detail jsonb,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    # FlyVault events from finalized Robinhood Chain blocks; locked_after is the account's earning balance after it.
    """CREATE TABLE IF NOT EXISTS vault_events (
        chain_id int NOT NULL,
        block_number bigint NOT NULL,
        log_index int NOT NULL,
        tx_hash text NOT NULL,
        block_time bigint NOT NULL,
        event text NOT NULL,
        account text,
        request_id numeric(78),
        amount numeric(78),
        locked_after numeric(78),
        ready_at bigint,
        PRIMARY KEY (chain_id, block_number, log_index))""",
    "CREATE INDEX IF NOT EXISTS vault_events_account ON vault_events (account, block_number, log_index)",
    """CREATE TABLE IF NOT EXISTS vault_settlements (
        id bigserial PRIMARY KEY,
        period_start timestamptz NOT NULL,
        period_end timestamptz NOT NULL UNIQUE,
        status text NOT NULL CHECK (status IN ('snapshotted', 'allocated', 'failed')),
        snapshot_slot bigint,
        snapshot_at timestamptz,
        native bigint, token_acct bigint, open_cost bigint,
        deposits bigint, withdrawals bigint, payouts bigint,
        realized bigint, allocated_before bigint, pot bigint, liquid_cap bigint, allocated bigint, carried bigint,
        total_weight numeric(100), earners int,
        rh_block_end bigint, impl text, impl_codehash text,
        inputs_sha256 text, error text,
        created_at timestamptz NOT NULL DEFAULT now(),
        allocated_at timestamptz)""",
    """CREATE TABLE IF NOT EXISTS vault_allocations (
        settlement_id bigint NOT NULL REFERENCES vault_settlements(id),
        evm text NOT NULL,
        weight numeric(100) NOT NULL,
        lamports bigint NOT NULL CHECK (lamports >= 0),
        PRIMARY KEY (settlement_id, evm))""",
    """CREATE TABLE IF NOT EXISTS vault_claims (
        id bigserial PRIMARY KEY,
        relay_id bigint UNIQUE,
        evm text NOT NULL,
        sol text NOT NULL,
        nonce text NOT NULL,
        domain text, uri text, chain_id int, sol_chain text, issued_at text, expires_at text,
        evm_sig text, sol_sig text, sig_kind text,
        status text NOT NULL CHECK (status IN ('received', 'verified', 'waiting_liquidity', 'sending', 'paid', 'rejected', 'failed')),
        reason text,
        lamports bigint,
        fee_lamports bigint,
        tx_signature text,
        blockhash text,
        last_valid_block_height bigint,
        slot bigint,
        attempts int NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        reported_at timestamptz,
        UNIQUE (evm, nonce))""",
    # one claim in flight per holder: a second one waits until the first is paid, rejected or failed
    "CREATE UNIQUE INDEX IF NOT EXISTS vault_claims_inflight ON vault_claims (evm) "
    "WHERE status IN ('verified', 'waiting_liquidity', 'sending')",
    # per-minute marks of the trading wallet for the flow-adjusted performance index (kill switch, NAV chart)
    """CREATE TABLE IF NOT EXISTS vault_nav (
        ts timestamptz PRIMARY KEY,
        slot bigint,
        native bigint NOT NULL,
        token_acct bigint NOT NULL,
        positions_value bigint NOT NULL,
        exit_cost bigint NOT NULL,
        wealth bigint NOT NULL,
        reserved bigint NOT NULL DEFAULT 0,
        consistent boolean NOT NULL DEFAULT true)""",
    """CREATE TABLE IF NOT EXISTS vault_kv (
        key text PRIMARY KEY,
        value jsonb,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    # owed per holder: everything allocated minus what was paid or is being paid
    """CREATE OR REPLACE VIEW vault_accounts AS
        SELECT a.evm, a.allocated, COALESCE(c.claimed, 0)::bigint AS claimed, COALESCE(c.in_flight, 0)::bigint AS in_flight,
               (a.allocated - COALESCE(c.claimed, 0) - COALESCE(c.in_flight, 0))::bigint AS owed
        FROM (SELECT evm, SUM(lamports)::bigint AS allocated FROM vault_allocations GROUP BY evm) a
        LEFT JOIN (SELECT evm,
                          SUM(lamports) FILTER (WHERE status = 'paid') AS claimed,
                          SUM(lamports) FILTER (WHERE status IN ('verified', 'waiting_liquidity', 'sending')) AS in_flight
                   FROM vault_claims GROUP BY evm) c USING (evm)""",
]
