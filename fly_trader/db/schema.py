"""Versioned DDL ladder for the fly_trader database.

Pattern from better_bot code/db_schema.py: a singleton ``schema_version`` row, a cross-process
advisory lock around migrations, and an idempotent list of statements. Adding a migration means
appending a version-gated block AND bumping SCHEMA_VERSION, or it never runs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg import sql

from .. import config
from .connection import connect, database_name, database_url

log = logging.getLogger(__name__)

SCHEMA_VERSION = 6          # 5: the plastic fly's tables and per-book halts (BASE_DDL); 6: the selector's strategy stack (MIGRATIONS[6])
_LOCK_KEY = 0x666C795F6D6967  # "fly_mig"

BASE_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS schema_version (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        version bigint NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS wallet_events (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(),
        kind text NOT NULL, pubkey text, detail jsonb)""",
    """CREATE TABLE IF NOT EXISTS tokens (
        mint text PRIMARY KEY, symbol text, name text, decimals int, token_program text,
        launchpad text, partner_config text, graduated_pool text, graduated_at timestamptz,
        first_pool_id text, first_pool_created_at timestamptz,
        mint_auth_disabled boolean, freeze_auth_disabled boolean,
        first_seen timestamptz NOT NULL DEFAULT now(), last_seen timestamptz NOT NULL DEFAULT now(),
        watch_status text NOT NULL DEFAULT 'candidate', raw jsonb)""",
    "CREATE INDEX IF NOT EXISTS tokens_graduated_at_idx ON tokens (graduated_at DESC)",
    "CREATE INDEX IF NOT EXISTS tokens_watch_status_idx ON tokens (watch_status)",
    """CREATE TABLE IF NOT EXISTS token_stats (
        id bigserial PRIMARY KEY, mint text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
        organic_score real, organic_label text, holder_count int, liquidity_usd double precision,
        usd_price double precision, mcap double precision, top_holders_pct real, dev_balance_pct real,
        dev_mints int, is_sus boolean, mint_auth_disabled boolean, freeze_auth_disabled boolean,
        is_verified boolean, stats jsonb)""",
    "CREATE INDEX IF NOT EXISTS token_stats_mint_ts_idx ON token_stats (mint, ts DESC)",
    """CREATE TABLE IF NOT EXISTS watch_pools (
        pool text PRIMARY KEY, mint text NOT NULL, quote_mint text, program_id text, program_label text,
        base_vault text, quote_vault text, base_decimals int, quote_decimals int, source text,
        active boolean NOT NULL DEFAULT true, tradable boolean NOT NULL DEFAULT true,
        added_at timestamptz NOT NULL DEFAULT now(), deactivated_at timestamptz, reason text)""",
    "CREATE INDEX IF NOT EXISTS watch_pools_active_idx ON watch_pools (active)",
    "CREATE INDEX IF NOT EXISTS watch_pools_mint_idx ON watch_pools (mint)",
    """CREATE TABLE IF NOT EXISTS swap_tape (
        id bigserial, ts timestamptz NOT NULL, slot bigint, sig text, tx_index int,
        pool text NOT NULL, mint text, side smallint NOT NULL,
        amount_base numeric(30,0), amount_quote bigint, price_sol double precision,
        signer text, res_base numeric(30,0), res_quote bigint, program_label text,
        PRIMARY KEY (id, ts)) PARTITION BY RANGE (ts)""",
    "CREATE INDEX IF NOT EXISTS swap_tape_pool_ts_idx ON swap_tape (pool, ts)",
    """CREATE TABLE IF NOT EXISTS capture_status (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        updated_at timestamptz NOT NULL DEFAULT now(), last_swap_ts timestamptz, rows_total bigint DEFAULT 0,
        pools_subscribed int DEFAULT 0, subscription_id bigint, reconnects int DEFAULT 0,
        decode_failures bigint DEFAULT 0, last_error text)""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id uuid PRIMARY KEY, kind text NOT NULL, started_at timestamptz NOT NULL DEFAULT now(),
        ended_at timestamptz, git_sha text, config jsonb, connectome_sha256 text, encoder_sha256 text,
        brain_snapshot_id bigint, corpus text, split jsonb, status text NOT NULL DEFAULT 'running',
        metrics jsonb)""",
    """CREATE TABLE IF NOT EXISTS populations (
        id smallint PRIMARY KEY, name text NOT NULL UNIQUE, n int NOT NULL, kind text, source text)""",
    """CREATE TABLE IF NOT EXISTS beats (
        id bigserial PRIMARY KEY, run_id uuid, ts timestamptz NOT NULL DEFAULT now(), beat_no bigint,
        sim_ts timestamptz, tape_last_id bigint, n_slots_active int, n_held_live int, n_held_paper int,
        ticks int, gpu_ms int, total_ms int, feed_age_ms int, notes jsonb)""",
    "CREATE INDEX IF NOT EXISTS beats_run_ts_idx ON beats (run_id, ts)",
    "CREATE INDEX IF NOT EXISTS beats_ts_idx ON beats (ts)",
    """CREATE TABLE IF NOT EXISTS beat_slots (
        beat_id bigint NOT NULL, slot smallint NOT NULL, ts timestamptz NOT NULL,
        mint text, pool text, dwell_beats int, features real[], feature_mask bigint, danger real,
        portfolio real[], glomeruli real[], stim real[], kc_active_frac real, m_hat real,
        rho_app real, rho_av real, dan_rew real, dan_pun real, delta_in real,
        decision_kind text, decision_size real, softmax_p real,
        PRIMARY KEY (beat_id, slot, ts)) PARTITION BY RANGE (ts)""",
    "CREATE INDEX IF NOT EXISTS beat_slots_mint_ts_idx ON beat_slots (mint, ts)",
    """CREATE TABLE IF NOT EXISTS brain_activity (
        beat_id bigint PRIMARY KEY, group_rates real[], kc_sparsity real, mbon_app_rate real,
        mbon_av_rate real, dan_rew_rate real, dan_pun_rate real, total_spikes bigint,
        v_mean real, v_max real, nan_flag boolean DEFAULT false)""",
    """CREATE TABLE IF NOT EXISTS decisions (
        id bigserial PRIMARY KEY, beat_id bigint, run_id uuid, ts timestamptz NOT NULL DEFAULT now(),
        slot smallint, mint text, pool text, kind text NOT NULL, m_hat real, size_sol real,
        forced boolean NOT NULL DEFAULT false, rail text, reason text, softmax_p real,
        book_targets text[], detail jsonb)""",
    "CREATE INDEX IF NOT EXISTS decisions_ts_idx ON decisions (ts)",
    "CREATE INDEX IF NOT EXISTS decisions_mint_ts_idx ON decisions (mint, ts)",
    "CREATE INDEX IF NOT EXISTS decisions_kind_ts_idx ON decisions (kind, ts)",
    """CREATE TABLE IF NOT EXISTS orders (
        id bigserial PRIMARY KEY, decision_id bigint, book text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
        attempt int NOT NULL DEFAULT 1, side text NOT NULL, input_mint text, output_mint text,
        amount_in numeric(30,0), slippage_bps int, request jsonb, order_response jsonb,
        execute_request jsonb, execute_response jsonb, request_id text, signature text,
        last_valid_block_height bigint, router text, fee_bps int, program_ids text[],
        status text, error_code int, error text, latency_ms int)""",
    "CREATE INDEX IF NOT EXISTS orders_decision_idx ON orders (decision_id)",
    "CREATE INDEX IF NOT EXISTS orders_signature_idx ON orders (signature)",
    "CREATE INDEX IF NOT EXISTS orders_ts_idx ON orders (ts)",
    """CREATE TABLE IF NOT EXISTS fills (
        id bigserial PRIMARY KEY, order_id bigint, book text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
        signature text, slot bigint, mint text, side text, token_delta numeric(30,0),
        sol_delta_lamports bigint, price_sol double precision, fee_lamports bigint, platform_fee jsonb,
        verified_by text, pre_snapshot jsonb, post_snapshot jsonb)""",
    "CREATE INDEX IF NOT EXISTS fills_book_ts_idx ON fills (book, ts)",
    "CREATE INDEX IF NOT EXISTS fills_mint_ts_idx ON fills (mint, ts)",
    """CREATE TABLE IF NOT EXISTS positions (
        id bigserial PRIMARY KEY, book text NOT NULL, mint text NOT NULL, pool text,
        opened_at timestamptz NOT NULL DEFAULT now(), closed_at timestamptz,
        entry_decision_id bigint, exit_decision_id bigint, qty numeric(30,0) NOT NULL DEFAULT 0,
        cost_sol double precision NOT NULL DEFAULT 0, entry_price double precision, peak_price double precision,
        last_mark_price double precision, last_mark_ts timestamptz, exit_price double precision,
        realized_sol double precision, fees_sol double precision NOT NULL DEFAULT 0,
        status text NOT NULL DEFAULT 'open', forced_exit_kind text, last_swap_ts timestamptz, satiety double precision NOT NULL DEFAULT 0)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS positions_open_uniq ON positions (book, mint) WHERE status = 'open'",
    "CREATE INDEX IF NOT EXISTS positions_book_status_idx ON positions (book, status)",
    """CREATE TABLE IF NOT EXISTS wealth_marks (
        beat_id bigint NOT NULL, book text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
        sol_free double precision, positions_value double precision, exit_cost double precision,
        wealth double precision, peak double precision, drawdown real, exposure double precision, n_open int,
        PRIMARY KEY (beat_id, book))""",
    "CREATE INDEX IF NOT EXISTS wealth_marks_book_ts_idx ON wealth_marks (book, ts)",
    """CREATE TABLE IF NOT EXISTS rewards (
        beat_id bigint NOT NULL, slot smallint NOT NULL, book text, mint text,
        r_slot real, r_global real, r_tilde real, m_prev real, m_now real, delta real, source text,
        PRIMARY KEY (beat_id, slot))""",
    """CREATE TABLE IF NOT EXISTS synapse_updates (
        beat_id bigint PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), eta real, gamma real,
        n_slots int, reward_source text, sum_abs real, max_abs real, frob real, frob_capped boolean,
        n_pos int, n_neg int, n_clipped int, w_mean real, w_min real, w_max real,
        delta_path text, delta_sha256 text)""",
    """CREATE TABLE IF NOT EXISTS brain_snapshots (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), run_id uuid, beat_id bigint,
        path text NOT NULL, sha256 text, kind text NOT NULL, promoted_at timestamptz, promoted_by text, note text)""",
    """CREATE TABLE IF NOT EXISTS brain_state (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        live_snapshot_id bigint, pending_snapshot_id bigint, updated_at timestamptz NOT NULL DEFAULT now())""",
    "INSERT INTO brain_state (singleton) VALUES (true) ON CONFLICT DO NOTHING",
    """CREATE TABLE IF NOT EXISTS encoder_state (
        name text PRIMARY KEY, version int NOT NULL DEFAULT 1, seed bigint, path text, sha256 text,
        running_mean real[], running_var real[], n bigint NOT NULL DEFAULT 0,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS circuit_state (
        id int PRIMARY KEY CHECK (id = 1), fail_count int NOT NULL DEFAULT 0, last_failure_ts timestamptz,
        tripped boolean NOT NULL DEFAULT false, kill_switch boolean NOT NULL DEFAULT false, kill_reason text,
        peak_wealth double precision, entries_paused boolean NOT NULL DEFAULT false,
        updated_at timestamptz NOT NULL DEFAULT now())""",
    "INSERT INTO circuit_state (id) VALUES (1) ON CONFLICT DO NOTHING",
    """CREATE TABLE IF NOT EXISTS circuit_events (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), kind text NOT NULL, detail jsonb)""",
    """CREATE TABLE IF NOT EXISTS notional_ledger (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), sol double precision NOT NULL, kind text)""",
    "CREATE INDEX IF NOT EXISTS notional_ledger_ts_idx ON notional_ledger (ts)",
    """CREATE TABLE IF NOT EXISTS api_calls (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), service text NOT NULL, endpoint text,
        method text, status int, latency_ms int, ok boolean, error text, request jsonb, response_bytes int)""",
    "CREATE INDEX IF NOT EXISTS api_calls_ts_idx ON api_calls (ts)",
    "CREATE INDEX IF NOT EXISTS api_calls_service_ok_ts_idx ON api_calls (service, ok, ts)",
    """CREATE TABLE IF NOT EXISTS events (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), level text NOT NULL DEFAULT 'info',
        source text, message text NOT NULL, detail jsonb)""",
    "CREATE INDEX IF NOT EXISTS events_ts_idx ON events (ts)",
    """CREATE TABLE IF NOT EXISTS slot_visits (
        mint text PRIMARY KEY, last_visit_ts timestamptz, visits bigint NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS processes (
        id bigserial PRIMARY KEY, name text NOT NULL, pid int NOT NULL, cmd text[],
        started_at timestamptz NOT NULL DEFAULT now(), stopped_at timestamptz, exit_code int,
        log_path text, started_by text)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS processes_live_uniq ON processes (name) WHERE stopped_at IS NULL",
    """CREATE TABLE IF NOT EXISTS corpus_tokens (
        mint text PRIMARY KEY, graduated_at timestamptz NOT NULL, grad_sig text, grad_slot bigint,
        source text NOT NULL DEFAULT 'migration_wallet', status text NOT NULL DEFAULT 'pending',
        life_h double precision, candles_1m int, candles_5m int, trades int, trades_through timestamptz,
        candle_path text, trade_path text, last_error text,
        added_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS corpus_tokens_status_idx ON corpus_tokens (status, graduated_at DESC)",
    """CREATE TABLE IF NOT EXISTS corpus_features (
        mint text PRIMARY KEY, built_at timestamptz NOT NULL DEFAULT now(), rows int, trade_rows int, has_trades boolean, path text)""",
    """CREATE TABLE IF NOT EXISTS replay_hours (
        hour timestamptz PRIMARY KEY, status text NOT NULL DEFAULT 'done', trades int, events int, bytes bigint, took_s real,
        last_error text, done_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS replay_days (
        day date PRIMARY KEY, graduations int, tokens_written int, took_s real, assembled_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS corpus_meta (
        mint text PRIMARY KEY, create_ts timestamptz, creator text, dev_sol double precision, dev_tokens double precision, dev_share real,
        supply double precision, mayhem boolean, uri text, name text, symbol text, ttg_min double precision, rq0 double precision, pool_id text,
        prior_launches int, prior_grads int, prior_known int, prior_rug_share real, prior_moon_share real,
        own_dd60 real, own_max60 real, own_alive6h boolean, updated_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS corpus_meta_creator_idx ON corpus_meta (creator)",
    "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS graduated_at timestamptz",
    "CREATE INDEX IF NOT EXISTS corpus_meta_creator_g_idx ON corpus_meta (creator, graduated_at)",
    """CREATE TABLE IF NOT EXISTS mature_days (
        day date PRIMARY KEY, mints int, rows int, took_s real, built_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS pump_minutes (
        mint text NOT NULL, ts timestamptz NOT NULL, pool_id text, open double precision, high double precision, low double precision, close double precision,
        buy_sol double precision, sell_sol double precision, n_buys int, n_sells int, n_traders int, resq_sol double precision, fee_rate double precision,
        PRIMARY KEY (mint, ts))""",
    "CREATE INDEX IF NOT EXISTS pump_minutes_ts_idx ON pump_minutes (ts)",
    """CREATE TABLE IF NOT EXISTS pump_events (
        sig text PRIMARY KEY, ts timestamptz NOT NULL, action text NOT NULL, pool text, mint text, pool_id text, signer text, dev_sol double precision,
        dev_tokens double precision, supply double precision, mayhem boolean, quote_in_pool double precision, name text, symbol text, uri text)""",
    "CREATE INDEX IF NOT EXISTS pump_events_mint_idx ON pump_events (mint)",
    "CREATE INDEX IF NOT EXISTS pump_events_creates_idx ON pump_events (signer, ts) WHERE action = 'create'",
    # PumpSwap pools NOT created by a pump.fun migration (createPool by anyone): unburned liquidity, blocked from the universe
    """CREATE TABLE IF NOT EXISTS pump_pools (
        pool_id text PRIMARY KEY, mint text, created_by text, ts timestamptz, source text NOT NULL, added_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS pump_pools_mint_idx ON pump_pools (mint)",
    """CREATE TABLE IF NOT EXISTS ui_settings (
        key text PRIMARY KEY, value jsonb, updated_at timestamptz NOT NULL DEFAULT now())""",
    # the plastic fly (agent/fly_session.py): every scored minute (pending rows are its synaptic tags; x is dropped once
    # the label resolves), daily calibrations per arm (plastic / frozen), hourly learning statistics, rollbacks
    """CREATE TABLE IF NOT EXISTS fly_scored (
        ts timestamptz NOT NULL, mint text NOT NULL, x real[], score real, frozen_score real, line real, frozen_line real,
        label real, state text NOT NULL DEFAULT 'pending', resolved_at timestamptz, bootstrap_id bigint,
        PRIMARY KEY (ts, mint))""",
    "CREATE INDEX IF NOT EXISTS fly_scored_state_ts_idx ON fly_scored (state, ts)",
    """CREATE TABLE IF NOT EXISTS fly_calibrations (
        day date NOT NULL, arm text NOT NULL, line real, sizing jsonb, trades int, total real, mean real, window_days int,
        created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (day, arm))""",
    """CREATE TABLE IF NOT EXISTS fly_updates (
        hour timestamptz PRIMARY KEY, n int, mean_delta real, mean_abs_delta real, step real, capped int, drift real, ic real, detail jsonb)""",
    """CREATE TABLE IF NOT EXISTS fly_rollbacks (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), reason text, checks jsonb, from_snapshot bigint, to_snapshot bigint)""",
    # per-book entry halts of the paper race books (agent/rails.check_book_drawdown); the live book uses circuit_state
    """CREATE TABLE IF NOT EXISTS book_state (
        book text PRIMARY KEY, halted boolean NOT NULL DEFAULT false, reason text, peak double precision,
        updated_at timestamptz NOT NULL DEFAULT now())""",
]

# Version-gated migrations: {version: [statements]} applied when stored version < version.
MIGRATIONS: dict[int, list[str]] = {
    2: ["ALTER TABLE beat_slots ALTER COLUMN feature_mask TYPE bigint"],
    3: ["ALTER TABLE positions ADD COLUMN IF NOT EXISTS satiety double precision NOT NULL DEFAULT 0"],
    4: ["ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS fee_rate double precision"],       # the pool fee the stream reports as charged
    # 6: the selector's strategy stack — per-minute wallet flow (train/flow.py), graduation-time curve facts and insiders
    # (train/corpus_meta.py), per-position hold and strategy, the fly's per-strategy scored rows
    6: ["ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS n_buyers int",
        "ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS wash_sol double precision",
        "ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS wash_buy_sol double precision",
        "ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS top_sell_sol double precision",
        "ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS insider_sell_sol double precision",
        "ALTER TABLE pump_minutes ADD COLUMN IF NOT EXISTS skill_buy real[]",
        "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS bundle_share real",
        "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS dev_hold_share real",
        "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS dev_sold_frac real",
        "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS grad_hhi real",
        "ALTER TABLE corpus_meta ADD COLUMN IF NOT EXISTS curve_known boolean",
        "CREATE TABLE IF NOT EXISTS token_insiders (mint text NOT NULL, wallet text NOT NULL, kind text NOT NULL, PRIMARY KEY (mint, wallet))",
        "ALTER TABLE positions ADD COLUMN IF NOT EXISTS hold_s double precision",
        "ALTER TABLE positions ADD COLUMN IF NOT EXISTS strategy text",
        "ALTER TABLE fly_scored ADD COLUMN IF NOT EXISTS strategy text NOT NULL DEFAULT 'ev'",
        "ALTER TABLE fly_scored ADD COLUMN IF NOT EXISTS hold_min int",
        "ALTER TABLE fly_scored DROP CONSTRAINT IF EXISTS fly_scored_pkey",
        "ALTER TABLE fly_scored ADD PRIMARY KEY (ts, mint, strategy)"],
}


def ensure_database(url: str | None = None) -> bool:
    """Create the database named in DATABASE_URL if it does not exist. Returns True if created."""
    url = url or database_url()
    name = database_name(url)
    admin_url = psycopg.conninfo.make_conninfo(url, dbname="postgres")   # same user/host/password, URL or key=value DSN
    with psycopg.connect(admin_url, autocommit=True) as conn:
        row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if row:
            return False
        conn.execute(f'CREATE DATABASE "{name}"')
        log.info("created database %s", name)
        return True


def apply_schema(url: str | None = None) -> int:
    """Apply the ladder under an advisory lock. Returns the resulting version."""
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
            try:
                for stmt in BASE_DDL:
                    cur.execute(stmt)
                cur.execute("SELECT version FROM schema_version WHERE singleton")
                row = cur.fetchone()
                current = int(row["version"]) if row else 0
                for v in sorted(MIGRATIONS):
                    if current < v:
                        for stmt in MIGRATIONS[v]:
                            cur.execute(stmt)
                        current = v
                from ..vault.schema import VAULT_DDL     # idempotent, outside the ladder: same DDL on master and distribution
                for stmt in VAULT_DDL:
                    cur.execute(stmt)
                target = max(current, SCHEMA_VERSION)
                cur.execute(
                    "INSERT INTO schema_version (singleton, version) VALUES (true, %s) "
                    "ON CONFLICT (singleton) DO UPDATE SET version = EXCLUDED.version, updated_at = now()",
                    (target,),
                )
                conn.commit()
                ensure_partitions(conn)
                conn.commit()
                return target
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
                conn.commit()


def _partition_name(base: str, start: datetime, granularity: str) -> str:
    fmt = "%Y%m%d%H" if granularity == "hour" else "%Y%m%d"
    return f"{base}_{start.strftime(fmt)}"


def ensure_partition(conn: psycopg.Connection, base: str, start: datetime, granularity: str) -> None:
    start = start.astimezone(timezone.utc)
    if granularity == "hour":
        start = start.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
    else:
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    name = _partition_name(base, start, granularity)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_class WHERE relname = %s", (name,))
        if cur.fetchone():
            return
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})").format(
                sql.Identifier(name), sql.Identifier(base), sql.Literal(start), sql.Literal(end)
            )
        )


def ensure_partitions(conn: psycopg.Connection, now: datetime | None = None) -> None:
    """Partitions for the near past and future so writers never hit a missing partition."""
    now = now or datetime.now(timezone.utc)
    for h in range(-2, 4):
        ensure_partition(conn, "swap_tape", now + timedelta(hours=h), "hour")
    for d in range(-1, 3):
        ensure_partition(conn, "beat_slots", now + timedelta(days=d), "day")


def ensure_partitions_for(conn: psycopg.Connection, ts: datetime) -> None:
    """Partitions around an arbitrary (e.g. simulated) timestamp."""
    ensure_partition(conn, "swap_tape", ts, "hour")
    ensure_partition(conn, "swap_tape", ts + timedelta(hours=1), "hour")
    ensure_partition(conn, "beat_slots", ts, "day")
    ensure_partition(conn, "beat_slots", ts + timedelta(days=1), "day")


def table_names(url: str | None = None) -> list[str]:
    with connect(url) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "AND tablename NOT LIKE 'swap_tape_%%' AND tablename NOT LIKE 'beat_slots_%%' ORDER BY 1"
        ).fetchall()
    return [r["tablename"] for r in rows]
