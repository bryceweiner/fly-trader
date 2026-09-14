"""swap_tape: the IPC table between the capture worker and the runner (plan, Postgres schema item 5).

``swap_tape`` is range-partitioned by hour on ``ts``; the capture worker creates partitions ahead of
its writes (cached here so the pg_class probe runs once per hour), the runner tails ``id > last_id``.
Partitions older than ``TAPE_HOT_HOURS`` are exported to Parquet and DETACHED, never dropped;
``purge_archived(confirm=True)`` is the only path that drops anything and it is never scheduled.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import sql

from .. import config
from ..db import schema
from ..db.connection import connect

log = logging.getLogger(__name__)

COLUMNS = ("ts", "slot", "sig", "tx_index", "pool", "mint", "side", "amount_base", "amount_quote",
           "price_sol", "signer", "res_base", "res_quote", "program_label")


@dataclass(slots=True)
class TapeRow:
    ts: datetime                      # UTC; receipt time for websocket rows, blockTime for RPC fixtures
    slot: int
    sig: str
    tx_index: int
    pool: str
    mint: str | None                  # base mint
    side: int                         # +1 buy of base with quote, -1 sell of base, 0 lp/other
    amount_base: int                  # raw base units (absolute)
    amount_quote: int                 # raw quote units, lamports for WSOL (absolute)
    price_sol: float | None           # quote per base in SOL; None unless the quote is WSOL
    signer: str | None
    res_base: int | None              # vault post balances
    res_quote: int | None
    program_label: str | None
    price_quote: float | None = None  # quote per base in the pool's own quote units (any quote)
    quote_mint: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def db_values(self) -> tuple:
        return tuple(getattr(self, c) for c in COLUMNS)


TAPE_FIELDS = tuple(f.name for f in fields(TapeRow))

# ---- partition cache -------------------------------------------------------------------------
_known_hours: set[str] = set()


def _hour_key(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%d%H")


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def ensure_partition_cached(conn: psycopg.Connection, ts: datetime, *, cache: bool = True) -> bool:
    """Create the hourly swap_tape partition for ``ts`` (and the next hour) unless already seen.

    ``cache=False`` still creates the partition but does not remember it — for callers that may roll
    the transaction back (tests), so a vanished partition is never assumed present.
    """
    key = _hour_key(ts)
    if key in _known_hours:
        return False
    schema.ensure_partitions_for(conn, _as_utc(ts))
    if cache:
        _known_hours.add(key)
    return True


def forget_partitions() -> None:
    _known_hours.clear()


# ---- writes ----------------------------------------------------------------------------------
def insert_rows(conn: psycopg.Connection, rows: list[TapeRow], *, commit: bool = True) -> int:
    """COPY ``rows`` into swap_tape. Creates any missing hourly partition first. Returns rows written."""
    if not rows:
        return 0
    for r in rows:
        r.ts = _as_utc(r.ts)
        ensure_partition_cached(conn, r.ts, cache=commit)
    try:
        with conn.cursor() as cur:
            with cur.copy(sql.SQL("COPY swap_tape ({}) FROM STDIN").format(
                    sql.SQL(", ").join(sql.Identifier(c) for c in COLUMNS))) as copy:
                for r in rows:
                    copy.write_row(r.db_values())
        if commit:
            conn.commit()
    except Exception:
        forget_partitions()  # a partition may have been detached under us; re-probe next time
        if commit:
            conn.rollback()
        raise
    return len(rows)


# ---- reads -----------------------------------------------------------------------------------
def _rows_as_dicts(cur) -> list[dict]:
    out = cur.fetchall()
    if out and not isinstance(out[0], dict):
        names = [d.name for d in cur.description]
        out = [dict(zip(names, r)) for r in out]
    return out


TAIL_SQL = (
    "SELECT t.id, " + ", ".join(f"t.{c}" for c in COLUMNS) + ", "
    "w.quote_mint, w.quote_decimals, w.base_decimals, "
    "CASE WHEN t.amount_base > 0 AND t.amount_quote > 0 AND w.quote_decimals IS NOT NULL AND w.base_decimals IS NOT NULL "
    "THEN (t.amount_quote::double precision / power(10, w.quote_decimals)) "
    "/ (t.amount_base::double precision / power(10, w.base_decimals)) END AS price_quote "
    "FROM (SELECT * FROM swap_tape WHERE id > %s ORDER BY id LIMIT %s) t "
    "LEFT JOIN watch_pools w ON w.pool = t.pool ORDER BY t.id"
)


def tail(conn: psycopg.Connection, last_id: int, limit: int = 50000) -> list[dict]:
    """Rows with ``id > last_id`` in id order (the runner's per-beat catch-up read).

    Each dict carries the swap_tape columns plus ``quote_mint``, ``quote_decimals``, ``base_decimals``
    and ``price_quote`` (quote units per base token) from ``watch_pools``; the latter are NULL for a
    pool whose vaults are not learned yet, which cannot have tape rows anyway.
    """
    with conn.cursor() as cur:
        cur.execute(TAIL_SQL, (int(last_id), int(limit)))
        return _rows_as_dicts(cur)


def latest_id(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(id), 0) AS m FROM swap_tape")
        row = cur.fetchone()
    return int(row["m"] if isinstance(row, dict) else row[0])


def last_swap_ts(conn: psycopg.Connection) -> datetime | None:
    """Timestamp of the newest row (walks the (id, ts) primary key, no full scan)."""
    with conn.cursor() as cur:
        cur.execute("SELECT ts FROM swap_tape ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        return None
    return row["ts"] if isinstance(row, dict) else row[0]


# ---- archive ---------------------------------------------------------------------------------
_PARTITIONED = {"swap_tape": "hour", "beat_slots": "day"}


def _partition_start(name: str, base: str, granularity: str) -> datetime | None:
    suffix = name[len(base) + 1:]
    try:
        if granularity == "hour":
            return datetime.strptime(suffix, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return datetime.strptime(suffix, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def list_partitions(conn: psycopg.Connection, base: str) -> list[str]:
    """Names of the tables currently attached as partitions of ``base``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname AS name FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
            "JOIN pg_class p ON p.oid = i.inhparent WHERE p.relname = %s ORDER BY 1",
            (base,),
        )
        return [r["name"] if isinstance(r, dict) else r[0] for r in cur.fetchall()]


def list_detached(conn: psycopg.Connection, base: str) -> list[str]:
    """Tables named like partitions of ``base`` that are no longer attached to it."""
    attached = set(list_partitions(conn, base))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tablename AS name FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE %s "
            "AND tablename <> %s ORDER BY 1",
            (base + r"\_%", base),
        )
        names = [r["name"] if isinstance(r, dict) else r[0] for r in cur.fetchall()]
    return [n for n in names if n not in attached and _partition_start(n, base, _PARTITIONED[base]) is not None]


def _count(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table)))
        row = cur.fetchone()
    return int(row["n"] if isinstance(row, dict) else row[0])


def _parquet_rows(path: Path) -> int | None:
    import pyarrow.parquet as pq
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def export_table(conn: psycopg.Connection, table: str, path: Path) -> int:
    """SELECT * of ``table`` to ``path`` (Parquet, snappy, atomic rename). Returns rows exported."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table)))
        names = [d.name for d in cur.description]
        rows = cur.fetchall()
    if rows and isinstance(rows[0], dict):
        cols = {n: [r[n] for r in rows] for n in names}
    else:
        cols = {n: [r[i] for r in rows] for i, n in enumerate(names)}
    arrays = {}
    for n, vals in cols.items():
        if any(isinstance(v, Decimal) for v in vals):
            arrays[n] = pa.array([None if v is None else Decimal(v) for v in vals], type=pa.decimal128(30, 0))
        else:
            arrays[n] = pa.array(vals)
    table_ = pa.table(arrays) if arrays else pa.table({"_empty": pa.array([], pa.int8())})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table_, tmp, compression="snappy")
    tmp.replace(path)
    return len(rows)


def archive_partitions(url: str | None = None, now: datetime | None = None,
                       archive_dir: Path | None = None) -> list[dict]:
    """Export-then-DETACH partitions older than the hot window (never DROP).

    swap_tape partitions whose hour ended more than ``TAPE_HOT_HOURS`` ago and beat_slots partitions
    whose day ended more than ``PG_ARCHIVE_DAYS`` ago are written to
    ``PG_ARCHIVE_DIR/<table>/<partition>.parquet`` (skipped when a file with the same row count is
    already there) and then detached from the parent. Returns one action dict per partition.
    """
    now = now or datetime.now(timezone.utc)
    archive_dir = Path(archive_dir or config.PG_ARCHIVE_DIR)
    cutoffs = {
        "swap_tape": now - timedelta(hours=config.TAPE_HOT_HOURS),
        "beat_slots": now - timedelta(days=config.PG_ARCHIVE_DAYS),
    }
    actions: list[dict] = []
    with connect(url) as conn:
        for base, gran in _PARTITIONED.items():
            span = timedelta(hours=1) if gran == "hour" else timedelta(days=1)
            for name in list_partitions(conn, base):
                start = _partition_start(name, base, gran)
                if start is None or start + span >= cutoffs[base]:
                    continue
                path = archive_dir / base / f"{name}.parquet"
                n_db = _count(conn, name)
                n_file = _parquet_rows(path) if path.exists() else None
                if n_file is not None and n_file == n_db:
                    exported = False
                else:
                    export_table(conn, name, path)
                    exported = True
                    n_file = _parquet_rows(path)
                    if n_file != n_db:
                        raise RuntimeError(f"archive mismatch for {name}: db={n_db} parquet={n_file}")
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("ALTER TABLE {} DETACH PARTITION {}").format(
                        sql.Identifier(base), sql.Identifier(name)))
                conn.commit()
                forget_partitions()
                actions.append({"table": base, "partition": name, "rows": n_db, "path": str(path),
                                "exported": exported, "detached": True})
                log.info("archived %s (%d rows) -> %s; detached", name, n_db, path)
    for a in actions:
        print(f"{a['table']}: {a['partition']} rows={a['rows']} exported={a['exported']} detached -> {a['path']}")
    if not actions:
        print("archive-partitions: nothing older than the hot window")
    return actions


def purge_archived(confirm: bool, url: str | None = None, archive_dir: Path | None = None) -> list[dict]:
    """DROP detached partitions whose Parquet archive exists with a matching row count.

    Only acts when ``confirm`` is True; otherwise prints what it would drop. Never scheduled.
    """
    archive_dir = Path(archive_dir or config.PG_ARCHIVE_DIR)
    actions: list[dict] = []
    with connect(url) as conn:
        for base in _PARTITIONED:
            for name in list_detached(conn, base):
                path = archive_dir / base / f"{name}.parquet"
                n_db = _count(conn, name)
                n_file = _parquet_rows(path) if path.exists() else None
                ok = n_file is not None and n_file == n_db
                action = {"table": base, "partition": name, "rows": n_db, "path": str(path),
                          "archive_rows": n_file, "safe": ok, "dropped": False}
                if not ok:
                    print(f"KEEP {name}: archive missing or row count differs (db={n_db}, parquet={n_file})")
                elif confirm:
                    with conn.cursor() as cur:
                        cur.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(name)))
                    conn.commit()
                    action["dropped"] = True
                    print(f"DROPPED {name} ({n_db} rows; archive {path})")
                    log.warning("purged detached partition %s (%d rows)", name, n_db)
                else:
                    print(f"WOULD DROP {name} ({n_db} rows; archive {path}) -- rerun with --confirm")
                actions.append(action)
    if not actions:
        print("purge-archived: no detached partitions")
    return actions
