"""The signer's own records (SQLite in its own volume, which the brain container cannot reach): swaps it signed (for
the rate and daily caps), claims it signed (one payment per claim id, whatever the brain's database says), and flags
(panic). Nothing here is trusted from the brain."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS swaps (ts REAL NOT NULL, direction TEXT NOT NULL, in_mint TEXT, out_mint TEXT, in_amount INTEGER,
                                  sol_spent INTEGER, signature TEXT);
CREATE INDEX IF NOT EXISTS swaps_ts ON swaps(ts);
CREATE TABLE IF NOT EXISTS claims (claim_id TEXT PRIMARY KEY, dest TEXT NOT NULL, lamports INTEGER NOT NULL, signature TEXT NOT NULL,
                                   last_valid_block_height INTEGER NOT NULL, ts REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS flags (key TEXT PRIMARY KEY, value TEXT, ts REAL);
"""


class Ledger:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(DDL)

    # swaps
    def swaps_since(self, since: float) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM swaps WHERE ts >= ?", (since,)).fetchall()

    def record_swap(self, direction: str, in_mint: str, out_mint: str, in_amount: int, sol_spent: int, signature: str) -> None:
        self.db.execute("INSERT INTO swaps VALUES (?,?,?,?,?,?,?)", (time.time(), direction, in_mint, out_mint, in_amount, sol_spent, signature))

    # claims
    def claim(self, claim_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()

    def record_claim(self, claim_id: str, dest: str, lamports: int, signature: str, lvbh: int) -> None:
        self.db.execute("INSERT INTO claims (claim_id, dest, lamports, signature, last_valid_block_height, ts) VALUES (?,?,?,?,?,?) "
                        "ON CONFLICT(claim_id) DO UPDATE SET signature = excluded.signature, last_valid_block_height = excluded.last_valid_block_height, "
                        "ts = excluded.ts, attempts = attempts + 1", (claim_id, dest, lamports, signature, lvbh, time.time()))

    # flags
    def flag(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM flags WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else None

    def set_flag(self, key: str, value: str | None) -> None:
        if value is None:
            self.db.execute("DELETE FROM flags WHERE key = ?", (key,))
        else:
            self.db.execute("INSERT INTO flags VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
                            (key, value, time.time()))
