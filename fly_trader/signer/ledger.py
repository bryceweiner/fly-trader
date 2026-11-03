"""The signer's own records (SQLite): the swaps it signed, for the rate and daily caps."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS swaps (ts REAL NOT NULL, direction TEXT NOT NULL, in_mint TEXT, out_mint TEXT, in_amount INTEGER,
                                  sol_spent INTEGER, signature TEXT);
CREATE INDEX IF NOT EXISTS swaps_ts ON swaps(ts);
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
