"""ReplayFeed: swap parquet → simulated tape rows, in time order, at a simulated beat cadence.

Corpus A (meteora): VOC data/live_meteora/swaps (schema sig, slot, ts(ms), pool, sbin, ebin, amount_in,
amount_out, swap_for_y, fee, fee_bps, tvl, res_base, res_quote, qmint, qdec). swap_for_y=True means
base X sold for quote Y (side −1); False means quote in, base out (side +1). Only SOL-quoted pools
(qdec 9) are replayed. Token identity = pool address (base mint is not stored); base decimals are
unknown, so prices are quote-per-raw-base-unit — all downstream math is scale-free (log returns,
P&L = qty × price), so this is exact for rewards. Corpus B (capture): our own data/capture/swaps.
"""
from __future__ import annotations

import glob
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .. import config
from ..market.features import TokenMeta

log = logging.getLogger(__name__)


class ReplayFeed:
    def __init__(self, corpus: str, t_start: float | None = None, t_end: float | None = None, files: list[str] | None = None):
        self.corpus = corpus
        if files is None:
            if corpus == "meteora":
                files = sorted(glob.glob(str(config.VOC_CORPUS_DIR / "swaps" / "**" / "*.parquet"), recursive=True))
            else:
                files = sorted(glob.glob(str(config.CAPTURE_DIR / "swaps" / "**" / "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no parquet files for corpus {corpus}")
        self.files = files
        self.con = duckdb.connect()
        self.con.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet({files!r}, union_by_name=true)")
        if corpus == "meteora":
            # quote decimals were only captured mid-run; infer each pool's quote from any of its rows
            self.con.execute("CREATE TABLE pool_q AS SELECT pool, max(qdec) AS qdec FROM s WHERE qdec IS NOT NULL GROUP BY pool")
            self.sql = """SELECT s.ts/1000.0 AS t, s.pool, s.swap_for_y, s.amount_in::DOUBLE AS amount_in, s.amount_out::DOUBLE AS amount_out,
                                 s.res_base, s.res_quote, pq.qdec, s.sig FROM s JOIN pool_q pq ON pq.pool = s.pool WHERE pq.qdec = 9"""
        else:
            self.sql = """SELECT epoch(ts) AS t, pool, mint, side, amount_base::DOUBLE AS amount_base, amount_quote::DOUBLE AS amount_quote,
                                 price_sol, signer, res_quote, sig FROM s"""
        lo, hi = self.con.execute(f"SELECT min(t), max(t) FROM ({self.sql})").fetchone()
        self.t0 = float(t_start if t_start is not None else lo)
        self.t1 = float(t_end if t_end is not None else hi)
        self.cur = None
        self.buf: list[dict] = []
        self.exhausted = False
        self.meta: dict[str, TokenMeta] = {}
        self.reserves: dict[str, tuple[list[float], list[float]]] = {}
        if corpus == "meteora":
            self._load_reserves()

    def _load_reserves(self) -> None:
        """Reserve snapshots (live_meteora/reserves) fill res_quote for early swap files that lack it."""
        files = sorted(glob.glob(str(config.VOC_CORPUS_DIR / "reserves" / "**" / "*.parquet"), recursive=True))
        if not files:
            return
        try:
            rows = self.con.execute(
                f"SELECT pool, ts/1000.0 AS t, res_quote FROM read_parquet({files!r}, union_by_name=true) "
                f"WHERE res_quote IS NOT NULL AND qdec = 9 ORDER BY pool, t").fetchall()
        except Exception as e:
            log.warning("reserve snapshots not loaded: %s", e)
            return
        for pool, t, rq in rows:
            ts_list, rq_list = self.reserves.setdefault(pool, ([], []))
            ts_list.append(float(t)); rq_list.append(float(rq))
        log.info("reserve snapshots loaded for %d pools", len(self.reserves))

    def _reserve_at(self, pool: str, t: float) -> int | None:
        r = self.reserves.get(pool)
        if not r:
            return None
        from bisect import bisect_right
        ts_list, rq_list = r
        i = bisect_right(ts_list, t) - 1
        return int(rq_list[max(i, 0)])

    def time_range(self) -> tuple[float, float]:
        return self.t0, self.t1

    def build_meta(self) -> dict[str, TokenMeta]:
        if self.corpus == "meteora":
            rows = self.con.execute(f"SELECT pool, min(t) FROM ({self.sql}) WHERE t >= {self.t0} AND t <= {self.t1} GROUP BY pool").fetchall()
            self.meta = {p: TokenMeta(mint=p, pool=p, program_label="Meteora DLMM", graduated_at=float(first), token2022=False, stats=None)
                         for p, first in rows}
        else:
            rows = self.con.execute(f"SELECT mint, pool, min(t) FROM ({self.sql}) WHERE t >= {self.t0} AND t <= {self.t1} GROUP BY 1,2").fetchall()
            self.meta = {m: TokenMeta(mint=m, pool=p, program_label="Pump.fun Amm", graduated_at=float(first), token2022=False, stats=None)
                         for m, p, first in rows}
        for m in self.meta.values():
            m.decimals = 0 if self.corpus == "meteora" else 6   # type: ignore[attr-defined]
            m.tradable = True                                    # type: ignore[attr-defined]
        return self.meta

    def _open(self) -> None:
        self.cur = self.con.execute(f"SELECT * FROM ({self.sql}) WHERE t >= {self.t0 - 3 * 3600} AND t <= {self.t1} ORDER BY t")

    def _fetch(self) -> None:
        if self.cur is None:
            self._open()
        rows = self.cur.fetchmany(200_000)
        if not rows:
            self.exhausted = True
            return
        cols = [d[0] for d in self.cur.description]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            out.append(self._to_tape(d))
        self.buf.extend(x for x in out if x is not None)

    def _to_tape(self, d: dict) -> dict | None:
        if self.corpus == "meteora":
            if d["swap_for_y"]:
                side, amount_base, amount_quote = -1, d["amount_in"], d["amount_out"]
            else:
                side, amount_base, amount_quote = 1, d["amount_out"], d["amount_in"]
            if not amount_base or amount_base <= 0 or not amount_quote or amount_quote <= 0:
                return None
            rq = d.get("res_quote")
            rq = int(rq) if rq is not None else self._reserve_at(d["pool"], d["t"])
            return {"id": None, "ts": d["t"], "side": side, "amount_quote": amount_quote, "amount_base": amount_base,
                    "price_sol": (amount_quote / 1e9) / amount_base, "signer": None,
                    "res_quote": rq, "mint": d["pool"], "pool": d["pool"], "quote_decimals": 9}
        return {"id": None, "ts": d["t"], "side": int(d["side"]), "amount_quote": d["amount_quote"], "amount_base": d["amount_base"],
                "price_sol": d["price_sol"], "signer": d["signer"], "res_quote": int(d["res_quote"]) if d["res_quote"] is not None else None,
                "mint": d["mint"], "pool": d["pool"], "quote_decimals": 9}

    def rows_until(self, t: float) -> list[dict]:
        """All swaps with ts <= t not yet delivered."""
        out = []
        while True:
            while self.buf and self.buf[0]["ts"] <= t:
                out.append(self.buf.pop(0))
            if self.buf or self.exhausted:
                break
            self._fetch()
        return out
