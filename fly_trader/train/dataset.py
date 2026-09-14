"""Build the offline training dataset from the recorded pump.fun swap tape.

For every watched token with enough activity and every BEAT_S_TRAIN-second beat inside the window, the same
feature vector the live runner computes (market/features.py) plus the token's last trade price. Saved as
float16/float32 arrays so PPO episodes can be replayed without touching Postgres:
    obs   [T, M, D]  features (0 where the token has no history yet)
    mask  [T, M]     1 if the token had ≥1 swap in the last 3 h at that beat (tradable)
    price [T, M]     last trade price (NaN before the first swap)
    resq  [T, M]     quote reserve in SOL (NaN if unknown)
    age_h [T, M]     hours since graduation (NaN if unknown)
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from ..db.connection import connect
from ..market.features import D, FeatureBank, TokenMeta

log = logging.getLogger(__name__)
BEAT_S_TRAIN = 5.0


def build(out_path: Path | None = None, hours: float | None = None, min_swaps: int = 200, beat_s: float = BEAT_S_TRAIN) -> Path:
    t_start = time.time()
    conn = connect()
    span = conn.execute("SELECT min(ts) AS t0, max(ts) AS t1 FROM swap_tape").fetchone()
    t1 = span["t1"].timestamp()
    t0 = span["t0"].timestamp() if hours is None else max(span["t0"].timestamp(), t1 - hours * 3600)
    toks = conn.execute("""SELECT mint, count(*) AS n FROM swap_tape WHERE ts >= to_timestamp(%s) AND ts <= to_timestamp(%s) AND side <> 0
                           GROUP BY mint HAVING count(*) >= %s ORDER BY n DESC""", (t0, t1, min_swaps)).fetchall()
    mints = [r["mint"] for r in toks]
    meta_rows = conn.execute("SELECT t.mint, t.decimals, t.graduated_at, t.token_program, wp.program_label, wp.pool FROM tokens t "
                             "LEFT JOIN watch_pools wp ON wp.mint = t.mint WHERE t.mint = ANY(%s)", (mints,)).fetchall()
    meta = {}
    for r in meta_rows:
        meta[r["mint"]] = TokenMeta(mint=r["mint"], pool=r["pool"], program_label=r["program_label"],
                                    graduated_at=r["graduated_at"].timestamp() if r["graduated_at"] else None,
                                    token2022=(r["token_program"] or "").startswith("Tokenz"), stats=None)
    stats = conn.execute("""SELECT DISTINCT ON (mint) mint, organic_score, holder_count, liquidity_usd, top_holders_pct, dev_balance_pct,
                            is_sus, is_verified, stats FROM token_stats WHERE mint = ANY(%s) ORDER BY mint, ts DESC""", (mints,)).fetchall()
    for r in stats:
        if r["mint"] in meta:
            st = dict(r); st["stats"] = st["stats"] if isinstance(st["stats"], dict) else json.loads(st["stats"] or "{}")
            meta[r["mint"]].stats = st
    T = int((t1 - t0) // beat_s) + 1
    M = len(mints)
    log.info("dataset: %d tokens, %d beats of %.0fs (%.1f h)", M, T, beat_s, (t1 - t0) / 3600)
    obs = np.zeros((T, M, D), dtype=np.float16)
    mask = np.zeros((T, M), dtype=np.bool_)
    price = np.full((T, M), np.nan, dtype=np.float32)
    resq = np.full((T, M), np.nan, dtype=np.float32)
    age = np.full((T, M), np.nan, dtype=np.float32)
    bank = FeatureBank()
    midx = {m: i for i, m in enumerate(mints)}
    cur = conn.cursor(name="tape_stream")
    cur.itersize = 50000
    cur.execute("""SELECT ts, mint, side, amount_quote, price_sol, signer, res_quote, 9 AS quote_decimals FROM swap_tape
                   WHERE ts >= to_timestamp(%s) AND ts <= to_timestamp(%s) AND mint = ANY(%s) ORDER BY ts""", (t0, t1, mints))
    beat = 0
    t_beat = t0
    n_rows = 0
    def _snapshot(bi: int, tb: float):
        for m, i in midx.items():
            st = bank.states.get(m)
            if st is None or st.last_ts is None:
                continue
            f, mk = st.features(tb, meta.get(m) or TokenMeta(m))
            obs[bi, i] = np.asarray(f, dtype=np.float16)
            mask[bi, i] = bank.activity_sol(m, tb) > 0.0
            price[bi, i] = st.last_price
            resq[bi, i] = st.last_res_quote_sol if st.last_res_quote_sol is not None else np.nan
            g = meta.get(m).graduated_at if m in meta else None
            age[bi, i] = (tb - g) / 3600.0 if g else np.nan
    for row in cur:
        r = dict(row)
        ts = r["ts"].timestamp()
        while ts > t_beat + beat_s and beat < T - 1:
            _snapshot(beat, t_beat)
            beat += 1
            t_beat = t0 + beat * beat_s
            if beat % 500 == 0:
                log.info("beat %d/%d (%.0fs elapsed)", beat, T, time.time() - t_start)
        bank.ingest_tape_row(r)
        n_rows += 1
    while beat < T:
        _snapshot(beat, t_beat)
        beat += 1
        t_beat = t0 + beat * beat_s
    cur.close(); conn.close()
    out_path = out_path or (config.DATA_DIR / "train" / f"obs_{datetime.fromtimestamp(t0, tz=timezone.utc):%Y%m%dT%H%M}_{int(beat_s)}s.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs, mask=mask, price=price, resq=resq, age=age, mints=np.array(mints), t0=t0, beat_s=beat_s,
                        feature_names=np.array([n for n in __import__("fly_trader.market.features", fromlist=["FEATURES"]).FEATURES]))
    log.info("dataset written %s: T=%d M=%d rows=%d (%.0fs)", out_path, T, M, n_rows, time.time() - t_start)
    return out_path


class Dataset:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.obs = z["obs"]; self.mask = z["mask"]; self.price = z["price"]; self.resq = z["resq"]; self.age = z["age"]
        self.mints = [str(m) for m in z["mints"]]; self.t0 = float(z["t0"]); self.beat_s = float(z["beat_s"])
        self.T, self.M, self.D = self.obs.shape
        # feature standardisation from the whole dataset (masked)
        flat = self.obs[self.mask].astype(np.float32)
        self.mean = flat.mean(axis=0) if len(flat) else np.zeros(self.D, np.float32)
        self.std = flat.std(axis=0) + 1e-6 if len(flat) else np.ones(self.D, np.float32)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x.astype(np.float32) - self.mean) / self.std, -5, 5)
