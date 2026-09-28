"""The Robinhood Chain training corpus, day by day, beside the Solana one (data/corpus/rh/...):

1. ``rh/wallet_skill.write_table(D)``: D's skill table, from wallet days ending 3 days before D (earlier days only);
2. ``aggregate(D)``: D's minutes recomputed from the indexed swaps with that table (rh/minutes.aggregate: the same bands,
   ETH conversion and wallet columns as live) → data/corpus/rh/mature/<D>.parquet, in train/mature's aggregate columns
   (the ``*_sol`` columns hold ETH on RH rows: native units + the chain input);
3. ``train/mature.build_day(D)`` with the RH directories, program label "Pons v4" (the Pons cost model), each launch's
   graduation time, the 1e9 supply and its measured hook fee → data/corpus/rh/features_mature/<D>/part.parquet;
4. ``rh/wallet_skill.build_wallet_day(D − 1)`` once D's aggregate and part exist.

A day is built once complete (the minutes cursor past its end); a part records ``fly_chain = rh`` and ``fly_rh_agg``.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..market.exit_cost import PONS_LABEL
from ..train import mature
from . import minutes, wallet_skill
from .wallet_skill import FEAT_DIR, MATURE_DIR

log = logging.getLogger(__name__)
RH_AGG_VERSION = 1
AGG_COLS = ["mint", "ts", "open", "high", "low", "close", "buy_sol", "sell_sol", "n_buys", "n_sells", "n_traders", "resq_sol", "fee_rate",
            "n_buyers", "wash_sol", "wash_buy_sol", "top_sell_sol", "insider_sell_sol", "skill_buy"]


def aggregate(d: date) -> int:
    t0 = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    skill = wallet_skill.table_for(d)
    with transaction() as conn:
        rows = minutes.aggregate(conn, t0, t0 + 86400, skill)
    recs = [{"mint": r[0], "ts": r[1], "open": r[5], "high": r[6], "low": r[7], "close": r[8], "buy_sol": r[11], "sell_sol": r[12], "n_buys": r[13],
             "n_sells": r[14], "n_traders": r[15], "resq_sol": r[16], "fee_rate": r[18], "n_buyers": r[19], "wash_sol": r[20], "wash_buy_sol": r[21],
             "top_sell_sol": r[22], "insider_sell_sol": r[23], "skill_buy": r[24]} for r in rows]
    MATURE_DIR.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("mint", pa.string()), ("ts", pa.timestamp("us", tz="UTC"))] + [(c, pa.float64()) for c in AGG_COLS[2:-1]] + [("skill_buy", pa.list_(pa.float64()))])
    t = pa.Table.from_pylist(recs, schema=schema).replace_schema_metadata({b"fly_version": str(mature.AGG_VERSION).encode(), b"fly_chain": b"rh",
                                                                          b"fly_rh_agg": str(RH_AGG_VERSION).encode(), b"fly_skill": wallet_skill.skill_version().encode()})
    tmp = MATURE_DIR / f"{d.isoformat()}.parquet.tmp"; pq.write_table(t, tmp, compression="zstd"); tmp.replace(MATURE_DIR / f"{d.isoformat()}.parquet")
    return len(recs)


def _meta_extra() -> tuple[dict, dict, dict]:
    """(graduations, supplies, TokenMeta extras) for every graduated launch."""
    k = config.RH_ETH_PER_SOL
    with transaction() as conn:
        g = conn.execute("SELECT mint, graduated_at FROM rh_meta").fetchall()
        fees = conn.execute("SELECT mint, percentile_cont(0.5) WITHIN GROUP (ORDER BY fee_rate) AS f FROM rh_minutes WHERE fee_rate IS NOT NULL GROUP BY mint").fetchall()
    grads = {r["mint"]: r["graduated_at"] for r in g}
    fee = {r["mint"]: float(r["f"]) for r in fees}
    extra = {m: {"ec_size": 0.1 * k, "pool_fee": fee.get(m, config.RH_HOOK_FEE)} for m in grads}
    return grads, {m: 1e9 for m in grads}, extra


def build_day(d: date) -> int:
    grads, supplies, extra = _meta_extra()
    return mature.build_day(d, grads=grads, supplies=supplies, src_dir=MATURE_DIR, out_dir=FEAT_DIR, program_label=PONS_LABEL, meta_extra=extra,
                            extra_md={"fly_chain": "rh", "fly_rh_agg": RH_AGG_VERSION})


def part_current(path) -> bool:
    if not mature.part_current(path):
        return False
    md = pq.read_schema(path).metadata or {}
    return md.get(b"fly_chain") == b"rh" and int(md.get(b"fly_rh_agg", b"0")) == RH_AGG_VERSION


def complete_through() -> date | None:
    """The last UTC day whose minutes are all written (the minutes cursor past its end)."""
    with transaction() as conn:
        r = conn.execute("SELECT detail FROM rh_scan WHERE name = 'rh_minutes'").fetchone()
    t = ((r["detail"] or {}) if r else {}).get("through")
    if not t:
        return None
    return datetime.fromtimestamp(int(t), timezone.utc).date() - timedelta(days=1)


def run_once(max_days: int = 3) -> dict:
    last = complete_through()
    if last is None:
        return {"days": 0}
    d = date.fromisoformat(config.RH_START_DAY); done = 0; out = {}
    while d <= last and done < max_days:
        part = FEAT_DIR / d.isoformat() / "part.parquet"; agg = MATURE_DIR / f"{d.isoformat()}.parquet"
        if not agg.exists() or not part_current(part):
            wallet_skill.write_table(d)
            n_agg = aggregate(d)
            n_part = build_day(d) if n_agg else 0
            wallet_skill.build_wallet_day(d - timedelta(days=1))
            out[d.isoformat()] = {"minutes": n_agg, "rows": n_part}; done += 1
        d += timedelta(days=1)
    return {"days": done, "built": out}
