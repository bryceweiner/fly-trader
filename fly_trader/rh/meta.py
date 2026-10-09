"""Per-launch facts for graduated Pons launches (``rh_meta``): corpus_meta's columns and definitions (train/corpus_meta.py),
so both chains feed the same model inputs; amounts in ETH.

- ttg_min: launch → graduation, minutes; dev_sol / dev_tokens / dev_share: the deployer's own first buy (the launch
  router's ``Launched``: quote in ETH at the launch's base mark, tokens, share of the 1e9 supply); mayhem: 0 (Pons has none);
- rq0: the quote the graduation seeded the pool with, in ETH (Solana: the ~79 SOL reserve at migration);
- prior_launches / prior_grads / prior_known / prior_rug_share / prior_moon_share: the deployer's earlier launches and
  graduations, and among the graduations whose 60-minute outcome was known by this graduation, the share that fell 90 %
  (rug) or doubled (moon) — corpus_meta.creator_history's definitions on Pons rows;
- own_dd60 / own_max60: the launch's own first-hour low / high after graduation (from rh_minutes, filled once 61 minutes old);
- bundle_share / dev_hold_share / dev_sold_frac / grad_hhi: corpus_meta.curve_facts on the curve trades (the launch block is
  the "create slot"; the deployer's first buy is a curve leg on Pons, so no separate initial buy), and its insiders
  (deployer, bundle wallets) → rh_insiders.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from ..db.connection import transaction
from ..train.corpus_meta import curve_facts
from . import prices

log = logging.getLogger(__name__)
SUPPLY = 1e9
TOKEN = 1e18


def _qty(raw, dec: int) -> float:
    return float(raw or 0) / 10 ** dec


def build(conn, token: str) -> dict | None:
    t = conn.execute("SELECT t.*, a.decimals AS qdec FROM rh_tokens t JOIN rh_assets a ON a.asset = t.quote_asset WHERE t.token = %s AND t.status = 'graduated'",
                     (token,)).fetchone()
    if not t or t["graduated_at"] is None:
        return None
    qdec = int(t["qdec"])
    b_launch, _ = prices.base_eth(conn, t["quote_asset"], t["launch_ts"]) if t["launch_ts"] else (None, True)
    b_grad, _ = prices.base_eth(conn, t["quote_asset"], t["graduated_at"])
    dev_tokens = _qty(t["dev_tokens_raw"], 18)
    out = {"mint": token, "graduated_at": t["graduated_at"], "create_ts": t["launch_ts"], "creator": t["creator"], "quote_asset": t["quote_asset"],
           "quote_class": t["quote_class"], "supply": SUPPLY, "pool_id": t["pool_id"], "mayhem": False,
           "ttg_min": (t["graduated_at"] - t["launch_ts"]).total_seconds() / 60.0 if t["launch_ts"] else None,
           "dev_sol": _qty(t["dev_quote_in_raw"], qdec) * b_launch if b_launch is not None and t["dev_quote_in_raw"] is not None else None,
           "dev_tokens": dev_tokens, "dev_share": dev_tokens / SUPPLY,
           "rq0": _qty(t["grad_quote_raw"], qdec) * b_grad if b_grad is not None and t["grad_quote_raw"] is not None else None}
    # the deployer's history, as corpus_meta.creator_history
    c, g = t["creator"], t["graduated_at"]
    out["prior_launches"] = int(conn.execute("SELECT count(*) AS n FROM rh_tokens WHERE creator = %s AND launch_ts < %s", (c, t["launch_ts"])).fetchone()["n"]) \
        if t["launch_ts"] else None
    r = conn.execute("""SELECT count(*) AS grads,
                               count(*) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS known,
                               avg(CASE WHEN own_dd60 <= -0.9 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS rug,
                               avg(CASE WHEN own_max60 >= 1.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS moon
                        FROM rh_meta WHERE creator = %(c)s AND graduated_at < %(g)s AND mint <> %(m)s""", {"c": c, "g": g, "m": token}).fetchone()
    out.update(prior_grads=int(r["grads"]), prior_known=int(r["known"]), prior_rug_share=float(r["rug"]) if r["rug"] is not None else None,
               prior_moon_share=float(r["moon"]) if r["moon"] is not None else None)
    # curve facts from the curve trades up to graduation
    wallets: dict[str, list] = {}
    for x in conn.execute("SELECT block, trader, side, tokens_raw FROM rh_curve_trades WHERE token = %s AND ts <= %s ORDER BY block, log_index", (token, g)).fetchall():
        w = wallets.setdefault(x["trader"], [None, 0.0, 0.0]); q = float(x["tokens_raw"] or 0) / TOKEN
        if x["side"] > 0:
            w[0] = x["block"] if w[0] is None else w[0]; w[1] += q
        else:
            w[2] += q
    if t["curve_indexed"]:
        facts, insiders = curve_facts(t["launch_block"], c, SUPPLY, 0.0, wallets)
        out.update(facts); out["curve_known"] = 1.0
        for w, kind in insiders:
            conn.execute("INSERT INTO rh_insiders (mint, wallet, kind) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", (token, w, kind))
    else:
        out.update(bundle_share=None, dev_hold_share=None, dev_sold_frac=None, grad_hhi=None, curve_known=0.0)
    return out


def outcomes(conn, token: str, graduated_at) -> tuple[float, float] | None:
    """(own_dd60, own_max60) from the first hour of rh_minutes after graduation, or None when it has no minute."""
    rows = conn.execute("SELECT close FROM rh_minutes WHERE mint = %s AND ts >= %s AND ts < %s ORDER BY ts", (token, graduated_at, graduated_at + timedelta(minutes=60))).fetchall()
    cl = [float(r["close"]) for r in rows if r["close"]]
    if not cl or cl[0] <= 0:
        return None
    return min(cl) / cl[0] - 1.0, max(cl) / cl[0] - 1.0


COLS = ("mint", "graduated_at", "create_ts", "creator", "quote_asset", "quote_class", "supply", "ttg_min", "dev_sol", "dev_tokens", "dev_share", "mayhem", "rq0",
        "pool_id", "prior_launches", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share", "bundle_share", "dev_hold_share", "dev_sold_frac",
        "grad_hhi", "curve_known")


def run_once(limit: int = 500) -> dict:
    """New graduations → rh_meta (in graduation order, so a deployer's earlier launches are there first); first-hour
    outcomes for those 61+ minutes past graduation."""
    n_new = n_out = 0
    with transaction() as conn:
        todo = conn.execute("SELECT t.token FROM rh_tokens t LEFT JOIN rh_meta m ON m.mint = t.token WHERE t.status = 'graduated' AND t.graduated_at IS NOT NULL "
                            "AND m.mint IS NULL ORDER BY t.graduated_at LIMIT %s", (limit,)).fetchall()
        for r in todo:
            row = build(conn, r["token"])
            if row is None:
                continue
            conn.execute(f"INSERT INTO rh_meta ({', '.join(COLS)}) VALUES ({', '.join(['%s'] * len(COLS))}) ON CONFLICT (mint) DO NOTHING", [row.get(k) for k in COLS])
            n_new += 1
        due = conn.execute("SELECT m.mint, m.graduated_at FROM rh_meta m WHERE m.own_dd60 IS NULL AND m.graduated_at + interval '61 minutes' <= "
                           "(SELECT max(ts) FROM rh_minutes) LIMIT %s", (limit,)).fetchall()
        for r in due:
            o = outcomes(conn, r["mint"], r["graduated_at"])
            if o is not None:
                conn.execute("UPDATE rh_meta SET own_dd60 = %s, own_max60 = %s, updated_at = now() WHERE mint = %s", (o[0], o[1], r["mint"])); n_out += 1
    return {"meta": n_new, "outcomes": n_out}


def load_features():
    """rh_meta in corpus_meta.load_features' shape (mint + FEATURE_COLS; mayhem and curve_known as 0/1 floats)."""
    import pandas as pd

    from ..train.corpus_meta import FEATURE_COLS
    with transaction() as conn:
        rows = conn.execute("SELECT mint, " + ", ".join(FEATURE_COLS) + " FROM rh_meta").fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["mint"] + FEATURE_COLS)
    df["mayhem"] = df["mayhem"].map({True: 1.0, False: 0.0})
    df["curve_known"] = df["curve_known"].astype(float)
    return df
