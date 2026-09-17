"""Position sizing: how much of the bankroll a buy gets, from how certain the model is that the trade will profit.

Certainty is measured, not assumed. During training the backtest's out-of-sample trades are grouped by how far their
score cleared the buy line (``build_table``: equal-count bands of ``score − threshold``). For each band the growth-optimal
(Kelly) fraction is found on the band's actual net returns — the bet f that maximises the average of log(1 + f·r), so the
rare −100 % rugs count in full; a band with no edge gets 0. Live (``size_position``): the current score's band →
``KELLY_FRACTION`` of that fraction of the deployable bankroll (wealth at cost minus the gas reserve), capped at
``MAX_POSITION_FRACTION`` of it, ``MAX_POOL_SHARE`` of the pool's quote reserve (bigger buys pay more price impact than
the 0.1 SOL the backtest measured) and the free cash above the reserve; below ``MIN_POSITION_SOL`` the buy is skipped.
A model without a table (trained before sizing) trades the fixed ``MAX_POSITION_SOL``.
"""
from __future__ import annotations

import numpy as np

from .. import config

N_BANDS = 4
MIN_BAND_TRADES = 50
_F_GRID = np.linspace(0.0, 0.99, 100)


def kelly_fraction(r: np.ndarray) -> float:
    """Fraction of the bankroll per bet that maximises mean log growth over the empirical net returns ``r`` (0 if no edge)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0 or r.mean() <= 0:
        return 0.0
    growth = np.log1p(np.clip(_F_GRID[:, None] * r[None, :], -0.999999, None)).mean(axis=1)
    return float(_F_GRID[int(np.argmax(growth))])


def build_table(margins: np.ndarray, returns: np.ndarray, n_bands: int = N_BANDS) -> list[dict]:
    """Bands of score margin (score − the model's threshold) over out-of-sample trades, lowest first:
    [{lo, n, mean, win, kelly}] where ``lo`` is the band's lowest margin (the first band starts at 0)."""
    m = np.asarray(margins, dtype=float); r = np.asarray(returns, dtype=float)
    ok = np.isfinite(m) & np.isfinite(r); m, r = m[ok], r[ok]
    if len(r) < MIN_BAND_TRADES:
        return []
    k = max(1, min(n_bands, len(r) // MIN_BAND_TRADES))
    edges = np.quantile(m, np.linspace(0, 1, k + 1)); edges[0] = 0.0
    out = []
    for b in range(k):
        sel = (m >= edges[b]) & ((m < edges[b + 1]) if b < k - 1 else np.ones(len(m), bool))
        rb = r[sel]
        out.append({"lo": float(edges[b]), "n": int(len(rb)), "mean": float(rb.mean()) if len(rb) else None,
                    "win": float((rb > 0).mean()) if len(rb) else None, "kelly": kelly_fraction(rb)})
    return out


def band_for(table: list[dict], margin: float) -> dict | None:
    best = None
    for b in table:
        if margin >= b["lo"]:
            best = b
    return best


def size_position(score: float, threshold: float, table: list[dict] | None, bankroll_sol: float, cash_sol: float,
                  res_quote_sol: float | None) -> tuple[float, str]:
    """(size in SOL, why). Size 0 means skip the buy."""
    reserve = config.GAS_RESERVE_SOL; free = cash_sol - reserve
    if not table:
        size = config.MAX_POSITION_SOL
        return (size, f"fixed {size:g} SOL (model has no sizing table)") if free >= size else (0.0, "not enough free cash")
    b = band_for(table, score - threshold)
    if b is None or b["kelly"] <= 0:
        return 0.0, "no edge in this score band"
    deployable = max(0.0, bankroll_sol - reserve)
    size = config.KELLY_FRACTION * b["kelly"] * deployable
    caps = {"bankroll cap": config.MAX_POSITION_FRACTION * deployable, "free cash": free}
    if res_quote_sol:
        caps["pool depth"] = config.MAX_POOL_SHARE * res_quote_sol
    limit = min(caps, key=caps.get)
    why = f"band from +{b['lo']:.3f} over the line: Kelly {b['kelly']:.2f} × {config.KELLY_FRACTION:g} of {deployable:.2f} SOL"
    if caps[limit] < size:
        size = caps[limit]; why += f", capped by {limit}"
    if size < config.MIN_POSITION_SOL:
        return 0.0, why + f" → {size:.3f} SOL is below the {config.MIN_POSITION_SOL:g} SOL minimum"
    return float(size), why + f" → {size:.3f} SOL"


def simulate_bankroll(ts: np.ndarray, horizon_s, margins: np.ndarray, returns: np.ndarray, table: list[dict] | None,
                      start_sol: float | None = None, res_quote: np.ndarray | None = None, tables: list | None = None) -> dict:
    """Replay out-of-sample trades in entry order with the sizing rule (positions overlap for ``horizon_s``; cash, the
    reserve and — given ``res_quote``, each trade's pool quote reserve — the pool-depth cap are enforced, as live).
    Returns the final bankroll, its multiple and the worst drawdown."""
    start = float(start_sol or config.CAPITAL_SOL or 1.0); cash = start; open_: list[tuple[float, float, float]] = []   # (exit_ts, cost, return)
    hv = np.asarray(horizon_s, dtype=np.float64); hold = np.full(len(ts), float(hv)) if hv.ndim == 0 else hv    # per-trade holds allowed
    peak = start; mdd = 0.0; n = 0
    for i in np.argsort(ts, kind="stable"):
        t = float(ts[i])
        still = []
        for x in open_:
            if x[0] <= t:
                cash += x[1] * (1 + x[2])
            else:
                still.append(x)
        open_ = still
        bankroll = cash + sum(x[1] for x in open_)
        size, _ = size_position(float(margins[i]), 0.0, tables[i] if tables is not None else table, bankroll, cash, float(res_quote[i]) if res_quote is not None else None)
        if size > 0:
            cash -= size; open_.append((t + float(hold[i]), size, float(returns[i]))); n += 1
        w = cash + sum(x[1] for x in open_); peak = max(peak, w); mdd = max(mdd, 1 - w / peak if peak > 0 else 0.0)
    final = cash + sum(x[1] * (1 + x[2]) for x in open_)
    return {"start_sol": start, "final_sol": final, "multiple": final / start if start > 0 else None, "max_drawdown": mdd, "trades": n}
