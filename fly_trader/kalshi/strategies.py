"""The Kalshi selector's strategy stack: the literature's candidate sets as a ``train.strategies.StackSpec``, fitted,
filtered and combined by the memecoin machinery (``fit_stack`` / ``final_models`` / ``decide``) unchanged.

Rows are (market, minute, side); a "hold" is the arm — ``taker`` (IOC at the ask, label ``fwd_pess``) or ``maker``
(rest one tick inside the ask, label ``fwd_h["maker"]``, NaN when unfilled) — and every position runs to settlement
(``ds.hold_s`` per row). The out-of-sample score of a strategy is the walk-forward classifier's ``p̂(side pays) −
effective price``: the edge per dollar of payout, which is what the lines, the sizing bands and the fly's own head are
calibrated on (kalshi/fly.py). Candidate sets (plan of 2026-09-22, "Literature basis"):
- ``favorite``: the side priced ≥ 70c (Bürgi–Deng–Whelan 2026: positive post-fee returns above 70c, every horizon ≤ 10 d);
  fitted thresholds on the side's ask, closeness to close, spread and open interest;
- ``momentum``: the side whose mid rose over 6 h with taker flow behind it (Ottaviani–Sørensen 2015 underreaction;
  Reichenbach–Walther 2025 order-flow skill); thresholds on ``ret_6h``, ``taker_imb_1h``, ``log_vol_24h``;
- ``constraint``: in a mutually exclusive event, a side cheaper than its siblings imply, or a strike-ladder violation
  (Nunes 2026, Saguillo 2025); thresholds on ``−implied_gap`` and ``ladder_resid``;
- ``ev``: every eligible row.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..train import strategies as S
from ..train.scaling import RobustScaler
from .decisions import MAKER_DISCOUNT, X_COLS, maker_fee_cents
from .features import KIDX

ARMS = ("taker", "maker")
STRATEGIES = {
    "ev": {"holds": ARMS, "vars": (), "highs": (None,)},
    "favorite": {"holds": ARMS, "vars": ("side_ask", "neg_log_h_to_close", "neg_spread", "log_oi"), "highs": (None,)},
    "momentum": {"holds": ARMS, "vars": ("ret_6h", "taker_imb_1h", "log_vol_24h"), "highs": (None,)},
    "constraint": {"holds": ARMS, "vars": ("neg_implied_gap", "ladder_resid"), "highs": (None,)},
}
FAVORITE_MIN_ASK = 70.0


def _c(cols, X, name):
    return X[:, cols.index(name)]


def base_mask(name: str, X: np.ndarray, cols: list[str], high: str | None = None) -> np.ndarray:
    if name == "ev":
        return np.ones(len(X), bool)
    if name == "favorite":
        return _c(cols, X, "side_ask") >= FAVORITE_MIN_ASK
    if name == "momentum":
        return (_c(cols, X, "ret_6h") > 0) & (_c(cols, X, "taker_imb_1h") > 0)
    if name == "constraint":
        return (_c(cols, X, "mutually_exclusive") > 0) & ((_c(cols, X, "implied_gap") < 0) | (_c(cols, X, "ladder_resid") > 0))
    raise KeyError(name)


def trigger_value(name: str, var: str, X: np.ndarray, cols: list[str]) -> np.ndarray:
    if var.startswith("neg_"):
        return -_c(cols, X, var[4:])
    return _c(cols, X, var)


def in_universe(X: np.ndarray, cols: list[str]) -> np.ndarray:
    X = np.atleast_2d(X); a = X[:, cols.index("side_ask")]
    return (a >= 1.0) & (a <= 99.0)


def label(ds, H):
    if H == "taker":
        return ds.fwd_pess
    if H == "maker":
        return ds.fwd_h.get("maker")
    return None


def hold_s(ds, H):
    return ds.hold_s


def effective(X: np.ndarray, cols: list[str], H) -> np.ndarray:
    """The arm's fee-inclusive price per contract (cents): the ask for the taker, one tick inside it plus the maker fee for the maker."""
    X = np.atleast_2d(X)
    if H == "taker":
        return X[:, cols.index("eff_price")].astype(np.float64)
    rest = np.clip(np.round(X[:, cols.index("side_ask")]) - MAKER_DISCOUNT, 1.0, 99.0)
    return rest + maker_fee_cents(rest, X[:, cols.index("fee_mult")], X[:, cols.index("maker_fee")])


def oos(ds, rows: np.ndarray, y: np.ndarray, ci: np.ndarray, H, stop, label_: str) -> np.ndarray:
    """Walk-forward P(side pays) on ``rows`` (NaN elsewhere) minus the arm's effective price: the edge per dollar of payout."""
    fin = rows & np.isfinite(y)
    p = S.wf_classify(ds, fin, lambda idx: ds.X[np.ix_(idx, ci)], ds.outcome.astype(np.int8), 2, stop)[:, 1]
    return np.where(fin & np.isfinite(p), p - effective(ds.X, ds.cols, H) / 100.0, np.nan).astype(np.float32)


def fit_final(ds, rows: np.ndarray, ci: np.ndarray, H):
    """The deployed model of a strategy: a classifier of the side paying, on every candidate row."""
    scaler = RobustScaler.fit(ds.X[np.ix_(rows, ci)], seed=99)
    m = HistGradientBoostingClassifier(**S._gbm_params(), random_state=99); m.fit(scaler.transform(ds.X[np.ix_(rows, ci)]), ds.outcome[rows].astype(np.int8))
    return m, scaler


def score_final(m: dict, Xc: np.ndarray, Xfull: np.ndarray, cols: list[str], H) -> np.ndarray:
    """Live: the deployed classifier's p̂ minus the arm's effective price (the same score the walk-forward produced)."""
    pr = m["gbm"].predict_proba(m["scaler"].transform(Xc)); p = pr[:, list(m["gbm"].classes_).index(1)] if 1 in m["gbm"].classes_ else np.zeros(len(Xc))
    return p - effective(Xfull, cols, H) / 100.0


def probability(m: dict, Xc: np.ndarray) -> np.ndarray:
    pr = m["gbm"].predict_proba(m["scaler"].transform(Xc))
    return pr[:, list(m["gbm"].classes_).index(1)] if 1 in m["gbm"].classes_ else np.zeros(len(Xc))


KALSHI = S.StackSpec(name="kalshi", strategies=STRATEGIES, base_mask=base_mask, trigger_value=trigger_value, groups={}, legacy_cols=list(X_COLS),
                     ev_hold="taker", in_universe=in_universe, label=label, hold_s=hold_s, oos=oos, random_hold=lambda ds: ds.hold_s)
KALSHI.fit_final = fit_final
KALSHI.score_final = score_final
KALSHI.hold_seconds = lambda H: 10 * 365 * 86400.0          # positions run to settlement; the engine never exits on a timer
