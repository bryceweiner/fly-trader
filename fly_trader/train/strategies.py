"""The selector's strategy stack (2026-09-15): three sub-strategies, a per-strategy win filter (meta-label), a dump
veto and hour/market gates, every parameter chosen by the walk-forward and one objective.

Objective (``Score``): on the walk-forward's selection half, among settings with ≥ ``WIN_MIN`` winning trades, profit
factor ≥ ``PF_MIN`` and ≥ ``selector.MIN_LINE_TRADES`` trades, the one with the
most total net profit (fixed size, one position per token, each position's own hold). When no setting reaches the win
rate, the most profitable one that meets the other conditions is kept and flagged (``fallback``). Only the chosen
setting of a component is ever scored on the evaluation half, where it must meet the same bars; a filter (meta-label,
veto, gates) must also beat removing the same number of trades at random (``RANDOM_SEEDS`` draws). Search ranges come
from the data: thresholds over deciles and tail percentiles (``QUANTILES``) of the variable among the selection half's
candidates, holds over ``HOLDS_MIN``.

Strategies (each: a definitional candidate set, fitted trigger thresholds, its own GBM of the net return over its own
hold, its own line):
- ``ev``: every tradable minute (today's selector), hold ``EV_HOLD``;
- ``capitulation``: a falling minute without insider selling (ret_1m < 0, insider_sell_share_1m = 0); thresholds on the
  drop size, the top seller's share and returning buyers (small-coin short-term reversal: Fičura & Colak 2023);
- ``breakout``: organic buying pressure at a new high (org_imb_5m > 0, buyers_slope > 0, at the 1 h or 3 h high);
  thresholds on organic imbalance, buyer growth and organic volume (order-flow and attention evidence, wash removed).
A per-strategy classifier of P(net return > 0), trained only on that strategy's out-of-sample candidates (nested
walk-forward; the primary score is one of its inputs), filters trades (Joubert 2022 meta-labeling). A multiclass
GBM of the loss tail gives P(return ≤ d) for dump levels d at tail quantiles of the selection half's pick returns
and vetoes buys (MemeTrans/MELT: rug risk). Hours and market-volume deciles whose selection-half profit factor is below 1
are blocked. Where several strategies fire on a token-minute, the one with the larger measured Kelly fraction (its
sizing band) or the higher score wins — whichever rule made more on the selection half.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from datetime import timedelta

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .. import config
from ..agent import sizing
from . import progress as prog
from .decisions import GROUPS, LEGACY_COLS, DecisionSet, taken_idx
from .scaling import RobustScaler

log = logging.getLogger(__name__)
HOLDS_MIN = (10, 20, 30, 45, 60, 90, 120, 180, 240)


@dataclass
class StackSpec:
    """What a trading type supplies to the stack machinery: its candidate sets, inputs, universe, labels and holds. The
    memecoin selector is ``MEMECOIN`` (the default of every function here); the Kalshi selector supplies its own
    (fly_trader/kalshi/strategies.py), where a "hold" is the arm (taker | maker) and every position is held to settlement."""
    name: str
    strategies: dict                 # name -> {"holds", "vars", "highs"}
    base_mask: object                # (name, X, cols, high) -> mask
    trigger_value: object            # (name, var, X, cols) -> values
    groups: dict                     # input groups the ev step tries, on top of legacy_cols
    legacy_cols: list
    ev_hold: object                  # the hold the input-group step fits at
    in_universe: object              # (X, cols) -> mask
    label: object                    # (ds, H) -> net return per row for hold H, or None when the set has no label for it
    hold_s: object                   # (ds, H) -> seconds: a scalar, or one per row
    oos: object                      # (ds, rows, y, col_idx, H, stop, label) -> out-of-sample score per row (NaN elsewhere)
    random_hold: object = None       # (ds) -> hold_s for the random baseline (None: ds.horizon_s)
EV_HOLD = 120
WIN_MIN, PF_MIN, SIGN_ALPHA = 0.65, 1.3, 0.05
# The bars a setting must clear, as a key: results fitted under another objective (train/wallet_skill.py's saved candidates)
# are discarded rather than compared with these. The weekly sign test is recorded but no longer required — with 9 weeks in
# each walk-forward half it demanded 8 winning weeks (89 %), a stricter bar than the 65 % winners it was meant to protect,
# and nothing in the corpus passed it (operator decision, 2026-09-16).
OBJECTIVE = f"win{WIN_MIN}-pf{PF_MIN}-trades-noweekly"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
LOSS_QUANTILES = (0.05, 0.10, 0.20, 0.30)
RANDOM_SEEDS = 20
PASSES = 2
PREDICT_CHUNK = 2_000_000       # rows scored at once out of sample: the block's own copy stays small
# The last three weeks are never fitted on: no strategy, filter, gate or input group may see them. The finished book is
# scored on them exactly once, for the record — never to decide anything, or they would stop being a holdout. At the
# book's observed rate (~48 trades a day) that is ~1,000 trades, an order of magnitude above MIN_LINE_TRADES.
HOLDOUT_DAYS = 21
CACHE_DIR = config.CORPUS_DIR / "stack_cache"


def _rss_gb() -> float:
    """The process's resident size now (ru_maxrss is a high-water mark and hides a climb)."""
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def _gbm_params():
    from .selector import GBM_PARAMS
    return GBM_PARAMS


def _min_trades() -> int:
    from .selector import MIN_LINE_TRADES
    return MIN_LINE_TRADES


# ---------------------------------------------------------------- objective
@dataclass
class Score:
    n: int = 0
    win: float | None = None
    pf: float | None = None
    total: float = 0.0
    mean: float | None = None
    weeks: int = 0
    weeks_pos: int = 0
    sign_ok: bool = False

    @property
    def base_ok(self) -> bool:
        return self.n >= _min_trades() and self.pf is not None and self.pf >= PF_MIN

    @property
    def admissible(self) -> bool:
        return self.base_ok and self.win is not None and self.win >= WIN_MIN

    def dict(self) -> dict:
        return {"n": self.n, "win": self.win, "pf": self.pf, "total": self.total, "mean": self.mean, "weeks": self.weeks, "weeks_pos": self.weeks_pos,
                "sign_ok": self.sign_ok, "admissible": self.admissible, "base_ok": self.base_ok}


def score_trades(ds: DecisionSet, tr: np.ndarray, returns: np.ndarray, day0) -> Score:
    r = np.asarray(returns[tr], dtype=np.float64)
    ok = np.isfinite(r); r = r[ok]; tr = np.asarray(tr)[ok]
    if len(r) == 0:
        return Score()
    g, l = r[r > 0].sum(), -r[r <= 0].sum()
    wk = np.array([(d - day0).days // 7 for d in ds.day[tr]])
    tot = np.bincount(wk - wk.min(), weights=r) if len(wk) else np.array([])
    has = np.bincount(wk - wk.min()) > 0
    weeks = int(has.sum()); pos = int(((tot > 0) & has).sum())
    p = float(binom.sf(pos - 1, weeks, 0.5)) if weeks else 1.0          # P(X ≥ pos) under no edge
    return Score(n=len(r), win=float((r > 0).mean()), pf=float(g / l) if l > 0 else float("inf"), total=float(r.sum()), mean=float(r.mean()),
                 weeks=weeks, weeks_pos=pos, sign_ok=p <= SIGN_ALPHA)


def score_pick(ds: DecisionSet, pick: np.ndarray, hold_s, returns: np.ndarray, day0) -> Score:
    return score_trades(ds, taken_idx(ds.ts, ds.mint, hold_s, np.flatnonzero(pick)), returns, day0)


def better(a: Score, b: Score | None) -> bool:
    """a beats b under the objective: admissible first, then the fallback conditions, then total profit."""
    if b is None:
        return a.base_ok
    ka = (a.admissible, a.base_ok, a.total); kb = (b.admissible, b.base_ok, b.total)
    return ka > kb


def passes(s: Score, fallback: bool) -> bool:
    return s.admissible or (fallback and s.base_ok)


def beats_random_removal(ds: DecisionSet, pick_before: np.ndarray, pick_after: np.ndarray, hold_s, returns, day0,
                         hold_before=None, returns_before=None) -> tuple[bool, dict]:
    """A filter must beat dropping the same number of trades at random from the unfiltered trades. Each book is scored
    with its own holds and returns: the filtered book's arrays hold nothing for a row the filter removed — a zero there
    would let a token re-enter within its own hold, and its return would count as zero profit rather than what it made."""
    hb = hold_s if hold_before is None else hold_before
    rb = returns if returns_before is None else returns_before
    t0 = taken_idx(ds.ts, ds.mint, hb, np.flatnonzero(pick_before)); t1 = taken_idx(ds.ts, ds.mint, hold_s, np.flatnonzero(pick_after))
    r0 = np.nan_to_num(np.asarray(rb[t0], dtype=np.float64)); after = float(np.nansum(returns[t1]))
    k = max(0, len(t0) - len(t1))
    if k == 0 or len(t0) == 0:
        return after >= float(r0.sum()), {"filtered_total": after, "random_total": float(r0.sum()), "removed": 0}
    rnd = [float(np.delete(r0, np.random.default_rng(s).choice(len(r0), k, replace=False)).sum()) for s in range(RANDOM_SEEDS)]
    return after > float(np.mean(rnd)), {"filtered_total": after, "random_total": float(np.mean(rnd)), "removed": k}


# ---------------------------------------------------------------- walk-forward primitives
def _cache_path(ds: DecisionSet, cols: list[str], tag: str) -> "Path":
    from .selector import DATA_VERSION
    k = repr((str(ds.days[0]), str(ds.days[-1]), int(len(ds.y)), float(ds.horizon_s), list(cols), tag,
              DATA_VERSION.get("agg"), DATA_VERSION.get("features"), DATA_VERSION.get("costs"), OBJECTIVE))
    return CACHE_DIR / f"{hashlib.sha256(k.encode()).hexdigest()[:20]}.npy"


def _cached(ds: DecisionSet, cols: list[str], tag: str, stop, compute):
    """One strategy-hold walk-forward is ~1.5 h of the fit; keep its out-of-sample scores on disk, keyed by the corpus,
    the inputs and the hold, so a fit that is interrupted resumes in minutes instead of starting again. A run that was
    stopped part-way is never written."""
    f = _cache_path(ds, cols, tag)
    if f.exists():
        try:
            v = np.load(f)
            if len(v) == len(ds.y):
                log.info("walk-forward %s: reusing cached scores", tag)
                return v
            log.warning("cached walk-forward %s has %d rows, not %d; recomputing", tag, len(v), len(ds.y))
        except (OSError, ValueError):
            log.warning("cached walk-forward %s unreadable; recomputing", tag)
    v = compute()
    if stop is not None and stop.is_set():
        return v                                                  # partial: not worth keeping
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npy"); np.save(tmp, v); tmp.replace(f)
    return v



def wf_blocks(days: list) -> list[list]:
    from .selector import BLOCK_DAYS, WARMUP_DAYS
    return [days[k:k + BLOCK_DAYS] for k in range(WARMUP_DAYS, len(days), BLOCK_DAYS)]


def fit_regressor(X: np.ndarray, y: np.ndarray, seed: int):
    scaler = RobustScaler.fit(X, seed=seed)
    m = HistGradientBoostingRegressor(**_gbm_params(), random_state=seed); m.fit(scaler.transform(X), np.clip(y, -1.0, 1.0))
    return m, scaler


def fit_classifier(X: np.ndarray, y: np.ndarray, seed: int):
    scaler = RobustScaler.fit(X, seed=seed)
    m = HistGradientBoostingClassifier(**_gbm_params(), random_state=seed); m.fit(scaler.transform(X), y)
    return m, scaler


def wf_regress(ds: DecisionSet, rows: np.ndarray, y: np.ndarray, cols: np.ndarray, stop: threading.Event | None = None, label: str = "") -> np.ndarray:
    """Out-of-sample predictions of y on ``rows`` (NaN elsewhere): each 7-day block by a model fit on rows at least a day earlier."""
    out = np.full(len(ds.y), np.nan, np.float32); fin = rows & np.isfinite(y); blocks = wf_blocks(ds.days)
    for k, blk in enumerate(blocks):
        if stop is not None and stop.is_set():
            break
        tr = np.flatnonzero(fin & (ds.day < (min(blk) - timedelta(days=1)))); te = np.flatnonzero(rows & np.isin(ds.day, blk))
        if len(tr) < 10 * _gbm_params()["min_samples_leaf"] or len(te) == 0:
            continue
        Xtr = ds.X[np.ix_(tr, cols)]                                  # this block's own copy: scaled in place, freed before the next
        sc = RobustScaler.fit(Xtr, seed=k)
        m = HistGradientBoostingRegressor(**_gbm_params(), random_state=k); m.fit(sc.transform_into(Xtr), np.clip(y[tr], -1.0, 1.0))
        del Xtr
        for i in range(0, len(te), PREDICT_CHUNK):
            ci = te[i:i + PREDICT_CHUNK]
            out[ci] = m.predict(sc.transform_into(ds.X[np.ix_(ci, cols)]))
        prog.update(f"selector: {label} walk-forward", k + 1, len(blocks), force=True)
    return out


def wf_classify(ds: DecisionSet, rows: np.ndarray, feats, y: np.ndarray, n_classes: int = 2, stop: threading.Event | None = None) -> np.ndarray:
    """Out-of-sample class probabilities [N, n_classes] on ``rows`` (NaN elsewhere); ``feats(idx)`` builds the inputs."""
    out = np.full((len(ds.y), n_classes), np.nan, np.float32); blocks = wf_blocks(ds.days)
    for k, blk in enumerate(blocks):
        if stop is not None and stop.is_set():
            break
        tr = np.flatnonzero(rows & (ds.day < (min(blk) - timedelta(days=1)))); te = np.flatnonzero(rows & np.isin(ds.day, blk))
        if len(tr) < 10 * _gbm_params()["min_samples_leaf"] or len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        Xtr = np.asarray(feats(tr), dtype=np.float32)                 # as above: one block-sized array at a time
        log.info("classifier block %d: train %d rows (%.2f GB), score %d rows", k, len(tr), Xtr.nbytes / 2**30, len(te))
        sc = RobustScaler.fit(Xtr, seed=k)
        m = HistGradientBoostingClassifier(**_gbm_params(), random_state=k); m.fit(sc.transform_into(Xtr), y[tr])
        del Xtr
        for i in range(0, len(te), PREDICT_CHUNK):
            ci = te[i:i + PREDICT_CHUNK]
            pr = m.predict_proba(sc.transform_into(np.asarray(feats(ci), dtype=np.float32)))
            full = np.zeros((len(ci), n_classes), np.float32); full[:, m.classes_.astype(int)] = pr
            out[ci] = full
    return out


def quantile_grid(v: np.ndarray, qs=QUANTILES) -> list[float]:
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    return sorted({float(x) for x in np.quantile(v, qs)}) if len(v) else []


# ---------------------------------------------------------------- strategies
def _col(ds_cols: list[str], X: np.ndarray, name: str) -> np.ndarray:
    return X[:, ds_cols.index(name)]


STRATEGIES = {
    "ev": {"holds": HOLDS_MIN, "vars": (), "highs": (None,)},        # the hold is fitted; EV_HOLD is only the default label horizon
    "capitulation": {"holds": HOLDS_MIN, "vars": ("neg_ret_1m", "top_sell_share_1m", "buyers_ret"), "highs": (None,)},
    "breakout": {"holds": HOLDS_MIN, "vars": ("org_imb_5m", "buyers_slope", "log_org_vol_15m"), "highs": ("dd_1h", "dd_3h")},
}


def base_mask(name: str, X: np.ndarray, cols: list[str], high: str | None = None) -> np.ndarray:
    """The definitional candidates of a strategy (no fitted parameter)."""
    if name == "ev":
        return np.ones(len(X), bool)
    if name == "capitulation":
        return (_col(cols, X, "ret_1m") < 0) & (_col(cols, X, "insider_sell_share_1m") == 0)
    if name == "breakout":
        return (_col(cols, X, "org_imb_5m") > 0) & (_col(cols, X, "buyers_slope") > 0) & (_col(cols, X, high or "dd_1h") >= 0)
    raise KeyError(name)


def trigger_value(name: str, var: str, X: np.ndarray, cols: list[str]) -> np.ndarray:
    return -_col(cols, X, "ret_1m") if var == "neg_ret_1m" else _col(cols, X, var)


def trigger_mask(name: str, thr: dict, X: np.ndarray, cols: list[str], spec: "StackSpec | None" = None) -> np.ndarray:
    tv = (spec or memecoin_spec()).trigger_value
    m = np.ones(len(X), bool)
    for var, t in thr.items():
        if t is not None and np.isfinite(t):
            m &= tv(name, var, X, cols) >= t
    return m


@dataclass
class StrategyFit:
    name: str
    hold_min: int
    line: float
    thr: dict
    high: str | None
    cols: list[str]
    oos: np.ndarray = field(repr=False, default=None)            # OOS primary score (chosen hold), NaN off candidates
    selection: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    trials: int = 0
    best_seen: dict = field(default_factory=dict)      # the most profitable setting tried, whether or not it cleared the bars
    meta_cut: float | None = None
    meta_p: np.ndarray = field(repr=False, default=None)
    hours: list = field(default_factory=list)
    regimes: list = field(default_factory=list)                   # blocked mkt_vol_1h decile bins [(lo, hi)]
    passed: bool = False
    reason: str = ""
    spec: object = None                                            # the StackSpec it was fitted under (None: MEMECOIN)

    @property
    def hold(self):
        """Seconds each position is held, for ``ds``-independent scalars; per-row holds come from ``spec.hold_s(ds, H)``."""
        return self.hold_min

    def pick(self, ds: DecisionSet, rows: np.ndarray, uni: np.ndarray) -> np.ndarray:
        sp = self.spec or memecoin_spec(); X, c = ds.X, ds.cols
        p = rows & uni & sp.base_mask(self.name, X, c, self.high) & trigger_mask(self.name, self.thr, X, c, sp) & (self.oos >= self.line)
        if self.meta_cut is not None and self.meta_p is not None:
            p &= self.meta_p >= self.meta_cut
        if self.hours:
            hr = ((ds.ts // 3600) % 24).astype(int); p &= ~np.isin(hr, self.hours)
        for lo, hi in self.regimes:
            mv = _col(c, X, "mkt_vol_1h"); p &= ~((mv >= lo) & (mv < hi))
        return p


def fit_strategy(ds: DecisionSet, name: str, cols: list[str], uni: np.ndarray, sel: np.ndarray, day0, stop=None, holds=None,
                 cache_tag: str = "", spec: "StackSpec | None" = None) -> StrategyFit | None:
    """Coordinate ascent over (hold, line, trigger thresholds, high window) on the selection half; each hold's OOS scores
    come from its own walk-forward (models fit on the strategy's candidates, label = net return over that hold).
    ``holds``: fit these holds instead of the strategy's whole grid — one walk-forward per hold is the expensive part, so
    the input-group comparison runs at a single hold and only the strategies themselves scan the grid. ``cache_tag``
    distinguishes fits whose inputs share their column names but hold different values (the wallet-skill candidates), which
    would otherwise collide in the walk-forward cache."""
    sp = spec or memecoin_spec(); st = sp.strategies[name]; X, c = ds.X, ds.cols; ci = np.asarray([c.index(x) for x in cols])
    oos = {}
    loose = uni & np.zeros(len(ds.y), bool)
    for high in st["highs"]:
        loose |= uni & sp.base_mask(name, X, c, high)
    for H in (holds or st["holds"]):
        y = sp.label(ds, H)
        if y is None:
            continue
        oos[H] = _cached(ds, cols, f"{sp.name}|{name}|hold{H}{cache_tag}", stop, lambda H=H, y=y: sp.oos(ds, loose, y, ci, H, stop, f"{name} {H}"))
    if not oos:
        return None
    ret = lambda H: sp.label(ds, H)
    best = None; best_p = None; trials = 0; seen = None
    thr = {v: None for v in st["vars"]}; high = st["highs"][0]; H = next(iter(oos))
    g0 = quantile_grid(oos[H][sel & loose]); line = g0[0] if g0 else float("inf")      # the loosest line until a trial passes
    grids = {v: quantile_grid(sp.trigger_value(name, v, X[sel & loose], c)) for v in st["vars"]}

    def trial(H_, line_, thr_, high_):
        nonlocal best, best_p, trials, seen
        trials += 1
        pk = sel & uni & sp.base_mask(name, X, c, high_) & trigger_mask(name, thr_, X, c, sp) & (oos[H_] >= line_)
        s = score_pick(ds, pk, sp.hold_s(ds, H_), ret(H_), day0)
        if s.n > 0 and (seen is None or s.total > seen.total):
            seen = s                    # the most profitable setting that actually traded (an empty pick scores 0 and must not win this)
        if better(s, best):
            best = s; best_p = (H_, line_, dict(thr_), high_)
    for _ in range(PASSES):
        for H_ in oos:
            for line_ in quantile_grid(oos[H_][sel & loose]):
                trial(H_, line_, thr, high)
        if best_p:
            H, line, thr, high = best_p
        for v in st["vars"]:
            for t in [None] + grids[v]:
                trial(H, line, {**thr, v: t}, high)
            if best_p:
                H, line, thr, high = best_p
        for high_ in st["highs"]:
            trial(H, line, thr, high_)
        if best_p:
            H, line, thr, high = best_p
    if best_p is None:
        return StrategyFit(name, H, float("inf"), thr, high, cols, oos=oos[H], trials=trials, best_seen=seen.dict() if seen else {},
                           reason="no setting met PF and the trade count on the selection half", spec=sp)
    H, line, thr, high = best_p
    return StrategyFit(name, H, float(line), thr, high, cols, oos=oos[H], selection=best.dict(), trials=trials, best_seen=seen.dict() if seen else {}, spec=sp)


# ---------------------------------------------------------------- meta-label, veto, gates
def fit_meta(ds: DecisionSet, s: StrategyFit, uni: np.ndarray, sel: np.ndarray, day0, stop=None) -> tuple[np.ndarray, float | None, dict]:
    sp = s.spec or memecoin_spec()
    """Nested: a classifier of P(net return > 0 over the strategy's hold) on its OOS candidates (primary score ≥ 0 and
    the fitted trigger), walk-forward, the primary score among its inputs; the cut-off (deciles among the selection
    half's candidates) jointly with the line."""
    X, c = ds.X, ds.cols; ci = np.asarray([c.index(x) for x in s.cols]); y_r = sp.label(ds, s.hold_min)
    rows = uni & sp.base_mask(s.name, X, c, s.high) & trigger_mask(s.name, s.thr, X, c, sp) & np.isfinite(s.oos) & (s.oos >= 0) & np.isfinite(y_r)
    log.info("meta-label %s: %d candidate rows of %d, %d inputs, hold %d min", s.name, int(rows.sum()), len(ds.y), len(ci) + 1, s.hold_min)
    feats = lambda idx: np.c_[X[np.ix_(idx, ci)], s.oos[idx]]
    p = wf_classify(ds, rows, feats, (y_r > 0).astype(np.int8), 2, stop)[:, 1]
    log.info("meta-label %s: classifiers done, %.1f GB resident; choosing the line and cut-off", s.name, _rss_gb())
    best = None; best_p = None; trials = 0; seen = None
    lines = [s.line] + quantile_grid(s.oos[sel & rows]); cuts = quantile_grid(p[sel & rows])
    log.info("meta-label %s: %d lines x %d cut-offs to try", s.name, len(lines), len(cuts))
    for line in lines:
        for cut in cuts:
            trials += 1
            if trials % 25 == 0:
                log.info("meta-label %s: trial %d/%d, %.1f GB resident", s.name, trials, len(lines) * len(cuts), _rss_gb())
            pk = sel & rows & (s.oos >= line) & (p >= cut)
            sc = score_pick(ds, pk, sp.hold_s(ds, s.hold_min), y_r, day0)
            if sc.n > 0 and (seen is None or sc.total > seen.total):
                seen = sc                                     # the most profitable cut-off that actually traded
            if better(sc, best):
                best = sc; best_p = (line, cut)
    return p, (best_p[1] if best_p else None), {"line": best_p[0] if best_p else None, "selection": best.dict() if best else None, "trials": trials,
                                                "best_seen": seen.dict() if seen else None}


def system_pick(ds: DecisionSet, fits: list[StrategyFit], rows: np.ndarray, uni: np.ndarray, combine: str, tables: dict | None = None):
    """(pick, hold_s, returns, strategy index) of the combined book: where several strategies fire, ``combine`` decides."""
    N = len(ds.y); pick = np.zeros(N, bool); hold = np.full(N, np.nan); ret = np.full(N, np.nan, np.float32); who = np.full(N, -1)
    key = np.full(N, -np.inf)
    for k, f in enumerate(fits):
        sp = f.spec or memecoin_spec()
        pk = f.pick(ds, rows, uni)
        if combine == "kelly" and tables and tables.get(f.name):
            v = np.array([((sizing.band_for(tables[f.name], m) or {}).get("kelly") or 0.0) for m in (f.oos[pk] - f.line)])
            kv = np.full(N, -np.inf); kv[pk] = v + 1e-9 * f.oos[pk]
        else:
            kv = np.where(pk, f.oos, -np.inf)
        take = pk & (kv > key)
        key = np.where(take, kv, key); who = np.where(take, k, who); pick |= pk
        hold = np.where(take, sp.hold_s(ds, f.hold_min), hold); ret = np.where(take, sp.label(ds, f.hold_min), ret)
    return pick, np.where(np.isnan(hold), 0.0, hold), ret, who


def fit_veto(ds: DecisionSet, pick: np.ndarray, hold: np.ndarray, ret: np.ndarray, sel: np.ndarray, day0, cols: list[str], stop=None):
    """Multiclass GBM of the loss tail on the book's candidate trades (nested walk-forward): P(return ≤ d) for dump levels d
    at the selection half's loss-tail quantiles; the level and the cut-off are chosen jointly by the objective."""
    base_r = ret[sel & pick & np.isfinite(ret)]
    if len(base_r) < 10 * _gbm_params()["min_samples_leaf"]:
        return None
    levels = sorted({float(x) for x in np.quantile(base_r, LOSS_QUANTILES)})
    y = np.searchsorted(levels, np.nan_to_num(ret, nan=0.0), side="left").astype(np.int8)      # class k: levels[k-1] < r ≤ levels[k]
    ci = np.asarray([ds.cols.index(x) for x in cols])
    rows = pick & np.isfinite(ret)
    probs = wf_classify(ds, rows, lambda idx: ds.X[np.ix_(idx, ci)], y, len(levels) + 1, stop)
    cum = np.cumsum(probs, axis=1)                                   # P(r ≤ levels[k])
    best = None; best_p = None; trials = 0; seen = None
    for k, d in enumerate(levels):
        pd_ = cum[:, k]
        for v in quantile_grid(pd_[sel & rows]):
            trials += 1
            pk = sel & rows & ~(pd_ >= v)
            sc = score_pick(ds, pk, hold, ret, day0)
            if sc.n > 0 and (seen is None or sc.total > seen.total):
                seen = sc                                     # the most profitable level and cut-off that actually traded
            if better(sc, best):
                best = sc; best_p = (k, d, v)
    if best_p is None:
        return {"cut": None, "levels": levels, "best_seen": seen.dict() if seen else None, "trials": trials}
    return {"level_index": best_p[0], "dump": best_p[1], "cut": best_p[2], "levels": levels, "p": cum[:, best_p[0]], "selection": best.dict(), "trials": trials,
            "best_seen": seen.dict() if seen else None}


# ---------------------------------------------------------------- the whole stack
@dataclass
class Stack:
    groups: list
    cols: list
    fits: list
    combine: str
    veto: dict | None
    components: list
    selection: dict
    evaluation: dict
    deployable: bool
    fallback: bool
    reason: str
    tables: dict = field(default_factory=dict)
    holdout_days: list = field(default_factory=list)    # withheld from every fit; scored once afterwards by score_holdout


def _comp(name, passed, reason, sel=None, ev=None, params=None, trials=0, best=None) -> dict:
    """One component's verdict. Logged as it is decided: the stack spends hours per component, and a run that dies must
    leave behind what it had already settled."""
    c = {"name": name, "passed": bool(passed), "reason": reason, "selection": sel, "evaluation": ev, "params": params or {}, "trials": int(trials),
         "best_seen": best}
    log.info("component %s: %s (%s) | selection %s | evaluation %s | best reached %s | %d settings tried",
             name, "kept" if c["passed"] else "dropped", reason, sel, ev, best, c["trials"])
    return c


def fit_stack(ds: DecisionSet, stop: threading.Event | None = None, holdout_days_n: int | None = None, spec: "StackSpec | None" = None) -> Stack:
    """Every component in order, each on top of the accepted ones; see the module docstring. ``holdout_days_n``: how many
    of the last days to withhold from every fit (default ``HOLDOUT_DAYS``). A caller whose ``ds`` is already restricted to
    days before some later date -- the fly's bootstrap teacher (train/fly_selector.py) -- passes 0: its out-of-sample
    proof is the replay that follows, and withholding again would only cost it its most recent three weeks."""
    from .selector import deploy_decision
    sp = spec or memecoin_spec()
    uni = sp.in_universe(ds.X, ds.cols)
    n_hold = HOLDOUT_DAYS if holdout_days_n is None else int(holdout_days_n)
    holdout_days = list(ds.days[-n_hold:]) if n_hold and len(ds.days) > n_hold + 4 else []
    fit_days = [d for d in ds.days if d not in set(holdout_days)]
    tdays = sorted({d for blk in wf_blocks(fit_days) for d in blk})
    if len(tdays) < 4:
        raise RuntimeError("too few walk-forward days for the strategy stack")
    half = tdays[len(tdays) // 2]; day0 = tdays[0]
    tested = np.isin(ds.day, tdays); sel = tested & (ds.day < half); ev = tested & (ds.day >= half)
    log.info("fitting on %d days (%s..%s); holdout of %d days (%s..%s) is never fitted on", len(tdays), tdays[0], tdays[-1],
             len(holdout_days), holdout_days[0] if holdout_days else "-", holdout_days[-1] if holdout_days else "-")
    comps = []
    # 1. input groups, judged on the ev strategy: all groups at once; if that is rejected, each group on its own, accumulating
    best = fit_strategy(ds, "ev", sp.legacy_cols, uni, sel, day0, stop, holds=(sp.ev_hold,), spec=sp)
    groups, cols = [], list(sp.legacy_cols)
    trial_sets = ([list(sp.groups)] if sp.groups else []) + [[g] for g in sp.groups]

    def _score_of(f):
        return Score(**{k: v for k, v in f.selection.items() if k in Score.__dataclass_fields__}) if f is not None and f.selection else None

    for gs in trial_sets:
        if stop is not None and stop.is_set():
            break
        if gs != list(sp.groups) and set(groups) == set(sp.groups):
            break
        add = [g for g in gs if g not in groups]
        if not add:
            continue
        gi = [ds.cols.index(x) for x in (x for g in add for x in sp.groups[g]) if x in ds.cols]
        sample = np.flatnonzero(tested)[:200_000]
        if gi and len(sample) and not (np.ptp(ds.X[np.ix_(sample, gi)], axis=0) > 0).any():
            comps.append(_comp(f"inputs: {'+'.join(add)}", False, "these inputs are empty in this corpus (every value identical), so there is nothing to learn from",
                               params={"groups": add})); continue
        cand_cols = cols + [x for g in add for x in sp.groups[g]]
        f = fit_strategy(ds, "ev", cand_cols, uni, sel, day0, stop, holds=(sp.ev_hold,), spec=sp)
        if f is None:
            continue
        sel_s = _score_of(f) or Score()
        if better(sel_s, _score_of(best)):
            evs = score_pick(ds, f.pick(ds, ev, uni), sp.hold_s(ds, f.hold_min), sp.label(ds, f.hold_min), day0)
            ok = passes(evs, fallback=not sel_s.admissible)
            comps.append(_comp(f"inputs: {'+'.join(add)}", ok, "improves the ev strategy on the selection half" + ("" if ok else ", fails on the evaluation half"),
                               f.selection, evs.dict(), {"groups": add}, f.trials, best=f.best_seen))
            if ok:
                groups, cols, best = groups + add, cand_cols, f
        else:
            comps.append(_comp(f"inputs: {'+'.join(add)}", False, "does not improve the ev strategy on the selection half", f.selection, None, {"groups": add}, f.trials,
                               best=f.best_seen))
    # 2. strategies
    fits = []
    for name in sp.strategies:
        log.info("fitting strategy %s over holds %s", name, sp.strategies[name]["holds"])
        f = fit_strategy(ds, name, cols, uni, sel, day0, stop, spec=sp)         # the full hold grid: the group step above fitted one hold only
        if f is None or not f.selection:
            comps.append(_comp(f"strategy: {name}", False, (f.reason if f else "no candidates") or "no admissible setting", trials=f.trials if f else 0,
                               best=f.best_seen if f else None)); continue
        y_r = sp.label(ds, f.hold_min)
        evs = score_pick(ds, f.pick(ds, ev, uni), sp.hold_s(ds, f.hold_min), y_r, day0)
        sel_adm = f.selection.get("admissible")
        f.evaluation = evs.dict(); f.passed = passes(evs, fallback=not sel_adm)
        f.reason = ("meets ≥65 % winners and PF ≥1.3 on both halves" if sel_adm and evs.admissible else
                    "profit fallback: below 65 % winners, PF ≥1.3 on both halves" if f.passed else "fails on the evaluation half")
        comps.append(_comp(f"strategy: {name}", f.passed, f.reason, f.selection, f.evaluation, {"hold_min": f.hold_min, "line": f.line, "thr": f.thr, "high": f.high}, f.trials,
                           best=f.best_seen))
        if f.passed:
            fits.append(f)
    if not fits:
        return Stack(groups, cols, [], "score", None, comps, {}, {}, False, False, "no strategy passed its walk-forward test")
    # sizing tables from each strategy's out-of-sample trades (selection half: used to choose the combination rule)
    tables = {}
    for f in fits:
        y_r = sp.label(ds, f.hold_min); t = taken_idx(ds.ts, ds.mint, sp.hold_s(ds, f.hold_min), np.flatnonzero(f.pick(ds, sel, uni)))
        tables[f.name] = sizing.build_table(f.oos[t] - f.line, y_r[t])
    # 3. combination rule
    best_c = None; combine = "score"
    for rule in ("kelly", "score"):
        pk, hd, rt, _ = system_pick(ds, fits, sel, uni, rule, tables)
        s = score_pick(ds, pk, hd, rt, day0)
        if better(s, best_c):
            best_c = s; combine = rule
    comps.append(_comp("combination", True, f"'{combine}' made more on the selection half", best_c.dict() if best_c else None, None, {"rule": combine}))
    # 4. meta-label per strategy
    for f in fits:
        p, cut, info = fit_meta(ds, f, uni, sel, day0, stop)
        if cut is None:
            comps.append(_comp(f"meta-label: {f.name}", False, "no cut-off cleared the bars on the selection half", trials=info["trials"], best=info.get("best_seen"))); continue
        log.info("meta-label %s: scoring the book before the filter (%.1f GB)", f.name, _rss_gb())
        before_s = score_pick(ds, *system_pick(ds, fits, sel, uni, combine, tables)[:3], day0)
        old = (f.meta_p, f.meta_cut, f.line); f.meta_p, f.meta_cut = p, cut; f.line = info["line"] if info["line"] is not None else f.line
        log.info("meta-label %s: scoring the book after the filter (%.1f GB)", f.name, _rss_gb())
        after_sel = score_pick(ds, *system_pick(ds, fits, sel, uni, combine, tables)[:3], day0)
        log.info("meta-label %s: evaluation half, unfiltered book (%.1f GB)", f.name, _rss_gb())
        pk0, hd0, rt0, _ = system_pick(ds, [*(g for g in fits if g is not f), _without_meta(f, old)], ev, uni, combine, tables)
        log.info("meta-label %s: evaluation half, filtered book (%.1f GB)", f.name, _rss_gb())
        pk1, hd1, rt1, _ = system_pick(ds, fits, ev, uni, combine, tables)
        log.info("meta-label %s: scoring the evaluation half (%.1f GB)", f.name, _rss_gb())
        ev_s = score_pick(ds, pk1, hd1, rt1, day0)
        log.info("meta-label %s: random-removal test (%.1f GB)", f.name, _rss_gb())
        rnd_ok, rnd = beats_random_removal(ds, pk0, pk1, hd1, rt1, day0, hold_before=hd0, returns_before=rt0)
        ok = better(after_sel, before_s) and passes(ev_s, fallback=not after_sel.admissible) and rnd_ok
        comps.append(_comp(f"meta-label: {f.name}", ok, "improves the book on the selection half, passes on the evaluation half and beats random removal" if ok
                           else "rejected (selection, evaluation or random-removal test)", after_sel.dict(), {**ev_s.dict(), "random_removal": rnd}, {"cut": cut, "line": f.line},
                           info["trials"], best=info.get("best_seen")))
        if not ok:
            f.meta_p, f.meta_cut, f.line = old
    # 5. veto
    veto = None
    pk, hd, rt, who = system_pick(ds, fits, tested, uni, combine, tables)
    v = fit_veto(ds, pk, hd, rt, sel, day0, cols, stop)
    if v is not None and v.get("cut") is not None:
        before_sel = score_pick(ds, pk & sel, hd, rt, day0); after_sel = score_pick(ds, pk & sel & ~(v["p"] >= v["cut"]), hd, rt, day0)
        ev_before = pk & ev; ev_after = ev_before & ~(v["p"] >= v["cut"])
        ev_s = score_pick(ds, ev_after, hd, rt, day0); rnd_ok, rnd = beats_random_removal(ds, ev_before, ev_after, hd, rt, day0)
        ok = better(after_sel, before_sel) and passes(ev_s, fallback=not after_sel.admissible) and rnd_ok
        comps.append(_comp("dump veto", ok, "improves the book on the selection half, passes on the evaluation half and beats random removal" if ok else "rejected",
                           after_sel.dict(), {**ev_s.dict(), "random_removal": rnd}, {"dump": v["dump"], "cut": v["cut"]}, v["trials"], best=v.get("best_seen")))
        if ok:
            veto = v
    elif v is None:
        comps.append(_comp("dump veto", False, "too few candidate trades to fit it"))
    else:
        comps.append(_comp("dump veto", False, "no dump level and cut-off cleared the bars on the selection half", trials=v["trials"], best=v.get("best_seen")))
    # (hour and market-volume gates were removed on 2026-09-20: fitted on the selection half they blocked 8 of 24 hours
    # and 2 volume deciles, cutting the book from ~27 trades a day to under 1 on days nothing was fitted on. Strategies
    # still carry empty ``hours``/``regimes`` so models saved with gates keep working.)
    keep = ~(veto["p"] >= veto["cut"]) if veto else np.ones(len(ds.y), bool)
    # the whole book
    pk, hd, rt, _ = system_pick(ds, fits, tested, uni, combine, tables); pk &= keep
    s_sel = score_pick(ds, pk & sel, hd, rt, day0); s_ev = score_pick(ds, pk & ev, hd, rt, day0)
    tr_ev = taken_idx(ds.ts, ds.mint, hd, np.flatnonzero(pk & ev))
    from .decisions import random_trades, summarize
    rnd = summarize(random_trades(ds, ev & uni, max(1, len(tr_ev)), hold_s=sp.random_hold(ds) if sp.random_hold else None,
                                  returns=sp.label(ds, sp.ev_hold) if sp.random_hold else None))
    dep, why = deploy_decision({"n": s_ev.n, "mean": s_ev.mean}, rnd)
    strict = s_sel.admissible and s_ev.admissible
    fb = (not strict) and s_sel.base_ok and s_ev.base_ok
    deployable = dep and (strict or fb)
    reason = (why + ("; meets ≥65 % winners and PF ≥1.3 on both halves" if strict else "; profit fallback (below 65 % winners)" if fb else "; fails the stack's bars"))
    return Stack(groups, cols, fits, combine, veto, comps, s_sel.dict(), {**s_ev.dict(), "random_mean": rnd.get("mean")}, deployable, fb and deployable, reason, tables,
                 holdout_days=holdout_days)


def _without_meta(f: StrategyFit, old) -> StrategyFit:
    import copy
    g = copy.copy(f); g.meta_p, g.meta_cut, g.line = old
    return g


def score_holdout(ds: DecisionSet, models: dict, days: list, spec: "StackSpec | None" = None) -> dict:
    """The deployed decision (``decide``: strategies, win filters, dump veto and gates as they would trade) scored on the
    days ``fit_stack`` withheld from every fit. It decides nothing — recorded and reported only. Feeding it back into any
    choice would make it a second evaluation half rather than a holdout."""
    from .decisions import random_trades, summarize
    sp = spec or memecoin_spec()
    if not days or not (models.get("strategies") or {}):
        return {}
    eligible = np.isin(ds.day, days) & sp.in_universe(ds.X, ds.cols)
    rows = np.flatnonzero(eligible)
    out = {"days": [str(days[0]), str(days[-1])], "rows": int(len(rows))}
    if not len(rows):
        return out
    d = decide(models, ds.X[rows], ds.cols, ds.ts[rows], spec=sp)
    take = d["allow"]
    if not take.any():
        return {**out, "n": 0}
    idx = rows[take]
    hold_full = np.zeros(len(ds.y)); ret_full = np.full(len(ds.y), np.nan, np.float32)
    for name, m in models["strategies"].items():
        r = idx[d["strategy"][take] == name]
        if len(r):
            hs = sp.hold_s(ds, m["hold_min"]); hold_full[r] = hs[r] if np.ndim(hs) else hs
            ret_full[r] = sp.label(ds, m["hold_min"])[r]
    tr = taken_idx(ds.ts, ds.mint, hold_full, idx)
    s = score_trades(ds, tr, ret_full, days[0])
    rnd = summarize(random_trades(ds, eligible, max(1, len(tr)), hold_s=sp.random_hold(ds) if sp.random_hold else None,
                                  returns=sp.label(ds, sp.ev_hold) if sp.random_hold else None))
    return {**out, **s.dict(), "random_mean": rnd.get("mean")}


# ---------------------------------------------------------------- deployment: final models and live decisions
def final_models(ds: DecisionSet, stack: Stack, exclude_days: list | None = None, spec: "StackSpec | None" = None) -> dict:
    """Every accepted component refit on all days (the walk-forward only chose their settings): per strategy its GBM over
    its candidates and hold, its win classifier on all its out-of-sample candidates; the veto on all candidate trades.
    ``exclude_days``: leave those days out of every fit, so the result can be scored on days it has never seen. Deployment
    refits on everything; only the holdout measurement excludes, or it would be scoring itself in sample."""
    sp = spec or memecoin_spec()
    uni = sp.in_universe(ds.X, ds.cols); out = {"strategies": {}, "veto": None, "combine": stack.combine, "spec": sp.name}
    keep = ~np.isin(ds.day, exclude_days) if exclude_days else np.ones(len(ds.y), bool)
    for f in stack.fits:
        ci = np.asarray([ds.cols.index(x) for x in f.cols]); y = sp.label(ds, f.hold_min)
        loose = np.zeros(len(ds.y), bool)
        for high in sp.strategies[f.name]["highs"]:
            loose |= uni & sp.base_mask(f.name, ds.X, ds.cols, high)
        rows = np.flatnonzero(loose & np.isfinite(y) & keep)
        gbm, scaler = sp.fit_final(ds, rows, ci, f.hold_min) if getattr(sp, "fit_final", None) else fit_regressor(ds.X[np.ix_(rows, ci)], y[rows], seed=99)
        meta = None
        if f.meta_cut is not None:
            mr = np.flatnonzero(uni & sp.base_mask(f.name, ds.X, ds.cols, f.high) & trigger_mask(f.name, f.thr, ds.X, ds.cols, sp) & np.isfinite(f.oos) & (f.oos >= 0) & np.isfinite(y) & keep)
            mc, ms = fit_classifier(np.c_[ds.X[np.ix_(mr, ci)], f.oos[mr]], (y[mr] > 0).astype(np.int8), seed=99)
            meta = {"model": mc, "scaler": ms, "cut": f.meta_cut}
        out["strategies"][f.name] = {"gbm": gbm, "scaler": scaler, "cols": list(f.cols), "line": f.line, "hold_min": f.hold_min, "thr": dict(f.thr), "high": f.high,
                                     "meta": meta, "hours": list(f.hours), "regimes": list(f.regimes), "sizing": stack.tables.get(f.name) or [],
                                     "selection": f.selection, "evaluation": f.evaluation}
    if stack.veto is not None:
        pk, hd, rt, _ = system_pick(ds, stack.fits, np.ones(len(ds.y), bool), uni, stack.combine, stack.tables)
        rows = np.flatnonzero(pk & np.isfinite(rt) & keep); ci = np.asarray([ds.cols.index(x) for x in stack.cols])
        y = np.searchsorted(stack.veto["levels"], rt[rows], side="left").astype(np.int8)
        vm, vs = fit_classifier(ds.X[np.ix_(rows, ci)], y, seed=99)
        out["veto"] = {"model": vm, "scaler": vs, "cols": list(stack.cols), "k": stack.veto["level_index"], "cut": stack.veto["cut"], "dump": stack.veto["dump"]}
    return out


def decide_all(models: dict, X: np.ndarray, cols: list[str], t_start, spec: "StackSpec | None" = None) -> dict:
    """Every accepted strategy's own view of each row: ``trig`` (its fitted trigger fires), ``score`` (its predicted net
    return over its hold, −inf off the trigger), ``allow`` (at/above its line and not blocked by its win filter, its
    gates or the dump veto), ``why`` (the blocking filter), plus its line, hold and sizing table. ``t_start``: the minute
    start (a scalar, or one per row) — the hour gates use it."""
    sp = spec or memecoin_spec()
    X = np.atleast_2d(X); n = len(X)
    hour = ((np.broadcast_to(np.asarray(t_start, dtype=np.float64), (n,)) // 3600) % 24).astype(int)
    out = {}
    for name, m in (models.get("strategies") or {}).items():
        ci = np.asarray([cols.index(x) for x in m["cols"]])
        trig = sp.base_mask(name, X, cols, m["high"]) & trigger_mask(name, m["thr"], X, cols, sp)
        sc = np.full(n, -np.inf); why = np.full(n, "", dtype=object)
        if trig.any():
            ti = np.flatnonzero(trig)
            sc[ti] = sp.score_final(m, X[np.ix_(ti, ci)], X[ti], cols, m["hold_min"]) if getattr(sp, "score_final", None) else m["gbm"].predict(m["scaler"].transform(X[np.ix_(ti, ci)]))
        ok = trig & (sc >= m["line"])
        if m.get("meta") is not None and ok.any():
            idx = np.flatnonzero(ok)
            p = m["meta"]["model"].predict_proba(m["meta"]["scaler"].transform(np.c_[X[np.ix_(idx, ci)], sc[idx]]))[:, list(m["meta"]["model"].classes_).index(1)]
            low = np.zeros(n, bool); low[idx] = p < m["meta"]["cut"]; why[low] = "win filter (meta-label)"
        gated = ok & np.isin(hour, m.get("hours", [])) & (why == "")
        why[gated] = [f"hour {h} UTC gated" for h in hour[gated]]
        if m.get("regimes") and "mkt_vol_1h" in cols:
            mv = X[:, cols.index("mkt_vol_1h")]
            for lo, hi in m["regimes"]:
                why[ok & (mv >= lo) & (mv < hi) & (why == "")] = "market-volume regime gated"
        out[name] = {"trig": trig, "score": sc, "allow": ok & (why == ""), "why": why, "line": m["line"], "hold_min": m["hold_min"], "sizing": m["sizing"]}
    v = models.get("veto")
    if v is not None and out:
        any_ok = np.zeros(n, bool)
        for d in out.values():
            any_ok |= d["allow"]
        if any_ok.any():
            idx = np.flatnonzero(any_ok); ci = np.asarray([cols.index(x) for x in v["cols"]])
            pr = v["model"].predict_proba(v["scaler"].transform(X[np.ix_(idx, ci)]))
            full = np.zeros((len(idx), int(max(v["model"].classes_)) + 1)); full[:, v["model"].classes_.astype(int)] = pr
            bad = idx[np.cumsum(full, axis=1)[:, v["k"]] >= v["cut"]]
            for d in out.values():
                hit = bad[d["allow"][bad]]
                d["allow"][hit] = False; d["why"][hit] = f"dump veto (P(≤{v['dump'] * 100:.0f} %) ≥ {v['cut']:.2f})"
    return out


def decide(models: dict, X: np.ndarray, cols: list[str], t_start, spec: "StackSpec | None" = None) -> dict:
    """The selector's decision for each row of one minute (``t_start`` = the minute's start): which strategy trades it,
    its predicted net return, hold, line and sizing table, and whether a filter blocks it (with the reason)."""
    sp = spec or memecoin_spec()
    X = np.atleast_2d(X); n = len(X); per = decide_all(models, X, cols, t_start, sp)
    best_key = np.full(n, -np.inf); strat = np.full(n, None, dtype=object); score = np.full(n, -1.0); hold = np.zeros(n)
    thr = np.full(n, np.inf); tables = [[] for _ in range(n)]; allow = np.zeros(n, bool); reason = np.full(n, "no strategy fires", dtype=object)
    for name, d in per.items():
        sc = d["score"]; margin = sc - d["line"]
        key = (np.array([((sizing.band_for(d["sizing"], v) or {}).get("kelly") or 0.0) if np.isfinite(v) else -np.inf for v in margin]) + 1e-9 * sc
               if models.get("combine") == "kelly" and d["sizing"] else sc)
        take = d["trig"] & (sc >= d["line"]) & (key > best_key)
        best_key = np.where(take, key, best_key)
        for i in np.flatnonzero(take):
            strat[i] = name; score[i] = sc[i]; hold[i] = sp.hold_seconds(d["hold_min"]) if getattr(sp, "hold_seconds", None) else d["hold_min"] * 60.0; thr[i] = d["line"]; tables[i] = d["sizing"]
            allow[i] = bool(d["allow"][i]); reason[i] = d["why"][i] or "trade"
    return {"strategy": strat, "score": score, "hold_s": hold, "threshold": thr, "tables": tables, "allow": allow, "reason": reason}


# ---------------------------------------------------------------- the memecoin stack's spec (every default above)
def _memecoin_label(ds: DecisionSet, H):
    if H in ds.fwd_h:
        return ds.fwd_h[H]
    return ds.fwd_pess if H * 60 == ds.horizon_s else None


def _memecoin_in_universe(X, cols):
    from .selector import in_universe
    return in_universe(X, cols)


def memecoin_spec() -> StackSpec:
    """The memecoin spec, read from this module's globals at call time (tests monkeypatch STRATEGIES, GROUPS, LEGACY_COLS)."""
    return StackSpec(name="memecoin", strategies=STRATEGIES, base_mask=base_mask, trigger_value=trigger_value, groups=GROUPS, legacy_cols=list(LEGACY_COLS),
                     ev_hold=EV_HOLD, in_universe=_memecoin_in_universe, label=_memecoin_label, hold_s=lambda ds, H: H * 60.0,
                     oos=lambda ds, rows, y, ci, H, stop, label: wf_regress(ds, rows, y, ci, stop, label=label + " min"))


MEMECOIN = memecoin_spec()
