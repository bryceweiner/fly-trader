"""The Kalshi selector: the strategy stack (kalshi/strategies.py) fitted walk-forward by the shared machinery, judged on
its evaluation half against random picks at the same fee-inclusive prices, and saved as ``brain_snapshots`` kind
'kalshi_selector' (``data/brain/kalshi_selectors/kalshi_<ts>.joblib``). Its deployed models are the visual fly's
teacher (kalshi/fly.py); the selector never trades a book of its own (decision of 2026-09-22: no race, no handover).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..train import progress as prog
from ..train import strategies as S
from ..train.decisions import DecisionSet, random_trades, summarize
from ..train.selector import MIN_EV, MIN_LINE_TRADES, deploy_decision
from . import decisions as KD
from .features import K_COLS, KALSHI_FEATURE_VERSION
from .mature import STRIDE_MIN
from .strategies import KALSHI, effective, probability

log = logging.getLogger(__name__)
SELECTOR_DIR = config.BRAIN_DIR / "kalshi_selectors"
KIND = "kalshi_selector"
DATA_VERSION = {"features": KALSHI_FEATURE_VERSION, "stride": STRIDE_MIN, "selector": "kalshi-1", "arms": "taker+maker",
                "cols": hashlib.sha1(",".join(K_COLS).encode()).hexdigest()[:8]}


def is_current(meta: dict | None) -> bool:
    d = (meta or {}).get("data")
    return bool(meta) and isinstance(d, dict) and all(d.get(k) == v for k, v in DATA_VERSION.items())


def is_deployable(meta: dict | None) -> bool:
    return is_current(meta) and bool(meta.get("deployable"))


@dataclass
class KalshiSelectorModel:
    """The deployed stack (``train/strategies.final_models`` under the Kalshi spec): per strategy its classifier, line
    (a minimum edge), arm and sizing; ``decide`` is the shared decision with the Kalshi spec."""
    stack: dict
    cols: list[str]
    trained_through: str
    metrics: dict = field(default_factory=dict)

    @property
    def strategies(self) -> list[str]:
        return list((self.stack.get("strategies") or {}).keys())

    def decide(self, X: np.ndarray, cols: list[str], t_start) -> dict:
        return S.decide(self.stack, X, cols, t_start, spec=KALSHI)

    def decide_all(self, X: np.ndarray, cols: list[str], t_start) -> dict:
        return S.decide_all(self.stack, X, cols, t_start, spec=KALSHI)

    def probabilities(self, X: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
        """Each strategy's p̂(side pays) on every row (the fly's distillation targets)."""
        out = {}
        for name, m in (self.stack.get("strategies") or {}).items():
            ci = [cols.index(c) for c in m["cols"]]
            out[name] = probability(m, np.atleast_2d(X)[:, ci])
        return out


def save(m: KalshiSelectorModel, run_id: str | None = None) -> tuple[Path, int]:
    SELECTOR_DIR.mkdir(parents=True, exist_ok=True)
    path = SELECTOR_DIR / f"kalshi_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.joblib"
    joblib.dump(m, path); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                           (run_id, str(path), sha, KIND, json.dumps({"trained_through": m.trained_through, **m.metrics}, default=str))).fetchone()
    return path, int(row["id"])


def latest_current(conn) -> dict | None:
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC", (KIND,)).fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if is_deployable(meta) and Path(r["path"]).exists():
            return {**dict(r), "meta": meta}
    return None


def load_snapshot(snapshot_id: int) -> KalshiSelectorModel | None:
    with transaction() as conn:
        row = conn.execute("SELECT path FROM brain_snapshots WHERE id = %s AND kind = %s", (snapshot_id, KIND)).fetchone()
    return joblib.load(row["path"]) if row and Path(row["path"]).exists() else None


def load_latest() -> KalshiSelectorModel | None:
    with transaction() as conn:
        r = latest_current(conn)
    return joblib.load(r["path"]) if r else None


def main(days: int | None = None, stop_event: threading.Event | None = None) -> dict:
    from ..ops.reset import reset_training_stats
    reset_training_stats("kalshi_selector", reason="kalshi selector training")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("kalshi selector: building decision points", 0, 1, force=True)
    t0 = time.time(); ds = KD.build(days=days)
    log.info("kalshi decision points: %d rows, %d days, taker universe mean %+.2f%%, maker filled %.0f%% (%.0fs)", len(ds.y), len(ds.days),
             ds.fwd_pess.mean() * 100, np.isfinite(ds.fwd_h["maker"]).mean() * 100, time.time() - t0)
    stack = S.fit_stack(ds, stop_event, spec=KALSHI)
    for c in stack.components:
        log.info("component %-28s %s — %s", c["name"], "PASSED" if c["passed"] else "dropped", c["reason"])
        record_event("info", KIND, f"component {c['name']}: {'passed' if c['passed'] else 'dropped'}", c)
    if stop_event is not None and stop_event.is_set():
        return {"stopped": True}
    prog.update("kalshi selector: fitting the deployable models on every day", 0, 1, force=True)
    models = S.final_models(ds, stack, spec=KALSHI) if stack.fits else {"strategies": {}, "spec": "kalshi"}
    holdout = S.score_holdout(ds, S.final_models(ds, stack, exclude_days=stack.holdout_days, spec=KALSHI), stack.holdout_days, spec=KALSHI) if stack.holdout_days and stack.fits else {}
    final = KalshiSelectorModel(stack=models, cols=list(ds.cols), trained_through=str(ds.days[-1]))
    final.metrics = {"walk_forward": {**stack.evaluation, "days": None}, "selection": stack.selection, "random_baseline": {"mean": stack.evaluation.get("random_mean")},
                     "components": stack.components,
                     "strategies": {k: {x: v[x] for x in ("line", "hold_min", "thr", "high", "sizing", "selection", "evaluation")} for k, v in models["strategies"].items()},
                     "combine": stack.combine, "veto": {k: v for k, v in (stack.veto or {}).items() if k not in ("p",)} or None, "fallback": stack.fallback,
                     "holdout": holdout, "costs": "Kalshi taker fee (factor x series multiplier x P(1-P)) at the ask; maker one tick inside the ask + maker fee where charged",
                     "data": DATA_VERSION, "deployable": stack.deployable, "deploy_reason": stack.reason,
                     "model": {"kind": "kalshi-stack", "max_days_to_close": config.KALSHI_MAX_DAYS_TO_CLOSE, "max_spread": config.KALSHI_MAX_SPREAD_CENTS},
                     "rows": int(len(ds.y)), "days": len(ds.days), "first_day": str(ds.days[0]), "last_day": str(ds.days[-1])}
    path, sid = save(final)
    record_event("info", KIND, f"selector saved (snapshot {sid})", {"path": str(path), "deployable": stack.deployable, "reason": stack.reason, **stack.evaluation})
    prog.update("kalshi selector: saved", 1, 1, force=True, snapshot_id=sid, deployable=stack.deployable, deploy_reason=stack.reason)
    log.info("kalshi selector saved: %s (snapshot %d) — %s: %s", path, sid, "put to work" if stack.deployable else "NOT put to work", stack.reason)
    return {"snapshot_id": sid, "deployable": stack.deployable, "deploy_reason": stack.reason, "components": stack.components}
