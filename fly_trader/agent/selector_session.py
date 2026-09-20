"""The selector's book: the GBM selector trading ``paper_selector`` on the minute engine (agent/minute_engine.py).

Each minute the engine's eligible rows (full feature vectors) are scored by the deployed selector (``brain_snapshots``
kind 'selector') with its columns picked by name; the book trades them with the shared paper rules
(agent/paper_trading.py): scores at or above the model's line open positions sized from its measured certainty, held
``horizon_min`` minutes, then sold at the last traded price (the training label's exit). A newer deployable model is
picked up without a restart (``maybe_reload``). After the handover (agent/handover.py) the book stops entering while the
fly trades, exits to flat and retires.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..train.decisions import X_COLS
from . import handover, paper_trading

log = logging.getLogger(__name__)
BOOK, KIND = "paper_selector", "selector"


class NoModel(RuntimeError):
    """No selector has qualified to trade (current data and a profitable backtest after costs)."""


def _columns(model) -> np.ndarray:
    missing = [c for c in model.cols if c not in X_COLS]
    if missing:
        raise RuntimeError(f"selector expects features the live engine does not produce: {missing}")
    return np.asarray([X_COLS.index(c) for c in model.cols], dtype=int)


def pinned_snapshot() -> int | None:
    """The selector snapshot the operator has pinned, or None. A pin freezes the book on one model: a forward test that
    silently swapped models mid-run could not say which one earned the result."""
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'pinned_selector_snapshot'").fetchone()
    if not r:
        return None
    v = r["value"]
    v = v.get("id") if isinstance(v, dict) else json.loads(v or "{}").get("id")
    return int(v) if v is not None else None


class SelectorBook:
    name = "selector"

    def __init__(self):
        from ..train import selector as sel
        pin = pinned_snapshot()
        self.model = sel.load_snapshot(pin) if pin is not None else sel.load_latest()
        if self.model is None:
            raise NoModel("no model has qualified to trade yet (trained on the current data, with a backtest that made money after costs and beat random picks)")
        self.idx = _columns(self.model)
        self.horizon_s = self.model.horizon_min * 60
        self.broker = PaperBroker(BOOK); self.done = False
        self.run_id = str(uuid.uuid4()); self.beat_no = 0
        with transaction() as conn:
            self.snapshot_id = pin if pin is not None else sel.latest_current(conn)["id"]   # the snapshot actually loaded
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,'running')",
                         (self.run_id, KIND, json.dumps({"threshold": self.model.threshold, "horizon_min": self.model.horizon_min, "book": BOOK}, default=str), self.snapshot_id))
        record_event("info", "selector", "selector session started", {"run_id": self.run_id, "snapshot": self.snapshot_id, "threshold": self.model.threshold, "horizon_min": self.model.horizon_min})
        log.info("selector session: snapshot %d threshold %.4f horizon %d min", self.snapshot_id, self.model.threshold, self.model.horizon_min)

    def maybe_reload(self) -> bool:
        """Switch to a newer deployable model without a restart (the paper book and open positions carry on). While a
        snapshot is pinned (``ui_settings['pinned_selector_snapshot']``) the book stays on it, so a forward test measures
        one model instead of a blend of every retrain that lands during it."""
        import joblib
        from ..train import selector as sel
        if pinned_snapshot() is not None:
            return False
        with transaction() as conn:
            r = sel.latest_current(conn)
        if not r or r["id"] == self.snapshot_id:
            return False
        m = joblib.load(r["path"])
        try:
            idx = _columns(m)
        except RuntimeError as e:
            log.error("model #%d: %s; keeping #%s", r["id"], e, self.snapshot_id)
            return False
        old = self.snapshot_id; self.model, self.idx, self.snapshot_id, self.horizon_s = m, idx, r["id"], m.horizon_min * 60
        with transaction() as conn:
            conn.execute("UPDATE runs SET brain_snapshot_id = %s, config = %s WHERE run_id = %s",
                         (r["id"], json.dumps({"threshold": m.threshold, "horizon_min": m.horizon_min, "book": BOOK}, default=str), self.run_id))
        record_event("info", "selector", f"switched to model #{r['id']}", {"from": old, "to": r["id"], "threshold": m.threshold})
        log.info("switched from model #%s to #%d (threshold %.4f)", old, r["id"], m.threshold)
        return True

    def on_bars(self, t_start: float, bars: dict) -> None:
        pass

    def open_mints(self, conn) -> set[str]:
        return {p["mint"] for p in ledger.open_positions(conn, BOOK)}

    def _decide(self, ctx) -> dict | None:
        """The strategy stack's per-row decision (train/strategies.decide), with the live fail-closed rules: rows whose
        inputs training had but live lacks are blocked — the skill table of the day missing while skill inputs are used, a
        token graduated inside the archive whose curve facts are not known yet while rug inputs are used."""
        if not (getattr(self.model, "stack", None) or {}).get("strategies") or not len(ctx.mints):
            return None
        d = self.model.decide(ctx.X, X_COLS, ctx.t_start)
        groups = set((getattr(self.model, "metrics", None) or {}).get("groups") or [])
        for k, i in enumerate(ctx.infos):
            if not d["allow"][k]:
                continue
            if "skill" in groups and i.get("skill_missing"):
                d["allow"][k] = False; d["reason"][k] = "fail closed: no wallet-skill table for today"
            elif "rug" in groups and i.get("age_h") is not None and not i.get("curve_known"):
                d["allow"][k] = False; d["reason"][k] = "fail closed: curve facts not known yet"
        return d

    def on_minute(self, ctx) -> dict:
        conn = ctx.conn
        d = self._decide(ctx)
        if d is not None:
            scores = np.where(np.isfinite(d["score"]), d["score"], -1.0); thr = d["threshold"]
        else:
            scores = self.model.score(ctx.X[:, self.idx]) if len(ctx.mints) else np.array([]); thr = self.model.threshold
        summary = {"minute": ctx.m1.isoformat(), "mints_traded": len(ctx.agg), "eligible": len(ctx.mints),
                   "picks": int((scores >= thr).sum()) if len(scores) else 0,
                   "score_p99": float(np.percentile(scores, 99)) if len(scores) else None, "score_max": float(scores.max()) if len(scores) else None,
                   "threshold": self.model.threshold, "snapshot": self.snapshot_id,
                   "strategies": sorted(((getattr(self.model, "stack", None) or {}).get("strategies") or {"ev": None}).keys())}
        if not ctx.trade:
            return summary
        if not ctx.fresh:
            status = {**summary, "stage": "stream stale: holding (no entries or exits)", "updated_at": datetime.now(timezone.utc).isoformat()}
            self._write_status(conn, status)
            return status
        handed = handover.state(conn) is not None and ctx.engine.has_book("fly")
        self.beat_no += 1
        extra = {} if d is None else {"holds": d["hold_s"], "strategies": d["strategy"], "allow": d["allow"], "reasons": d["reason"], "tables": d["tables"]}
        st = paper_trading.trade_minute(ctx, book=BOOK, run_id=self.run_id, beat_no=self.beat_no, broker=self.broker, kind=KIND, mints=ctx.mints, infos=ctx.infos,
                                        scores=scores, threshold=thr, table=getattr(self.model, "sizing", None), horizon_s=self.horizon_s,
                                        block="handed over to the fly" if handed else None, **extra)
        if handed and st["open"] == 0:
            self.done = True
            record_event("info", "selector", "selector retired: the fly holds the seat and the book is flat", {"run_id": self.run_id})
        status = {**summary, **st, "stage": "retired (handed over to the fly)" if self.done else ("exiting to flat (handed over to the fly)" if handed else "trading"),
                  "latency_s": (datetime.now(timezone.utc) - ctx.m1).total_seconds(), "updated_at": datetime.now(timezone.utc).isoformat()}
        self._write_status(conn, status)
        return status

    @staticmethod
    def _write_status(conn, status: dict) -> None:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps(status, default=str),))

    def finish(self) -> None:
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = 'finished' WHERE run_id = %s", (self.run_id,))
        record_event("info", "selector", "selector session stopped", {"run_id": self.run_id, "minutes": self.beat_no})
