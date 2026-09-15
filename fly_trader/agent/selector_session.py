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


class SelectorBook:
    name = "selector"

    def __init__(self):
        from ..train import selector as sel
        self.model = sel.load_latest()
        if self.model is None:
            raise NoModel("no model has qualified to trade yet (trained on the current data, with a backtest that made money after costs and beat random picks)")
        self.idx = _columns(self.model)
        self.horizon_s = self.model.horizon_min * 60
        self.broker = PaperBroker(BOOK); self.done = False
        self.run_id = str(uuid.uuid4()); self.beat_no = 0
        with transaction() as conn:
            self.snapshot_id = sel.latest_current(conn)["id"]                  # the snapshot load_latest returned
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,'running')",
                         (self.run_id, KIND, json.dumps({"threshold": self.model.threshold, "horizon_min": self.model.horizon_min, "book": BOOK}, default=str), self.snapshot_id))
        record_event("info", "selector", "selector session started", {"run_id": self.run_id, "snapshot": self.snapshot_id, "threshold": self.model.threshold, "horizon_min": self.model.horizon_min})
        log.info("selector session: snapshot %d threshold %.4f horizon %d min", self.snapshot_id, self.model.threshold, self.model.horizon_min)

    def maybe_reload(self) -> bool:
        """Switch to a newer deployable model without a restart (the paper book and open positions carry on)."""
        import joblib
        from ..train import selector as sel
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

    def on_minute(self, ctx) -> dict:
        conn = ctx.conn
        scores = self.model.score(ctx.X[:, self.idx]) if len(ctx.mints) else np.array([])
        summary = {"minute": ctx.m1.isoformat(), "mints_traded": len(ctx.agg), "eligible": len(ctx.mints),
                   "picks": int((scores >= self.model.threshold).sum()) if len(scores) else 0,
                   "score_p99": float(np.percentile(scores, 99)) if len(scores) else None, "score_max": float(scores.max()) if len(scores) else None,
                   "threshold": self.model.threshold, "snapshot": self.snapshot_id}
        if not ctx.trade:
            return summary
        if not ctx.fresh:
            status = {**summary, "stage": "stream stale: holding (no entries or exits)", "updated_at": datetime.now(timezone.utc).isoformat()}
            self._write_status(conn, status)
            return status
        handed = handover.state(conn) is not None and ctx.engine.has_book("fly")
        self.beat_no += 1
        st = paper_trading.trade_minute(ctx, book=BOOK, run_id=self.run_id, beat_no=self.beat_no, broker=self.broker, kind=KIND, mints=ctx.mints, infos=ctx.infos,
                                        scores=scores, threshold=self.model.threshold, table=getattr(self.model, "sizing", None), horizon_s=self.horizon_s,
                                        block="handed over to the fly" if handed else None)
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
