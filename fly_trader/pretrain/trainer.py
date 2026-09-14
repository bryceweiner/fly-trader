"""Pretraining: replay a corpus through the SAME Beat with the paper broker.

Walk-forward folds (train/val with learning on, then the test window twice from the fold snapshot:
η = 0 headline and η on as-live). Every run is a `runs` row; metrics are counted from rows. Results are
reported, not gating (operator decision: live from day one).
"""
from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timezone

from .. import config
from ..brain import checkpoint
from ..brain.lif import Connectome
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from ..agent.beat import Session, SessionOptions
from .metrics import run_metrics
from .replay import ReplayFeed
from .splits import walk_forward

log = logging.getLogger(__name__)
_STOP = {"flag": False}


def _run_window(c, corpus: str, t_start: float, t_end: float, *, beat_s: float, ticks: int, slots: int, learn: bool,
                initial_snapshot: int | None, label: str, split: dict, max_beats: int | None) -> tuple[Session, dict]:
    feed = ReplayFeed(corpus, t_start, t_end)
    book = f"pretrain:{label}"
    opts = SessionOptions(run_kind="pretrain", ticks=ticks, learn=learn, persist_slots=True, initial_snapshot=initial_snapshot)
    s = Session(c, opts, corpus=corpus, split=split)
    g = s.add_replay_group(slots, feed, book, name="pretrain")
    g.clock = t_start - 3 * 3600 - beat_s  # context warm-up
    s.start()
    n = 0
    t_wall = time.time()
    log.info("window %s: %s → %s (%d tokens)", label, datetime.fromtimestamp(t_start, tz=timezone.utc), datetime.fromtimestamp(t_end, tz=timezone.utc), len(g.meta))
    saved_sim = config.BEAT_S_SIM
    config.BEAT_S_SIM = beat_s
    while g.clock + beat_s <= t_end and not _STOP["flag"]:
        out = s.run_beat(time.time())
        t = g.clock
        n += 1
        if n % 500 == 0:
            rate = n / max(time.time() - t_wall, 1e-6)
            log.info("%s beat %d sim=%s active=%d dec=%d gpu_ms=%d total_ms=%d mean_mbon=%.3f kc=%.3f beats/s=%.1f", label, n,
                     datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m-%d %H:%M"), out["active"], out["decisions"], out["gpu_ms"],
                     out["total_ms"], out["mean_mbon"], out["kc"], rate)
        if max_beats and n >= max_beats:
            break
        t += beat_s
    config.BEAT_S_SIM = saved_sim
    metrics = run_metrics(s.run_id, book)
    metrics["wall_s"] = time.time() - t_wall
    metrics["label"] = label
    s.finish("stopped" if _STOP["flag"] else "done", metrics)
    return s, metrics


def main(corpus: str = "meteora", beat_s: float | None = None, ticks: int | None = None, slots: int | None = None,
         max_beats: int | None = None, days: float | None = None) -> None:
    setup("pretrain")
    beat_s = beat_s or config.BEAT_S_SIM
    ticks = ticks or config.BEAT_TICKS
    slots = slots or config.SLOTS

    def _stop(*_):
        _STOP["flag"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    c = Connectome.load()
    feed = ReplayFeed(corpus)
    t0, t1 = feed.time_range()
    if days:
        t1 = min(t1, t0 + days * 86400)
    total_d = (t1 - t0) / 86400.0
    if corpus == "meteora":
        folds = walk_forward(t0, t1, train_d=5, val_d=2, test_d=3, embargo_h=6, step_d=3)
    else:
        folds = walk_forward(t0, t1, train_d=7, val_d=2, test_d=3, embargo_h=6, step_d=3)
    if not folds:  # short corpus: 60/20/20 split
        tr = t0 + 0.6 * (t1 - t0)
        va = t0 + 0.8 * (t1 - t0)
        from .splits import Fold
        folds = [Fold((t0, tr), (tr, va), (va, t1))]
    record_event("info", "pretrain", "pretrain started", {"corpus": corpus, "days": total_d, "folds": [f.as_dict() for f in folds],
                                                          "beat_s": beat_s, "ticks": ticks, "slots": slots})
    print(f"corpus={corpus} span={total_d:.1f}d folds={len(folds)} beat_s={beat_s} ticks={ticks} slots={slots}")
    results = []
    for i, f in enumerate(folds):
        if _STOP["flag"]:
            break
        label = f"{corpus[:3]}-f{i}"
        s_train, m_train = _run_window(c, corpus, f.train[0], f.val[1], beat_s=beat_s, ticks=ticks, slots=slots, learn=True,
                                       initial_snapshot=None, label=f"{label}-train", split=f.as_dict(), max_beats=max_beats)
        with transaction() as conn:
            snap_id = s_train.save_snapshot(conn, "pretrained", None, note=f"{label} train+val")
        _, m_frozen = _run_window(c, corpus, f.test[0], f.test[1], beat_s=beat_s, ticks=ticks, slots=slots, learn=False,
                                  initial_snapshot=snap_id, label=f"{label}-test-frozen", split=f.as_dict(), max_beats=max_beats)
        _, m_live = _run_window(c, corpus, f.test[0], f.test[1], beat_s=beat_s, ticks=ticks, slots=slots, learn=True,
                                initial_snapshot=snap_id, label=f"{label}-test-aslive", split=f.as_dict(), max_beats=max_beats)
        results.append({"fold": i, "snapshot_id": snap_id, "train": m_train, "test_frozen": m_frozen, "test_aslive": m_live})
        print(json.dumps(results[-1], indent=1, default=str))
    record_event("info", "pretrain", "pretrain finished", {"results": results})
    print(json.dumps({"results": results}, indent=1, default=str))
