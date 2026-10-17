"""The plastic fly: trading its own book (``paper_fly``) on the minute engine and learning from the market every minute.

It starts once the fly design passed its replay (``train/fly_replay.verdict``, current definitions) and a deployable
bootstrap exists (``train/fly_selector.latest_deployable``); the replay's chosen configuration (α, half-life) is the
learning rate and forgetting of its mushroom body (``brain/plastic.py``). Each minute (agent/minute_engine.py), live money
first (CLAUDE.md): ``on_minute`` runs before any other book's minute, ``after_minute`` once every chain's live orders are out.

1. bars: minute start, open, close and exit cost of every traded mint with a pending tag feed the label resolver;
2. scoring (``on_minute``): eligible minutes are scored by the plastic fly and its shadow and tagged; only minutes at or
   above the line teach it (brain/plastic.py);
3. trading (``on_minute``; fresh stream, trade minutes): the race rules (``agent/paper_trading.py``) at the plastic line
   and sizing plan the paper book's minute, and once the fly holds the seat the wallet's orders on its entries go out
   (agent/fly_live.py, rh/live.py; the wallet's scheduled sales left at the minute boundary, agent/live_first.py);
4. after the live orders (``after_minute``): the scored rows stored (``fly_scored``: pending rows keep the feature
   vector, so tags survive restarts), the paper trades booked, then outcomes: tags whose exit minute has now been seen
   teach the mushroom body's outcome pathway what the row made -- the book's realized return when it traded the row,
   else its label (``resolve_label``, exactly as ``train/decisions.build`` labels a row); a row the wallet bought too
   then teaches the fill pathway what the real fills added (live realized − outcome) once its live position has closed
   (``fly_scored`` state 'awaiting_fill' until then); a row with neither outcome nor label is dropped;
5. calibration: at the first minute of each UTC day, the line and sizing of the plastic fly and of its frozen shadow
   (the bootstrap, D = 0) from their scores of the minutes resolved in the last seven days (``train/fly_calibrate.py``).

Hourly: learning statistics (``fly_updates``), a snapshot of the plastic state (kind 'fly_plastic'; hourly for 7 days,
then one per day), the rollback checks (``train/fly_governance.py``; a rollback restores the newest good snapshot at
least a day old, or D = 0, pauses the checks a day, and the third within 7 days freezes learning and requests a
re-bootstrap), and the handover (``HANDOVER_DAYS`` of racing, at least ``HANDOVER_MIN_TRADES`` trades, realized P&L
at least the selector's). The state file ``data/brain/plastic/state.pt`` is written atomically every minute, before
tags are marked resolved: on restart, pending tags at or before its watermark count as applied.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .. import config, markets
from ..brain import activity, device as brain_device, plastic
from ..brain.connectome import current_connectome_path
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..ops.reset import activity_dir, fly_state_dir
from ..train import fly_calibrate, fly_governance as gov, fly_selector, selector, strategies
from ..train.decisions import X_COLS
from . import handover, paper_trading

log = logging.getLogger(__name__)
BOOK, KIND = "paper_fly", "fly"            # the Solana book's names (another chain's book takes its own from fly_trader/markets)
BAR_KEEP = 270               # minutes of bars kept per watched mint (the longest strategy hold, 240, + next-open window + margin)
NEXT_OPEN_S = 120.0          # entry at the next traded minute's open when it comes within two minutes (decisions.build)
JUMP = 50.0
LABEL_LAG_S = 120.0          # a row teaches a minute after its exit minute, when the book has sold it
OUTCOME_WAIT_S = 600.0       # a traded row whose book position is still open waits up to this long past its hold, else its label
FILL_RETRY_S = 60.0          # a row whose live position is still open is looked at again a minute later
FILL_WAIT_S = 4 * 3600.0     # ... until this long after its hold (a live position unsold 3 h past its hold is a dead-bag)
FILL_IC_DAYS, FILL_IC_MIN = 7, 30   # the fill pathway's check: its predictions against the fills it learned
ROLLBACK_PAUSE_S, ROLLBACK_LIMIT, ROLLBACK_WINDOW_S, ROLLBACK_MIN_AGE_S = 86400.0, 3, 7 * 86400.0, 86400.0
SNAP_HOURLY_DAYS, SCORED_KEEP_DAYS = 7, 35
COMMAND_KEY = "fly_command"
COMMAND_MAX_AGE_S = 600.0          # an operator command older than this (queued while the engine was down) is dropped


def send_command(cmd: str) -> None:
    """Queue an operator command for the running fly (applied at its next minute): 'pause', 'resume' or 'rollback'."""
    assert cmd in ("pause", "resume", "rollback"), cmd
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (COMMAND_KEY, json.dumps({"cmd": cmd, "at": datetime.now(timezone.utc).isoformat()})))


def resolve_label(bars: list, t: float, horizon_s: float) -> float | None:
    """Net return of a row scored at minute start ``t`` from the mint's bars [(minute start, open, close, exit cost)]
    in time order — ``decisions.build``'s ``fwd_pess``: entry at the next traded minute's open when it comes within
    ``NEXT_OPEN_S`` (else the close), exit at the last close at or before ``t + horizon_s``, the one-side exit cost at
    both ends. None when the row's own bar is missing, a price is not positive, or the exit minute jumped 50× from the
    one before (the archive drops such exits)."""
    rows = [b for b in bars if b[0] >= t]
    if not rows or rows[0][0] != t:
        return None
    j = 0
    for i, b in enumerate(rows):
        if b[0] <= t + horizon_s:
            j = i
        else:
            break
    entry = rows[1][1] if len(rows) > 1 and rows[1][0] - t <= NEXT_OPEN_S and rows[1][1] else rows[0][2]
    exit_px = rows[j][2]
    if not entry or entry <= 0 or not exit_px or exit_px <= 0:
        return None
    if j > 0 and rows[j - 1][2] and not (1 / JUMP <= exit_px / rows[j - 1][2] <= JUMP):
        return None
    return float(exit_px / entry * (1 - rows[0][3]) * (1 - rows[j][3]) - 1)


def chain_passed(v: dict, chain: str) -> tuple[bool, str]:
    """Whether the replay lets this chain's book trade: its own per-chain verdict (a combined replay), else the one verdict
    for Solana (a Solana-only replay says nothing about another chain)."""
    if "chains" in v:
        c = v["chains"].get(chain)
        return (bool(c and c.get("passed")), (c or {}).get("reason") or f"the replay has no {chain} verdict")
    return (bool(v.get("passed")) and chain == "sol", v.get("reason") if chain == "sol" else f"the replay judged no {chain} trades")


def try_start(live: bool = False, chain: str = "sol", brain: "FlyBook | None" = None) -> tuple["FlyBook | None", str]:
    """A FlyBook for ``chain`` when the fly may trade there, else (None, why). ``brain``: the Solana book whose fly and
    plastic memory another chain's book shares (one model learns from both chains)."""
    from ..train import fly_replay
    v = fly_replay.verdict()
    if not v:
        return None, "the fly's replay has not run on the current definitions (fly-trader fly-replay)"
    ok, why = chain_passed(v, chain)
    if not ok:
        return None, f"the fly's replay did not pass on {markets.for_chain(chain).name}: {why}"
    with transaction() as conn:
        boot = fly_selector.latest_deployable(conn)
    if boot is None:
        return None, "no deployable fly bootstrap (the training pipeline bootstraps one)"
    try:
        return FlyBook(boot, v, live=live, chain=chain, brain=brain), "ok"
    except Exception as e:
        log.exception("fly book failed to start")
        return None, f"fly book failed to start: {type(e).__name__}: {e}"


def no_teacher(n: int) -> dict:
    """The decision when the teacher gate is on (FLY_TEACHER_GATE) but the fly's teacher could not be loaded: no row is
    allowed, each recorded with why. A fresh install once lost the teacher's snapshot id (ops/release.remap_teacher)."""
    return {"strategy": np.full(n, None, dtype=object), "score": np.full(n, -1.0), "hold_s": np.zeros(n), "threshold": np.full(n, np.inf),
            "tables": [[] for _ in range(n)], "allow": np.zeros(n, bool),
            "reason": np.full(n, "fail closed: the fly's teacher (the selector that taught it) is not loaded", dtype=object)}


class FlyBook:
    def __init__(self, boot: dict, verdict: dict, live: bool = False, chain: str = "sol", brain: "FlyBook | None" = None):
        self.live = live; self.done = False; self.mirror = None
        self._phase: dict | None = None; self._scored_rows: list = []; self._activity: tuple | None = None   # the live phase's, written after it
        self.verdict = verdict
        self.chain = chain; self.market = markets.for_chain(chain); self.brain = brain
        self.name = "fly" if chain == "sol" else f"fly_{chain}"
        self.BOOK = self.market.fly_book; self.KIND = self.name; self.status_key = self.market.fly_status_key
        self.state_path = fly_state_dir() / ("state.pt" if chain == "sol" else f"state_{chain}.pt")
        self._load_bootstrap(boot)
        self.run_id = str(uuid.uuid4()); self.beat_no = 0; self.broker = PaperBroker(self.BOOK)
        with transaction() as conn:
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,'running')",
                         (self.run_id, self.KIND, json.dumps({"channels": self.cfg, "book": self.BOOK, "chain": chain}, default=str), self.boot_id))
            self._restore_tags(conn)
        record_event("info", "fly", "fly session started", {"run_id": self.run_id, "bootstrap": self.boot_id, "channels": self.cfg, "lines": self.lines["plastic"],
                                                            "pending": len(self.pending), "learning_frozen": self.learning_frozen})
        log.info("fly session: bootstrap %d on %s, channels %s, lines %s, %d pending tags", self.boot_id, brain_device.describe(self.fly.net.dev), self.cfg,
                 self.lines["plastic"], len(self.pending))

    # ---- model and state ----
    def _load_bootstrap(self, boot: dict | None) -> None:
        b = self.brain
        if b is not None:        # another chain's book: the Solana book's fly and plastic memory, this chain's own lines and tags
            self.boot_id = b.boot_id; self.fly = b.fly; self.connectome_name = b.connectome_name; self.capture_ok = False
            self.teacher_models = b.teacher_models; self.idx = b.idx; self.names = b.names; self.holds = b.holds; self.H = b.H
            self.teacher_live, self.teacher_groups = b.teacher_live, b.teacher_groups
            self.cfg = b.cfg; self.alpha = b.alpha; self.half_life = b.half_life; self.bank = b.bank
            net = self.fly.net
        else:
            self.boot_id = int(boot["id"])
            self.fly = fly_selector.load(boot["path"])
            try:
                self.connectome_name = current_connectome_path().name       # stamped on the activity files the console draws
            except Exception:
                self.connectome_name = ""
            self.capture_ok = True
            self.teacher_models = self._teacher_models(boot)
            self.idx = np.asarray([X_COLS.index(c) for c in self.fly.cols], dtype=int)
            self.names = list(self.fly.strategies); self.holds = {k: float(self.fly.rules[k]["hold_min"]) * 60.0 for k in self.names}
            self.H = max(self.holds.values())
            per = self.verdict.get("per_strategy") or {self.names[0]: {"alpha": self.verdict.get("alpha", 0.0), "half_life_days": self.verdict.get("half_life_days")}}
            self.cfg = {k: (float((per.get(k) or {}).get("alpha") or 0.0), (per.get(k) or {}).get("half_life_days")) for k in self.names}
            self.alpha = self.cfg[self.names[0]][0]; self.half_life = float(self.cfg[self.names[0]][1]) if self.cfg[self.names[0]][1] is not None else math.inf
            net = self.fly.net
            self.bank = plastic.PlasticBank(net, [(self.alpha, self.half_life)], fly_selector.SCALE, learn=net.learn.cpu().numpy(), read=net.read.cpu().numpy())
            self._channel_rates()
        self.pending = plastic.PendingTags(net.n_kc, net.k_active)
        self.pending_fill = plastic.PendingTags(net.n_kc, net.k_active); self.fill_base: dict[tuple, float] = {}    # rows waiting for their live fill: their outcome
        self.fill_choice = False                                 # the fill pathways are chosen at the first scored minute (the Solana book's memory)
        self.bars: dict[str, deque] = {}; self.watch: dict[str, int] = {}
        own = {k: self.fly.line_for(k, self.chain) for k in self.names}; own_s = {k: list(self.fly.sizing_for(k, self.chain)) for k in self.names}
        if config.FLY_SCORE == "teacher" and self.teacher_live:   # it scores on its teacher's scale: it starts on its teacher's lines and bands
            ts_ = self.teacher_live.get("strategies") or {}
            own = {k: float(ts_[k]["line"]) if k in ts_ else v for k, v in own.items()}
            own_s = {k: list(ts_[k].get("sizing") or v) if k in ts_ else v for k, v in own_s.items()}
        self.lines = {"plastic": dict(own), "frozen": dict(own)}
        self.sizing = {"plastic": {k: list(v) for k, v in own_s.items()}, "frozen": {k: list(v) for k, v in own_s.items()}}
        self.applied_through = -math.inf; self.learning_frozen = False; self.checks_paused_until = {k: 0.0 for k in self.names}
        self.race_started_at: float | None = None; self.last_day: int | None = None; self.last_hour: int | None = None
        self.rollbacks: dict = {k: [] for k in self.names}; self.nu_set = False; self.hour_stats = self._empty_stats(); self.last_checks: dict = {}
        s = self._read_state()
        if b is not None:
            self.nu_set = True                                   # the shared memory is the Solana book's to load and save
        if s and s.get("bootstrap_id") == self.boot_id and s.get("cfg") == self.cfg and s.get("score", "own") != config.FLY_SCORE:
            self._archive_state(s)                               # learned under another scoring: it starts from its bootstrap
            s = None
        if s and s.get("bootstrap_id") == self.boot_id and s.get("cfg") == self.cfg:
            if b is None:
                self.bank.load_state(s["bank"])
            self.nu_set = True
            self.lines, self.sizing = s["lines"], s["sizing"]
            self.applied_through, self.learning_frozen, self.checks_paused_until = s["applied_through"], s["learning_frozen"], s["checks_paused_until"]
            self.race_started_at, self.last_day, self.last_hour, self.rollbacks = s["race_started_at"], s["last_day"], s["last_hour"], s["rollbacks"]
        elif s:
            self.race_started_at = s.get("race_started_at")        # a new bootstrap keeps racing the same book
            with transaction() as conn:
                conn.execute("UPDATE fly_scored SET state = 'dropped', x = NULL WHERE state = 'pending' AND chain = %s AND bootstrap_id IS DISTINCT FROM %s",
                             (self.chain, self.boot_id))
        if b is None:                                            # its fill pathways: kept from the state, switched off, or chosen at the first scored minute
            if bool(self.bank.fill.any()) and not config.FLY_FILL_PATHWAY:
                self.bank.drop_fill(); self._channel_rates()
                record_event("info", "fly", "fill pathways switched off (FLY_FILL_PATHWAY=0): their compartments learn outcomes again from the bootstrap")
            elif bool(self.bank.fill.any()):
                self._channel_rates(outcome=False)
            self.fill_choice = config.FLY_FILL_PATHWAY and not bool(self.bank.fill.any())

    def _channel_rates(self, outcome: bool = True) -> None:
        """Each strategy's learning rates: its outcome pathway's the replay chose (``outcome``: set them; else they keep
        the saved ones, raised by the energy their fill pathway took), its fill pathway's this run's configuration (no
        replay has fills to choose them by)."""
        for j, k in enumerate(self.names):
            fc = self.bank.fill[j]
            if outcome:
                a, h = self.cfg[k]; cols = self.bank.learn[j] & ~fc
                self.bank.alpha[0, cols] = a; self.bank.half_life_s[0, cols] = float(h) * 86400.0 if h is not None else float("inf")
            self.bank.alpha[0, fc] = config.FLY_FILL_ALPHA; self.bank.half_life_s[0, fc] = config.FLY_FILL_HALF_LIFE_DAYS * 86400.0

    def _choose_fill(self, u0: torch.Tensor, k: torch.Tensor) -> None:
        """Choose the fill pathways on this minute's rows (brain/plastic.PlasticBank.choose_fill), once; the state keeps them."""
        comp, names = plastic.net_compartments(self.fly.net)
        info = self.bank.choose_fill(u0, k, comp, config.FLY_FILL_RANGE)
        self._channel_rates(outcome=False); self.fill_choice = False
        detail = {self.names[j]: {**v, "compartment": names[v["compartment"]] if comp is not None and 0 <= v["compartment"] < len(names) else v["compartment"]}
                  for j, v in info.items()}
        record_event("info", "fly", "fill pathways chosen: " + ", ".join(f"{n} {d['compartment']} (lowers by up to {d['lowers_by'] * 100:.1f} %, "
                                                                           f"{d['energy_share'] * 100:.0f} % of its channel)" for n, d in detail.items()), detail)

    teacher_live: dict | None = None
    teacher_groups: set = set()

    def _teacher_models(self, boot: dict) -> dict | None:
        """The stack that taught this fly (its snapshot's ``teacher_snapshot``: the holdout-blind copy when that is what
        taught it), whose final decision on each scored row is stored as ``teacher_allow`` -- the match calibration
        rule's budget. None (with a warning) when the fly was taught by a refit stack or the snapshot is gone."""
        meta = boot.get("meta") or {}
        sid = meta.get("teacher_snapshot")
        if sid is None:
            if fly_calibrate.RULE == "match":
                log.warning("fly #%s names no teacher snapshot: the match rule keeps its bootstrap line", self.boot_id)
            return None
        try:
            m = selector.load_snapshot(int(sid))
        except Exception:
            log.exception("fly #%s: teacher snapshot #%s could not be loaded", self.boot_id, sid)
            return None
        blind = getattr(m, "blind", None) or {}
        models = blind["models"] if "blind to the calibration week" in str(meta.get("teacher") or "") and blind.get("models") else getattr(m, "stack", None)
        live = getattr(m, "stack", None)                         # the gate: the teacher's full stack, as its selector book trades it
        if live and live.get("strategies") and all(c in X_COLS for v in live["strategies"].values() for c in v["cols"]):
            self.teacher_live = live
            self.teacher_groups = set((getattr(m, "metrics", None) or {}).get("groups") or [])
        if not models or not models.get("strategies") or any(c not in X_COLS for v in models["strategies"].values() for c in v["cols"]):
            log.warning("fly #%s: teacher #%s has no usable strategy stack", self.boot_id, sid)
            return None
        return models

    def _teacher_allow(self, X: np.ndarray, t: float) -> dict[str, np.ndarray]:
        """Per strategy: would the teacher trade each row (trigger, line and every filter)? Empty when unknown."""
        if self.teacher_models is None:
            return {}
        try:
            per = strategies.decide_all(self.teacher_models, X, X_COLS, t)
            return {k: np.asarray(v["allow"], bool) for k, v in per.items()}
        except Exception:
            log.exception("teacher decisions failed for this minute (the fly still trades)")
            return {}

    def _read_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            return torch.load(self.state_path, map_location="cpu", weights_only=True)
        except Exception:
            log.exception("fly state unreadable; starting from the bootstrap")
            return None

    def _archive_state(self, s: dict) -> None:
        """Keep a state learned under another scoring mode beside the new one (never loaded again)."""
        try:
            dst = self.state_path.with_name(f"{self.state_path.stem}_{s.get('score', 'own')}_{time.strftime('%Y%m%dT%H%M%S')}.pt")
            os.replace(self.state_path, dst)
            record_event("info", "fly", f"fly state learned with FLY_SCORE={s.get('score', 'own')} set aside: the fly starts from its bootstrap",
                         {"path": str(dst), "drift": None})
        except OSError:
            log.exception("could not set the old fly state aside")

    def _state(self) -> dict:
        return {"bootstrap_id": self.boot_id, "cfg": self.cfg, "score": config.FLY_SCORE, "bank": self.bank.state() if self.brain is None else None, "lines": self.lines, "sizing": self.sizing,
                "applied_through": self.applied_through, "learning_frozen": self.learning_frozen, "checks_paused_until": self.checks_paused_until,
                "race_started_at": self.race_started_at, "last_day": self.last_day, "last_hour": self.last_hour, "rollbacks": self.rollbacks,
                "saved_at": time.time()}

    def _write_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        torch.save(self._state(), tmp); os.replace(tmp, self.state_path)

    def _restore_tags(self, conn) -> None:
        """Tags from ``fly_scored``: pending rows wait for their outcome again, 'awaiting_fill' rows (outcome learned) for
        their live position to close; rows past their time are resolved without."""
        rows = conn.execute("SELECT ts, mint, strategy, x, score, line, state, outcome FROM fly_scored WHERE state IN ('pending', 'awaiting_fill') "
                            "AND bootstrap_id = %s AND chain = %s ORDER BY ts", (self.boot_id, self.chain)).fetchall()
        rows = [r for r in rows if r["strategy"] in self.holds]; now = time.time()
        waiting = [r for r in rows if r["state"] == "awaiting_fill"]; rows = [r for r in rows if r["state"] == "pending"]
        applied = [(r["ts"], r["mint"], r["strategy"]) for r in rows if r["ts"].timestamp() + self.holds[r["strategy"]] + LABEL_LAG_S <= self.applied_through]
        expired = [(r["ts"], r["mint"], r["strategy"]) for r in waiting
                   if r["x"] is None or r["outcome"] is None or r["ts"].timestamp() + self.holds[r["strategy"]] + FILL_WAIT_S <= now]
        if applied or expired:
            with conn.cursor() as cur:
                cur.executemany("UPDATE fly_scored SET state = 'resolved', resolved_at = now(), x = NULL WHERE ts = %s AND mint = %s AND strategy = %s", applied + expired)
        gone = set(expired)
        todo = [r for r in rows if r["ts"].timestamp() + self.holds[r["strategy"]] + LABEL_LAG_S > self.applied_through and r["x"] is not None]
        fills = [r for r in waiting if (r["ts"], r["mint"], r["strategy"]) not in gone]
        for queue, rs_all in ((self.pending, todo), (self.pending_fill, fills)):
            groups: dict[tuple, list] = {}
            for r in rs_all:
                groups.setdefault((r["ts"].timestamp(), r["strategy"]), []).append(r)
            for (t, name), rs in sorted(groups.items()):
                parts = self._restored_parts(t, name, rs)
                if parts is None:
                    continue
                rs, y, u0, k, w, j = parts
                keys = [(t, r["mint"], name) for r in rs]
                if queue is self.pending:
                    self.pending.push(keys, np.full(len(rs), t), t + self.holds[name] + LABEL_LAG_S, y, u0, k, w, s=j)
                    for r in rs:
                        self.watch[r["mint"]] = self.watch.get(r["mint"], 0) + 1
                else:                                            # a traded row: it learns its fill whatever the teacher said
                    self.pending_fill.push(keys, np.full(len(rs), t), now, y, u0, k, torch.ones_like(w), s=j)
                    self.fill_base.update({key: float(r["outcome"]) for key, r in zip(keys, rs)})

    def _restored_parts(self, t: float, name: str, rs: list):
        """(rows kept, head output, MBON pre-activation, KC code, weight, strategy index) of stored rows of one minute and
        strategy, recomputed from their feature vectors; None when the teacher scores none of them."""
        j = self.names.index(name)
        Xr = np.asarray([r["x"] for r in rs], dtype=np.float32)
        Y, u0, k = self.fly.parts_all(Xr[:, self.idx])
        tv = self._teacher_scores(Xr, t); loc = np.arange(len(rs))
        if tv is not None:
            loc = loc[np.isfinite(tv[:, j])]
            if not len(loc):
                return None
            rs = [rs[i] for i in loc]; Xr = Xr[loc]; li = torch.as_tensor(loc, device=Y.device); Y, u0, k = Y[li], u0[li], k[li]
        y = self._head(Y[:, j], tv, loc, j, u0)
        if config.FLY_TEACHER_GATE and self.teacher_live is not None:      # gated: it learns from its teacher's trades, as live
            d = strategies.decide(self.teacher_live, Xr, X_COLS, t)
            w = torch.tensor([[float(d["strategy"][i] == name and bool(d["allow"][i])) for i in range(len(rs))]])
        elif config.FLY_TEACHER_GATE:
            w = torch.zeros((1, len(rs)))
        else:
            w = plastic.row_weights(torch.tensor([[float(r["score"]) for r in rs]]), torch.tensor([float(rs[0]["line"])]))
        return rs, y, u0, k, w, j

    def maybe_reload(self) -> bool:
        """A newer deployable bootstrap (a re-bootstrap) replaces this one at once: D = 0, pending tags dropped, the book carries on.
        Another chain's book follows its brain (the Solana book reloads first)."""
        if self.brain is not None:
            if self.brain.boot_id == self.boot_id:
                return False
            self._write_state(); self._load_bootstrap(None)
            return True
        from ..train import fly_replay
        with transaction() as conn:
            boot = fly_selector.latest_deployable(conn)
        v = fly_replay.verdict()
        if boot is None or v is None or not v.get("passed") or int(boot["id"]) == self.boot_id:
            return False
        old = self.boot_id; self.verdict = v
        self._write_state()
        self._load_bootstrap(boot)
        with transaction() as conn:
            conn.execute("UPDATE runs SET brain_snapshot_id = %s WHERE run_id = %s", (self.boot_id, self.run_id))
        self._write_state()
        record_event("info", "fly", f"switched to fly bootstrap #{self.boot_id}", {"from": old, "to": self.boot_id, "lines": self.lines["plastic"]})
        return True

    # ---- engine interface ----
    def on_bars(self, t_start: float, bars: dict) -> None:
        for m in self.watch:
            b = bars.get(m)
            if b is not None:
                q = self.bars.setdefault(m, deque(maxlen=BAR_KEEP))
                if not q or q[-1][0] < b[0]:
                    q.append(b)

    def open_mints(self, conn) -> set[str]:
        return {p["mint"] for p in ledger.open_positions(conn, self.BOOK)}

    LIVE = True                  # the engine runs this book's minute before any other book's (agent/minute_engine.py)

    def on_minute(self, ctx) -> dict:
        """The minute's live part (live money first, CLAUDE.md): operator commands, the fly's scores, its paper book's plan
        (agent/paper_trading.plan_minute) and the wallet's orders on it (``_go_live``). Nothing else of the minute runs
        before them: ``after_minute`` books the paper trades, learns and writes the rest once every chain's live orders
        are out."""
        conn = ctx.conn; self._phase = None
        self._commands(conn, ctx.m1_epoch)
        scored = self._score(conn, ctx)
        plan = live = None
        if ctx.trade and ctx.fresh:
            if self.race_started_at is None:
                self.race_started_at = ctx.m1_epoch
            self.beat_no += 1
            d = scored["decision"]
            plan = paper_trading.plan_minute(ctx, book=self.BOOK, run_id=self.run_id, beat_no=self.beat_no, broker=self.broker, kind=self.KIND, mints=ctx.mints,
                                             infos=ctx.infos, scores=d["score"], threshold=d["threshold"], table=None, horizon_s=self.holds[self.names[0]],
                                             holds=d["hold_s"], strategies=d["strategy"], tables=d["tables"], flat=config.FLY_SIZING == "flat",
                                             allow=d.get("allow"), reasons=d.get("reason"))
            if self.trades_live(conn):
                live = self._go_live(ctx, plan["beat_id"], plan["entries"])
        self._phase = {"m1": ctx.m1, "picks": scored["picks"], "plan": plan, "live": live}
        return {"minute": ctx.m1.isoformat(), "picks": scored["picks"], "live": live}

    def after_minute(self, ctx) -> dict:
        """The rest of the minute, once every chain's live orders are out (agent/minute_engine.py): the scored rows stored,
        the paper trades booked (agent/paper_trading.book_minute), outcomes learned, the daily calibration, the live book's
        bookkeeping, the hourly checks, the state and the status."""
        conn = ctx.conn
        ph = self._phase if self._phase is not None and self._phase["m1"] == ctx.m1 else {"picks": 0, "plan": None, "live": None}
        self._phase = None
        self._store_scored(conn)
        st = paper_trading.book_minute(ctx, ph["plan"]) if ph["plan"] is not None else None
        learned = self._learn(conn, ctx.m1_epoch)
        day = int(ctx.t_start // 86400)
        if self.last_day is None:
            self.last_day = day
        elif day > self.last_day:
            self.last_day = day; self._calibrate(conn, ctx.m1_epoch)
        drift = {k: float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0]) for j, k in enumerate(self.names)}
        out = {"minute": ctx.m1.isoformat(), "eligible": len(ctx.mints), "picks": ph["picks"], "lines": self.lines["plastic"], "frozen_lines": self.lines["frozen"],
               "line": self.lines["plastic"][self.names[0]], "frozen_line": self.lines["frozen"][self.names[0]],
               "drift": max(drift.values()) if drift else 0.0, "drift_by_strategy": drift, "pending": len(self.pending), "learned": learned,
               "learning_frozen": self.learning_frozen, "channels": self.cfg, "holds_min": {k: v / 60.0 for k, v in self.holds.items()}, "bootstrap": self.boot_id,
               "device": str(self.fly.net.dev)}
        if st is not None:
            out.update({k: v for k, v in st.items() if k != "entries"}); out["stage"] = "trading"
            if ph["live"] is not None:
                out["live"] = {**ph["live"], **self._live_after(ctx, st["beat_id"])}
        else:
            out["stage"] = "learning (not trading this minute)" if ctx.trade else "catching up"
        hour = int(ctx.m1_epoch // 3600)
        if self.last_hour is None:
            self.last_hour = hour
        elif hour > self.last_hour:
            self.last_hour = hour; self._hourly(conn, ctx.m1_epoch)
        self._write_capture()
        self._write_state()
        if learned.get("keys"):          # a row still waiting for its live fill keeps its features (tags survive restarts)
            with conn.cursor() as cur:
                cur.executemany("UPDATE fly_scored SET label = %s, state = %s, outcome = %s, resolved_at = now(), x = CASE WHEN %s = 'awaiting_fill' THEN x END "
                                "WHERE ts = %s AND mint = %s AND strategy = %s", [(lb, st_, oc, st_, ts_, m, s_) for lb, st_, oc, ts_, m, s_ in learned.pop("keys")])
        if learned.get("fill_keys"):
            with conn.cursor() as cur:
                cur.executemany("UPDATE fly_scored SET fill = %s, state = 'resolved', resolved_at = now(), x = NULL WHERE ts = %s AND mint = %s AND strategy = %s",
                                learned.pop("fill_keys"))
        learned.pop("keys", None); learned.pop("fill_keys", None)
        out["checks"] = self.last_checks; out["updated_at"] = datetime.now(timezone.utc).isoformat()
        out["race_started_at"] = self.race_started_at          # the console's "racing for N days" (the race outlives a paper reset)
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (self.status_key, json.dumps({**out, "chain": self.chain}, default=str)))
        return out

    def trades_live(self, conn) -> bool:
        """Mirror this minute on the wallet: the runner may sign, and the fly holds the seat or LIVE_WITHOUT_HANDOVER says
        live money trades whoever holds it (the signing gates in ``_live`` still apply)."""
        return bool(self.live) and (config.LIVE_WITHOUT_HANDOVER or handover.state(conn, self.chain) is not None)

    def _signing(self) -> str | None:
        """Why the wallet may not sign now (None: it may)."""
        if self.chain != "sol":
            missing = config.rh_live_prerequisites_missing()
            return ", ".join(missing) if missing else None
        from ..chain.cluster_guard import signing_allowed
        return None if signing_allowed() else "signing is not allowed (LIVE_ENABLED=1 and the live prerequisites)"

    def _go_live(self, ctx, beat_id: int, entries: list) -> dict:
        """The wallet's orders of this minute, in their own savepoint: a wallet RPC timeout must not roll back the paper plan."""
        try:
            with ctx.conn.transaction():
                return self._live(ctx, beat_id, entries)
        except Exception as e:
            log.exception("%s: live minute %s failed; the paper minute stands", self.name, ctx.m1.isoformat())
            record_event("error", "fly_live", f"live minute failed: {type(e).__name__}: {str(e)[:160]}", {"chain": self.chain, "minute": ctx.m1.isoformat()})
            return {"stage": f"live minute failed: {type(e).__name__}"}

    def _live(self, ctx, beat_id: int, entries: list) -> dict:
        """Mirror this minute's entries on the bot wallet (agent/fly_live.py; Robinhood Chain: the ETH wallet, rh/live.py)
        while signing is allowed."""
        why = self._signing()
        if why:
            return {"stage": f"live gated: {why}"}
        if self.mirror is None:
            if self.chain != "sol":
                from ..rh.live import RhLiveMirror
                self.mirror = RhLiveMirror(self.H)
                record_event("info", "rh_live", "the RH fly trades the ETH wallet", {"run_id": self.run_id})
            else:
                from .fly_live import LiveMirror
                self.mirror = LiveMirror(self.H)
                record_event("info", "fly_live", "the fly trades the bot wallet", {"run_id": self.run_id})
        if self.chain != "sol":
            return self.mirror.minute(ctx, run_id=self.run_id, beat_id=beat_id, entries=entries)
        return self.mirror.minute(ctx, run_id=self.run_id, beat_id=beat_id, entries=entries, line=self.lines["plastic"][self.names[0]],
                                  table=self.sizing["plastic"][self.names[0]])

    def _live_after(self, ctx, beat_id: int) -> dict:
        """The live book's bookkeeping of the minute (dead-bags, the gap monitor), in its own savepoint."""
        if self.mirror is None:
            return {}
        try:
            with ctx.conn.transaction():
                return self.mirror.after(ctx.conn, ctx, run_id=self.run_id, beat_id=beat_id)
        except Exception as e:
            log.exception("%s: live bookkeeping of minute %s failed", self.name, ctx.m1.isoformat())
            record_event("error", "fly_live", f"live bookkeeping failed: {type(e).__name__}: {str(e)[:160]}", {"chain": self.chain, "minute": ctx.m1.isoformat()})
            return {"bookkeeping": f"failed: {type(e).__name__}"}

    def live_exits(self, m1_epoch: float) -> dict | None:
        """The wallet's sales due at the minute boundary ``m1_epoch``, sent from the live clock's thread the moment it passes
        (agent/live_first.py): before the stream has written the minute, whatever the engine is busy with."""
        mirror = self.mirror
        if mirror is None or self.done or self._signing():
            return None
        with transaction() as conn:
            if not self.trades_live(conn):
                return None
            return mirror.exits(conn, m1=datetime.fromtimestamp(m1_epoch, timezone.utc), run_id=self.run_id)

    def idle(self) -> None:
        """Between minutes: the live book's bookkeeping that waits for an idle wallet (agent/live_first.py)."""
        if self.mirror is not None and not self.done:
            self.mirror.idle()

    def finish(self) -> None:
        if self.mirror is not None:
            self.mirror.stop()
        try:
            self._write_state()
        except Exception:
            log.exception("fly state write failed at finish")
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = 'finished' WHERE run_id = %s", (self.run_id,))
        record_event("info", "fly", "fly session stopped", {"run_id": self.run_id, "minutes": self.beat_no})

    # ---- learning ----
    @staticmethod
    def _empty_stats() -> dict:
        return {"n": 0, "sum_delta": 0.0, "sum_abs": 0.0, "step": 0.0, "capped": 0, "unknown": 0, "fill_n": 0, "fill_sum_delta": 0.0}

    def _learn(self, conn, now: float) -> dict:
        """Due tags teach the outcome pathway what their rows made: the book's realized return of a row it traded (a
        position still open waits for its sale, up to ``OUTCOME_WAIT_S``), else the label. A row the wallet bought too
        then teaches the fill pathway what the real fills added (live − outcome): at once when the live position has
        closed, else from ``pending_fill`` once it does (never, past ``FILL_WAIT_S``)."""
        tags = self.pending.pop_due(now, device=self.bank.dev)
        waiting = self.pending_fill.pop_due(now, device=self.bank.dev)
        stats: dict = {"n": 0, "unknown": 0}; keys: list = []; fill_keys: list = []
        fill_on = bool(self.bank.fill.any())
        if tags is not None:
            book, live = self._outcomes(conn, tags.keys)
            later = np.array([book.get(key, (None,))[0] == "open" and key[0] + self.holds[key[2]] + OUTCOME_WAIT_S > now for key in tags.keys], dtype=bool)
            if config.FLY_OUTCOME_FROM_BOOK and later.any():     # its book has not sold it yet: its outcome next minute
                h_ = tags.subset(later); self.pending.push(h_.keys, h_.ts, now + FILL_RETRY_S, h_.y_dn, h_.u0, h_.k, h_.w, s=h_.s)
                tags = tags.subset(~later)
        if tags is not None and len(tags):
            labels = np.array([np.nan if (lb := resolve_label(list(self.bars.get(m, ())), t, self.holds[name])) is None else lb for t, m, name in tags.keys], dtype=np.float64)
            r = np.array([b[1] if config.FLY_OUTCOME_FROM_BOOK and (b := book.get(key, (None, None)))[0] == "closed" and b[1] is not None else lb
                          for key, lb in zip(tags.keys, labels)], dtype=np.float64)
            ok = np.isfinite(r)
            for t, m, _ in tags.keys:
                c = self.watch.get(m, 0) - 1
                if c > 0:
                    self.watch[m] = c
                else:
                    self.watch.pop(m, None); self.bars.pop(m, None)
            stats.update(n=int(ok.sum()), unknown=int((~ok).sum()))
            if ok.any() and (st := self._teach(tags.subset(ok), r[ok], now, "outcome")) is not None:
                stats.update(mean_delta=st["mean_delta"][0], mean_abs_delta=st["mean_abs_delta"][0], step=st["step"][0], capped=bool(st["capped"][0]))
                h = self.hour_stats; h["n"] += stats["n"]; h["sum_delta"] += st["mean_delta"][0] * stats["n"]; h["sum_abs"] += st["mean_abs_delta"][0] * stats["n"]
                h["step"] += st["step"][0]; h["capped"] += int(st["capped"][0])
            self.hour_stats["unknown"] += stats["unknown"]
            lv = [live.get(key, (None, None)) for key in tags.keys]
            now_fill = np.array([fill_on and ok[i] and lv[i][0] == "closed" and lv[i][1] is not None for i in range(len(tags))], dtype=bool)
            wait = np.array([fill_on and ok[i] and lv[i][0] == "open" for i in range(len(tags))], dtype=bool)
            if wait.any():                                       # the wallet has not sold it yet: its fill when it has
                w_ = tags.subset(wait); self.pending_fill.push(w_.keys, w_.ts, now + FILL_RETRY_S, w_.y_dn, w_.u0, w_.k, w_.w, s=w_.s)
                self.fill_base.update({key: float(r[i]) for i, key in enumerate(tags.keys) if wait[i]})
            if now_fill.any():
                res = np.array([lv[i][1] - r[i] for i in range(len(tags)) if now_fill[i]])
                self._teach_fill(tags.subset(now_fill), res, now, stats)
                fill_keys += [(float(v), datetime.fromtimestamp(key[0], timezone.utc), key[1], key[2]) for key, v in zip([k_ for k_, f in zip(tags.keys, now_fill) if f], res)]
            keys = [(float(lb) if np.isfinite(lb) else None, ("awaiting_fill" if wait[i] else "resolved") if ok[i] else "unknown", float(r[i]) if ok[i] else None,
                     datetime.fromtimestamp(t, timezone.utc), m, name) for i, ((t, m, name), lb) in enumerate(zip(tags.keys, labels))]
        if waiting is not None:
            _, live = self._outcomes(conn, waiting.keys)
            lv = [live.get(key, (None, None)) for key in waiting.keys]; base = [self.fill_base.get(key) for key in waiting.keys]
            done = np.array([b is not None and v[0] == "closed" and v[1] is not None for v, b in zip(lv, base)], dtype=bool)
            again = np.array([b is not None and v[0] == "open" and key[0] + self.holds[key[2]] + FILL_WAIT_S > now for key, v, b in zip(waiting.keys, lv, base)], dtype=bool)
            if again.any():
                a_ = waiting.subset(again); self.pending_fill.push(a_.keys, a_.ts, now + FILL_RETRY_S, a_.y_dn, a_.u0, a_.k, a_.w, s=a_.s)
            res = np.array([v[1] - b for v, b, d in zip(lv, base, done) if d])
            if done.any():
                self._teach_fill(waiting.subset(done), res, now, stats)
            fills = iter(res)
            for i, key in enumerate(waiting.keys):
                if not again[i]:
                    fill_keys.append((float(next(fills)) if done[i] else None, datetime.fromtimestamp(key[0], timezone.utc), key[1], key[2]))
                    self.fill_base.pop(key, None)
        if tags is None and waiting is None:
            return {"n": 0}
        self.applied_through = now
        stats["keys"] = keys; stats["fill_keys"] = fill_keys
        return stats

    def _teach(self, tags, r: np.ndarray, now: float, pathway: str) -> dict | None:
        """One update of ``pathway`` (None while learning is frozen: the memory only decays)."""
        if self.learning_frozen:
            self.bank.decay_to(now)
            return None
        return self.bank.update(tags, torch.tensor(r, dtype=torch.float32), now, pathway=pathway)

    def _teach_fill(self, tags, res: np.ndarray, now: float, stats: dict) -> None:
        """The fill pathway learns the live fill residuals ``res`` of ``tags`` (rows the wallet traded: weight 1)."""
        tags.w = torch.ones_like(tags.w)
        st = self._teach(tags, res, now, "fill")
        stats["fill_n"] = stats.get("fill_n", 0) + len(tags)
        if st is not None:
            stats["fill_mean_delta"] = st["mean_delta"][0]
            self.hour_stats["fill_n"] += len(tags); self.hour_stats["fill_sum_delta"] += st["mean_delta"][0] * len(tags)

    def _outcomes(self, conn, keys: list) -> tuple[dict, dict]:
        """For scored rows (t, mint, strategy) this book bought: {key: (status, realized return once closed)} of its own
        position and of the wallet's (the live mirror buys under the same decision, stamped at the row's minute end). The
        return is realized / cost, as the live-vs-mirror gap counts it."""
        if not keys:
            return {}, {}
        ts = sorted({datetime.fromtimestamp(t + 60.0, timezone.utc) for t, _, _ in keys}); mints = sorted({m for _, m, _ in keys})
        rows = conn.execute("SELECT d.ts, d.mint, d.detail->>'strategy' AS strategy, p.book, p.status, p.realized_sol / NULLIF(p.cost_sol, 0) AS ret "
                            "FROM decisions d JOIN positions p ON p.entry_decision_id = d.id AND p.book = ANY(%s) "     # an aborted RH buy (no tokens, gas only) was no trade
                            "AND COALESCE(p.forced_exit_kind, '') <> 'aborted' WHERE d.kind = %s AND d.ts = ANY(%s) AND d.mint = ANY(%s)",
                            ([self.BOOK, self.market.live_book], f"{self.KIND}_enter", ts, mints)).fetchall()
        book: dict = {}; live: dict = {}
        for r in rows:
            key = (r["ts"].timestamp() - 60.0, r["mint"], r["strategy"])
            (book if r["book"] == self.BOOK else live)[key] = (r["status"], float(r["ret"]) if r["status"] == "closed" and r["ret"] is not None else None)
        return book, live

    def _score(self, conn, ctx) -> dict:
        n = len(ctx.mints); empty = {"strategy": np.full(n, None, dtype=object), "score": np.full(n, -1.0), "hold_s": np.zeros(n), "threshold": np.full(n, np.inf),
                                     "tables": [[] for _ in range(n)]}
        if not n:
            return {"decision": empty, "picks": 0}
        X = ctx.X[:, self.idx]
        Y, u0, k, H = self.fly.parts_all_h(X); trig = self.fly.triggers(ctx.X, X_COLS)
        self._capture(H, trig, ctx.t_start); del H
        if not self.nu_set:
            self.bank.estimate_nu(Y[:, 0], u0, k); self.nu_set = True
        if self.fill_choice:
            self._choose_fill(u0, k)
        t = ctx.t_start; ts = datetime.fromtimestamp(t, timezone.utc); rows = []; ta = self._teacher_allow(ctx.X, t)
        fill_on = bool(self.bank.fill.any())
        V = np.full((n, len(self.names)), -np.inf); F = np.full((n, len(self.names)), np.nan)
        gated = config.FLY_TEACHER_GATE
        if config.FLY_SCORE == "teacher" and self.teacher_live is None:      # it starts from its teacher's score: none, no score
            return {"decision": no_teacher(n), "picks": 0}
        td = self._teacher_decision(ctx) if gated and self.teacher_live is not None else None
        tv = self._teacher_scores(ctx.X, t)
        for j, name in enumerate(self.names):
            loc = np.flatnonzero(trig[:, j])
            if tv is not None:
                loc = loc[np.isfinite(tv[loc, j])]                   # a candidate its teacher scores
            if not len(loc):
                continue
            li = torch.as_tensor(loc, device=Y.device)
            y = self._head(Y[li, j], tv, loc, j, u0[li])
            with torch.no_grad():
                sc, _ = self.bank.predict(y, u0[li], k[li], s=np.full(len(loc), j)); sc = sc[0]
                fr = self.bank.frozen(y, u0[li], s=np.full(len(loc), j))
                fd = self.bank.pathway_scores(y, u0[li], k[li], s=np.full(len(loc), j))[1][0].float().cpu().numpy() if fill_on else None
            v, f = sc.float().cpu().numpy(), fr.float().cpu().numpy(); V[loc, j] = v; F[loc, j] = f
            line, fline = self.lines["plastic"][name], self.lines["frozen"][name]
            al = ta.get(name)
            rows += [(ts, ctx.mints[i], name, int(self.holds[name] // 60), ctx.X[i].tolist(), float(v[q]), float(f[q]), line, fline, self.boot_id,
                      None if al is None else bool(al[i]), None if fd is None else float(fd[q])) for q, i in enumerate(loc)]
            if td is not None:   # gated: its decisions are the teacher's trades of this strategy (before its own vetoes), so it learns from those
                mine = np.array([td["strategy"][i] == name and bool(td["allow"][i]) for i in loc], dtype=np.float32)
                w = torch.as_tensor(mine[None], dtype=sc.dtype, device=sc.device)
            elif gated:          # the gate is on but its teacher is not loaded: nothing to learn from until it is
                w = torch.zeros((1, len(loc)), dtype=sc.dtype, device=sc.device)
            else:
                w = plastic.row_weights(sc[None], torch.tensor([line], dtype=sc.dtype, device=sc.device))
            self.pending.push([(t, ctx.mints[i], name) for i in loc], np.full(len(loc), t), t + self.holds[name] + LABEL_LAG_S, y, u0[li], k[li], w, s=j)
            for i in loc:
                m = ctx.mints[i]; self.watch[m] = self.watch.get(m, 0) + 1
                q_ = self.bars.setdefault(m, deque(maxlen=BAR_KEEP)); b = ctx.bars.get(m)
                if b is not None and (not q_ or q_[-1][0] < b[0]):
                    q_.append(b)
        self._scored_rows = [r + (self.chain,) for r in rows]   # stored by after_minute (_store_scored): no row write before the live orders
        if self.chain == "sol":
            self.fly.lines = dict(self.lines["plastic"]); self.fly.sizings = {k: list(v) for k, v in self.sizing["plastic"].items()}
        else:                                                    # the shared model decides this chain's rows on this chain's lines
            self.fly.chain_lines[self.chain] = dict(self.lines["plastic"]); self.fly.chain_sizings[self.chain] = {k: list(v) for k, v in self.sizing["plastic"].items()}
        if td is not None:
            d = self._gated(ctx, V, F, td)
            return {"decision": d, "picks": int(d["allow"].sum())}
        if gated:                # never the ungated fly: it bought what the selector rejected and lost (2026-10-08)
            return {"decision": no_teacher(n), "picks": 0}
        d = fly_selector.fly_decide(self.fly, ctx.X, X_COLS, np.where(np.isfinite(V), V, -np.inf))
        return {"decision": d, "picks": int(sum(1 for x in d["strategy"] if x is not None))}

    def _teacher_scores(self, X: np.ndarray, t) -> np.ndarray | None:
        """[B, S] the teacher's own score of each row per strategy (NaN off its trigger) when the fly scores as its teacher
        plus its learned correction (config.FLY_SCORE = teacher), else None."""
        if config.FLY_SCORE != "teacher" or self.teacher_live is None:
            return None
        return fly_selector.teacher_scores(self.teacher_live, X, t, self.names, X_COLS)

    def _head(self, y_own: torch.Tensor, tv: np.ndarray | None, loc: np.ndarray, j: int, u0: torch.Tensor) -> torch.Tensor:
        """The head output the bank scores with: the fly's own decoder output, or (FLY_SCORE = teacher) the one under which
        its frozen score is the teacher's score exactly (PlasticBank.head_for), so only its learning moves it."""
        if tv is None:
            return y_own
        return self.bank.head_for(torch.as_tensor(tv[loc, j], dtype=torch.float32, device=self.bank.dev), u0, s=np.full(len(loc), j))

    def _teacher_decision(self, ctx) -> dict:
        """The deployed selector's decision on this minute's rows, with its live fail-closed rules."""
        from .selector_session import fail_closed
        return fail_closed(self.teacher_groups, ctx.infos, strategies.decide(self.teacher_live, ctx.X, X_COLS, ctx.t_start))

    def _gated(self, ctx, V: np.ndarray, F: np.ndarray, td: dict | None = None) -> dict:
        """The fly's decision under its teacher: each row's strategy, trigger, line, win filter, gates, dump veto and
        fail-closed rules are the deployed selector's (agent/selector_session.fail_closed), so a fresh bootstrap trades the
        selector's set; the fly drops a teacher trade only where its learning has moved its score more than
        ``FLY_LEARNED_VETO`` below its bootstrap score. A row the teacher blocks is recorded with the teacher's reason."""
        n = len(ctx.mints)
        td = td if td is not None else self._teacher_decision(ctx)
        strat = np.full(n, None, dtype=object); score = np.full(n, -1.0); hold = np.zeros(n); thr = np.full(n, np.inf)
        tables = [[] for _ in range(n)]; allow = np.zeros(n, bool); reason = np.full(n, "", dtype=object)
        for i in range(n):
            s_ = td["strategy"][i]
            if s_ is None or s_ not in self.names:
                continue
            j = self.names.index(s_); v, f = V[i, j], F[i, j]
            # the selector's own score, line and sizing table for the row: a gated fly buy is sized exactly as the selector
            # sizes it (agent/sizing.size_position; FLY_SIZING=kelly: the same Kelly band, fraction and caps; flat: flat)
            strat[i] = s_; score[i] = float(td["score"][i]); thr[i] = float(td["threshold"][i]); hold[i] = self.holds[s_]; tables[i] = td["tables"][i]
            if not td["allow"][i]:
                reason[i] = f"selector: {td['reason'][i]}"
            elif np.isfinite(v) and np.isfinite(f) and v < f - config.FLY_LEARNED_VETO:
                reason[i] = f"learned veto: score {v:.3f} < bootstrap {f:.3f} - {config.FLY_LEARNED_VETO:g}"
            else:
                allow[i] = True; reason[i] = "trade"
        return {"strategy": strat, "score": score, "hold_s": hold, "threshold": thr, "tables": tables, "allow": allow, "reason": reason}

    def _capture(self, H: torch.Tensor, trig: np.ndarray, t: float) -> None:
        """The console's brain view: this minute's mean activity over the rows some strategy's trigger fired on (every
        row when none did), kept for ``_write_capture`` (one file per minute, brain/activity.py) after the live orders.
        Never allowed to interrupt scoring."""
        self._activity = None
        if not self.capture_ok:
            return
        try:
            sel = np.flatnonzero(trig.any(1)); n_c = int(len(sel))
            m = (H.index_select(0, torch.as_tensor(sel, device=H.device)) if n_c else H).mean(0)
            self._activity = (t, activity.quantise(m), int(len(trig)), n_c)
        except Exception:
            self.capture_ok = False
            log.exception("brain activity capture disabled for this session")

    def _write_capture(self) -> None:
        a, self._activity = self._activity, None
        if a is None or not self.capture_ok:
            return
        try:
            activity.write(activity_dir(), a[0], a[1], n_rows=a[2], n_cand=a[3], connectome=self.connectome_name)
        except Exception:
            self.capture_ok = False
            log.exception("brain activity capture disabled for this session")

    def _store_scored(self, conn) -> None:
        """The live phase's scored rows (``fly_scored``: pending rows keep the feature vector, so tags survive restarts)."""
        rows, self._scored_rows = self._scored_rows, []
        if rows:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO fly_scored (ts, mint, strategy, hold_min, x, score, frozen_score, line, frozen_line, bootstrap_id, teacher_allow, fill_score, chain) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ts, mint, strategy) DO NOTHING", rows)

    # ---- daily and hourly ----
    def _resolved(self, conn, since_resolved: float, cols: str, name: str) -> list[dict]:
        """A strategy's resolved rows whose labels became known after ``since_resolved`` (epoch s); a row still waiting for
        its live fill has its label."""
        return conn.execute(f"SELECT {cols} FROM fly_scored WHERE state IN ('resolved', 'awaiting_fill') AND label IS NOT NULL AND bootstrap_id = %s AND strategy = %s AND chain = %s "
                            "AND ts >= %s ORDER BY ts", (self.boot_id, name, self.chain,
                                                         datetime.fromtimestamp(since_resolved - self.holds[name] - LABEL_LAG_S, timezone.utc))).fetchall()

    def _calibrate(self, conn, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).date()
        rule = fly_calibrate.RULE
        for name in self.names:
            rows = self._resolved(conn, now - fly_calibrate.WINDOW_DAYS * 86400.0, "ts, mint, score, frozen_score, label, teacher_allow", name)
            target = None
            if rule == "match":
                # the teacher's trades over the same rows are the budget; rows scored before it was recorded cannot count
                rows = [r for r in rows if r["teacher_allow"] is not None]
                if not rows:
                    log.info("fly %s: no scored rows with the teacher's decision yet; the line stays %.4f", name, self.lines["plastic"][name])
                    continue
            if not rows:
                continue
            ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([r["mint"] for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
            if rule == "match":
                al = np.array([bool(r["teacher_allow"]) for r in rows]) & np.isfinite(lab)
                target = len(fly_calibrate.taken_idx(ts, mint, self.holds[name], np.flatnonzero(al)))
            for arm, col in (("plastic", "score"), ("frozen", "frozen_score")):
                cal = fly_calibrate.calibrate(ts, mint, self.holds[name], np.array([r[col] for r in rows], dtype=np.float64), lab,
                                              prev_line=self.lines[arm][name], prev_sizing=self.sizing[arm][name], rule=rule, target_trades=target)
                if cal.changed:
                    record_event("info", "fly", f"{name} {arm} line {self.lines[arm][name] * 100:+.2f}% → {cal.line * 100:+.2f}%",
                                 {"trades": cal.trades, "mean": cal.mean, "total": cal.total, "rule": rule, "teacher_trades": target})
                self.lines[arm][name], self.sizing[arm][name] = cal.line, cal.sizing
                conn.execute("INSERT INTO fly_calibrations (day, arm, line, sizing, trades, total, mean, window_days) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                             "ON CONFLICT (day, arm) DO UPDATE SET line = EXCLUDED.line, sizing = EXCLUDED.sizing, trades = EXCLUDED.trades, total = EXCLUDED.total, mean = EXCLUDED.mean",
                             (day, f"{arm}:{name}" if self.chain == "sol" else f"{arm}:{self.chain}:{name}", cal.line, json.dumps(cal.sizing), cal.trades,
                              cal.total, cal.mean, fly_calibrate.WINDOW_DAYS))

    def _drift(self, j: int) -> float:
        return float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0])

    def _checks(self, conn, now: float) -> dict:
        """The rollback checks per strategy (its own trades, its own channel's drift). They judge the outcome pathway
        against the market's labels, so its score here leaves out what the fill pathway adds (a fill veto drops trades
        whose labels are fine: counted, it would read as the plastic fly trailing its shadow); ``_fill_checks`` judges that."""
        out = {}
        for j, name in enumerate(self.names):
            rows = self._resolved(conn, now - gov.SHADOW_DAYS * 86400.0, "ts, mint, score, fill_score, line, frozen_score, frozen_line, label", name)
            if rows:
                ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([r["mint"] for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
                g = lambda c: np.array([np.nan if r[c] is None else r[c] for r in rows], dtype=np.float64)
                H = self.holds[name]; sc = g("score") - np.nan_to_num(g("fill_score"))
                rp = gov.picks_returns(ts, mint, H, sc, g("line"), lab); rf = gov.picks_returns(ts, mint, H, g("frozen_score"), g("frozen_line"), lab)
                recent = ts + H + LABEL_LAG_S > now - gov.IC_HOURS * 3600.0
                out[name] = gov.check((rp, rf), (sc[recent], lab[recent]), self._drift(j))
            else:
                out[name] = gov.check(None, None, self._drift(j))
        return out

    def _fill_checks(self, conn, now: float) -> dict:
        """The fill pathways against the fills they learned (every chain: the memory is shared): over ``FILL_IC_DAYS``, a
        strategy's fill pathway whose score changes correlate negatively with the live fill residuals that followed (at
        least ``FILL_IC_MIN`` of them) goes back to its bootstrap."""
        out: dict = {}
        if not bool(self.bank.fill.any()):
            return out
        for j, name in enumerate(self.names):
            rows = conn.execute("SELECT fill_score, fill FROM fly_scored WHERE fill IS NOT NULL AND fill_score IS NOT NULL AND bootstrap_id = %s AND strategy = %s "
                                "AND ts >= %s", (self.boot_id, name, datetime.fromtimestamp(now - FILL_IC_DAYS * 86400.0, timezone.utc))).fetchall()
            a = np.array([r["fill_score"] for r in rows], dtype=np.float64); b = np.array([r["fill"] for r in rows], dtype=np.float64)
            ic = float(np.corrcoef(a, b)[0, 1]) if len(rows) >= 3 and a.std() > 0 and b.std() > 0 else None
            out[name] = {"n": len(rows), "ic": ic, "drift": float(self.bank.drift(j, pathway="fill")[0])}
            if len(rows) >= FILL_IC_MIN and ic is not None and ic < 0 and not self.learning_frozen:
                self.bank.reset_columns(self.bank.fill[j]); out[name]["reset"] = True
                record_event("warning", "fly", f"{name} fill pathway back to its bootstrap: its fill predictions ran against the fills (IC {ic:+.2f} over {len(rows)})", out[name])
        return out

    def _hourly(self, conn, now: float) -> None:
        if self.brain is not None:
            # another chain's book: its own handover; the shared memory's checks, snapshots and rollbacks are the Solana book's
            self._handover_check(conn, now)
            return
        hour = datetime.fromtimestamp(math.floor(now / 3600) * 3600 - 3600, timezone.utc)
        checks = self._checks(conn, now); self.last_checks = checks; fills = self._fill_checks(conn, now)
        h = self.hour_stats; first = checks.get(self.names[0]) or {}
        conn.execute("INSERT INTO fly_updates (hour, n, mean_delta, mean_abs_delta, step, capped, drift, ic, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (hour) DO NOTHING",
                     (hour, h["n"], h["sum_delta"] / h["n"] if h["n"] else None, h["sum_abs"] / h["n"] if h["n"] else None, h["step"], h["capped"],
                      max((c["drift"] for c in checks.values()), default=0.0), first.get("ic"),
                      json.dumps({"unknown": h["unknown"], "checks": checks, "fill": {"n": h["fill_n"], "mean_delta": h["fill_sum_delta"] / h["fill_n"] if h["fill_n"] else None,
                                                                                     "checks": fills}}, default=str)))
        self.hour_stats = self._empty_stats()
        good = not any(c["triggers"] for c in checks.values())
        sid = self._snapshot(conn, now, good, checks)
        for name, c in checks.items():
            if c["triggers"] and now >= self.checks_paused_until.get(name, 0.0) and not self.learning_frozen:
                self._rollback(conn, now, c, sid, name=name)
        self._prune(conn, now)
        activity.prune(activity_dir(), now - activity.KEEP_S)
        self._handover_check(conn, now)

    def _snapshot(self, conn, now: float, good: bool, checks: dict) -> int:
        d = fly_state_dir(); d.mkdir(parents=True, exist_ok=True)
        path = d / f"snap_{datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M')}.pt"
        torch.save({"bank": self.bank.state(), "bootstrap_id": self.boot_id, "lines": self.lines, "sizing": self.sizing}, path)
        note = {"bootstrap_id": self.boot_id, "data": fly_selector.FLY_VERSION, "score": config.FLY_SCORE, "good": good, "checks": checks,
                "drift": max((c["drift"] for c in checks.values()), default=0.0), "ic24": {k: c.get("ic") for k, c in checks.items()}}
        return int(conn.execute("INSERT INTO brain_snapshots (run_id, path, kind, note) VALUES (%s,%s,'fly_plastic',%s) RETURNING id",
                                (self.run_id, str(path), json.dumps(note, default=str))).fetchone()["id"])

    def _commands(self, conn, now: float) -> None:
        """Operator commands from the console (``send_command``): pause / resume learning, roll back now."""
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (COMMAND_KEY,)).fetchone()
        if not r or not r["value"]:
            return
        v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])
        conn.execute("DELETE FROM ui_settings WHERE key = %s", (COMMAND_KEY,))
        cmd = v.get("cmd")
        try:
            age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(v["at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            age = 0.0
        if age > COMMAND_MAX_AGE_S:          # queued while the engine was down: an operator's click must not land days later
            record_event("warning", "fly", f"ignored a stale operator command {cmd!r} ({age / 60:.0f} min old)")
            return
        if cmd == "pause":
            self.learning_frozen = True; record_event("info", "fly", "learning paused by the operator")
        elif cmd == "resume":
            self.learning_frozen = False; self.rollbacks = {}; record_event("info", "fly", "learning resumed by the operator")
        elif cmd == "rollback":
            for j, name in enumerate(self.names):
                self._rollback(conn, now, {"triggers": ["operator"], "drift": self._drift(j)}, None, count=False, name=name)

    def _rollback(self, conn, now: float, checks: dict, from_sid: int | None, count: bool = True, name: str | None = None) -> None:
        """Restore one strategy's channel (its KC→MBON columns) from the newest good snapshot at least a day old, or the bootstrap."""
        name = name or self.names[0]; j = self.names.index(name); cols = self.bank.learn[j]
        target = None
        for r in conn.execute("SELECT id, path, note, ts FROM brain_snapshots WHERE kind = 'fly_plastic' AND ts <= %s ORDER BY id DESC",
                              (datetime.fromtimestamp(now - ROLLBACK_MIN_AGE_S, timezone.utc),)).fetchall():
            n = json.loads(r["note"] or "{}")
            ok = n.get("good") or not ((n.get("checks") or {}).get(name) or {}).get("triggers")
            if ok and n.get("bootstrap_id") == self.boot_id and n.get("score", "own") == config.FLY_SCORE and Path(r["path"]).exists():
                target = r; break
        if target is not None:
            snap = plastic.PlasticBank(self.fly.net, [(0.0, math.inf)], fly_selector.SCALE, learn=self.bank.learn.cpu().numpy(), read=self.bank.read.cpu().numpy())
            st = torch.load(target["path"], map_location="cpu", weights_only=True)["bank"]
            snap.D = torch.zeros_like(snap.D); snap.D[:, snap.mask.bool()] = st["D"].to(snap.dev).float()
            self.bank.D[0][:, cols] = snap.D[0][:, cols]
        else:
            self.bank.D[0][:, cols] = 0.0
        self.checks_paused_until[name] = now + ROLLBACK_PAUSE_S
        self.rollbacks[name] = [t for t in self.rollbacks.get(name, []) if t > now - ROLLBACK_WINDOW_S] + ([now] if count else [])
        conn.execute("INSERT INTO fly_rollbacks (reason, checks, from_snapshot, to_snapshot) VALUES (%s,%s,%s,%s)",
                     (f"{name}: " + ", ".join(checks["triggers"]), json.dumps(checks, default=str), from_sid, target["id"] if target is not None else None))
        record_event("warning", "fly", f"{name} channel rolled back ({', '.join(checks['triggers'])})", {"to_snapshot": target["id"] if target is not None else "bootstrap (D = 0)", "checks": checks})
        if count and len(self.rollbacks[name]) >= ROLLBACK_LIMIT:
            self.learning_frozen = True
            from ..train import pipeline
            pipeline.request_run(fly=True, reason=f"{name}: {len(self.rollbacks[name])} rollbacks in 7 days")
            record_event("error", "fly", f"learning frozen after {len(self.rollbacks[name])} {name} rollbacks in 7 days; re-bootstrap requested", {"rollbacks": self.rollbacks})

    def _prune(self, conn, now: float) -> None:
        rows = conn.execute("SELECT id, path, ts FROM brain_snapshots WHERE kind = 'fly_plastic' AND ts < %s ORDER BY ts DESC",
                            (datetime.fromtimestamp(now - SNAP_HOURLY_DAYS * 86400.0, timezone.utc),)).fetchall()
        kept_days: set = set(); drop = []
        for r in rows:
            d = r["ts"].date()
            if d in kept_days:
                drop.append(r)
            else:
                kept_days.add(d)
        for r in drop:
            conn.execute("DELETE FROM brain_snapshots WHERE id = %s", (r["id"],))
            try:
                Path(r["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute("DELETE FROM fly_scored WHERE state <> 'pending' AND ts < %s", (datetime.fromtimestamp(now - SCORED_KEEP_DAYS * 86400.0, timezone.utc),))

    def _handover_check(self, conn, now: float) -> None:
        """The fly takes the selector's seat once it has raced ``config.HANDOVER_DAYS`` (0: no minimum, the window is the
        whole race), closed ``config.HANDOVER_MIN_TRADES`` paper trades and, when ``config.HANDOVER_BEAT_SELECTOR``,
        realized at least the paper selector's P&L over that window."""
        days = config.HANDOVER_DAYS
        if handover.state(conn, self.chain) is not None or self.race_started_at is None or now - self.race_started_at < days * 86400.0:
            return
        since = datetime.fromtimestamp(self.race_started_at if days <= 0 else now - days * 86400.0, timezone.utc)
        fb, sb = self.BOOK, self.market.selector_book                 # this chain's race: its fly book against its selector book
        pnl = {b: conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s, count(*) AS n FROM positions WHERE book = %s AND status = 'closed' AND closed_at >= %s",
                               (b, since)).fetchone() for b in (fb, sb)}
        fly_pnl, sel_pnl, n = float(pnl[fb]["s"]), float(pnl[sb]["s"]), int(pnl[fb]["n"])
        if n >= config.HANDOVER_MIN_TRADES and (fly_pnl >= sel_pnl or not config.HANDOVER_BEAT_SELECTOR):
            detail = {"at": datetime.fromtimestamp(now, timezone.utc).isoformat(), "fly_pnl_sol": fly_pnl, "selector_pnl_sol": sel_pnl, "fly_trades": n,
                      "selector_trades": int(pnl[sb]["n"]), "days": days, "beat_selector": config.HANDOVER_BEAT_SELECTOR, "bootstrap": self.boot_id,
                      "chain": self.chain, "unit": self.market.unit}
            handover.record(conn, detail, self.chain)
            record_event("info", "handover", f"the fly takes the selector's seat: {fly_pnl:+.3f} SOL vs {sel_pnl:+.3f} SOL over {n} trades"
                         + (f" in {days} days" if days > 0 else ""), detail)
