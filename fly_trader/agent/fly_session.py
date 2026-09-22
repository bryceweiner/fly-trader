"""The plastic fly: trading its own book (``paper_fly``) on the minute engine and learning from the market every minute.

It starts once the fly design passed its replay (``train/fly_replay.verdict``, current definitions) and a deployable
bootstrap exists (``train/fly_selector.latest_deployable``); the replay's chosen configuration (α, half-life) is the
learning rate and forgetting of its mushroom body (``brain/plastic.py``). Each minute (agent/minute_engine.py):

1. bars: minute start, open, close and exit cost of every traded mint with a pending tag feed the label resolver;
2. labels: tags whose exit minute has now been seen are resolved exactly as ``train/decisions.build`` labels a row
   (``resolve_label``) and teach the mushroom body; a label that cannot be resolved is dropped;
3. calibration: at the first minute of each UTC day, the line and sizing of the plastic fly and of its frozen shadow
   (the bootstrap, D = 0) from their scores of the minutes resolved in the last seven days (``train/fly_calibrate.py``);
4. scoring: eligible minutes are scored by the plastic fly and its shadow, stored (``fly_scored``: pending rows keep the
   feature vector, so tags survive restarts) and tagged; only minutes at or above the line teach it (brain/plastic.py);
5. trading (fresh stream, trade minutes): the race rules (``agent/paper_trading.py``) at the plastic line and sizing.

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

from .. import config
from ..brain import plastic
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..ops.reset import fly_state_dir
from ..train import fly_calibrate, fly_governance as gov, fly_selector
from ..train.decisions import X_COLS
from . import handover, paper_trading

log = logging.getLogger(__name__)
BOOK, KIND = "paper_fly", "fly"
BAR_KEEP = 270               # minutes of bars kept per watched mint (the longest strategy hold, 240, + next-open window + margin)
NEXT_OPEN_S = 120.0          # entry at the next traded minute's open when it comes within two minutes (decisions.build)
JUMP = 50.0
LABEL_LAG_S = 60.0
ROLLBACK_PAUSE_S, ROLLBACK_LIMIT, ROLLBACK_WINDOW_S, ROLLBACK_MIN_AGE_S = 86400.0, 3, 7 * 86400.0, 86400.0
SNAP_HOURLY_DAYS, SCORED_KEEP_DAYS = 7, 35
COMMAND_KEY = "fly_command"


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


def try_start(live: bool = False) -> tuple["FlyBook | None", str]:
    """A FlyBook when the fly may trade, else (None, why)."""
    from ..train import fly_replay
    v = fly_replay.verdict()
    if not v:
        return None, "the fly's replay has not run on the current definitions (fly-trader fly-replay)"
    if not v.get("passed"):
        return None, f"the fly's replay did not pass: {v.get('reason')}"
    with transaction() as conn:
        boot = fly_selector.latest_deployable(conn)
    if boot is None:
        return None, "no deployable fly bootstrap (the training pipeline bootstraps one)"
    try:
        return FlyBook(boot, v, live=live), "ok"
    except Exception as e:
        log.exception("fly book failed to start")
        return None, f"fly book failed to start: {type(e).__name__}: {e}"


class FlyBook:
    name = "fly"

    def __init__(self, boot: dict, verdict: dict, live: bool = False):
        self.live = live; self.done = False; self.mirror = None
        self.verdict = verdict
        self.state_path = fly_state_dir() / "state.pt"
        self._load_bootstrap(boot)
        self.run_id = str(uuid.uuid4()); self.beat_no = 0; self.broker = PaperBroker(BOOK)
        with transaction() as conn:
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,'running')",
                         (self.run_id, KIND, json.dumps({"channels": self.cfg, "book": BOOK}, default=str), self.boot_id))
            self._restore_tags(conn)
        record_event("info", "fly", "fly session started", {"run_id": self.run_id, "bootstrap": self.boot_id, "channels": self.cfg, "lines": self.lines["plastic"],
                                                            "pending": len(self.pending), "learning_frozen": self.learning_frozen})
        log.info("fly session: bootstrap %d, channels %s, lines %s, %d pending tags", self.boot_id, self.cfg, self.lines["plastic"], len(self.pending))

    # ---- model and state ----
    def _load_bootstrap(self, boot: dict) -> None:
        self.boot_id = int(boot["id"])
        self.fly = fly_selector.load(boot["path"])
        self.idx = np.asarray([X_COLS.index(c) for c in self.fly.cols], dtype=int)
        self.names = list(self.fly.strategies); self.holds = {k: float(self.fly.rules[k]["hold_min"]) * 60.0 for k in self.names}
        self.H = max(self.holds.values())
        per = self.verdict.get("per_strategy") or {self.names[0]: {"alpha": self.verdict.get("alpha", 0.0), "half_life_days": self.verdict.get("half_life_days")}}
        self.cfg = {k: (float((per.get(k) or {}).get("alpha") or 0.0), (per.get(k) or {}).get("half_life_days")) for k in self.names}
        self.alpha = self.cfg[self.names[0]][0]; self.half_life = float(self.cfg[self.names[0]][1]) if self.cfg[self.names[0]][1] is not None else math.inf
        net = self.fly.net
        self.bank = plastic.PlasticBank(net, [(self.alpha, self.half_life)], fly_selector.SCALE, learn=net.learn.cpu().numpy(), read=net.read.cpu().numpy())
        for j, k in enumerate(self.names):                      # each strategy's channel learns and forgets at its own replay-chosen rate
            a, h = self.cfg[k]; cols = self.bank.learn[j]
            self.bank.alpha[0, cols] = a; self.bank.half_life_s[0, cols] = float(h) * 86400.0 if h is not None else float("inf")
        self.pending = plastic.PendingTags(net.n_kc, net.k_active)
        self.bars: dict[str, deque] = {}; self.watch: dict[str, int] = {}
        self.lines = {"plastic": dict(self.fly.lines), "frozen": dict(self.fly.lines)}
        self.sizing = {"plastic": {k: list(v) for k, v in self.fly.sizings.items()}, "frozen": {k: list(v) for k, v in self.fly.sizings.items()}}
        self.applied_through = -math.inf; self.learning_frozen = False; self.checks_paused_until = {k: 0.0 for k in self.names}
        self.race_started_at: float | None = None; self.last_day: int | None = None; self.last_hour: int | None = None
        self.rollbacks: dict = {k: [] for k in self.names}; self.nu_set = False; self.hour_stats = self._empty_stats(); self.last_checks: dict = {}
        s = self._read_state()
        if s and s.get("bootstrap_id") == self.boot_id and s.get("cfg") == self.cfg:
            self.bank.load_state(s["bank"]); self.nu_set = True
            self.lines, self.sizing = s["lines"], s["sizing"]
            self.applied_through, self.learning_frozen, self.checks_paused_until = s["applied_through"], s["learning_frozen"], s["checks_paused_until"]
            self.race_started_at, self.last_day, self.last_hour, self.rollbacks = s["race_started_at"], s["last_day"], s["last_hour"], s["rollbacks"]
        elif s:
            self.race_started_at = s.get("race_started_at")        # a new bootstrap keeps racing the same book
            with transaction() as conn:
                conn.execute("UPDATE fly_scored SET state = 'dropped', x = NULL WHERE state = 'pending' AND bootstrap_id IS DISTINCT FROM %s", (self.boot_id,))

    def _read_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            return torch.load(self.state_path, map_location="cpu", weights_only=False)
        except Exception:
            log.exception("fly state unreadable; starting from the bootstrap")
            return None

    def _state(self) -> dict:
        return {"bootstrap_id": self.boot_id, "cfg": self.cfg, "bank": self.bank.state(), "lines": self.lines, "sizing": self.sizing,
                "applied_through": self.applied_through, "learning_frozen": self.learning_frozen, "checks_paused_until": self.checks_paused_until,
                "race_started_at": self.race_started_at, "last_day": self.last_day, "last_hour": self.last_hour, "rollbacks": self.rollbacks,
                "saved_at": time.time()}

    def _write_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        torch.save(self._state(), tmp); os.replace(tmp, self.state_path)

    def _restore_tags(self, conn) -> None:
        rows = conn.execute("SELECT ts, mint, strategy, x, score, line FROM fly_scored WHERE state = 'pending' AND bootstrap_id = %s ORDER BY ts", (self.boot_id,)).fetchall()
        rows = [r for r in rows if r["strategy"] in self.holds]
        applied = [(r["ts"], r["mint"], r["strategy"]) for r in rows if r["ts"].timestamp() + self.holds[r["strategy"]] + LABEL_LAG_S <= self.applied_through]
        if applied:
            with conn.cursor() as cur:
                cur.executemany("UPDATE fly_scored SET state = 'resolved', resolved_at = now(), x = NULL WHERE ts = %s AND mint = %s AND strategy = %s", applied)
        todo = [r for r in rows if r["ts"].timestamp() + self.holds[r["strategy"]] + LABEL_LAG_S > self.applied_through and r["x"] is not None]
        groups: dict[tuple, list] = {}
        for r in todo:
            groups.setdefault((r["ts"].timestamp(), r["strategy"]), []).append(r)
        for (t, name), rs in sorted(groups.items()):
            j = self.names.index(name)
            Y, u0, k = self.fly.parts_all(np.asarray([r["x"] for r in rs], dtype=np.float32))
            w = plastic.row_weights(torch.tensor([[float(r["score"]) for r in rs]]), torch.tensor([float(rs[0]["line"])]))
            self.pending.push([(t, r["mint"], name) for r in rs], np.full(len(rs), t), t + self.holds[name] + LABEL_LAG_S, Y[:, j], u0, k, w, s=j)
            for r in rs:
                self.watch[r["mint"]] = self.watch.get(r["mint"], 0) + 1

    def maybe_reload(self) -> bool:
        """A newer deployable bootstrap (a re-bootstrap) replaces this one at once: D = 0, pending tags dropped, the book carries on."""
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
        return {p["mint"] for p in ledger.open_positions(conn, BOOK)}

    def on_minute(self, ctx) -> dict:
        conn, t_now = ctx.conn, ctx.t_start
        self._commands(conn, ctx.m1_epoch)
        learned = self._learn(conn, ctx.m1_epoch)
        day = int(t_now // 86400)
        if self.last_day is None:
            self.last_day = day
        elif day > self.last_day:
            self.last_day = day; self._calibrate(conn, ctx.m1_epoch)
        scored = self._score(conn, ctx)
        drift = {k: float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0]) for j, k in enumerate(self.names)}
        out = {"minute": ctx.m1.isoformat(), "eligible": len(ctx.mints), "picks": scored["picks"], "lines": self.lines["plastic"], "frozen_lines": self.lines["frozen"],
               "line": self.lines["plastic"][self.names[0]], "frozen_line": self.lines["frozen"][self.names[0]],
               "drift": max(drift.values()) if drift else 0.0, "drift_by_strategy": drift, "pending": len(self.pending), "learned": learned,
               "learning_frozen": self.learning_frozen, "channels": self.cfg, "holds_min": {k: v / 60.0 for k, v in self.holds.items()}, "bootstrap": self.boot_id}
        if ctx.trade and ctx.fresh:
            if self.race_started_at is None:
                self.race_started_at = ctx.m1_epoch
            self.beat_no += 1
            d = scored["decision"]
            st = paper_trading.trade_minute(ctx, book=BOOK, run_id=self.run_id, beat_no=self.beat_no, broker=self.broker, kind=KIND, mints=ctx.mints, infos=ctx.infos,
                                            scores=d["score"], threshold=d["threshold"], table=None, horizon_s=self.holds[self.names[0]], holds=d["hold_s"],
                                            strategies=d["strategy"], tables=d["tables"])
            out.update({k: v for k, v in st.items() if k != "entries"}); out["stage"] = "trading"
            if self.live and handover.state(conn) is not None:
                out["live"] = self._live(ctx, st)
        else:
            out["stage"] = "learning (not trading this minute)" if ctx.trade else "catching up"
        hour = int(ctx.m1_epoch // 3600)
        if self.last_hour is None:
            self.last_hour = hour
        elif hour > self.last_hour:
            self.last_hour = hour; self._hourly(conn, ctx.m1_epoch)
        self._write_state()
        if learned.get("keys"):
            with conn.cursor() as cur:
                cur.executemany("UPDATE fly_scored SET label = %s, state = %s, resolved_at = now(), x = NULL WHERE ts = %s AND mint = %s AND strategy = %s", learned.pop("keys"))
        learned.pop("keys", None)
        out["checks"] = self.last_checks; out["updated_at"] = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('fly_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps(out, default=str),))
        return out

    def _live(self, ctx, st: dict) -> dict:
        """The fly holds the seat: mirror this minute on the bot wallet (agent/fly_live.py) while signing is allowed."""
        from ..chain.cluster_guard import signing_allowed
        if not signing_allowed():
            return {"stage": "live gated: signing is not allowed (LIVE_ENABLED=1 and the live prerequisites)"}
        if self.mirror is None:
            from .fly_live import LiveMirror
            self.mirror = LiveMirror(self.H)
            record_event("info", "fly_live", "the fly trades the bot wallet", {"run_id": self.run_id})
        return self.mirror.minute(ctx, run_id=self.run_id, beat_id=st["beat_id"], entries=st["entries"], line=self.lines["plastic"][self.names[0]],
                                  table=self.sizing["plastic"][self.names[0]])

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
        return {"n": 0, "sum_delta": 0.0, "sum_abs": 0.0, "step": 0.0, "capped": 0, "unknown": 0}

    def _learn(self, conn, now: float) -> dict:
        tags = self.pending.pop_due(now, device=self.bank.dev)
        if tags is None:
            return {"n": 0}
        labels = np.array([np.nan if (lb := resolve_label(list(self.bars.get(m, ())), t, self.holds[name])) is None else lb for t, m, name in tags.keys], dtype=np.float64)
        ok = np.isfinite(labels)
        for t, m, _ in tags.keys:
            c = self.watch.get(m, 0) - 1
            if c > 0:
                self.watch[m] = c
            else:
                self.watch.pop(m, None); self.bars.pop(m, None)
        stats = {"n": int(ok.sum()), "unknown": int((~ok).sum())}
        if ok.any():
            tv = tags.subset(ok)
            if self.learning_frozen:
                self.bank.decay_to(now)
            else:
                st = self.bank.update(tv, torch.tensor(labels[ok], dtype=torch.float32), now)
                stats.update(mean_delta=st["mean_delta"][0], mean_abs_delta=st["mean_abs_delta"][0], step=st["step"][0], capped=bool(st["capped"][0]))
                h = self.hour_stats; h["n"] += stats["n"]; h["sum_delta"] += st["mean_delta"][0] * stats["n"]; h["sum_abs"] += st["mean_abs_delta"][0] * stats["n"]
                h["step"] += st["step"][0]; h["capped"] += int(st["capped"][0])
        self.hour_stats["unknown"] += stats["unknown"]
        self.applied_through = now
        stats["keys"] = [(float(lb) if np.isfinite(lb) else None, "resolved" if np.isfinite(lb) else "unknown", datetime.fromtimestamp(t, timezone.utc), m, name)
                         for (t, m, name), lb in zip(tags.keys, labels)]
        return stats

    def _score(self, conn, ctx) -> dict:
        n = len(ctx.mints); empty = {"strategy": np.full(n, None, dtype=object), "score": np.full(n, -1.0), "hold_s": np.zeros(n), "threshold": np.full(n, np.inf),
                                     "tables": [[] for _ in range(n)]}
        if not n:
            return {"decision": empty, "picks": 0}
        X = ctx.X[:, self.idx]
        Y, u0, k = self.fly.parts_all(X); trig = self.fly.triggers(ctx.X, X_COLS)
        if not self.nu_set:
            self.bank.estimate_nu(Y[:, 0], u0, k); self.nu_set = True
        t = ctx.t_start; ts = datetime.fromtimestamp(t, timezone.utc); rows = []
        V = np.full((n, len(self.names)), -np.inf)
        for j, name in enumerate(self.names):
            loc = np.flatnonzero(trig[:, j])
            if not len(loc):
                continue
            li = torch.as_tensor(loc, device=Y.device)
            with torch.no_grad():
                sc, _ = self.bank.predict(Y[li, j], u0[li], k[li], s=np.full(len(loc), j)); sc = sc[0]
                fr = self.bank.frozen(Y[li, j], u0[li], s=np.full(len(loc), j))
            v, f = sc.float().cpu().numpy(), fr.float().cpu().numpy(); V[loc, j] = v
            line, fline = self.lines["plastic"][name], self.lines["frozen"][name]
            rows += [(ts, ctx.mints[i], name, int(self.holds[name] // 60), ctx.X[i].tolist(), float(v[q]), float(f[q]), line, fline, self.boot_id) for q, i in enumerate(loc)]
            w = plastic.row_weights(sc[None], torch.tensor([line], dtype=sc.dtype, device=sc.device))
            self.pending.push([(t, ctx.mints[i], name) for i in loc], np.full(len(loc), t), t + self.holds[name] + LABEL_LAG_S, Y[li, j], u0[li], k[li], w, s=j)
            for i in loc:
                m = ctx.mints[i]; self.watch[m] = self.watch.get(m, 0) + 1
                q_ = self.bars.setdefault(m, deque(maxlen=BAR_KEEP)); b = ctx.bars.get(m)
                if b is not None and (not q_ or q_[-1][0] < b[0]):
                    q_.append(b)
        if rows:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO fly_scored (ts, mint, strategy, hold_min, x, score, frozen_score, line, frozen_line, bootstrap_id) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ts, mint, strategy) DO NOTHING", rows)
        self.fly.lines = dict(self.lines["plastic"]); self.fly.sizings = {k: list(v) for k, v in self.sizing["plastic"].items()}
        d = fly_selector.fly_decide(self.fly, ctx.X, X_COLS, np.where(np.isfinite(V), V, -np.inf))
        return {"decision": d, "picks": int(sum(1 for x in d["strategy"] if x is not None))}

    # ---- daily and hourly ----
    def _resolved(self, conn, since_resolved: float, cols: str, name: str) -> list[dict]:
        """A strategy's resolved rows whose labels became known after ``since_resolved`` (epoch s)."""
        return conn.execute(f"SELECT {cols} FROM fly_scored WHERE state = 'resolved' AND label IS NOT NULL AND bootstrap_id = %s AND strategy = %s AND ts >= %s ORDER BY ts",
                            (self.boot_id, name, datetime.fromtimestamp(since_resolved - self.holds[name] - LABEL_LAG_S, timezone.utc))).fetchall()

    def _calibrate(self, conn, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).date()
        for name in self.names:
            rows = self._resolved(conn, now - fly_calibrate.WINDOW_DAYS * 86400.0, "ts, mint, score, frozen_score, label", name)
            if not rows:
                continue
            ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([r["mint"] for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
            for arm, col in (("plastic", "score"), ("frozen", "frozen_score")):
                cal = fly_calibrate.calibrate(ts, mint, self.holds[name], np.array([r[col] for r in rows], dtype=np.float64), lab,
                                              prev_line=self.lines[arm][name], prev_sizing=self.sizing[arm][name])
                if cal.changed:
                    record_event("info", "fly", f"{name} {arm} line {self.lines[arm][name] * 100:+.2f}% → {cal.line * 100:+.2f}%", {"trades": cal.trades, "mean": cal.mean, "total": cal.total})
                self.lines[arm][name], self.sizing[arm][name] = cal.line, cal.sizing
                conn.execute("INSERT INTO fly_calibrations (day, arm, line, sizing, trades, total, mean, window_days) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                             "ON CONFLICT (day, arm) DO UPDATE SET line = EXCLUDED.line, sizing = EXCLUDED.sizing, trades = EXCLUDED.trades, total = EXCLUDED.total, mean = EXCLUDED.mean",
                             (day, f"{arm}:{name}", cal.line, json.dumps(cal.sizing), cal.trades, cal.total, cal.mean, fly_calibrate.WINDOW_DAYS))

    def _drift(self, j: int) -> float:
        return float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0])

    def _checks(self, conn, now: float) -> dict:
        """The rollback checks per strategy (its own trades, its own channel's drift)."""
        out = {}
        for j, name in enumerate(self.names):
            rows = self._resolved(conn, now - gov.SHADOW_DAYS * 86400.0, "ts, mint, score, line, frozen_score, frozen_line, label", name)
            if rows:
                ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([r["mint"] for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
                g = lambda c: np.array([r[c] for r in rows], dtype=np.float64)
                H = self.holds[name]
                rp = gov.picks_returns(ts, mint, H, g("score"), g("line"), lab); rf = gov.picks_returns(ts, mint, H, g("frozen_score"), g("frozen_line"), lab)
                recent = ts + H + LABEL_LAG_S > now - gov.IC_HOURS * 3600.0
                out[name] = gov.check((rp, rf), (g("score")[recent], lab[recent]), self._drift(j))
            else:
                out[name] = gov.check(None, None, self._drift(j))
        return out

    def _hourly(self, conn, now: float) -> None:
        hour = datetime.fromtimestamp(math.floor(now / 3600) * 3600 - 3600, timezone.utc)
        checks = self._checks(conn, now); self.last_checks = checks
        h = self.hour_stats; first = checks.get(self.names[0]) or {}
        conn.execute("INSERT INTO fly_updates (hour, n, mean_delta, mean_abs_delta, step, capped, drift, ic, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (hour) DO NOTHING",
                     (hour, h["n"], h["sum_delta"] / h["n"] if h["n"] else None, h["sum_abs"] / h["n"] if h["n"] else None, h["step"], h["capped"],
                      max((c["drift"] for c in checks.values()), default=0.0), first.get("ic"), json.dumps({"unknown": h["unknown"], "checks": checks}, default=str)))
        self.hour_stats = self._empty_stats()
        good = not any(c["triggers"] for c in checks.values())
        sid = self._snapshot(conn, now, good, checks)
        for name, c in checks.items():
            if c["triggers"] and now >= self.checks_paused_until.get(name, 0.0) and not self.learning_frozen:
                self._rollback(conn, now, c, sid, name=name)
        self._prune(conn, now)
        self._handover_check(conn, now)

    def _snapshot(self, conn, now: float, good: bool, checks: dict) -> int:
        d = fly_state_dir(); d.mkdir(parents=True, exist_ok=True)
        path = d / f"snap_{datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M')}.pt"
        torch.save({"bank": self.bank.state(), "bootstrap_id": self.boot_id, "lines": self.lines, "sizing": self.sizing}, path)
        note = {"bootstrap_id": self.boot_id, "data": fly_selector.FLY_VERSION, "good": good, "checks": checks,
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
            if ok and n.get("bootstrap_id") == self.boot_id and Path(r["path"]).exists():
                target = r; break
        if target is not None:
            snap = plastic.PlasticBank(self.fly.net, [(0.0, math.inf)], fly_selector.SCALE, learn=self.bank.learn.cpu().numpy(), read=self.bank.read.cpu().numpy())
            st = torch.load(target["path"], map_location="cpu", weights_only=False)["bank"]
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
        if handover.state(conn) is not None or self.race_started_at is None or now - self.race_started_at < days * 86400.0:
            return
        since = datetime.fromtimestamp(self.race_started_at if days <= 0 else now - days * 86400.0, timezone.utc)
        pnl = {b: conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s, count(*) AS n FROM positions WHERE book = %s AND status = 'closed' AND closed_at >= %s",
                               (b, since)).fetchone() for b in (BOOK, "paper_selector")}
        fly_pnl, sel_pnl, n = float(pnl[BOOK]["s"]), float(pnl["paper_selector"]["s"]), int(pnl[BOOK]["n"])
        if n >= config.HANDOVER_MIN_TRADES and (fly_pnl >= sel_pnl or not config.HANDOVER_BEAT_SELECTOR):
            detail = {"at": datetime.fromtimestamp(now, timezone.utc).isoformat(), "fly_pnl_sol": fly_pnl, "selector_pnl_sol": sel_pnl, "fly_trades": n,
                      "selector_trades": int(pnl["paper_selector"]["n"]), "days": days, "beat_selector": config.HANDOVER_BEAT_SELECTOR, "bootstrap": self.boot_id}
            handover.record(conn, detail)
            record_event("info", "handover", f"the fly takes the selector's seat: {fly_pnl:+.3f} SOL vs {sel_pnl:+.3f} SOL over {n} trades"
                         + (f" in {days} days" if days > 0 else ""), detail)
