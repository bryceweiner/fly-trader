"""The plastic Kalshi fly: trading the two paper arms (``paper_kalshi_taker``, ``paper_kalshi_maker``) on the Kalshi minute
engine and learning from every settlement (agent/fly_session.py for prediction markets).

It starts once its design passed the replay (kalshi/fly_replay.verdict) and a deployable bootstrap exists
(kalshi/fly.latest_deployable); the replay's chosen (α, half-life) per strategy is its mushroom body's learning rate
and forgetting (brain/plastic.py). Each minute (kalshi/engine.py):

1. labels: tags whose market has now settled (the stream writes ``kalshi_markets.result``) teach the mushroom body the
   outcome y ∈ {0, 1} of their side (δ = y − p̂); a tag due before its market resolved is re-armed hourly, and dropped
   as unknown a week after close;
2. calibration: at the first minute of each UTC day, every strategy's edge line and sizing, plastic and frozen, from the
   rows resolved in the last seven days (``kalshi_fly_scored``; the taker label is the return at the effective ask, the
   maker label the return of a bid one tick inside the ask when a later minute's ask reached it);
3. scoring: eligible (market, minute, side) rows are scored by the plastic fly and its frozen shadow — p̂ per strategy,
   the edge p̂ − the arm's fee-inclusive price — stored and tagged; only rows at or above the line teach it;
4. trading (fresh stream, trade minutes): the taker arm (kalshi/paper.taker_entries) and the maker arm (kalshi/maker.plan
   + adjudicate) from the same decision, then settlements and marks per book; the live mirror (kalshi/live.py) when
   ``KALSHI_LIVE_ENABLED`` and the prerequisites hold.

Hourly: learning statistics (``kalshi_fly_updates``), a snapshot of the plastic state (kind 'kalshi_fly_plastic'), the
rollback checks (train/fly_governance.py; a third rollback in 7 days freezes learning and asks kalshi/pipeline.py for a
re-bootstrap). State: ``data/brain/plastic_kalshi/state.pt``; commands ``ui_settings['kalshi_fly_command']``; status
``ui_settings['kalshi_fly_status']``; activity for the 3D view in ``data/brain/activity_kalshi/``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..brain import activity, device as brain_device, plastic
from ..brain.connectome import current_connectome_path
from ..db.apilog import record_event
from ..db.connection import transaction
from ..ops.reset import kalshi_activity_dir, kalshi_fly_state_dir
from ..train import fly_calibrate, fly_governance as gov
from . import fly as KF, maker, paper as P
from .features import K_COLS, KIDX
from .strategies import effective

log = logging.getLogger(__name__)
NAME, KIND = "kalshi_fly", "kalshi_fly"
STATUS_KEY, COMMAND_KEY = "kalshi_fly_status", "kalshi_fly_command"
SNAP_KIND = "kalshi_fly_plastic"
LABEL_LAG_S = 60.0
REARM_S, UNKNOWN_AFTER_S = 3600.0, 7 * 86400.0
ROLLBACK_PAUSE_S, ROLLBACK_LIMIT, ROLLBACK_WINDOW_S, ROLLBACK_MIN_AGE_S = 86400.0, 3, 7 * 86400.0, 86400.0
SNAP_HOURLY_DAYS, SCORED_KEEP_DAYS = 7, 35


def send_command(cmd: str) -> None:
    assert cmd in ("pause", "resume", "rollback"), cmd
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (COMMAND_KEY, json.dumps({"cmd": cmd, "at": datetime.now(timezone.utc).isoformat()})))


def try_start() -> tuple["KalshiFlyBook | None", str]:
    from . import fly_replay
    v = fly_replay.verdict()
    if not v:
        return None, "the Kalshi fly's replay has not run on the current definitions (fly-trader kalshi-fly-replay)"
    if not v.get("passed"):
        return None, f"the Kalshi fly's replay did not pass: {v.get('reason')}"
    with transaction() as conn:
        boot = KF.latest_deployable(conn)
    if boot is None:
        return None, "no deployable Kalshi fly bootstrap (the Kalshi training pipeline bootstraps one)"
    try:
        return KalshiFlyBook(boot, v), "ok"
    except Exception as e:
        log.exception("kalshi fly book failed to start")
        return None, f"kalshi fly book failed to start: {type(e).__name__}: {e}"


def arm_label(x: np.ndarray, side_won: float, arm: str, ticker: str, filled: bool | None) -> float | None:
    """The realised net return per dollar of the arm on a row (the training labels): taker at the effective ask; maker
    one tick inside the ask plus the maker fee, when the resting bid filled (None when it did not, or is not known)."""
    eff = float(effective(x[None], K_COLS, arm)[0])
    if arm == "maker" and not filled:
        return None
    if eff <= 0:
        return None
    return float((100.0 * side_won - eff) / eff)


class KalshiFlyBook:
    name = NAME

    def __init__(self, boot: dict, verdict: dict):
        self.done = False; self.mirror = None; self.verdict = verdict
        self.state_path = kalshi_fly_state_dir() / "state.pt"
        self._load_bootstrap(boot)
        self.run_id = str(uuid.uuid4()); self.beat_no = 0
        with transaction() as conn:
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,'running')",
                         (self.run_id, KIND, json.dumps({"channels": self.cfg, "books": list(P.BOOKS.values())}, default=str), self.boot_id))
            self._restore_tags(conn)
        record_event("info", NAME, "kalshi fly session started", {"run_id": self.run_id, "bootstrap": self.boot_id, "channels": self.cfg, "lines": self.lines["plastic"],
                                                                   "pending": len(self.pending), "learning_frozen": self.learning_frozen})
        log.info("kalshi fly session: bootstrap %d on %s, channels %s, lines %s, %d pending tags", self.boot_id, brain_device.describe(self.fly.net.dev), self.cfg,
                 self.lines["plastic"], len(self.pending))

    # ---- model and state ----
    def _load_bootstrap(self, boot: dict) -> None:
        self.boot_id = int(boot["id"])
        self.fly = KF.load(boot["path"])
        try:
            self.connectome_name = current_connectome_path().name
        except Exception:
            self.connectome_name = ""
        self.capture_ok = True
        self.idx = np.asarray([K_COLS.index(c) for c in self.fly.cols], dtype=int)
        self.names = list(self.fly.strategies); self.arms = {k: self.fly.arm(j) for j, k in enumerate(self.names)}
        per = self.verdict.get("per_strategy") or {self.names[0]: {"alpha": self.verdict.get("alpha", 0.0), "half_life_days": self.verdict.get("half_life_days")}}
        self.cfg = {k: (float((per.get(k) or {}).get("alpha") or 0.0), (per.get(k) or {}).get("half_life_days")) for k in self.names}
        a0, h0 = self.cfg[self.names[0]]
        net = self.fly.net
        self.bank = plastic.PlasticBank(net, [(a0, float(h0) if h0 is not None else math.inf)], KF.SCALE, learn=net.learn.cpu().numpy(), read=net.read.cpu().numpy())
        for j, k in enumerate(self.names):
            a, h = self.cfg[k]; cols = self.bank.learn[j]
            self.bank.alpha[0, cols] = a; self.bank.half_life_s[0, cols] = float(h) * 86400.0 if h is not None else float("inf")
        self.pending = plastic.PendingTags(net.n_kc, net.k_active)
        self.watch: dict[str, int] = {}                                  # ticker → pending tags on it
        self.lines = {"plastic": dict(self.fly.lines), "frozen": dict(self.fly.lines)}
        self.sizing = {"plastic": {k: list(v) for k, v in self.fly.sizings.items()}, "frozen": {k: list(v) for k, v in self.fly.sizings.items()}}
        self.applied_through = -math.inf; self.learning_frozen = False; self.checks_paused_until = {k: 0.0 for k in self.names}
        self.last_day: int | None = None; self.last_hour: int | None = None
        self.rollbacks: dict = {k: [] for k in self.names}; self.nu_set = False; self.hour_stats = self._empty_stats(); self.last_checks: dict = {}
        s = self._read_state()
        if s and s.get("bootstrap_id") == self.boot_id and s.get("cfg") == self.cfg:
            self.bank.load_state(s["bank"]); self.nu_set = True
            self.lines, self.sizing = s["lines"], s["sizing"]
            self.applied_through, self.learning_frozen, self.checks_paused_until = s["applied_through"], s["learning_frozen"], s["checks_paused_until"]
            self.last_day, self.last_hour, self.rollbacks = s["last_day"], s["last_hour"], s["rollbacks"]
        elif s:
            with transaction() as conn:
                conn.execute("UPDATE kalshi_fly_scored SET state = 'dropped', x = NULL WHERE state = 'pending' AND bootstrap_id IS DISTINCT FROM %s", (self.boot_id,))

    def _read_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            return torch.load(self.state_path, map_location="cpu", weights_only=False)
        except Exception:
            log.exception("kalshi fly state unreadable; starting from the bootstrap")
            return None

    def _state(self) -> dict:
        return {"bootstrap_id": self.boot_id, "cfg": self.cfg, "bank": self.bank.state(), "lines": self.lines, "sizing": self.sizing, "applied_through": self.applied_through,
                "learning_frozen": self.learning_frozen, "checks_paused_until": self.checks_paused_until, "last_day": self.last_day, "last_hour": self.last_hour,
                "rollbacks": self.rollbacks, "saved_at": time.time()}

    def _write_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp"); torch.save(self._state(), tmp); os.replace(tmp, self.state_path)

    def _restore_tags(self, conn) -> None:
        rows = conn.execute("SELECT ts, ticker, side, strategy, x, score, line, due_ts FROM kalshi_fly_scored WHERE state = 'pending' AND bootstrap_id = %s AND x IS NOT NULL ORDER BY ts",
                            (self.boot_id,)).fetchall()
        groups: dict[tuple, list] = {}
        for r in rows:
            if r["strategy"] in self.arms:
                groups.setdefault((r["ts"].timestamp(), r["strategy"]), []).append(r)
        for (t, name), rs in sorted(groups.items()):
            j = self.names.index(name)
            Y, u0, k = self.fly.parts_all(np.asarray([r["x"] for r in rs], dtype=np.float32))
            w = plastic.row_weights(torch.tensor([[float(r["score"]) for r in rs]]), torch.tensor([float(rs[0]["line"])]))
            due = min(r["due_ts"].timestamp() if r["due_ts"] else t for r in rs)
            self.pending.push([(t, r["ticker"], r["side"], name) for r in rs], np.full(len(rs), t), max(due, time.time()), Y[:, j], u0, k, w, s=j)
            for r in rs:
                self.watch[r["ticker"]] = self.watch.get(r["ticker"], 0) + 1

    def maybe_reload(self) -> bool:
        from . import fly_replay
        with transaction() as conn:
            boot = KF.latest_deployable(conn)
        v = fly_replay.verdict()
        if boot is None or v is None or not v.get("passed") or int(boot["id"]) == self.boot_id:
            return False
        old = self.boot_id; self.verdict = v
        self._write_state(); self._load_bootstrap(boot)
        with transaction() as conn:
            conn.execute("UPDATE runs SET brain_snapshot_id = %s WHERE run_id = %s", (self.boot_id, self.run_id))
        self._write_state()
        record_event("info", NAME, f"switched to kalshi fly bootstrap #{self.boot_id}", {"from": old, "to": self.boot_id, "lines": self.lines["plastic"]})
        return True

    # ---- engine interface ----
    def tickers_watched(self) -> set[str]:
        return set(self.watch)

    def on_minute(self, ctx) -> dict:
        conn, t_now = ctx.conn, ctx.t_start
        self._commands(conn, ctx.m1_epoch)
        learned = self._learn(conn, ctx)
        day = int(t_now // 86400)
        if self.last_day is None:
            self.last_day = day
        elif day > self.last_day:
            self.last_day = day; self._calibrate(conn, ctx.m1_epoch)
        scored = self._score(conn, ctx)
        drift = {k: float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0]) for j, k in enumerate(self.names)}
        out = {"minute": ctx.m1.isoformat(), "eligible": len(ctx.keys), "markets_active": ctx.n_rows // 2, "picks": scored["picks"], "lines": self.lines["plastic"],
               "frozen_lines": self.lines["frozen"], "arms": self.arms, "drift": max(drift.values()) if drift else 0.0, "drift_by_strategy": drift, "pending": len(self.pending),
               "learned": learned, "learning_frozen": self.learning_frozen, "channels": self.cfg, "bootstrap": self.boot_id, "device": str(self.fly.net.dev)}
        self.beat_no += 1
        beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active, notes) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                            (self.run_id, ctx.m1, self.beat_no, len(ctx.keys), json.dumps({"minute": ctx.m1.isoformat(), "markets_active": ctx.n_rows // 2}))).fetchone()["id"]
        books = {}
        for arm, book in P.BOOKS.items():
            results = ctx.settled([p["ticker"] for p in P.open_positions(conn, book)])
            settled = P.settle(conn, book=book, results=results, ts=ctx.m1, run_id=self.run_id, beat_id=beat)
            books[arm] = {"settled": len(settled), "settled_pnl": sum(s["realized_cents"] for s in settled) / 100.0}
        adj = maker.adjudicate(conn, book=P.BOOKS["maker"], run_id=self.run_id, beat_id=beat, ts=ctx.m1, now=ctx.m1_epoch, extremes=ctx.extremes, metas=ctx.metas)
        books["maker"].update(filled=len(adj["filled"]), expired=adj["expired"])
        live_plan = {"taker_entries": [], "maker": {"posted": [], "canceled": 0, "replaced": 0}}
        if ctx.trade and ctx.fresh:
            tk = P.taker_entries(conn, book=P.BOOKS["taker"], run_id=self.run_id, beat_id=beat, ts=ctx.m1, picks=scored["taker"], quotes=ctx.quotes)
            mk = maker.plan(conn, book=P.BOOKS["maker"], run_id=self.run_id, beat_id=beat, ts=ctx.m1, now=ctx.m1_epoch, picks=scored["maker"], quotes=ctx.quotes, metas=ctx.metas)
            books["taker"].update(entered=tk["entered"], blocked=tk["blocked"]); books["maker"].update(posted=len(mk["posted"]), canceled=mk["canceled"], replaced=mk["replaced"], resting=mk["resting"])
            live_plan = {"taker_entries": tk["entries"], "maker": mk}
            out["stage"] = "trading"
        else:
            out["stage"] = "learning (not trading this minute)" if ctx.trade else "catching up"
        for arm, book in P.BOOKS.items():
            books[arm].update(P.mark(conn, book=book, quotes=ctx.quotes, ts=ctx.m1, beat_id=beat))
        out["books"] = books
        if config.KALSHI_LIVE_ENABLED:
            out["live"] = self._live(ctx, beat, live_plan)
        hour = int(ctx.m1_epoch // 3600)
        if self.last_hour is None:
            self.last_hour = hour
        elif hour > self.last_hour:
            self.last_hour = hour; self._hourly(conn, ctx.m1_epoch)
        self._write_state()
        if learned.get("keys"):
            with conn.cursor() as cur:
                cur.executemany("UPDATE kalshi_fly_scored SET label = %s, state = %s, resolved_at = now(), x = NULL WHERE ts = %s AND ticker = %s AND side = %s AND strategy = %s", learned.pop("keys"))
        if learned.get("rearmed"):
            with conn.cursor() as cur:
                cur.executemany("UPDATE kalshi_fly_scored SET due_ts = %s WHERE ts = %s AND ticker = %s AND side = %s AND strategy = %s", learned.pop("rearmed"))
        learned.pop("keys", None); learned.pop("rearmed", None)
        out["checks"] = self.last_checks; out["updated_at"] = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (STATUS_KEY, json.dumps(out, default=str)))
        return out

    def _live(self, ctx, beat_id: int, plan: dict) -> dict:
        missing = config.kalshi_live_prerequisites_missing()
        if missing:
            return {"stage": "live gated: " + ", ".join(missing)}
        if self.mirror is None:
            from .live import KalshiLiveMirror
            self.mirror = KalshiLiveMirror()
            record_event("info", "kalshi_live", "the Kalshi fly trades the subaccount", {"run_id": self.run_id, "subaccount": config.KALSHI_SUBACCOUNT})
        try:
            return self.mirror.minute(ctx, run_id=self.run_id, beat_id=beat_id, taker_entries=plan["taker_entries"], maker_plan=plan["maker"])
        except Exception as e:
            log.exception("kalshi live mirror failed this minute")
            return {"stage": "live error", "error": f"{type(e).__name__}: {e}"[:300]}

    def finish(self) -> None:
        try:
            self._write_state()
        except Exception:
            log.exception("kalshi fly state write failed at finish")
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = 'finished' WHERE run_id = %s", (self.run_id,))
        record_event("info", NAME, "kalshi fly session stopped", {"run_id": self.run_id, "minutes": self.beat_no})

    # ---- learning ----
    @staticmethod
    def _empty_stats() -> dict:
        return {"n": 0, "sum_delta": 0.0, "sum_abs": 0.0, "step": 0.0, "capped": 0, "unknown": 0}

    def _maker_filled(self, conn, ticker: str, side: str, t: float, rest: float) -> bool:
        """Did a bid at ``rest`` (cents) resting from minute ``t`` fill: a later minute's ask on the side reached it."""
        col = "min(yes_ask_low)" if side == "yes" else "100 - max(yes_bid_high)"
        r = conn.execute(f"SELECT {col} AS m FROM kalshi_minutes WHERE ticker = %s AND ts > %s", (ticker, datetime.fromtimestamp(t, timezone.utc))).fetchone()
        return r is not None and r["m"] is not None and float(r["m"]) <= rest

    def _learn(self, conn, ctx) -> dict:
        now = ctx.m1_epoch
        tags = self.pending.pop_due(now, device=self.bank.dev)
        if tags is None:
            return {"n": 0}
        results = ctx.settled({tk for _, tk, _, _ in tags.keys})
        labels = np.full(len(tags.keys), np.nan); keep = np.zeros(len(tags.keys), bool); rearm = []
        rows = {}
        need = [(datetime.fromtimestamp(t, timezone.utc), tk, side, name) for t, tk, side, name in tags.keys]
        if need:
            for r in conn.execute("SELECT ts, ticker, side, strategy, x FROM kalshi_fly_scored WHERE (ts, ticker, side, strategy) IN (SELECT * FROM unnest(%s::timestamptz[], %s::text[], %s::text[], %s::text[]))",
                                  ([n[0] for n in need], [n[1] for n in need], [n[2] for n in need], [n[3] for n in need])).fetchall():
                rows[(r["ts"].timestamp(), r["ticker"], r["side"], r["strategy"])] = r["x"]
        out_keys = []
        for i, (t, tk, side, name) in enumerate(tags.keys):
            res = results.get(tk)
            if res in ("yes", "no"):
                y = 1.0 if res == side else 0.0; labels[i] = y; keep[i] = True
                x = rows.get((t, tk, side, name)); arm = self.arms[name]; lab = None
                if x is not None:
                    xa = np.asarray(x, dtype=np.float32)
                    filled = self._maker_filled(conn, tk, side, t, float(np.clip(np.round(xa[KIDX["side_ask"]]) - 1, 1, 99))) if arm == "maker" else None
                    lab = arm_label(xa, y, arm, tk, filled)
                out_keys.append((lab, "resolved" if lab is not None else "unfilled", datetime.fromtimestamp(t, timezone.utc), tk, side, name))
            elif now - t > UNKNOWN_AFTER_S:
                out_keys.append((None, "unknown", datetime.fromtimestamp(t, timezone.utc), tk, side, name))
            else:
                rearm.append(i)
        for i, (t, tk, side, name) in enumerate(tags.keys):
            if i not in rearm:
                c = self.watch.get(tk, 0) - 1
                if c > 0:
                    self.watch[tk] = c
                else:
                    self.watch.pop(tk, None)
        rearmed = []
        if rearm:
            sub = tags.subset(np.isin(np.arange(len(tags.keys)), rearm))
            for j in sorted({int(s) for s in sub.s}):
                m = sub.s == j; ss = sub.subset(m)
                self.pending.push(ss.keys, ss.ts, now + REARM_S, ss.y_dn, ss.u0, ss.k, ss.w, s=j)
            rearmed = [(datetime.fromtimestamp(now + REARM_S, timezone.utc), datetime.fromtimestamp(t, timezone.utc), tk, side, name) for t, tk, side, name in sub.keys]
        stats = {"n": int(keep.sum()), "unknown": int(sum(1 for k in out_keys if k[1] == "unknown")), "rearmed": rearmed, "keys": out_keys, "waiting": len(rearm)}
        if keep.any():
            tv = tags.subset(keep)
            if self.learning_frozen:
                self.bank.decay_to(now)
            else:
                st = self.bank.update(tv, torch.tensor(labels[keep], dtype=torch.float32), now)
                stats.update(mean_delta=st["mean_delta"][0], mean_abs_delta=st["mean_abs_delta"][0], step=st["step"][0], capped=bool(st["capped"][0]))
                h = self.hour_stats; h["n"] += stats["n"]; h["sum_delta"] += st["mean_delta"][0] * stats["n"]; h["sum_abs"] += st["mean_abs_delta"][0] * stats["n"]
                h["step"] += st["step"][0]; h["capped"] += int(st["capped"][0])
        self.hour_stats["unknown"] += stats["unknown"]
        self.applied_through = now
        return stats

    def _score(self, conn, ctx) -> dict:
        n = len(ctx.keys); empty = {"taker": [], "maker": [], "picks": 0}
        if not n:
            return empty
        X = ctx.X[:, self.idx]
        Y, u0, k, H = self.fly.parts_all_h(X); trig = self.fly.triggers(ctx.X, K_COLS)
        self._capture(H, trig, ctx.t_start); del H
        if not self.nu_set:
            self.bank.estimate_nu(Y[:, 0], u0, k); self.nu_set = True
        t = ctx.t_start; ts = datetime.fromtimestamp(t, timezone.utc); rows = []
        E = np.full((n, len(self.names)), -np.inf); Pm = np.full((n, len(self.names)), np.nan)
        for j, name in enumerate(self.names):
            loc = np.flatnonzero(trig[:, j])
            if not len(loc):
                continue
            li = torch.as_tensor(loc, device=Y.device); arm = self.arms[name]
            eff = torch.as_tensor(effective(ctx.X[loc], K_COLS, arm) / 100.0, device=Y.device, dtype=Y.dtype)
            with torch.no_grad():
                p, _ = self.bank.predict(Y[li, j], u0[li], k[li], s=np.full(len(loc), j)); p = p[0].clamp(0.0, 1.0)
                fp = self.bank.frozen(Y[li, j], u0[li], s=np.full(len(loc), j)).clamp(0.0, 1.0)
                sc, fr = p - eff, fp - eff
            v, f = sc.float().cpu().numpy(), fr.float().cpu().numpy(); E[loc, j] = v; Pm[loc, j] = p.float().cpu().numpy()
            line, fline = self.lines["plastic"][name], self.lines["frozen"][name]
            due = [ctx.metas[ctx.keys[i][0]].close_ts + LABEL_LAG_S if ctx.metas.get(ctx.keys[i][0]) and ctx.metas[ctx.keys[i][0]].close_ts else t + ctx.hold_s[i] + LABEL_LAG_S for i in loc]
            rows += [(ts, ctx.keys[i][0], ctx.keys[i][1], name, ctx.X[i].tolist(), float(v[q]), float(f[q]), line, fline, datetime.fromtimestamp(due[q], timezone.utc), self.boot_id)
                     for q, i in enumerate(loc)]
            w = plastic.row_weights(sc[None], torch.tensor([line], dtype=sc.dtype, device=sc.device))
            keep = np.flatnonzero((w[0] > 0).cpu().numpy())
            if len(keep):
                for d_ in sorted({due[q] for q in keep}):
                    qq = [q for q in keep if due[q] == d_]; ki = torch.as_tensor(loc[qq], device=Y.device)
                    self.pending.push([(t, ctx.keys[loc[q]][0], ctx.keys[loc[q]][1], name) for q in qq], np.full(len(qq), t), float(d_), Y[ki, j], u0[ki], k[ki],
                                      w[:, torch.as_tensor(qq, device=w.device)], s=j)
                    for q in qq:
                        tk = ctx.keys[loc[q]][0]; self.watch[tk] = self.watch.get(tk, 0) + 1
        if rows:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO kalshi_fly_scored (ts, ticker, side, strategy, x, score, frozen_score, line, frozen_line, due_ts, bootstrap_id) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ts, ticker, side, strategy) DO NOTHING", rows)
        self.fly.lines = dict(self.lines["plastic"]); self.fly.sizings = {k_: list(v) for k_, v in self.sizing["plastic"].items()}
        picks = {"taker": [], "maker": []}
        for arm in ("taker", "maker"):
            d = KF.kalshi_decide(self.fly, ctx.X, K_COLS, E, ctx.hold_s, only=arm, trig=trig)
            for i, sname in enumerate(d["strategy"]):
                if sname is None:
                    continue
                j = self.names.index(sname)
                picks[arm].append({"ticker": ctx.keys[i][0], "side": ctx.keys[i][1], "p": float(Pm[i, j]), "edge": float(d["score"][i]), "line": float(d["threshold"][i]),
                                   "table": d["tables"][i], "strategy": sname, "hold_s": float(ctx.hold_s[i])})
        return {**picks, "picks": len(picks["taker"]) + len(picks["maker"])}

    def _capture(self, H: torch.Tensor, trig: np.ndarray, t: float) -> None:
        if not self.capture_ok:
            return
        try:
            sel = np.flatnonzero(trig.any(1)); n_c = int(len(sel))
            m = (H.index_select(0, torch.as_tensor(sel, device=H.device)) if n_c else H).mean(0)
            activity.write(kalshi_activity_dir(), t, activity.quantise(m), n_rows=int(len(trig)), n_cand=n_c, connectome=self.connectome_name)
        except Exception:
            self.capture_ok = False
            log.exception("kalshi brain activity capture disabled for this session")

    # ---- daily and hourly ----
    def _resolved(self, conn, since: float, cols: str, name: str) -> list[dict]:
        return conn.execute(f"SELECT {cols} FROM kalshi_fly_scored WHERE state = 'resolved' AND label IS NOT NULL AND bootstrap_id = %s AND strategy = %s AND resolved_at >= %s ORDER BY ts",
                            (self.boot_id, name, datetime.fromtimestamp(since, timezone.utc))).fetchall()

    def _calibrate(self, conn, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).date()
        for name in self.names:
            rows = self._resolved(conn, now - fly_calibrate.WINDOW_DAYS * 86400.0, "ts, ticker, side, score, frozen_score, label, due_ts", name)
            if not rows:
                continue
            ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([f"{r['ticker']}:{r['side']}" for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
            hold = np.array([max((r["due_ts"].timestamp() - r["ts"].timestamp()) if r["due_ts"] else 60.0, 60.0) for r in rows])
            for arm, col in (("plastic", "score"), ("frozen", "frozen_score")):
                cal = fly_calibrate.calibrate(ts, mint, hold, np.array([r[col] for r in rows], dtype=np.float64), lab, prev_line=self.lines[arm][name], prev_sizing=self.sizing[arm][name])
                if cal.changed:
                    record_event("info", NAME, f"{name} {arm} edge line {self.lines[arm][name]:.3f} → {cal.line:.3f}", {"trades": cal.trades, "mean": cal.mean, "total": cal.total})
                self.lines[arm][name], self.sizing[arm][name] = cal.line, cal.sizing
                conn.execute("INSERT INTO kalshi_fly_calibrations (day, arm, line, sizing, trades, total, mean, window_days) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                             "ON CONFLICT (day, arm) DO UPDATE SET line = EXCLUDED.line, sizing = EXCLUDED.sizing, trades = EXCLUDED.trades, total = EXCLUDED.total, mean = EXCLUDED.mean",
                             (day, f"{arm}:{name}", cal.line, json.dumps(cal.sizing), cal.trades, cal.total, cal.mean, fly_calibrate.WINDOW_DAYS))

    def _drift(self, j: int) -> float:
        return float(self.bank.drift(j)[0]) if self.bank.learn.shape[0] > 1 else float(self.bank.drift()[0])

    def _checks(self, conn, now: float) -> dict:
        out = {}
        for j, name in enumerate(self.names):
            rows = self._resolved(conn, now - gov.SHADOW_DAYS * 86400.0, "ts, ticker, side, score, line, frozen_score, frozen_line, label, due_ts, resolved_at", name)
            if rows:
                ts = np.array([r["ts"].timestamp() for r in rows]); mint = np.array([f"{r['ticker']}:{r['side']}" for r in rows]); lab = np.array([r["label"] for r in rows], dtype=np.float64)
                hold = np.array([max((r["due_ts"].timestamp() - r["ts"].timestamp()) if r["due_ts"] else 60.0, 60.0) for r in rows])
                g = lambda c: np.array([r[c] for r in rows], dtype=np.float64)
                rp = gov.picks_returns(ts, mint, hold, g("score"), g("line"), lab); rf = gov.picks_returns(ts, mint, hold, g("frozen_score"), g("frozen_line"), lab)
                recent = np.array([r["resolved_at"].timestamp() for r in rows]) > now - gov.IC_HOURS * 3600.0
                out[name] = gov.check((rp, rf), (g("score")[recent], lab[recent]), self._drift(j))
            else:
                out[name] = gov.check(None, None, self._drift(j))
        return out

    def _hourly(self, conn, now: float) -> None:
        hour = datetime.fromtimestamp(math.floor(now / 3600) * 3600 - 3600, timezone.utc)
        checks = self._checks(conn, now); self.last_checks = checks
        h = self.hour_stats; first = checks.get(self.names[0]) or {}
        conn.execute("INSERT INTO kalshi_fly_updates (hour, n, mean_delta, mean_abs_delta, step, capped, drift, ic, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (hour) DO NOTHING",
                     (hour, h["n"], h["sum_delta"] / h["n"] if h["n"] else None, h["sum_abs"] / h["n"] if h["n"] else None, h["step"], h["capped"],
                      max((c["drift"] for c in checks.values()), default=0.0), first.get("ic"), json.dumps({"unknown": h["unknown"], "checks": checks}, default=str)))
        self.hour_stats = self._empty_stats()
        good = not any(c["triggers"] for c in checks.values())
        sid = self._snapshot(conn, now, good, checks)
        for name, c in checks.items():
            if c["triggers"] and now >= self.checks_paused_until.get(name, 0.0) and not self.learning_frozen:
                self._rollback(conn, now, c, sid, name=name)
        self._prune(conn, now)
        activity.prune(kalshi_activity_dir(), now - activity.KEEP_S)

    def _snapshot(self, conn, now: float, good: bool, checks: dict) -> int:
        d = kalshi_fly_state_dir(); d.mkdir(parents=True, exist_ok=True)
        path = d / f"snap_{datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M')}.pt"
        torch.save({"bank": self.bank.state(), "bootstrap_id": self.boot_id, "lines": self.lines, "sizing": self.sizing}, path)
        note = {"bootstrap_id": self.boot_id, "data": KF.KALSHI_FLY_VERSION, "good": good, "checks": checks, "drift": max((c["drift"] for c in checks.values()), default=0.0),
                "ic24": {k: c.get("ic") for k, c in checks.items()}}
        return int(conn.execute("INSERT INTO brain_snapshots (run_id, path, kind, note) VALUES (%s,%s,%s,%s) RETURNING id", (self.run_id, str(path), SNAP_KIND, json.dumps(note, default=str))).fetchone()["id"])

    def _commands(self, conn, now: float) -> None:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (COMMAND_KEY,)).fetchone()
        if not r or not r["value"]:
            return
        v = r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])
        conn.execute("DELETE FROM ui_settings WHERE key = %s", (COMMAND_KEY,))
        cmd = v.get("cmd")
        if cmd == "pause":
            self.learning_frozen = True; record_event("info", NAME, "learning paused by the operator")
        elif cmd == "resume":
            self.learning_frozen = False; self.rollbacks = {}; record_event("info", NAME, "learning resumed by the operator")
        elif cmd == "rollback":
            for j, name in enumerate(self.names):
                self._rollback(conn, now, {"triggers": ["operator"], "drift": self._drift(j)}, None, count=False, name=name)

    def _rollback(self, conn, now: float, checks: dict, from_sid: int | None, count: bool = True, name: str | None = None) -> None:
        name = name or self.names[0]; j = self.names.index(name); cols = self.bank.learn[j]
        target = None
        for r in conn.execute("SELECT id, path, note, ts FROM brain_snapshots WHERE kind = %s AND ts <= %s ORDER BY id DESC",
                              (SNAP_KIND, datetime.fromtimestamp(now - ROLLBACK_MIN_AGE_S, timezone.utc))).fetchall():
            n = json.loads(r["note"] or "{}")
            ok = n.get("good") or not ((n.get("checks") or {}).get(name) or {}).get("triggers")
            if ok and n.get("bootstrap_id") == self.boot_id and Path(r["path"]).exists():
                target = r; break
        if target is not None:
            snap = plastic.PlasticBank(self.fly.net, [(0.0, math.inf)], KF.SCALE, learn=self.bank.learn.cpu().numpy(), read=self.bank.read.cpu().numpy())
            st = torch.load(target["path"], map_location="cpu", weights_only=False)["bank"]
            snap.D = torch.zeros_like(snap.D); snap.D[:, snap.mask.bool()] = st["D"].to(snap.dev).float()
            self.bank.D[0][:, cols] = snap.D[0][:, cols]
        else:
            self.bank.D[0][:, cols] = 0.0
        self.checks_paused_until[name] = now + ROLLBACK_PAUSE_S
        self.rollbacks[name] = [t for t in self.rollbacks.get(name, []) if t > now - ROLLBACK_WINDOW_S] + ([now] if count else [])
        conn.execute("INSERT INTO kalshi_fly_rollbacks (reason, checks, from_snapshot, to_snapshot) VALUES (%s,%s,%s,%s)",
                     (f"{name}: " + ", ".join(checks["triggers"]), json.dumps(checks, default=str), from_sid, target["id"] if target is not None else None))
        record_event("warning", NAME, f"{name} channel rolled back ({', '.join(checks['triggers'])})", {"to_snapshot": target["id"] if target is not None else "bootstrap (D = 0)", "checks": checks})
        if count and len(self.rollbacks[name]) >= ROLLBACK_LIMIT:
            self.learning_frozen = True
            from . import pipeline
            pipeline.request_run(fly=True, reason=f"{name}: {len(self.rollbacks[name])} rollbacks in 7 days")
            record_event("error", NAME, f"learning frozen after {len(self.rollbacks[name])} {name} rollbacks in 7 days; re-bootstrap requested", {"rollbacks": self.rollbacks})

    def _prune(self, conn, now: float) -> None:
        rows = conn.execute("SELECT id, path, ts FROM brain_snapshots WHERE kind = %s AND ts < %s ORDER BY ts DESC",
                            (SNAP_KIND, datetime.fromtimestamp(now - SNAP_HOURLY_DAYS * 86400.0, timezone.utc))).fetchall()
        kept: set = set(); drop = []
        for r in rows:
            d = r["ts"].date()
            if d in kept:
                drop.append(r)
            else:
                kept.add(d)
        for r in drop:
            conn.execute("DELETE FROM brain_snapshots WHERE id = %s", (r["id"],))
            try:
                Path(r["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute("DELETE FROM kalshi_fly_scored WHERE state <> 'pending' AND ts < %s", (datetime.fromtimestamp(now - SCORED_KEEP_DAYS * 86400.0, timezone.utc),))
