"""The proof before the fly trades: bootstrapped once, then left to learn from the market for months, on the corpus.

1. Decision points for the whole corpus (``train/decisions.build``); ``S`` = the ``start_day``-th day. The fly is
   bootstrapped for ``S`` (``fly_selector.bootstrap``: the selector teaches it once on the days before ``S − 8`` and it
   calibrates its own line on the week before ``S``). Nothing is retrained after that.
2. Every tradable minute from ``S`` on, in time order, under every configuration of ``CONFIGS`` side by side (α, τ½;
   α = 0 is the frozen fly) — exactly what the live fly does each minute:
   a. tags whose labels are known (scored at ``t`` with ``t + hold + 60 s`` ≤ now) are captured: the realized net
      return (the training label ``fwd_pess``) teaches the mushroom body (``brain/plastic.py``);
   b. at each UTC day boundary each configuration recalibrates its line and sizing (``train/fly_calibrate.py``) on its
      own scores of the minutes resolved in the last seven days (the week before ``S``, scored by the bootstrap fly,
      seeds the first windows);
   c. the minute's rows are scored and tagged; a row scored at/above the configuration's line counts ``TOP_WEIGHT`` ×;
   d. every hour the rollback checks (``train/fly_governance.py``) are evaluated and counted, not acted on: the replay
      measures how often they would trip.
3. The configuration with the most total net profit on the first half of the replay days (≥ 100 trades; plastic
   configurations only) is judged on the second half against its frozen twin and random picks with the selector's
   deploy rule (``selector.deploy_decision``): profitable after costs, better than random, ≥ 100 trades. The verdict is
   stored in ``ui_settings['fly_replay']`` (with ``FLY_VERSION``); the live fly needs a passed verdict.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ..brain import plastic
from ..db.apilog import record_event
from ..db.connection import transaction
from . import fly_calibrate, fly_governance as gov, fly_selector, selector
from . import progress as prog
from .decisions import DecisionSet, build, evaluate, random_trades, summarize, taken_idx

log = logging.getLogger(__name__)
KEY = "fly_replay"
START_DAY = 30
CHUNK = 2048
CONFIGS = [(0.0, math.inf)] + [(a, h) for a in (1e-4, 3e-4, 1e-3, 3e-3) for h in (1.0, 3.0, 7.0, 30.0)]
LABEL_LAG_S = 60.0           # a label is known once the exit minute has closed
HOUR_S, DAY_S = 3600.0, 86400.0


def _day_start(t: float) -> float:
    return math.floor(t / DAY_S) * DAY_S


def run(days: int | None = None, start_day: int = START_DAY, configs: list | None = None, stop: threading.Event | None = None, device=None,
        ds: DecisionSet | None = None, fly=None, boot: dict | None = None, graph=None, save_verdict: bool = True) -> dict:
    """The replay; ``ds``/``fly``/``boot`` may be given (tests, or a bootstrap already run)."""
    configs = [(float(a), float(h)) for a, h in (configs or CONFIGS)]
    if configs[0][0] != 0.0:
        configs = [(0.0, math.inf)] + configs                                          # configuration 0 is always the frozen fly
    t0 = time.time()
    if ds is None:
        prog.update("fly replay: building decision points", 0, 1, force=True)
        ds = build(days=days)
    dl = ds.days
    if len(dl) <= start_day + 2:
        raise RuntimeError(f"the corpus has {len(dl)} days; the replay needs more than {start_day + 2}")
    S = dl[start_day]
    if fly is None:
        fly, boot = fly_selector.bootstrap(ds, S, stop=stop, graph=graph, device=device)
        if fly is None:
            return {"stopped": True}
    boot = boot or {}
    H = float(ds.horizon_s); lag = H + LABEL_LAG_S
    S_epoch = datetime(S.year, S.month, S.day, tzinfo=timezone.utc).timestamp()
    uni = selector.in_universe(ds.X, ds.cols)
    hist_from = S_epoch - fly_calibrate.WINDOW_DAYS * DAY_S
    rows = np.flatnonzero(uni & (ds.ts >= hist_from))
    order = rows[np.argsort(ds.ts[rows], kind="stable")]
    ts_o = ds.ts[order]; n = len(order)
    live_from = int(np.searchsorted(ts_o, S_epoch, side="left"))                      # rows before it: history only (calibration seed)
    bank = plastic.PlasticBank(fly.net, configs, fly_selector.SCALE)
    C = bank.C
    scores = np.full((C, n), np.nan, np.float32)
    lines = np.full(C, fly.threshold, np.float64); sizings = [list(fly.sizing) for _ in range(C)]
    line_log: dict[str, list] = {}
    pending = plastic.PendingTags(fly.net.n_kc, fly.net.k_active)
    labels = torch.tensor(ds.fwd_pess, dtype=torch.float32)
    starts = np.flatnonzero(np.r_[True, ts_o[1:] != ts_o[:-1]]); ends = np.r_[starts[1:], n]
    day_done = _day_start(ts_o[live_from]) if live_from < n else None
    next_gov = (ts_o[live_from] + HOUR_S) if live_from < n else None
    gov_counts = {c: {"shadow": 0, "ic": 0, "drift": 0, "any": 0, "checks": 0} for c in range(C)}
    nu_set = False; g = 0; n_updates = 0
    fly.net.eval()
    while g < len(starts):
        if stop is not None and stop.is_set():
            return {"stopped": True}
        g1 = g; rows_in = 0
        while g1 < len(starts) and rows_in < CHUNK:
            rows_in += ends[g1] - starts[g1]; g1 += 1
        a, b = starts[g], ends[g1 - 1]
        y_dn, u0, k = fly.parts(ds.X[order[a:b]])
        if not nu_set:
            bank.estimate_nu(y_dn, u0, k); nu_set = True
        for gi in range(g, g1):
            s0, s1 = starts[gi], ends[gi]; now = float(ts_o[s0]); lo, hi = s0 - a, s1 - a
            if s0 >= live_from:
                tags = pending.pop_due(now, device=bank.dev)
                if tags is not None:
                    bank.update(tags, labels[np.asarray(tags.keys)], now); n_updates += len(tags)
                if day_done is not None and _day_start(now) > day_done:
                    day_done = _day_start(now)
                    _recalibrate(ds, order, ts_o, scores, lines, sizings, now, lag, line_log)
                if next_gov is not None and now >= next_gov:
                    next_gov = now + HOUR_S
                    _governance(ds, order, ts_o, scores, lines, line_log, bank, now, lag, gov_counts)
            with torch.no_grad():
                sc, _ = bank.predict(y_dn[lo:hi], u0[lo:hi], k[lo:hi])
            scores[:, s0:s1] = sc.float().cpu().numpy()
            if s0 >= live_from:
                w = plastic.row_weights(sc, torch.tensor(lines, dtype=sc.dtype, device=sc.device))
                pending.push(order[s0:s1].tolist(), ts_o[s0:s1], now + lag, y_dn[lo:hi], u0[lo:hi], k[lo:hi], w)
        if g == 0 or (g1 // 50) != (g // 50):
            prog.update("fly replay: learning from the market", int(b), n, day=str(datetime.fromtimestamp(float(ts_o[b - 1]), timezone.utc).date()),
                        updates=n_updates, pending=len(pending), drift=[round(float(x), 4) for x in bank.drift()], eta_s=(n - b) * (time.time() - t0) / max(b, 1))
        g = g1
    return _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, S, save_verdict, time.time() - t0)


def _window(ts_o: np.ndarray, now: float, lag: float, span_s: float) -> tuple[int, int]:
    """Positions of the rows whose labels resolved in (now − span, now]."""
    return int(np.searchsorted(ts_o, now - span_s - lag, side="right")), int(np.searchsorted(ts_o, now - lag, side="right"))


def _line_at(line_log: dict, c: int, ts: np.ndarray, default: float) -> np.ndarray:
    """The line configuration ``c`` had in force when each row was scored."""
    log_c = line_log.get(c) or []
    if not log_c:
        return np.full(len(ts), default)
    at = np.array([t for t, _ in log_c]); vals = np.array([v for _, v in log_c])
    i = np.searchsorted(at, ts, side="right") - 1
    return np.where(i >= 0, vals[np.clip(i, 0, None)], default)


def _recalibrate(ds, order, ts_o, scores, lines, sizings, now, lag, line_log) -> None:
    a, b = _window(ts_o, now, lag, fly_calibrate.WINDOW_DAYS * DAY_S)
    rows = order[a:b]
    for c in range(len(lines)):
        if c not in line_log:
            line_log[c] = [(-math.inf, float(lines[c]))]
        cal = fly_calibrate.calibrate(ds.ts[rows], ds.mint[rows], ds.horizon_s, scores[c, a:b], ds.fwd_pess[rows], prev_line=float(lines[c]), prev_sizing=sizings[c])
        lines[c] = cal.line; sizings[c] = cal.sizing
        line_log[c].append((now, float(cal.line)))


def _governance(ds, order, ts_o, scores, lines, line_log, bank, now, lag, counts) -> None:
    a3, b3 = _window(ts_o, now, lag, gov.SHADOW_DAYS * DAY_S); a1, b1 = _window(ts_o, now, lag, gov.IC_HOURS * HOUR_S)
    r3 = order[a3:b3]; r1 = order[a1:b1]
    lf = _line_at(line_log, 0, ts_o[a3:b3], float(lines[0]))
    rf = gov.picks_returns(ds.ts[r3], ds.mint[r3], ds.horizon_s, scores[0, a3:b3], lf, ds.fwd_pess[r3])
    drift = bank.drift().tolist()
    for c in range(1, len(lines)):
        lc = _line_at(line_log, c, ts_o[a3:b3], float(lines[c]))
        rp = gov.picks_returns(ds.ts[r3], ds.mint[r3], ds.horizon_s, scores[c, a3:b3], lc, ds.fwd_pess[r3])
        out = gov.check((rp, rf), (scores[c, a1:b1], ds.fwd_pess[r1]), drift[c])
        counts[c]["checks"] += 1
        for t in out["triggers"]:
            counts[c][t] += 1
        counts[c]["any"] += bool(out["triggers"])


def _arm(ds, order, ts_o, scores_c, line_log, c, default_line, mask_pos) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (N) scores and per-row lines of configuration ``c`` on the positions ``mask_pos`` (NaN elsewhere)."""
    S_full = np.full(len(ds.y), np.nan); L_full = np.full(len(ds.y), np.inf)
    pos = np.flatnonzero(mask_pos)
    S_full[order[pos]] = scores_c[pos]; L_full[order[pos]] = _line_at(line_log, c, ts_o[pos], default_line)
    return S_full, L_full


def _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, S, save_verdict, secs) -> dict:
    days_r = sorted(set(ds.day[order[live_from:]].tolist()))
    sel_days, ev_days = days_r[: len(days_r) // 2], days_r[len(days_r) // 2:]
    day_o = ds.day[order]
    live_pos = np.arange(len(order)) >= live_from
    sel_pos = live_pos & np.isin(day_o, sel_days); ev_pos = live_pos & np.isin(day_o, ev_days)
    table = []
    for c, (alpha, hl) in enumerate(configs):
        Sf, Lf = _arm(ds, order, ts_o, scores[c], line_log, c, boot.get("line", selector.MIN_EV), sel_pos)
        tr = taken_idx(ds.ts, ds.mint, ds.horizon_s, np.flatnonzero(Sf >= Lf)); r = ds.fwd_pess[tr]
        table.append({"config": c, "alpha": alpha, "half_life_days": hl if math.isfinite(hl) else None, "trades": int(len(r)), "total": float(r.sum()),
                      "mean": float(r.mean()) if len(r) else None, "governance": gov_counts.get(c)})
    ok = [t for t in table if t["alpha"] > 0 and t["trades"] >= selector.MIN_LINE_TRADES]
    best = max(ok, key=lambda t: t["total"])["config"] if ok else None

    def judge(c):
        Sf, Lf = _arm(ds, order, ts_o, scores[c], line_log, c, boot.get("line", selector.MIN_EV), ev_pos)
        test = np.zeros(len(ds.y), bool); test[order[ev_pos]] = True
        ev = evaluate(ds, Sf, test, Lf, "fly")
        rnd = summarize(random_trades(ds, test, max(1, int((test & (Sf >= Lf)).sum()))))
        return ev, rnd

    out = {"S": str(S), "selection_days": [str(sel_days[0]), str(sel_days[-1])] if sel_days else None, "evaluation_days": [str(ev_days[0]), str(ev_days[-1])] if ev_days else None,
           "configs": table, "bootstrap": {k: boot.get(k) for k in ("line", "calibration", "diagnostics", "gates_ok", "gate_failures")}, "data": fly_selector.FLY_VERSION,
           "secs": secs, "finished_at": datetime.now(timezone.utc).isoformat()}
    ev0, rnd0 = judge(0)
    out["frozen"] = {**ev0["pooled"], "random": rnd0}
    if best is None:
        out.update(passed=False, reason="no plastic configuration made 100 trades on the selection half")
    else:
        ev, rnd = judge(best)
        passed, why = selector.deploy_decision(ev["pooled"], rnd)
        out.update(passed=bool(passed), reason=why, chosen=table[best], evaluation=ev["pooled"], random=rnd, per_day=ev["per_day"],
                   alpha=table[best]["alpha"], half_life_days=table[best]["half_life_days"], plastic_beats_frozen=bool((ev["pooled"]["mean"] or -1) > (ev0["pooled"]["mean"] or -1)))
    log.info("fly replay from %s: %s — %s | frozen %s", S, "PASSED" if out.get("passed") else "FAILED", out.get("reason"), out["frozen"])
    if save_verdict:
        record_event("info" if out.get("passed") else "warning", "fly_replay", "fly replay " + ("passed" if out.get("passed") else "failed"),
                     {k: out.get(k) for k in ("S", "passed", "reason", "alpha", "half_life_days", "evaluation", "random", "frozen", "plastic_beats_frozen")})
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (KEY, json.dumps(out, default=str)))
    prog.update("fly replay: done", 1, 1, force=True, passed=out.get("passed"), reason=out.get("reason"))
    return out


def verdict() -> dict | None:
    """The stored replay verdict for the current fly definitions, or None."""
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (KEY,)).fetchone()
    v = (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")) if r else None
    return v if v and v.get("data") == fly_selector.FLY_VERSION else None


def main(days: int | None = None, start_day: int = START_DAY, stop_event: threading.Event | None = None) -> dict:
    """The replay; when it passes and no fly may trade yet, the live fly is bootstrapped at once (taught on the whole corpus)."""
    prog.set_stop_event(stop_event); prog.clear()
    out = run(days=days, start_day=start_day, stop=stop_event)
    if out.get("passed") and not (stop_event is not None and stop_event.is_set()):
        with transaction() as conn:
            have = fly_selector.latest_deployable(conn)
        if have is None:
            log.info("replay passed: bootstrapping the live fly")
            out["live_bootstrap"] = {k: v for k, v in fly_selector.main(days=days, stop_event=stop_event).items() if k in ("snapshot_id", "line", "gates_ok", "gate_failures", "calibration")}
    return out
