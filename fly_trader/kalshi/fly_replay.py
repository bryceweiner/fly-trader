"""The Kalshi fly's replay: the bootstrap (kalshi/fly.py) traded through the corpus under mushroom-body plasticity
(brain/plastic.py) in every configuration of (α, τ½), learning from each position's **settlement** (a tag comes due when
its market settles, not after a fixed hold) — the memecoin replay (train/fly_replay.py) with per-row holds, edges instead
of raw scores, and outcomes y ∈ {0, 1} as the reward. The verdict (``ui_settings['kalshi_fly_replay']``) chooses the
configuration and decides whether the Kalshi fly may trade a paper book.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import date, datetime, timezone

import numpy as np
import torch

from ..brain import device as brain_device
from ..brain import plastic
from ..db.apilog import record_event
from ..db.connection import transaction
from ..train import fly_calibrate
from ..train import fly_governance as gov
from ..train import fly_replay as R
from ..train import progress as prog
from ..train import selector
from ..train.decisions import DecisionSet, evaluate, random_trades, summarize, taken_idx
from . import decisions as KD
from . import fly as KF
from .strategies import effective, in_universe, label

log = logging.getLogger(__name__)
KEY = "kalshi_fly_replay"
DEFAULT_START = R.DEFAULT_START
CONFIGS = R.CONFIGS
LABEL_LAG_S = R.LABEL_LAG_S
HOUR_S, DAY_S = R.HOUR_S, R.DAY_S
CHUNK = R.CHUNK


def _resolved_window(ts_o: np.ndarray, res_o: np.ndarray, upto: int, now: float, span_s: float) -> np.ndarray:
    """Positions (< ``upto``) of the rows whose settlement label became known in (now − span, now]."""
    hi = int(np.searchsorted(ts_o[:upto], now, side="right"))
    r = res_o[:hi]
    return np.flatnonzero((r > now - span_s) & (r <= now))


def _recalibrate(ds, order, ts_o, res_o, upto, scores, lines, sizings, now, arms, line_log) -> None:
    """One edge line and sizing per strategy from the frozen fly's edges on the positions settled in the last window."""
    for s_, arm in enumerate(arms):
        pos = _resolved_window(ts_o, res_o, upto, now, fly_calibrate.WINDOW_DAYS * DAY_S)
        sc = scores[0, s_, pos]; has = np.isfinite(sc); pos = pos[has]; rows = order[pos]
        y = label(ds, arm)[rows]
        cal = fly_calibrate.calibrate(ds.ts[rows], ds.mint[rows], ds.hold_s[rows], sc[has], y, prev_line=float(lines[0, s_]), prev_sizing=sizings[0][s_])
        for c in range(lines.shape[0]):
            key = (c, s_)
            if key not in line_log:
                line_log[key] = [(-math.inf, float(lines[c, s_]))]
            lines[c, s_] = cal.line; sizings[c][s_] = cal.sizing
            line_log[key].append((now, float(cal.line)))


def _governance(ds, order, ts_o, res_o, upto, scores, lines, line_log, bank, now, arms, counts) -> None:
    for s_, arm in enumerate(arms):
        p3 = _resolved_window(ts_o, res_o, upto, now, gov.SHADOW_DAYS * DAY_S); p1 = _resolved_window(ts_o, res_o, upto, now, gov.IC_HOURS * HOUR_S)
        r3 = order[p3]; r1 = order[p1]; y = label(ds, arm)
        lf = R._line_at(line_log, (0, s_), ts_o[p3], float(lines[0, s_]))
        rf = gov.picks_returns(ds.ts[r3], ds.mint[r3], ds.hold_s[r3], np.nan_to_num(scores[0, s_, p3], nan=-np.inf), lf, y[r3])
        drift = bank.drift(s_).tolist() if bank.learn.shape[0] > 1 else bank.drift().tolist()
        for c in range(1, lines.shape[0]):
            lc = R._line_at(line_log, (c, s_), ts_o[p3], float(lines[c, s_]))
            rp = gov.picks_returns(ds.ts[r3], ds.mint[r3], ds.hold_s[r3], np.nan_to_num(scores[c, s_, p3], nan=-np.inf), lc, y[r3])
            s1 = scores[c, s_, p1]; has = np.isfinite(s1)
            out = gov.check((rp[np.isfinite(rp)], rf[np.isfinite(rf)]), (s1[has], ds.outcome[r1][has]), drift[c])
            counts[(c, s_)]["checks"] += 1
            for t in out["triggers"]:
                counts[(c, s_)][t] += 1
            counts[(c, s_)]["any"] += bool(out["triggers"])


def _book(ds, order, ts_o, scores, line_log, choice, boot_lines, arms, mask_pos):
    """The combined book of the per-strategy choices: per row, among the strategies at/above their line in force, the
    one with the largest margin; (pick, per-row hold s = time to settlement, per-row return of that strategy's arm)."""
    N = len(ds.y); key = np.full(N, -np.inf); pick = np.zeros(N, bool); ret = np.full(N, np.nan, np.float32)
    for s_, c in enumerate(choice):
        Sf, Lf = R._arm(ds, order, ts_o, scores[c, s_], line_log, (c, s_), boot_lines[s_], mask_pos)
        lab = label(ds, arms[s_])
        ok = (Sf >= Lf) & np.isfinite(lab)
        m = np.where(ok, Sf - Lf, -np.inf); take = ok & (m > key)
        key = np.where(take, m, key); pick |= ok
        ret = np.where(take, lab, ret)
    return pick, np.where(pick, ds.hold_s, 0.0), ret


def run(days: int | None = None, start_day: int = DEFAULT_START, configs: list | None = None, stop: threading.Event | None = None, device=None,
        ds: DecisionSet | None = None, fly=None, boot: dict | None = None, g=None, save_verdict: bool = True) -> dict:
    configs = [(float(a), float(h)) for a, h in (configs or CONFIGS)]
    if configs[0][0] != 0.0:
        configs = [(0.0, math.inf)] + configs
    t0 = time.time()
    if ds is None:
        prog.update("kalshi fly replay: building decision points", 0, 1, force=True)
        ds = KD.build(days=days)
    dl = ds.days
    if len(dl) <= start_day + 2:
        raise RuntimeError(f"the Kalshi corpus has {len(dl)} days; the replay needs more than {start_day + 2}")
    Sday = dl[start_day]
    if fly is None:
        fly, boot = KF.bootstrap(ds, Sday, stop=stop, g=g, device=device)
        if fly is None:
            return {"stopped": True, **(boot or {})}
    boot = boot or {}
    names = fly.strategies; NS = len(names); arms = [fly.arm(s_) for s_ in range(NS)]
    S_epoch = datetime(Sday.year, Sday.month, Sday.day, tzinfo=timezone.utc).timestamp()
    uni = in_universe(ds.X, ds.cols)
    hist_from = S_epoch - fly_calibrate.WINDOW_DAYS * DAY_S
    rows = np.flatnonzero(uni & (ds.ts >= hist_from))
    order = rows[np.argsort(ds.ts[rows], kind="stable")]
    ts_o = ds.ts[order]; n = len(order); res_o = ts_o + ds.hold_s[order] + LABEL_LAG_S          # when each row's settlement label is known
    eff_o = np.stack([effective(ds.X[order], ds.cols, a) / 100.0 for a in arms]).astype(np.float32)   # [NS, n] each arm's fee-inclusive price
    live_from = int(np.searchsorted(ts_o, S_epoch, side="left"))
    bank = plastic.PlasticBank(fly.net, configs, KF.SCALE, learn=fly.net.learn.cpu().numpy(), read=fly.net.read.cpu().numpy())
    brain_device.empty_cache(bank.dev); chunk = fly.net.batch_rows(ceiling=CHUNK)
    log.info("kalshi fly replay on %s: %d rows per batch, %d rows, %d strategies", brain_device.describe(bank.dev), chunk, n, NS)
    C = bank.C
    scores = np.full((C, NS, n), np.nan, np.float32)                   # edges (p̂ − the arm's price) per configuration
    lines = np.array([[fly.lines[k] for k in names]] * C, dtype=np.float64); sizings = [[list(fly.sizings[k]) for k in names] for _ in range(C)]
    line_log: dict = {}
    pending = plastic.PendingTags(fly.net.n_kc, fly.net.k_active)
    outcome = torch.tensor(ds.outcome, dtype=torch.float32)
    starts = np.flatnonzero(np.r_[True, ts_o[1:] != ts_o[:-1]]); ends = np.r_[starts[1:], n]
    day_done = R._day_start(ts_o[live_from]) if live_from < n else None
    next_gov = (ts_o[live_from] + HOUR_S) if live_from < n else None
    gov_counts = {(c, s): {"shadow": 0, "ic": 0, "drift": 0, "any": 0, "checks": 0} for c in range(C) for s in range(NS)}
    learn: dict = {}
    nu_set = False; g_ = 0; n_updates = 0
    fly.net.eval()
    while g_ < len(starts):
        if stop is not None and stop.is_set():
            return {"stopped": True}
        g1 = g_; rows_in = 0
        while g1 < len(starts) and rows_in < chunk:
            rows_in += ends[g1] - starts[g1]; g1 += 1
        a, b = starts[g_], ends[g1 - 1]
        Y, u0, k = fly.parts_all(ds.X[order[a:b]]); trig = fly.triggers(ds.X[order[a:b]], ds.cols)
        if not nu_set:
            bank.estimate_nu(Y[:, 0], u0, k); nu_set = True
        for gi in range(g_, g1):
            s0, s1 = starts[gi], ends[gi]; now = float(ts_o[s0]); lo, hi = s0 - a, s1 - a
            if s0 >= live_from:
                tags = pending.pop_due(now, device=bank.dev)
                if tags is not None:
                    r = torch.tensor([float(outcome[row]) for row, _ in tags.keys], dtype=torch.float32)
                    R._learn_add(learn, now, bank.update(tags, r, now)); n_updates += len(tags)
                if day_done is not None and R._day_start(now) > day_done:
                    day_done = R._day_start(now)
                    _recalibrate(ds, order, ts_o, res_o, s0, scores, lines, sizings, now, arms, line_log)
                if next_gov is not None and now >= next_gov:
                    next_gov = now + HOUR_S
                    _governance(ds, order, ts_o, res_o, s0, scores, lines, line_log, bank, now, arms, gov_counts)
            for s_ in range(NS):
                loc = np.flatnonzero(trig[lo:hi, s_])
                if not len(loc):
                    continue
                li = torch.as_tensor(lo + loc, device=Y.device)
                with torch.no_grad():
                    p, _ = bank.predict(Y[li, s_], u0[li], k[li], s=np.full(len(loc), s_))
                    edge = p.clamp(0.0, 1.0) - torch.as_tensor(eff_o[s_, s0 + loc], device=p.device)[None]
                scores[:, s_, s0 + loc] = edge.float().cpu().numpy()
                if s0 >= live_from:
                    w = plastic.row_weights(edge, torch.tensor(lines[:, s_], dtype=edge.dtype, device=edge.device))
                    keep = np.flatnonzero((w > 0).any(0).cpu().numpy())
                    if len(keep):
                        kl = loc[keep]; due = res_o[s0 + kl]
                        for d_ in np.unique(due):                                      # one push per settlement time
                            sub = np.flatnonzero(due == d_); kk = kl[sub]; ki = torch.as_tensor(lo + kk, device=Y.device)
                            pending.push([(int(order[s0 + j]), s_) for j in kk], ts_o[s0 + kk], float(d_),
                                         Y[ki, s_], u0[ki], k[ki], w[:, torch.as_tensor(keep[sub], device=w.device)], s=s_)
        if g_ == 0 or (g1 // 50) != (g_ // 50):
            prog.update("kalshi fly replay: learning from settlements", int(b), n, day=str(datetime.fromtimestamp(float(ts_o[b - 1]), timezone.utc).date()),
                        updates=n_updates, pending=len(pending), drift=[round(float(x), 4) for x in bank.drift()], eta_s=(n - b) * (time.time() - t0) / max(b, 1))
        g_ = g1
    return _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, Sday, save_verdict, time.time() - t0, fly, names, arms, learn)


def _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, Sday, save_verdict, secs, fly, names, arms, learn) -> dict:
    days_r = sorted(set(ds.day[order[live_from:]].tolist()))
    sel_days, ev_days = days_r[: len(days_r) // 2], days_r[len(days_r) // 2:]
    day_o = ds.day[order]
    live_pos = np.arange(len(order)) >= live_from
    sel_pos = live_pos & np.isin(day_o, sel_days); ev_pos = live_pos & np.isin(day_o, ev_days)
    boot_lines = [fly.lines[k] for k in names]
    per_strategy = {}; choice = []
    for s_, name in enumerate(names):
        y = label(ds, arms[s_]); table = []
        for c, (alpha, hl) in enumerate(configs):
            Sf, Lf = R._arm(ds, order, ts_o, scores[c, s_], line_log, (c, s_), boot_lines[s_], sel_pos)
            tr = taken_idx(ds.ts, ds.mint, ds.hold_s, np.flatnonzero((Sf >= Lf) & np.isfinite(y))); r = y[tr]
            table.append({"config": c, "alpha": alpha, "half_life_days": hl if math.isfinite(hl) else None, "trades": int(len(r)), "total": float(np.nansum(r)),
                          "mean": float(np.nanmean(r)) if len(r) else None, "governance": gov_counts.get((c, s_))})
        ok = [t for t in table if t["alpha"] > 0 and t["trades"] >= selector.MIN_LINE_TRADES]
        best = max(ok, key=lambda t: t["total"])["config"] if ok else 0
        choice.append(best)
        per_strategy[name] = {"configs": table, "chosen": table[best] if ok else None, "alpha": table[best]["alpha"], "half_life_days": table[best]["half_life_days"],
                              "arm": arms[s_], "hold_min": arms[s_], "plastic": bool(ok)}

    def judge(ch):
        pick, hold, ret = _book(ds, order, ts_o, scores, line_log, ch, boot_lines, arms, ev_pos)
        test = np.zeros(len(ds.y), bool); test[order[ev_pos]] = True
        ev = evaluate(ds, np.where(pick, 1.0, np.nan), test & pick, 0.5, "kalshi fly", hold_s=hold, returns=ret)
        rnd = summarize(random_trades(ds, test, max(1, int(ev["pooled"]["n"] or 1)), hold_s=ds.hold_s, returns=ds.fwd_pess))
        return ev, rnd

    out = {"S": str(Sday), "selection_days": [str(sel_days[0]), str(sel_days[-1])] if sel_days else None, "evaluation_days": [str(ev_days[0]), str(ev_days[-1])] if ev_days else None,
           "strategies": names, "arms": arms, "per_strategy": per_strategy, "configs": per_strategy[names[0]]["configs"],
           "bootstrap": {k: boot.get(k) for k in ("lines", "calibration", "diagnostics", "gates_ok", "gate_failures")}, "data": KF.KALSHI_FLY_VERSION,
           "secs": secs, "finished_at": datetime.now(timezone.utc).isoformat(), "learning": R._learning_series(learn or {})}
    ev0, rnd0 = judge([0] * len(names))
    out["frozen"] = {**ev0["pooled"], "random": rnd0}
    if not any(v["plastic"] for v in per_strategy.values()):
        out.update(passed=False, reason="no strategy's plastic configuration made 100 trades on the selection half")
    else:
        ev, rnd = judge(choice)
        passed, why = selector.deploy_decision(ev["pooled"], rnd)
        out.update(passed=bool(passed), reason=why, chosen=per_strategy[names[0]]["chosen"], evaluation=ev["pooled"], random=rnd, per_day=ev["per_day"],
                   alpha=per_strategy[names[0]]["alpha"], half_life_days=per_strategy[names[0]]["half_life_days"],
                   plastic_beats_frozen=bool((ev["pooled"]["mean"] or -1) > (ev0["pooled"]["mean"] or -1)))
    log.info("kalshi fly replay from %s: %s — %s | frozen %s", Sday, "PASSED" if out.get("passed") else "FAILED", out.get("reason"), out["frozen"])
    if save_verdict:
        record_event("info" if out.get("passed") else "warning", "kalshi_fly_replay", "kalshi fly replay " + ("passed" if out.get("passed") else "failed"),
                     {k: out.get(k) for k in ("S", "passed", "reason", "evaluation", "random", "frozen", "plastic_beats_frozen")})
        try:
            with transaction() as conn:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (KEY, json.dumps(prog._finite(out), default=str)))
        except Exception:
            log.exception("could not store the Kalshi replay verdict")
        try:
            c0 = choice[0] if choice else 0
            with transaction() as conn:
                for rec in out["learning"]:
                    d = date.fromisoformat(rec["day"]); pick = lambda k: (rec[k][c0] if len(rec.get(k) or []) > c0 else None)
                    conn.execute("INSERT INTO kalshi_fly_updates (hour, n, mean_delta, mean_abs_delta, step, capped, drift, ic, detail) "
                                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (hour) DO NOTHING",
                                 (datetime(d.year, d.month, d.day, tzinfo=timezone.utc), rec["n"], pick("mean_delta"), pick("mean_abs_delta"),
                                  pick("step"), pick("capped"), pick("drift"), None, json.dumps({"source": "replay", "config": c0, "per_config_drift": rec["drift"]}, default=str)))
        except Exception:
            log.exception("could not store the Kalshi replay's learning statistics (the verdict is stored)")
    prog.update("kalshi fly replay: done", 1, 1, force=True, passed=out.get("passed"), reason=out.get("reason"))
    return out


def verdict() -> dict | None:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = %s", (KEY,)).fetchone()
    v = (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"] or "{}")) if r else None
    return v if v and v.get("data") == KF.KALSHI_FLY_VERSION else None


def main(days: int | None = None, start_day: int = DEFAULT_START, stop_event: threading.Event | None = None) -> dict:
    """The replay; when it passes and no Kalshi fly may trade yet, the live fly is bootstrapped at once."""
    prog.set_stop_event(stop_event); prog.clear()
    out = run(days=days, start_day=start_day, stop=stop_event)
    if out.get("passed") and not (stop_event is not None and stop_event.is_set()):
        with transaction() as conn:
            have = KF.latest_deployable(conn)
        if have is None:
            log.info("kalshi replay passed: bootstrapping the live Kalshi fly")
            out["live_bootstrap"] = {k: v for k, v in KF.main(days=days, stop_event=stop_event).items() if k in ("snapshot_id", "lines", "gates_ok", "gate_failures", "calibration")}
    return out
