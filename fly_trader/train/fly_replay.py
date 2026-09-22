"""The proof before the fly trades: bootstrapped once, then left to learn from the market for weeks, on the corpus.

1. Decision points for the whole corpus with every strategy hold's labels (``train/decisions.build``); ``S`` = the
   ``start_day``-th day. The fly is bootstrapped for ``S`` (``fly_selector.bootstrap``: the selector's strategy stack,
   fitted on the days before ``S − 8`` only, teaches it once, one head per strategy on its own dopamine channel; each
   strategy calibrates its own line on the week before ``S``). The stack's walk-forward needs enough days in each of its
   two halves, so ``S`` is at least ``DEFAULT_START`` days in. Nothing is retrained after that.
2. Every tradable minute from ``S`` on, in time order, under every configuration of ``CONFIGS`` side by side (α, τ½;
   α = 0 is the frozen fly) — exactly what the live fly does each minute:
   a. tags whose labels are known (scored at ``t`` with ``t + the strategy's hold + 60 s`` ≤ now) are captured: the
      realized net return over that strategy's hold teaches only that strategy's channel (``brain/plastic.py``);
   b. at each UTC day boundary each strategy's line and sizing are recalibrated (``train/fly_calibrate.py``) on the
      frozen fly's scores of the minutes resolved in the last seven days and shared by every configuration, so the
      arms differ only in their weights (per-configuration lines once traded different slices: ``_recalibrate``);
   c. each strategy's candidate rows (the selector's trigger) are scored; only the rows a configuration would have
      traded (at or above its own line) teach it -- the candidate set is overwhelmingly losers and learning from it
      dragged every prediction toward that mean (brain/plastic.row_weights);
   d. every hour the rollback checks (``train/fly_governance.py``) are evaluated per strategy and counted, not acted on.
3. Each strategy's configuration is the one with the most total net profit on its own trades of the first half of the
   replay days (≥ 100 trades; plastic configurations only). Channels are disjoint, so the combination is exact. The
   combined book (per row the strategy the fly's own decision picks, each position on its own hold) is judged on the
   second half against the frozen fly and random picks with the selector's deploy rule. The verdict is stored in
   ``ui_settings['fly_replay']`` (with ``FLY_VERSION``); the live fly needs a passed verdict.
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

from ..brain import plastic
from ..db.apilog import record_event
from ..db.connection import transaction
from . import fly_calibrate, fly_governance as gov, fly_selector, selector
from . import progress as prog
from .decisions import DecisionSet, build, evaluate, random_trades, summarize, taken_idx

log = logging.getLogger(__name__)
KEY = "fly_replay"
DEFAULT_START = selector.WARMUP_DAYS + 2 * 35 + fly_selector.CALIB_DAYS + fly_selector.PURGE_DAYS    # 99: 5 weeks in each of the stack's walk-forward halves
START_DAY = DEFAULT_START
CHUNK = 2048
CONFIGS = [(0.0, math.inf)] + [(a, h) for a in (1e-4, 3e-4, 1e-3, 3e-3) for h in (1.0, 3.0, 7.0, 30.0)]
LABEL_LAG_S = 60.0           # a label is known once the exit minute has closed
HOUR_S, DAY_S = 3600.0, 86400.0


def _day_start(t: float) -> float:
    return math.floor(t / DAY_S) * DAY_S


def _labels(ds: DecisionSet, hold_min: int) -> np.ndarray:
    return ds.fwd_h[hold_min] if hold_min in ds.fwd_h else ds.fwd_pess


def run(days: int | None = None, start_day: int = START_DAY, configs: list | None = None, stop: threading.Event | None = None, device=None,
        ds: DecisionSet | None = None, fly=None, boot: dict | None = None, graph=None, save_verdict: bool = True) -> dict:
    """The replay; ``ds``/``fly``/``boot`` may be given (tests, or a bootstrap already run)."""
    from .strategies import HOLDS_MIN
    configs = [(float(a), float(h)) for a, h in (configs or CONFIGS)]
    if configs[0][0] != 0.0:
        configs = [(0.0, math.inf)] + configs                                          # configuration 0 is always the frozen fly
    t0 = time.time()
    if ds is None:
        prog.update("fly replay: building decision points", 0, 1, force=True)
        ds = build(days=days, holds=HOLDS_MIN)
    dl = ds.days
    if len(dl) <= start_day + 2:
        raise RuntimeError(f"the corpus has {len(dl)} days; the replay needs more than {start_day + 2}")
    S = dl[start_day]
    if fly is None:
        fly, boot = fly_selector.bootstrap(ds, S, stop=stop, graph=graph, device=device)
        if fly is None:
            return {"stopped": True, **(boot or {})}
    boot = boot or {}
    names = fly.strategies; NS = len(names); holds_min = [int(fly.rules[k]["hold_min"]) for k in names]
    lags = [h * 60.0 + LABEL_LAG_S for h in holds_min]
    S_epoch = datetime(S.year, S.month, S.day, tzinfo=timezone.utc).timestamp()
    uni = selector.in_universe(ds.X, ds.cols)
    hist_from = S_epoch - fly_calibrate.WINDOW_DAYS * DAY_S
    rows = np.flatnonzero(uni & (ds.ts >= hist_from))
    order = rows[np.argsort(ds.ts[rows], kind="stable")]
    ts_o = ds.ts[order]; n = len(order)
    live_from = int(np.searchsorted(ts_o, S_epoch, side="left"))                      # rows before it: history only (calibration seed)
    bank = plastic.PlasticBank(fly.net, configs, fly_selector.SCALE, learn=fly.net.learn.cpu().numpy(), read=fly.net.read.cpu().numpy())
    C = bank.C
    scores = np.full((C, NS, n), np.nan, np.float32)
    lines = np.array([[fly.lines[k] for k in names]] * C, dtype=np.float64); sizings = [[list(fly.sizings[k]) for k in names] for _ in range(C)]
    line_log: dict = {}
    pending = plastic.PendingTags(fly.net.n_kc, fly.net.k_active)
    labels = [torch.tensor(_labels(ds, h), dtype=torch.float32) for h in holds_min]
    starts = np.flatnonzero(np.r_[True, ts_o[1:] != ts_o[:-1]]); ends = np.r_[starts[1:], n]
    day_done = _day_start(ts_o[live_from]) if live_from < n else None
    next_gov = (ts_o[live_from] + HOUR_S) if live_from < n else None
    gov_counts = {(c, s): {"shadow": 0, "ic": 0, "drift": 0, "any": 0, "checks": 0} for c in range(C) for s in range(NS)}
    learn: dict = {}                     # per UTC day, per configuration: what the mushroom body actually did
    nu_set = False; g = 0; n_updates = 0
    fly.net.eval()
    while g < len(starts):
        if stop is not None and stop.is_set():
            return {"stopped": True}
        g1 = g; rows_in = 0
        while g1 < len(starts) and rows_in < CHUNK:
            rows_in += ends[g1] - starts[g1]; g1 += 1
        a, b = starts[g], ends[g1 - 1]
        Y, u0, k = fly.parts_all(ds.X[order[a:b]]); trig = fly.triggers(ds.X[order[a:b]], ds.cols)
        if not nu_set:
            bank.estimate_nu(Y[:, 0], u0, k); nu_set = True
        for gi in range(g, g1):
            s0, s1 = starts[gi], ends[gi]; now = float(ts_o[s0]); lo, hi = s0 - a, s1 - a
            if s0 >= live_from:
                tags = pending.pop_due(now, device=bank.dev)
                if tags is not None:
                    r = torch.tensor([float(labels[s_][row]) for row, s_ in tags.keys], dtype=torch.float32)
                    _learn_add(learn, now, bank.update(tags, r, now)); n_updates += len(tags)
                if day_done is not None and _day_start(now) > day_done:
                    day_done = _day_start(now)
                    _recalibrate(ds, order, ts_o, scores, lines, sizings, now, lags, holds_min, line_log)
                if next_gov is not None and now >= next_gov:
                    next_gov = now + HOUR_S
                    _governance(ds, order, ts_o, scores, lines, line_log, bank, now, lags, holds_min, gov_counts)
            for s_ in range(NS):
                loc = np.flatnonzero(trig[lo:hi, s_])
                if not len(loc):
                    continue
                li = torch.as_tensor(lo + loc, device=Y.device)
                with torch.no_grad():
                    sc, _ = bank.predict(Y[li, s_], u0[li], k[li], s=np.full(len(loc), s_))
                scores[:, s_, s0 + loc] = sc.float().cpu().numpy()
                if s0 >= live_from:
                    w = plastic.row_weights(sc, torch.tensor(lines[:, s_], dtype=sc.dtype, device=sc.device))
                    keep = np.flatnonzero((w > 0).any(0).cpu().numpy())      # a row no configuration would trade teaches none of them
                    if len(keep):
                        kl = loc[keep]; ki = torch.as_tensor(lo + kl, device=Y.device)
                        pending.push([(int(order[s0 + j]), s_) for j in kl], ts_o[s0 + kl], now + lags[s_],
                                     Y[ki, s_], u0[ki], k[ki], w[:, torch.as_tensor(keep, device=w.device)], s=s_)
        if g == 0 or (g1 // 50) != (g // 50):
            prog.update("fly replay: learning from the market", int(b), n, day=str(datetime.fromtimestamp(float(ts_o[b - 1]), timezone.utc).date()),
                        updates=n_updates, pending=len(pending), drift=[round(float(x), 4) for x in bank.drift()], eta_s=(n - b) * (time.time() - t0) / max(b, 1))
        g = g1
    return _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, S, save_verdict, time.time() - t0, fly, names, holds_min, lines, learn)


def _learn_add(learn: dict, now: float, st: dict) -> None:
    """Accumulate one update's statistics per UTC day, per configuration. A replay that leaves no record of its own
    learning cannot say why learning helped or hurt — which is how a degrading fly went unexplained for two runs."""
    if not st or not st.get("n"):
        return
    n = int(st["n"]); d = datetime.fromtimestamp(now, timezone.utc).date()
    e = learn.setdefault(d, {"n": 0, "delta": None, "abs": None, "step": None, "capped": None, "drift": None})
    e["n"] += n
    for key, src, scale in (("delta", "mean_delta", n), ("abs", "mean_abs_delta", n), ("step", "step", 1)):
        v = [float(x) * scale for x in st[src]]
        e[key] = v if e[key] is None else [a + b for a, b in zip(e[key], v)]
    cap = [int(bool(x)) for x in st["capped"]]
    e["capped"] = cap if e["capped"] is None else [a + b for a, b in zip(e["capped"], cap)]
    e["drift"] = [float(x) for x in st["drift"]]


def _learning_series(learn: dict) -> list:
    """The per-day series for the verdict: mean delta (how wrong it was), step size, and drift from the bootstrap."""
    out = []
    for d in sorted(learn):
        e = learn[d]; n = max(e["n"], 1)
        out.append({"day": str(d), "n": e["n"],
                    "mean_delta": [round(x / n, 6) for x in (e["delta"] or [])],
                    "mean_abs_delta": [round(x / n, 6) for x in (e["abs"] or [])],
                    "step": [round(x, 8) for x in (e["step"] or [])],
                    "capped": e["capped"] or [], "drift": [round(x, 5) for x in (e["drift"] or [])]})
    return out


def _window(ts_o: np.ndarray, now: float, lag: float, span_s: float) -> tuple[int, int]:
    """Positions of the rows whose labels resolved in (now − span, now]."""
    return int(np.searchsorted(ts_o, now - span_s - lag, side="right")), int(np.searchsorted(ts_o, now - lag, side="right"))


def _line_at(line_log: dict, key, ts: np.ndarray, default: float) -> np.ndarray:
    """The line (configuration, strategy) ``key`` had in force when each row was scored."""
    log_c = line_log.get(key) or []
    if not log_c:
        return np.full(len(ts), default)
    at = np.array([t for t, _ in log_c]); vals = np.array([v for _, v in log_c])
    i = np.searchsorted(at, ts, side="right") - 1
    return np.where(i >= 0, vals[np.clip(i, 0, None)], default)


def _recalibrate(ds, order, ts_o, scores, lines, sizings, now, lags, holds_min, line_log) -> None:
    """One line and sizing per strategy, calibrated on the frozen fly's scores and shared by every configuration.

    Until 2026-09-22 each configuration calibrated its own. With the plastic and frozen weights no more than 5 % apart,
    the frozen arm still took 4,216 selection-half trades to the chosen arm's 2,609: the arms traded different slices
    because of their lines, so "plastic beats frozen" measured line calibration, not plasticity. A shared line leaves
    the weights as the only difference between arms, which is the question the replay exists to answer."""
    for s_, (lag, H) in enumerate(zip(lags, holds_min)):
        a, b = _window(ts_o, now, lag, fly_calibrate.WINDOW_DAYS * DAY_S)
        rows = order[a:b]; y = _labels(ds, H)[rows]
        sc = scores[0, s_, a:b]; has = np.isfinite(sc)
        cal = fly_calibrate.calibrate(ds.ts[rows[has]], ds.mint[rows[has]], H * 60.0, sc[has], y[has], prev_line=float(lines[0, s_]), prev_sizing=sizings[0][s_])
        for c in range(lines.shape[0]):
            key = (c, s_)
            if key not in line_log:
                line_log[key] = [(-math.inf, float(lines[c, s_]))]
            lines[c, s_] = cal.line; sizings[c][s_] = cal.sizing
            line_log[key].append((now, float(cal.line)))


def _governance(ds, order, ts_o, scores, lines, line_log, bank, now, lags, holds_min, counts) -> None:
    for s_, (lag, H) in enumerate(zip(lags, holds_min)):
        a3, b3 = _window(ts_o, now, lag, gov.SHADOW_DAYS * DAY_S); a1, b1 = _window(ts_o, now, lag, gov.IC_HOURS * HOUR_S)
        r3 = order[a3:b3]; r1 = order[a1:b1]; y = _labels(ds, H)
        lf = _line_at(line_log, (0, s_), ts_o[a3:b3], float(lines[0, s_]))
        sf = scores[0, s_, a3:b3]
        rf = gov.picks_returns(ds.ts[r3], ds.mint[r3], H * 60.0, np.nan_to_num(sf, nan=-np.inf), lf, y[r3])
        drift = bank.drift(s_).tolist() if bank.learn.shape[0] > 1 else bank.drift().tolist()
        for c in range(1, lines.shape[0]):
            lc = _line_at(line_log, (c, s_), ts_o[a3:b3], float(lines[c, s_]))
            sc3 = np.nan_to_num(scores[c, s_, a3:b3], nan=-np.inf)
            rp = gov.picks_returns(ds.ts[r3], ds.mint[r3], H * 60.0, sc3, lc, y[r3])
            s1 = scores[c, s_, a1:b1]; has = np.isfinite(s1)
            out = gov.check((rp, rf), (s1[has], y[r1][has]), drift[c])
            counts[(c, s_)]["checks"] += 1
            for t in out["triggers"]:
                counts[(c, s_)][t] += 1
            counts[(c, s_)]["any"] += bool(out["triggers"])


def _arm(ds, order, ts_o, scores_cs, line_log, key, default_line, mask_pos) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (N) scores and per-row lines of (configuration, strategy) ``key`` on ``mask_pos`` (NaN elsewhere)."""
    S_full = np.full(len(ds.y), np.nan); L_full = np.full(len(ds.y), np.inf)
    pos = np.flatnonzero(mask_pos & np.isfinite(scores_cs))
    S_full[order[pos]] = scores_cs[pos]; L_full[order[pos]] = _line_at(line_log, key, ts_o[pos], default_line)
    return S_full, L_full


def _book(ds, order, ts_o, scores, line_log, choice: list[int], boot_lines: list[float], holds_min, mask_pos):
    """The combined book of the per-strategy choices: per row, among the strategies at/above their line in force, the one
    with the largest margin; returns (pick, per-row hold s, per-row return)."""
    N = len(ds.y); key = np.full(N, -np.inf); pick = np.zeros(N, bool); hold = np.zeros(N); ret = np.full(N, np.nan, np.float32)
    for s_, c in enumerate(choice):
        Sf, Lf = _arm(ds, order, ts_o, scores[c, s_], line_log, (c, s_), boot_lines[s_], mask_pos)
        lab = _labels(ds, holds_min[s_])
        ok = (Sf >= Lf) & np.isfinite(lab)          # a row whose hold runs past the corpus has no label: not a trade
        m = np.where(ok, Sf - Lf, -np.inf); take = ok & (m > key)
        key = np.where(take, m, key); pick |= ok
        hold = np.where(take, holds_min[s_] * 60.0, hold); ret = np.where(take, lab, ret)
    return pick, hold, ret


def _verdict(ds, order, ts_o, live_from, scores, line_log, configs, bank, gov_counts, boot, S, save_verdict, secs, fly, names, holds_min, lines, learn=None) -> dict:
    days_r = sorted(set(ds.day[order[live_from:]].tolist()))
    sel_days, ev_days = days_r[: len(days_r) // 2], days_r[len(days_r) // 2:]
    day_o = ds.day[order]
    live_pos = np.arange(len(order)) >= live_from
    sel_pos = live_pos & np.isin(day_o, sel_days); ev_pos = live_pos & np.isin(day_o, ev_days)
    boot_lines = [fly.lines[k] for k in names]
    per_strategy = {}; choice = []
    for s_, name in enumerate(names):
        y = _labels(ds, holds_min[s_]); table = []
        for c, (alpha, hl) in enumerate(configs):
            Sf, Lf = _arm(ds, order, ts_o, scores[c, s_], line_log, (c, s_), boot_lines[s_], sel_pos)
            tr = taken_idx(ds.ts, ds.mint, holds_min[s_] * 60.0, np.flatnonzero(Sf >= Lf)); r = y[tr]
            table.append({"config": c, "alpha": alpha, "half_life_days": hl if math.isfinite(hl) else None, "trades": int(len(r)), "total": float(np.nansum(r)),
                          "mean": float(np.nanmean(r)) if len(r) else None, "governance": gov_counts.get((c, s_))})
        ok = [t for t in table if t["alpha"] > 0 and t["trades"] >= selector.MIN_LINE_TRADES]
        best = max(ok, key=lambda t: t["total"])["config"] if ok else 0
        choice.append(best)
        per_strategy[name] = {"configs": table, "chosen": table[best] if ok else None, "alpha": table[best]["alpha"], "half_life_days": table[best]["half_life_days"],
                              "hold_min": holds_min[s_], "plastic": bool(ok)}

    def judge(ch):
        pick, hold, ret = _book(ds, order, ts_o, scores, line_log, ch, boot_lines, holds_min, ev_pos)
        test = np.zeros(len(ds.y), bool); test[order[ev_pos]] = True
        ev = evaluate(ds, np.where(pick, 1.0, np.nan), test & pick, 0.5, "fly", hold_s=hold, returns=ret)
        rnd = summarize(random_trades(ds, test, max(1, int(ev["pooled"]["n"] or 1))))
        return ev, rnd

    out = {"S": str(S), "selection_days": [str(sel_days[0]), str(sel_days[-1])] if sel_days else None, "evaluation_days": [str(ev_days[0]), str(ev_days[-1])] if ev_days else None,
           "strategies": names, "per_strategy": per_strategy, "configs": per_strategy[names[0]]["configs"],
           "bootstrap": {k: boot.get(k) for k in ("lines", "calibration", "diagnostics", "gates_ok", "gate_failures")}, "data": fly_selector.FLY_VERSION,
           "secs": secs, "finished_at": datetime.now(timezone.utc).isoformat(), "learning": _learning_series(learn or {})}
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
    log.info("fly replay from %s: %s — %s | frozen %s", S, "PASSED" if out.get("passed") else "FAILED", out.get("reason"), out["frozen"])
    log.info("fly replay verdict: %s", {k: out.get(k) for k in ("passed", "reason", "evaluation", "frozen", "random")})
    if save_verdict:
        record_event("info" if out.get("passed") else "warning", "fly_replay", "fly replay " + ("passed" if out.get("passed") else "failed"),
                     {k: out.get(k) for k in ("S", "passed", "reason", "evaluation", "random", "frozen", "plastic_beats_frozen")} |
                     {"per_strategy": {k: {x: v[x] for x in ("alpha", "half_life_days", "hold_min", "plastic")} for k, v in per_strategy.items()}})
        try:        # NaN is valid Python and invalid JSON: never let the store discard a finished replay
            with transaction() as conn:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (KEY, json.dumps(prog._finite(out), default=str)))
        except Exception:
            log.exception("could not store the replay verdict; it is in the log above and the fly cannot be armed until it is stored")
        try:        # the chosen configuration's learning, where the console already looks for it
            c0 = choice[0] if choice else 0
            with transaction() as conn:
                for rec in out["learning"]:
                    d = date.fromisoformat(rec["day"]); pick = lambda k: (rec[k][c0] if len(rec.get(k) or []) > c0 else None)
                    conn.execute("INSERT INTO fly_updates (hour, n, mean_delta, mean_abs_delta, step, capped, drift, ic, detail) "
                                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (hour) DO NOTHING",
                                 (datetime(d.year, d.month, d.day, tzinfo=timezone.utc), rec["n"], pick("mean_delta"), pick("mean_abs_delta"),
                                  pick("step"), pick("capped"), pick("drift"), None,
                                  json.dumps({"source": "replay", "config": c0, "per_config_drift": rec["drift"]}, default=str)))
        except Exception:
            log.exception("could not store the replay's learning statistics (the verdict is stored)")
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
            out["live_bootstrap"] = {k: v for k, v in fly_selector.main(days=days, stop_event=stop_event).items() if k in ("snapshot_id", "lines", "gates_ok", "gate_failures", "calibration")}
    return out
