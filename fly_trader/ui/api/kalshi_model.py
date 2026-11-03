"""Kalshi model & training: the strategy stack the visual fly was taught by, the automatic pipeline, the replay verdict,
the plastic visual fly's controls and health, and every saved Kalshi model.

The Kalshi modules (selector, fly, pipeline, fly_session, ops.reset) are imported inside the functions: kalshi/fly.py
pulls torch, which a console serving a dormant Kalshi never needs, and tests patch them on their modules."""
from __future__ import annotations

import json

from ...db.queries import q, q1, read_connection
from .. import runtime
from ..hub import ActionError, action
from ..state import kalshi_state, parse_note, pct, runs_elsewhere, setting, until
from .kalshi_overview import kalshi_topic, require_kalshi

FLY_COMMANDS = ("pause", "resume", "rollback")


def _half(x, k):
    return None if not x or x.get(k) is None else float(x[k])


def _ts(sid) -> object:
    """A snapshot's saved time: kalshi/selector.latest_current and train/fly_selector.latest_current return only id,
    path and note (the Streamlit page read ``['ts']`` off them and raised KeyError)."""
    r = q1("SELECT ts FROM brain_snapshots WHERE id = %s", (sid,)) if sid is not None else None
    return r["ts"] if r else None


def _meta(snap: dict) -> dict:
    return snap.get("meta") or parse_note(snap.get("note"))


def _comma(x) -> str:
    return f"{x:,}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "?"


def _training(alive: bool, ps: dict) -> bool:
    return alive and str(ps.get("stage", "")).startswith(("training", "bootstrapping"))


def _fly_running(fly: dict) -> bool:
    return bool(fly.get("stage")) and fly.get("stage") != "not trading"


@kalshi_topic("kalshi_model.stack", every=15)
def stack(params: dict) -> dict:
    """The newest *deployable* Kalshi selector on the current definitions (kalshi/selector.latest_current skips the
    rest despite its name): the stack that teaches the visual fly, its walk-forward evaluation, components and strategies."""
    from ...kalshi import selector as kselector
    with read_connection() as conn:
        sel = kselector.latest_current(conn)
    if not sel:
        return {"selector": None}
    meta = _meta(sel); wf = meta.get("walk_forward") or {}; rb = meta.get("random_baseline") or {}
    comps = [{"component": c.get("name"), "kept": c.get("passed"), "why": c.get("reason"),
              "win": (_half(c.get("evaluation"), "win") or 0) * 100 if _half(c.get("evaluation"), "win") is not None else None,
              "pf": _half(c.get("evaluation"), "pf"), "trades": _half(c.get("evaluation"), "n"), "trials": c.get("trials")} for c in meta.get("components") or []]
    strat = [{"strategy": name, "arm": v.get("hold_min"), "line": v.get("line"), "trigger": json.dumps(v.get("thr") or {}, default=str), "trades": _half(v.get("evaluation"), "n"),
              "win": (_half(v.get("evaluation"), "win") or 0) * 100 if _half(v.get("evaluation"), "win") is not None else None,
              "pf": _half(v.get("evaluation"), "pf")} for name, v in (meta.get("strategies") or {}).items()]
    return {"selector": {"id": sel["id"], "ts": _ts(sel["id"]), "first_day": meta.get("first_day"), "last_day": meta.get("last_day"), "rows": meta.get("rows"),
                         "deployable": bool(meta.get("deployable")), "deploy_reason": str(meta.get("deploy_reason") or "")},
            "wf": {"n": wf.get("n"), "mean": wf.get("mean"), "win": wf.get("win"), "pf": wf.get("pf"), "days_positive": wf.get("days_positive"), "days": wf.get("days")},
            "random_mean": rb.get("mean"), "components": comps, "strategies": strat}


@kalshi_topic("kalshi_model.retraining", every=3)
def retraining(params: dict) -> dict:
    from ...kalshi import fly as KF, pipeline, selector as kselector
    ps = pipeline.state(); alive = runtime.sup().alive("kalshi_train"); running = _training(alive, ps)
    pr = setting("training_status")[0] if running else {}
    with read_connection() as conn:
        sel = kselector.latest_current(conn); boot = KF.latest_current(conn); dep = KF.latest_deployable(conn)
    progress = None
    if running and pr.get("total"):
        progress = {"value": min(1.0, float(pr.get("step") or 0) / max(float(pr["total"]), 1.0)),
                    "text": f"{pr.get('stage', '')}" + (f" · about {float(pr['eta_s']) / 60:.0f} min left" if pr.get("eta_s") else "")}
    return {"interval_days": pipeline.INTERVAL_DAYS, "alive": alive, "running": running, "status": (ps.get("stage") or "idle").capitalize() if alive else "Off",
            "last_run_at": ps.get("last_run_at"),
            "next_run": until(ps.get("next_run_at")) if ps.get("next_run_at") else ("as soon as the corpus is built" if alive else "—"),
            "selector_id": sel["id"] if sel else None, "bootstrap_id": boot["id"] if boot else None, "may_trade": bool(dep) if boot else None, "progress": progress,
            "waiting": (f"Waiting for the corpus: {ps['waiting']}. Checking again {until(ps.get('retry_after'))}." if ps.get("stage") == "waiting for data" and ps.get("waiting") else None),
            "failed": (f"The last run failed: {ps['last_error']}. Retrying {until(ps.get('retry_after'))}." if ps.get("stage") == "failed" and ps.get("last_error") else None)}


@kalshi_topic("kalshi_model.visual_fly", every=15)
def visual_fly(params: dict) -> dict:
    from ...kalshi import fly as KF
    from ...train.fly_governance import DRIFT_MAX
    k = kalshi_state(); fly = k["fly"]; rp = k["replay"]
    running = _fly_running(fly); frozen = bool(fly.get("learning_frozen"))
    replay = None
    if rp:
        ev, rnd, fr = rp.get("evaluation") or {}, rp.get("random") or {}, rp.get("frozen") or {}; per = rp.get("per_strategy") or {}
        replay = {"passed": bool(rp.get("passed")), "reason": rp.get("reason"), "n": ev.get("n"), "mean": ev.get("mean"), "random_mean": rnd.get("mean"), "frozen_mean": fr.get("mean"),
                  "days_positive": ev.get("days_positive"), "days": ev.get("days"), "alpha": f"{rp.get('alpha') or 0:g}",
                  "half_life": f"{rp['half_life_days']:g} days" if rp.get("half_life_days") else "none",
                  "per_strategy": ("Per strategy: " + " · ".join(f"{s} ({v.get('arm')}): α {v.get('alpha')}, τ½ {v.get('half_life_days') or '∞'}" for s, v in per.items())) if per else None}
    out = {"running": running, "frozen": frozen, "runner": k["runner"], "replay": replay, "strategies": None, "updates": None, "rollbacks": None,
           "pending": fly.get("pending"), "learned": (fly.get("learned") or {}).get("n", 0), "device": fly.get("device"), "drift": float(fly.get("drift") or 0), "drift_max": DRIFT_MAX,
           "not_trading": None if running else f"Not trading: {fly.get('detail') or k['trading_why']}", "bootstrap": None}
    if running:
        ch = fly.get("checks") or {}; lines = fly.get("lines") or {}; fl = fly.get("frozen_lines") or {}; arms = fly.get("arms") or {}; drift = fly.get("drift_by_strategy") or {}
        out["strategies"] = [{"strategy": s, "arm": arms.get(s), "line": float(lines[s]), "shadow": float(fl.get(s, lines[s])), "drift": float(drift.get(s) or 0) * 100,
                              "ic": (ch.get(s) or {}).get("ic"), "checks": ", ".join((ch.get(s) or {}).get("triggers") or []) or "ok"} for s in lines]
        out["updates"] = q("SELECT hour, n, mean_abs_delta, drift, ic FROM kalshi_fly_updates ORDER BY hour DESC LIMIT 72")[::-1]
        out["rollbacks"] = q("SELECT ts, reason, from_snapshot, to_snapshot FROM kalshi_fly_rollbacks ORDER BY id DESC LIMIT 20")
    b = k["bootstrap"]
    if b:
        m = _meta(b); d = m.get("diagnostics") or {}; ok, why = KF.deployable(m)
        out["bootstrap"] = {"id": b["id"], "ts": _ts(b["id"]), "may_trade": ok,
                            "text": (f"{_comma(m.get('photoreceptors'))} photoreceptors carry the inputs into {_comma(m.get('neurons'))} neurons; the mushroom body carries "
                                     f"{pct(d.get('mbon_share'), 0, False)} of the score, {pct(d.get('mbon_saturated'), 0, False)} of MBONs saturated, slope {d.get('slope') or 0:.2f} "
                                     "against its teacher. " + ("May trade: " if ok else "May not trade: ") + why + ".")}
    return out


@kalshi_topic("kalshi_model.history", every=15)
def history(params: dict) -> dict:
    """Every saved Kalshi model (stacks and visual-fly bootstraps) and the Kalshi trainer's log."""
    from ...kalshi import fly as KF, selector as kselector
    rows = []
    for r in q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('kalshi_selector', 'kalshi_fly_selector') ORDER BY id DESC LIMIT 30"):
        m = parse_note(r["note"]); sel = r["kind"] == "kalshi_selector"
        ok = kselector.is_current(m) if sel else KF.is_current(m)
        wf = (m.get("walk_forward") or {}) if sel else {"n": (m.get("calibration") or {}).get("trades"), "mean": (m.get("calibration") or {}).get("mean")}
        rows.append({"id": r["id"], "model": f"#{r['id']}", "type": "stack" if sel else "visual fly bootstrap", "saved": r["ts"], "data": "current" if ok else "outdated",
                     "trades": wf.get("n") if ok else None, "per_trade": (wf["mean"] * 100) if ok and wf.get("mean") is not None else None,
                     "put_to_work": (bool(m.get("deployable")) if sel else KF.deployable(m)[0]) if ok else None})
    return {"models": rows, "log": runtime.sup().log_tail("kalshi_train", 40) or ""}


# ---------------------------------------------------------------- actions
def _start_trainer() -> bool:
    """Start the Kalshi trainer when it is not running (it picks up the pipeline's request); True when this started it."""
    s = runtime.sup()
    if s.alive("kalshi_train"):
        return False
    try:
        s.start("kalshi_train", started_by="console")
    except runtime.ObserverMode:
        raise
    except RuntimeError as e:
        raise ActionError(f"Requested, but the Kalshi trainer did not start: {e}") from e
    return True


@action("kalshi_model.retrain")
def retrain(body: dict) -> dict:
    from ...kalshi import pipeline
    require_kalshi()
    if _training(runtime.sup().alive("kalshi_train"), pipeline.state()):
        raise ActionError("Kalshi training is already running.")
    pipeline.request_run()
    started = _start_trainer()
    return {"message": "Kalshi retraining requested" + ("; the Kalshi trainer started" if started else "")}


@action("kalshi_model.train_stop")
def train_stop(body: dict) -> dict:
    from ...kalshi import pipeline
    require_kalshi()
    if not _training(runtime.sup().alive("kalshi_train"), pipeline.state()):
        raise ActionError("Kalshi training is not running.")
    runtime.sup().request_stop("kalshi_train")
    return {"message": "Stop requested: the Kalshi trainer stops when its current step ends"}


@action("kalshi_model.fly_command")
def fly_command(body: dict) -> dict:
    from ...kalshi import fly_session
    require_kalshi()
    cmd = body.get("cmd")
    if cmd not in FLY_COMMANDS:
        raise ActionError(f"cmd must be one of {', '.join(FLY_COMMANDS)}", 400)
    if not (runtime.sup().alive("kalshi_runner") or runs_elsewhere("kalshi_runner")) or not _fly_running(setting("kalshi_fly_status")[0]):     # its status outlives a stopped engine
        raise ActionError("The visual fly is not running.")
    fly_session.send_command(cmd)
    return {"message": "Sent: applied at the next minute"}


@action("kalshi_model.fly_rebootstrap")
def fly_rebootstrap(body: dict) -> dict:
    """The Kalshi stack refits and teaches a new visual fly. The Streamlit button only wrote the request, which waited for
    a trainer someone started; this starts the trainer too, as Retrain now does."""
    from ...kalshi import pipeline
    require_kalshi()
    pipeline.request_run(fly=True, reason="console")
    started = _start_trainer()
    return {"message": "Re-bootstrap requested" + ("; the Kalshi trainer started" if started else "")}


@action("kalshi_model.fly_reset")
def fly_reset(body: dict) -> dict:
    from ...ops import reset as ops_reset
    from .safety import engine_stopped, no_reset
    require_kalshi()
    engine_stopped("kalshi_runner", "Kalshi engine")
    with no_reset():
        engine_stopped("kalshi_runner", "Kalshi engine")
        r = ops_reset.reset_kalshi_fly(reason="console button")
    return {"message": f"Archived to {r.get('archived_to')}", "archived_to": r.get("archived_to"), "rows": r.get("rows")}
