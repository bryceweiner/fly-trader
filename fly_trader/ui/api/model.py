"""Model & training (memecoins): which model trades and how it tested
against random picks, the automatic retraining pipeline (the selector every 7 days until the fly holds its seat; the fly
bootstrapped once), the plastic fly, and every saved model.

The fly modules (train/fly_selector, agent/fly_session) import torch, so they are imported where used, not when the
console starts."""
from __future__ import annotations

import json

from ... import config, markets
from ...db.queries import q
from ...train import pipeline
from ...train.fly_governance import DRIFT_MAX
from ...train.selector import is_current, is_deployable
from .. import runtime
from ..hub import ActionError, action, topic
from ..state import backtest_line, jv, latest_snapshot, loaded_model, parse_note, pct, runs_elsewhere, setting, until

FLY_COMMANDS = ("pause", "resume", "rollback")


# ---------------------------------------------------------------- reads shared by topics and action guards
def _pipeline() -> dict:
    """train/pipeline.state() through the topic thread's reused connection."""
    return setting(pipeline.KEY)[0]


def _fly_status() -> dict:
    return setting(markets.SOL.fly_status_key)[0]


def _fly_running(fly: dict) -> bool:
    return bool(fly.get("stage")) and fly.get("stage") != "not trading"


def _training(alive: bool, ps: dict) -> bool:
    return alive and str(ps.get("stage", "")).startswith("training")


def _model_at_work() -> dict | None:
    return loaded_model() if runtime.sup().alive("runner") else None


# ---------------------------------------------------------------- Model at work
def _half(x: dict | None, k: str):
    return None if not x or x.get(k) is None else float(x[k])


def _pct100(x: dict | None, k: str):
    v = _half(x, k)
    return None if v is None else v * 100


def _best(b: dict | None) -> str:
    """The most profitable setting a component tried, whether or not it cleared the bars — why a dropped one was dropped."""
    if not b:
        return "—"
    win = f"{b['win'] * 100:.0f}% winners" if b.get("win") is not None else "—"
    pf = f"PF {b['pf']:.2f}" if b.get("pf") is not None else "PF ∞"
    return f"{win} · {pf} · {b.get('n', 0)} trades · total {b.get('total', 0):+.1f}"


def _sizing(meta: dict) -> dict:
    """How big each buy is: the model's certainty bands from its backtest, and the backtest replayed with that sizing."""
    tbl = meta.get("sizing") or []; bk = meta.get("bankroll") or {}
    if not tbl:
        return {"caption": f"Fixed {config.MAX_POSITION_SOL:g} SOL per buy: this model was trained before sizing and has no certainty bands.", "bands": [], "bankroll": None}
    caption = (f"Each buy's size comes from how certain the model is. Its backtest trades are split into bands by how far the score cleared the buy line; "
               f"each band gets {config.KELLY_FRACTION:g} × its growth-optimal share of the bankroll (wealth minus the {config.GAS_RESERVE_SOL:g} SOL gas reserve), "
               f"at most {config.MAX_POSITION_FRACTION:.0%} of the bankroll and {config.MAX_POOL_SHARE:.0%} of the pool. Buys under {config.MIN_POSITION_SOL:g} SOL are skipped.")
    bands = [{"band": f"+{float(b.get('lo') or 0):.3f} and up", "n": b.get("n"), "mean": (b.get("mean") or 0) * 100, "win": (b.get("win") or 0) * 100,
              "bet": min(config.KELLY_FRACTION * float(b.get("kelly") or 0), config.MAX_POSITION_FRACTION) * 100} for b in tbl]
    s, f = bk.get("sized") or {}, bk.get("fixed") or {}
    bankroll = (f"The backtest's trades replayed from {s.get('start_sol', 0):g} SOL: with this sizing → {s.get('final_sol', 0):.2f} SOL (worst drawdown {s.get('max_drawdown', 0):.0%}); "
                f"at a fixed {config.MAX_POSITION_SOL:g} SOL → {f.get('final_sol', 0):.2f} SOL (worst drawdown {f.get('max_drawdown', 0):.0%}).") if s and f else None
    return {"caption": caption, "bands": bands, "bankroll": bankroll}


def _stack(meta: dict) -> dict | None:
    """The selector's strategy stack: every component tried (kept or dropped, and why) and the strategies it trades."""
    comps = meta.get("components") or []
    if not comps:
        return None
    rows = [{"component": c.get("name"), "kept": c.get("passed"), "why": c.get("reason"), "win_sel": _pct100(c.get("selection"), "win"),
             "win_eval": _pct100(c.get("evaluation"), "win"), "pf_eval": _half(c.get("evaluation"), "pf"), "total_eval": _half(c.get("evaluation"), "total"),
             "n_eval": _half(c.get("evaluation"), "n"), "trials": c.get("trials"), "best": _best(c.get("best_seen")),
             "params": json.dumps(c.get("params") or {}, default=str)} for c in comps]
    strat = meta.get("strategies") or {}
    srows = [{"strategy": k, "hold_min": v.get("hold_min"), "line": (v.get("line") or 0) * 100, "trigger": json.dumps(v.get("thr") or {}, default=str),
              "n_eval": _half(v.get("evaluation"), "n"), "win_eval": _pct100(v.get("evaluation"), "win"), "pf_eval": _half(v.get("evaluation"), "pf")}
             for k, v in (strat.items() if isinstance(strat, dict) else ())]
    return {"fallback": bool(meta.get("fallback")), "components": rows, "strategies": srows}


def _halves(meta: dict) -> list[dict]:
    """The backtest's parts side by side (the chart above the model's numbers): the selection half that chose the settings,
    the evaluation half that judged them, and the holdout no fit ever saw, overall and per chain."""
    rb = meta.get("random_baseline") or {}
    parts = [("Selection", meta.get("selection"), None), ("Evaluation", meta.get("walk_forward"), rb.get("mean")), ("Holdout", meta.get("holdout"), None)]
    parts += [(f"Holdout · {markets.MARKETS[k].name if k in markets.MARKETS else k}", v, None) for k, v in sorted((meta.get("chains") or {}).items())]
    out = []
    for name, h, rnd in parts:
        if not isinstance(h, dict) or h.get("mean") is None:
            continue
        r = h.get("random_mean") if h.get("random_mean") is not None else rnd
        out.append({"name": name, "n": h.get("n"), "mean": h["mean"], "random_mean": r, "win": h.get("win"), "pf": h.get("pf"), "weeks": h.get("weeks"),
                    "weeks_pos": h.get("weeks_pos"), "days": h.get("days") if isinstance(h.get("days"), list) else None})
    return out


@topic("model.loaded", every=10)
def loaded(params: dict) -> dict:
    """The model at work (the running selector session's snapshot) and the newest model on the current definitions."""
    m, latest = _model_at_work(), latest_snapshot("selector")
    out: dict = {"model": None, "latest": None}
    if m:
        meta = m["meta"]; d: dict = {"id": m["id"], "current": is_current(meta)}
        if d["current"]:
            wf = meta.get("walk_forward") or {}; rb = meta.get("random_baseline") or {}; run = m.get("run") or {}; mi = meta.get("model") or {}
            d.update(ts=m["ts"], since=m["since"], first_day=meta.get("first_day"), trained_through=meta.get("trained_through"),
                     threshold=float(run.get("threshold") or meta.get("threshold") or 0), horizon_min=run.get("horizon_min") or meta.get("horizon_min"),
                     min_age_h=mi.get("min_age_h"), wf={k: wf.get(k) for k in ("n", "days", "mean", "win", "pf", "days_positive", "weeks", "weeks_pos")},
                     random_mean=rb.get("mean"), backtest_line=backtest_line(meta), sizing=_sizing(meta), stack=_stack(meta), halves=_halves(meta))
        out["model"] = d
    if latest:
        out["latest"] = {"id": latest["id"], "ts": latest["ts"], "deployable": is_deployable(latest["meta"]), "deploy_reason": latest["meta"].get("deploy_reason")}
    return out


# ---------------------------------------------------------------- Automatic retraining
@topic("model.retraining", every=3)
def retraining(params: dict) -> dict:
    alive = runtime.sup().alive("train"); ps = _pipeline(); tr, _ = setting("training_status"); ho, _ = setting(markets.SOL.handover_key)
    stage = ps.get("stage"); running = _training(alive, ps); sel = latest_snapshot("selector"); nxt = ps.get("next_run_at")
    progress = None
    if running and tr.get("step") is not None and tr.get("total"):
        progress = {"value": min(1.0, float(tr["step"]) / max(float(tr["total"]), 1.0)),
                    "text": f"{tr.get('stage', '')}" + (f" · about {float(tr['eta_s']) / 60:.0f} min left" if tr.get("eta_s") else "")}
    return {"handover": bool(ho), "interval_days": pipeline.INTERVAL_DAYS, "alive": alive, "running": running, "stage": stage,
            "status": str(stage or "idle").capitalize() if alive else "Off", "last_run_at": ps.get("last_run_at"),
            "next_run": until(nxt) if nxt else ("as soon as the data is ready" if alive else "—"),
            "latest": {"id": sel["id"], "deployable": is_deployable(sel["meta"]), "backtest_line": backtest_line(sel["meta"])} if sel else None,
            "progress": progress,
            "waiting": (f"Waiting for the training data: {ps['waiting']}. Checking again {until(ps.get('retry_after'))}."
                        if stage == "waiting for data" and ps.get("waiting") else None),
            "failed": f"The last run failed: {ps['last_error']}. Retrying {until(ps.get('retry_after'))}." if stage == "failed" and ps.get("last_error") else None}


def _start_trainer() -> bool:
    """Start the train worker unless it runs; True when this call started it. A refusal (another process hosts it) is shown as is."""
    s = runtime.sup()
    if s.alive("train"):
        return False
    try:
        s.start("train", started_by="console")
    except runtime.ObserverMode:
        raise
    except RuntimeError as e:
        raise ActionError(str(e)) from e
    return True


@action("model.retrain")
def retrain(body: dict) -> dict:
    """Retrain now: the whole pipeline runs at once (the 7-day schedule restarts from this run); starts the Trainer when it is off."""
    if _training(runtime.sup().alive("train"), _pipeline()):
        raise ActionError("A training run is already in progress.")
    pipeline.request_run()
    started = _start_trainer()
    return {"message": "Retraining requested" + (": the Trainer was off and has been started." if started else ": the Trainer runs it now.")}


@action("model.train_stop")
def train_stop(body: dict) -> dict:
    if not _training(runtime.sup().alive("train"), _pipeline()):
        raise ActionError("No training run is in progress.")
    runtime.sup().request_stop("train")
    return {"message": "Stopping the current run; the schedule resumes when the Trainer is started again."}


# ---------------------------------------------------------------- Latest selector backtest, day by day
@topic("model.results", every=15)
def results(params: dict) -> dict:
    wfd = q("SELECT detail FROM events WHERE source = 'selector' AND message LIKE 'walk-forward%%' ORDER BY id")
    if not wfd:
        return {"hidden": False, "rows": []}
    if not latest_snapshot("selector"):
        return {"hidden": True, "rows": []}
    rows = [{"day": d.get("day"), "half": d.get("half"), "ic": d.get("ic"), "n": d.get("n"),
             "model": (d["mean"] * 100) if d.get("mean") is not None else None, "random": (d["random_mean"] * 100) if d.get("random_mean") is not None else None,
             "winners": (d["win"] * 100) if d.get("win") is not None else None, "pf": d.get("pf")} for d in (jv(r["detail"]) for r in wfd)]
    return {"hidden": False, "rows": rows}


# ---------------------------------------------------------------- The plastic fly
def _fly_bootstrap() -> dict | None:
    """The newest bootstrap on the current fly definitions, with its network diagnostics and whether it may trade."""
    from ...train import fly_selector
    for r in q("SELECT id, ts, note FROM brain_snapshots WHERE kind = 'fly_selector' ORDER BY id DESC LIMIT 10"):
        m = parse_note(r["note"])
        if fly_selector.is_current(m):
            d = m.get("diagnostics") or {}; ok, why = fly_selector.deployable(m)
            return {"id": r["id"], "ts": r["ts"], "ok": ok,
                    "text": (f"the mushroom body carries {pct(d.get('mbon_share'), 0, False)} of the score, {pct(d.get('mbon_saturated'), 0, False)} of MBONs saturated, "
                             f"{pct(d.get('kc_active'), 0, False)} of Kenyon cells active, slope {d.get('slope') or 0:.2f} against its teacher. "
                             + ("May trade: " if ok else "May not trade: ") + why + ".")}
    return None


def _per_strategy(f: dict) -> list[dict]:
    """The fly's checks per strategy, when it trades several (fly_status.checks keyed by strategy instead of one ic/shadow),
    with each strategy's buy line and its frozen shadow's."""
    out, lines, frozen = [], f.get("lines") or {}, f.get("frozen_lines") or {}
    for k, v in (f.get("checks") or {}).items():
        if isinstance(v, dict) and ("ic" in v or "shadow" in v):
            sh = v.get("shadow") or {}
            out.append({"strategy": k, "ic": v.get("ic"), "drift": v.get("drift"), "mean_plastic": sh.get("mean_plastic"), "mean_frozen": sh.get("mean_frozen"),
                        "n_plastic": sh.get("n_plastic"), "n_frozen": sh.get("n_frozen"), "triggers": ", ".join(v.get("triggers") or []) or None,
                        "line": lines.get(k), "frozen_line": frozen.get(k)})
    return out


@topic("model.fly", every=15)
def fly(params: dict) -> dict:
    f = _fly_status(); rp, _ = setting("fly_replay"); running = _fly_running(f)
    out: dict = {"running": running, "frozen": bool(f.get("learning_frozen")), "runner_alive": runtime.sup().alive("runner"), "replay": None, "live": None,
                 "detail": None, "updates": [], "rollbacks": [], "bootstrap": _fly_bootstrap()}
    if rp:
        ev, rnd, fr = rp.get("evaluation") or {}, rp.get("random") or {}, rp.get("frozen") or {}
        out["replay"] = {"passed": bool(rp.get("passed")), "reason": rp.get("reason"), "n": ev.get("n"), "mean": ev.get("mean"), "random_mean": rnd.get("mean"),
                         "frozen_mean": fr.get("mean"), "days_positive": ev.get("days_positive"), "days": ev.get("days"), "alpha": f"{rp.get('alpha') or 0:g}",
                         "half_life": f"{rp['half_life_days']:g} days" if rp.get("half_life_days") else "none",
                         "caption": (f"Bootstrapped for {rp.get('S')} and never retrained: every scored minute taught its mushroom body with the realized return two hours later. "
                                     f"Configuration chosen on {' → '.join(rp.get('selection_days') or ['—'])}, judged on {' → '.join(rp.get('evaluation_days') or ['—'])}.")}
    if running:
        ch = f.get("checks") or {}; sh = ch.get("shadow") or {}
        out["live"] = {"line": f.get("line"), "frozen_line": f.get("frozen_line"), "drift": f.get("drift"), "pending": f.get("pending"), "ic": ch.get("ic"),
                       "shadow_plastic": sh.get("mean_plastic"), "shadow_frozen": sh.get("mean_frozen"), "device": f.get("device"), "drift_max": DRIFT_MAX,
                       "per_strategy": _per_strategy(f) if "ic" not in ch and "shadow" not in ch else []}
        out["updates"] = list(reversed(q("SELECT hour, n, mean_abs_delta, drift, ic FROM fly_updates ORDER BY hour DESC LIMIT 72")))
        out["rollbacks"] = q("SELECT ts, reason, from_snapshot, to_snapshot FROM fly_rollbacks ORDER BY id DESC LIMIT 20")
    else:
        out["detail"] = f"Not trading: {f.get('detail') or 'the trading engine is stopped'}."
    return out


@action("model.fly_command")
def fly_command(body: dict) -> dict:
    """Pause / resume the fly's learning, or roll it back: queued for the running fly, applied at its next minute."""
    cmd = body.get("cmd")
    if cmd not in FLY_COMMANDS:
        raise ActionError(f"cmd must be one of {', '.join(FLY_COMMANDS)}", 400)
    if not _fly_running(_fly_status()) or not (runtime.sup().alive("runner") or runs_elsewhere("runner")):
        raise ActionError("The fly is not running: start the trading engine first.")    # its status outlives a stopped engine
    from ...agent import fly_session
    fly_session.send_command(cmd)
    return {"message": "Sent: applied at the next minute"}


@action("model.fly_rebootstrap")
def fly_rebootstrap(body: dict) -> dict:
    """The selector retrains and teaches a new fly. Unlike the Streamlit button, this also starts the Trainer when it is off
    (as Retrain now does), so the request is not left waiting for someone to start it."""
    pipeline.request_run(fly=True, reason="console")
    started = _start_trainer()
    return {"message": "Re-bootstrap requested" + (": the Trainer was off and has been started." if started else ".")}


@action("model.fly_reset")
def fly_reset(body: dict) -> dict:
    """Archive and clear the fly's paper book and everything it learned (ops/reset.reset_fly); never under a running engine —
    wherever it runs — and never while an engine starts (the same guard as the Safety page's resets)."""
    from ...ops import reset
    from .safety import engine_stopped, no_reset
    engine_stopped("runner", "trading engine")
    with no_reset():
        engine_stopped("runner", "trading engine")
        r = reset.reset_fly(reason="console button")
    return {"message": f"Archived to {r['archived_to']}", "archived_to": r["archived_to"], "rows": r.get("rows")}


# ---------------------------------------------------------------- Saved models, trainer log
@topic("model.history", every=30)
def history(params: dict) -> dict:
    from ...train import fly_selector
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('selector', 'fly_selector') ORDER BY id DESC LIMIT 30")
    in_use = (_model_at_work() or {}).get("id")
    rows = []
    for r in snaps:
        m = parse_note(r["note"]); wf = m.get("walk_forward") or m.get("fly") or {}; rb = m.get("random_baseline") or m.get("random") or {}
        sel = r["kind"] == "selector"; ok = is_current(m) if sel else fly_selector.is_current(m)
        if not sel:
            cal = m.get("calibration") or {}
            wf, rb = {"n": cal.get("trades"), "mean": cal.get("mean")}, {}
        rows.append({"id": r["id"], "model": f"#{r['id']}", "type": "selector" if sel else "fly bootstrap", "saved": r["ts"], "data": "current" if ok else "outdated",
                     "trades": wf.get("n") if ok else None, "mean": (wf["mean"] * 100) if ok and wf.get("mean") is not None else None,
                     "random": (rb["mean"] * 100) if ok and rb.get("mean") is not None else None, "pf": wf.get("pf") if ok else None,
                     "deployed": (bool(m.get("deployable")) if sel else fly_selector.deployable(m)[0]) if ok else None, "in_use": r["id"] == in_use})
    return {"rows": rows}


@topic("model.trainer_log", every=5)
def trainer_log(params: dict) -> dict:
    return {"text": runtime.sup().log_tail("train", 40) or "(no log yet)"}

