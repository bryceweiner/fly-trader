"""A matched selector and fly (opt-in, 2026-10-23): both trained on exactly the same data, the fly taught by the very
model it is judged against, then both scored on the days neither saw.

Bryce, 2026-10-23: "I want them to match. Retrain when the data backfill is complete, so that the selector and fly are
trained on the exact same data. Then report the results of each strategy on the OOS data."

On one decision set pinned to ``last_day`` (decisions.build), into the scratch folder ``out`` and nowhere else: no
brain_snapshots row, no event, no training status, the walk-forward cache inside ``out``.
1. The selector: the strategy stack (strategies.fit_stack; its last ``HOLDOUT_DAYS`` days are withheld from every fit),
   its full fit and its holdout-blind fit (final_models) -> selector.skops (the full model with the blind fit inside, as
   the trainer saves it) and selector.json (what the trainer would note, the stack's own holdout record included).
2. The blind fly: taught by the blind fit on exactly that fit's rows (fly_selector.matched_teacher, bootstrap_matched)
   -> fly_blind/ (fly.pt, note.json, the per-epoch checkpoint).
3. The out-of-sample report: the blind selector's and the blind fly's picks on the holdout days, per strategy and per
   chain (fly_selector.oos_report) -> oos.json, oos.txt. The fly's own combined book there is its evidence.
4. With ``replay``: the blind fly learning over the holdout days as the live fly learns (train/fly_replay.run from the
   first holdout day; no verdict stored, its trades in replay/) -> replay.json.
5. The full fly: taught by the full fit on all its rows, carrying the blind pair's evidence (fly_selector.deployable
   reads it) -> fly_full/.
Each stage's result stays in ``out``: a rerun with the same ``last_day`` loads what is done and resumes the rest (the
walk-forward from its cache, a fly from its checkpoint). SIGTERM stops between steps; the stages already saved stay.

    fly-trader matched-pair OUT --last-day YYYY-MM-DD [--epochs N] [--match-days N] [--replay]

It needs the memory of a training run (the decision set ~52 GB at its peak, a fly 12-18 GB) and runs for a day or more
(a fly epoch ~2 h): start it apart from the console, under nice.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def _quiet_progress() -> None:
    """Training progress to this log, every two minutes, instead of ui_settings['training_status'] (the console's)."""
    from . import progress
    last = [0.0]

    def update(stage, step=None, total=None, force=False, **extra):
        if force or time.time() - last[0] > 120:
            last[0] = time.time()
            log.info("progress: %s %s/%s %s", stage, step, total, {k: extra[k] for k in ("epoch", "loss", "rows_per_s", "eta_s") if k in extra})
    progress.update = update


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(obj, default=str, indent=1)); tmp.replace(path)


def _save_fly(fly, info: dict, path: Path) -> None:
    """fly_selector.save()'s payload, written to ``path`` only (no brain_snapshots row)."""
    from . import model_io
    n = fly.net
    model_io.torch_save({"state_dict": n.state_dict(), "scaler": fly.scaler.state(), "cols": fly.cols, "threshold": fly.threshold, "sizing": fly.sizing,
                         "horizon_min": fly.horizon_min, "rules": fly.rules, "lines": fly.lines, "sizings": fly.sizings, "combine": fly.combine,
                         "chain_lines": fly.chain_lines, "chain_sizings": fly.chain_sizings,
                         "config": {"k_steps": n.k_steps, "leak": n.leak, "hidden": n.hidden, "obs_dim": n.obs_dim, "kc_active": n.kc_active, "scale": n.scale,
                                    "n_strategies": n.n_strategies, "aff_rows": None}, "metrics": info}, path)


def _selector(ds, out: Path, stop) -> tuple[object | None, list]:
    """Stage 1: the stack, its full and blind fits, saved; (model, holdout days), or (None, []) when stopped."""
    from . import model_io, selector as SEL, strategies as S
    path = out / "selector.skops"
    if path.exists():
        m = model_io.load_selector(path)
        if str(m.trained_through) != str(ds.days[-1]):
            raise RuntimeError(f"{path} was fit through {m.trained_through}, this set ends {ds.days[-1]}: another last day")
        lo, hi = (date.fromisoformat(x) for x in m.blind["holdout_days"])
        return m, [d for d in ds.days if lo <= d <= hi]
    S.CACHE_DIR = out / "stack_cache"
    t0 = time.time(); stack = S.fit_stack(ds, stop)
    for c in stack.components:
        log.info("component %-28s %s - %s", c["name"], "PASSED" if c["passed"] else "dropped", c["reason"])
    if stop is not None and stop.is_set():
        return None, []
    if not stack.fits or not stack.holdout_days:
        raise RuntimeError("the stack has no strategy or no holdout: there is nothing to match a fly to, or to score it on")
    models = S.final_models(ds, stack)
    blind = S.final_models(ds, stack, exclude_days=stack.holdout_days)
    holdout = S.score_holdout(ds, blind, stack.holdout_days)
    by_chain = S.score_holdout_by_chain(ds, blind, stack.holdout_days) if ds.chain is not None else {}
    ev = models["strategies"].get("ev") or next(iter(models["strategies"].values()))
    final = SEL.SelectorModel(gbm=ev["gbm"], scaler=ev["scaler"], cols=ev["cols"], threshold=ev["line"], horizon_min=ev["hold_min"],
                              trained_through=str(ds.days[-1]), sizing=ev["sizing"])
    hd = sorted(stack.holdout_days)
    final.stack = models
    final.blind = {"models": blind, "fit_through": str(hd[0] - timedelta(days=1)), "holdout_days": [str(hd[0]), str(hd[-1])]}
    final.metrics = {"walk_forward": {**stack.evaluation, "days": None}, "selection": stack.selection, "random_baseline": {"mean": stack.evaluation.get("random_mean")},
                     "components": stack.components, "strategies": {k: {x: v[x] for x in ("line", "hold_min", "thr", "high", "hours", "regimes", "sizing", "selection", "evaluation")}
                                                                    for k, v in models["strategies"].items()},
                     "groups": stack.groups, "combine": stack.combine, "veto": {k: v for k, v in (stack.veto or {}).items() if k not in ("p",)} or None,
                     "fallback": stack.fallback, "holdout": holdout, "chains": by_chain, "line": final.threshold,
                     "costs": "real fees (pool fee by market cap, Jupiter 10 bps, network fee) + impact", "data": SEL.DATA_VERSION,
                     "deployable": stack.deployable, "deploy_reason": stack.reason, "sizing": final.sizing, "matched": True,
                     "model": {"kind": "stack", "min_age_h": SEL.MIN_AGE_H, "min_resq_sol": SEL.MIN_RESQ_SOL, "min_vol_15m_sol": SEL.MIN_VOL_15M_SOL},
                     "rows": int(len(ds.y)), "days": len(ds.days), "first_day": str(ds.days[0]), "last_day": str(ds.days[-1])}
    model_io.save_selector(final, path)
    _write_json(out / "selector.json", {"trained_through": final.trained_through, "holdout_days": final.blind["holdout_days"], "fit_s": round(time.time() - t0),
                                        **final.metrics})
    log.info("selector saved to %s: %s (%s); holdout %s", path, "deployable" if stack.deployable else "NOT deployable", stack.reason,
             {k: holdout.get(k) for k in ("n", "mean", "win", "random_mean")})
    return final, hd


def _fly(ds, out: Path, sub: str, models: dict, exclude_days, epochs, match_days, stop) -> tuple[object | None, dict | None, object]:
    """A matched fly, trained once and loaded on a rerun: (fly with its teacher, its note, the teacher)."""
    from . import fly_selector as FS
    d = out / sub; d.mkdir(exist_ok=True)
    teacher, fit, note = FS.matched_teacher(ds, models, exclude_days=exclude_days)
    if (d / "fly.pt").exists():
        fly = FS.load(d / "fly.pt"); fly.teacher = teacher
        return fly, json.loads((d / "note.json").read_text()), teacher
    log.info("%s: %s", sub, note)
    fly, info = FS.bootstrap_matched(ds, teacher, fit, epochs=epochs, stop=stop, teacher_note=note, checkpoint=d / "bootstrap.ckpt.pt", match_days=match_days)
    if fly is None:
        return None, None, teacher
    _save_fly(fly, info, d / "fly.pt"); _write_json(d / "note.json", {**info, "data": FS.FLY_VERSION})
    (d / "bootstrap.ckpt.pt").unlink(missing_ok=True)
    return fly, info, teacher


def _fmt(book: dict) -> str:
    f = lambda v, p=2: "-" if v is None else f"{v * 100:+.{p}f}%"      # noqa: E731
    w = "-" if book.get("win") is None else f"{book['win'] * 100:.1f}%"
    return (f"{book.get('n') or 0:>6} trades  win {w:>6}  mean {f(book.get('mean')):>8}  "
            f"total {f(book.get('total'), 1):>9}  random {f(book.get('random_mean')):>8}")


def report_text(rep: dict) -> str:
    lines = [f"Out of sample: {rep['days'][0]} .. {rep['days'][1]} ({rep['rows']} tradable minutes); net returns per trade at fixed size.", rep.get("note", ""), ""]
    for chain, books in rep["chains"].items():
        lines.append(f"== {chain}")
        for who in ("selector", "fly"):
            b = books[who]
            lines.append(f"  {who:<9} all            {_fmt(b['all'])}")
            for k, v in b.items():
                if k != "all":
                    lines.append(f"  {'':<9} {k:<14} {_fmt(v)}")
        lines.append("")
    ev = rep.get("evidence") or {}; m = ev.get("mean")
    lines.append(f"Evidence (the fly's own picks, every chain): {ev.get('trades')} trades, mean " + ("-" if m is None else f"{m * 100:+.2f}%"))
    return "\n".join(lines) + "\n"


def main(out, last_day, epochs: int | None = None, match_days: int | None = None, replay: bool = False, stop: threading.Event | None = None) -> dict:
    """Stages 1-5 of the module docstring into ``out`` on the set ending ``last_day``; the summary (also summary.json)."""
    from . import decisions as D, fly_selector as FS, strategies as S
    from ..brain import device as brain_device
    out = Path(out); out.mkdir(parents=True, exist_ok=True); _quiet_progress()
    last = date.fromisoformat(str(last_day)); epochs = FS.EPOCHS if epochs is None else int(epochs)
    t0 = time.time(); ds = D.build(days=None, horizon_min=D.HOLD_MIN, holds=S.HOLDS_MIN, last_day=last)
    if ds.days[-1] != last:
        raise RuntimeError(f"the corpus ends {ds.days[-1]}, not {last}: build that day first, or pin the last day it has")
    log.info("decision points: %d rows, %d days (%s..%s), %.0f s", len(ds.y), len(ds.days), ds.days[0], ds.days[-1], time.time() - t0)
    summary: dict = {"last_day": str(last), "rows": int(len(ds.y)), "days": [str(ds.days[0]), str(ds.days[-1])]}
    sel, hold = _selector(ds, out, stop)
    if sel is None:
        return {**summary, "stopped": "selector"}
    summary["selector"] = {"holdout_days": sel.blind["holdout_days"], "holdout": sel.metrics.get("holdout"), "deployable": sel.metrics.get("deployable")}
    fly_b, info_b, _ = _fly(ds, out, "fly_blind", sel.blind["models"], hold, epochs, match_days, stop)
    if fly_b is None:
        return {**summary, "stopped": "fly_blind"}
    oos_path = out / "oos.json"
    if oos_path.exists():
        rep = json.loads(oos_path.read_text())
    else:
        rep = FS.oos_report(ds, sel.blind["models"], fly_b, hold)
        own = rep["chains"]["all"]["fly"]["all"]
        rep["evidence"] = {"book": "the blind fly's own picks, every chain", "days": rep["days"], "trades": own.get("n"), "mean": own.get("mean"),
                           "total": own.get("total"), "win": own.get("win"), "random_mean": own.get("random_mean")}
        stack_rec = (sel.metrics.get("holdout") or {})
        mine = rep["chains"]["all"]["selector"]["all"]
        if (stack_rec.get("n"), stack_rec.get("mean")) != (mine.get("n"), mine.get("mean")):
            log.warning("the report's selector book (%s, %s) differs from the stack's own holdout record (%s, %s)", mine.get("n"), mine.get("mean"),
                        stack_rec.get("n"), stack_rec.get("mean"))
        _write_json(oos_path, rep); (out / "oos.txt").write_text(report_text(rep))
        info_b = {**info_b, "oos": {"days": rep["days"], "evidence": rep["evidence"]},
                  "calibration": {"trades": rep["evidence"]["trades"], "mean": rep["evidence"]["mean"], "total": rep["evidence"]["total"], "source": "out of sample"}}
        _write_json(out / "fly_blind" / "note.json", {**info_b, "data": FS.FLY_VERSION})
    log.info("out of sample:\n%s", report_text(rep))
    summary["oos"] = {"evidence": rep["evidence"], "selector": rep["chains"]["all"]["selector"]["all"], "fly": rep["chains"]["all"]["fly"]["all"]}
    if replay and not (out / "replay.json").exists() and not (stop is not None and stop.is_set()):
        from . import fly_replay as FR
        FR.TRADES_DIR = out / "replay"
        res = FR.run(ds=ds, fly=fly_b, boot=info_b, start_day=ds.days.index(hold[0]), stop=stop, save_verdict=False, dump_trades=True)
        if res.get("stopped"):
            return {**summary, "stopped": "replay"}
        _write_json(out / "replay.json", res)
    if (out / "replay.json").exists():
        r = json.loads((out / "replay.json").read_text())
        summary["replay"] = {k: r.get(k) for k in ("passed", "reason", "S", "chains")}
    del fly_b; brain_device.empty_cache(brain_device.resolve())
    fly_f, info_f, _ = _fly(ds, out, "fly_full", sel.stack, None, epochs, match_days, stop)
    if fly_f is None:
        return {**summary, "stopped": "fly_full"}
    if not (info_f.get("oos") or {}).get("evidence"):
        info_f = {**info_f, "oos": {"days": rep["days"], "evidence": rep["evidence"], "source": "the blind pair on the holdout days (fly_blind)"},
                  "calibration": {"trades": rep["evidence"]["trades"], "mean": rep["evidence"]["mean"], "total": rep["evidence"]["total"],
                                  "source": "out of sample (the blind pair)"}}
        _save_fly(fly_f, info_f, out / "fly_full" / "fly.pt"); _write_json(out / "fly_full" / "note.json", {**info_f, "data": FS.FLY_VERSION})
    ok, why = FS.deployable({**info_f, "data": FS.FLY_VERSION})
    summary["fly_full"] = {"lines": info_f.get("lines"), "chain_lines": info_f.get("chain_lines"), "gates_ok": info_f.get("gates_ok"), "deployable": ok, "why": why}
    _write_json(out / "summary.json", summary)
    log.info("matched pair done in %s: the full fly is %s (%s)", out, "deployable" if ok else "NOT deployable", why)
    return summary


def cli(args) -> dict:
    """``fly-trader matched-pair``: logs to OUT/run.log and stderr; SIGTERM stops between steps."""
    import signal
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(); root.setLevel(logging.INFO); fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in (logging.FileHandler(out / "run.log"), logging.StreamHandler()):
        h.setFormatter(fmt); root.addHandler(h)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return main(out, args.last_day, epochs=args.epochs, match_days=args.match_days, replay=args.replay, stop=stop)
