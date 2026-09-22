"""Model & training: which model trades and how it tested against random picks, the automatic retraining pipeline (the
selector every 7 days until the fly holds its seat; the fly bootstrapped once), the plastic fly, and every saved model."""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.agent import fly_session
from fly_trader.db.queries import q, q1
from fly_trader.ops.reset import reset_fly
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.train import fly_selector, pipeline
from fly_trader.train.selector import is_current, is_deployable
from fly_trader.ui.common import ago, backtest_line, jv, latest_snapshot, parse_note, pct, system_state


def _pf(v) -> str:
    return "∞" if v is None else f"{float(v):.2f}"


def _until(iso: str | None) -> str:
    if not iso:
        return "—"
    s = (datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()
    return "due now" if s <= 0 else f"in {s / 3600:.0f} h" if s < 172800 else f"in {s / 86400:.1f} days"


def sizing_panel(meta: dict) -> None:
    """How big each buy is: the model's certainty bands from its backtest, and the backtest replayed with that sizing."""
    tbl = meta.get("sizing") or []; bk = meta.get("bankroll") or {}
    st.markdown("**Position sizing**")
    if not tbl:
        st.caption(f"Fixed {config.MAX_POSITION_SOL:g} SOL per buy: this model was trained before sizing and has no certainty bands.")
        return
    st.caption(f"Each buy's size comes from how certain the model is. Its backtest trades are split into bands by how far the score cleared the buy line; "
               f"each band gets {config.KELLY_FRACTION:g} × its growth-optimal share of the bankroll (wealth minus the {config.GAS_RESERVE_SOL:g} SOL gas reserve), "
               f"at most {config.MAX_POSITION_FRACTION:.0%} of the bankroll and {config.MAX_POOL_SHARE:.0%} of the pool. Buys under {config.MIN_POSITION_SOL:g} SOL are skipped.")
    st.dataframe(pd.DataFrame([{"score above the line": f"+{b['lo']:.3f} and up", "test trades": b["n"], "average per trade": (b["mean"] or 0) * 100,
                                "winning trades": (b["win"] or 0) * 100, "bet (share of bankroll)": min(config.KELLY_FRACTION * b["kelly"], config.MAX_POSITION_FRACTION) * 100}
                               for b in tbl]), hide_index=True,
                 column_config={"average per trade": st.column_config.NumberColumn(format="%+.2f%%"), "winning trades": st.column_config.NumberColumn(format="%.0f%%"),
                                "bet (share of bankroll)": st.column_config.NumberColumn(format="%.1f%%", help="0 % means that band has no edge: no buy.")})
    s, f = bk.get("sized") or {}, bk.get("fixed") or {}
    if s and f:
        st.caption(f"The backtest's trades replayed from {s.get('start_sol', 0):g} SOL: with this sizing → {s.get('final_sol', 0):.2f} SOL (worst drawdown {s.get('max_drawdown', 0):.0%}); "
                   f"at a fixed {config.MAX_POSITION_SOL:g} SOL → {f.get('final_sol', 0):.2f} SOL (worst drawdown {f.get('max_drawdown', 0):.0%}).")


def _best(b: dict | None) -> str:
    """The most profitable setting a component tried, whether or not it cleared the bars — why a dropped one was dropped."""
    if not b:
        return "—"
    win = f"{b['win'] * 100:.0f}% winners" if b.get("win") is not None else "—"
    pf = f"PF {b['pf']:.2f}" if b.get("pf") is not None else "PF ∞"
    return f"{win} · {pf} · {b.get('n', 0)} trades · total {b.get('total', 0):+.1f}"


def stack_panel(meta: dict) -> None:
    """The selector's strategy stack: every component tried (kept or dropped, and why) and the strategies it trades."""
    comps = meta.get("components") or []
    if not comps:
        return
    st.markdown("**Components**")
    if meta.get("fallback"):
        st.warning("No setting reached 65 % winning trades on both halves: this model trades the most profitable system that passes profit factor 1.3 and the deploy rule instead.",
                   icon=":material/warning:")

    def half(x: dict | None, k: str):
        return None if not x or x.get(k) is None else float(x[k])
    st.dataframe(pd.DataFrame([{"component": c["name"], "kept": c["passed"], "why": c["reason"],
                                "winning (selection)": (half(c.get("selection"), "win") or 0) * 100 if half(c.get("selection"), "win") is not None else None,
                                "winning (evaluation)": (half(c.get("evaluation"), "win") or 0) * 100 if half(c.get("evaluation"), "win") is not None else None,
                                "profit factor (evaluation)": half(c.get("evaluation"), "pf"), "total (evaluation)": half(c.get("evaluation"), "total"),
                                "trades (evaluation)": half(c.get("evaluation"), "n"), "settings tried": c.get("trials"), "best reached": _best(c.get("best_seen")),
                                "parameters": json.dumps(c.get("params") or {}, default=str)}
                               for c in comps]), hide_index=True,
                 column_config={"winning (selection)": st.column_config.NumberColumn(format="%.0f%%"), "winning (evaluation)": st.column_config.NumberColumn(format="%.0f%%"),
                                "profit factor (evaluation)": st.column_config.NumberColumn(format="%.2f"), "total (evaluation)": st.column_config.NumberColumn(format="%+.2f",
                                help="Sum of the net returns of the evaluation half's trades (1.00 = one position's cost)."),
                                "best reached": st.column_config.TextColumn(help="The most profitable setting this component tried on the selection half, whether or not it cleared the bars.")})
    strat = meta.get("strategies") or {}
    if strat:
        st.markdown("**Strategies**")
        st.dataframe(pd.DataFrame([{"strategy": k, "hold (min)": v.get("hold_min"), "buy line": (v.get("line") or 0) * 100, "trigger": json.dumps(v.get("thr") or {}, default=str),
                                    "trades (evaluation)": half(v.get("evaluation"), "n"), "winning (evaluation)": (half(v.get("evaluation"), "win") or 0) * 100 if half(v.get("evaluation"), "win") is not None else None,
                                    "profit factor (evaluation)": half(v.get("evaluation"), "pf")} for k, v in strat.items()]), hide_index=True,
                     column_config={"buy line": st.column_config.NumberColumn(format="%+.2f%%"), "winning (evaluation)": st.column_config.NumberColumn(format="%.0f%%"),
                                    "profit factor (evaluation)": st.column_config.NumberColumn(format="%.2f")})


@st.fragment(run_every="10s")
def loaded() -> None:
    s = system_state(); m, latest = s["model"], s["latest"]
    with st.container(border=True):
        st.markdown("**Model at work**")
        if m and not is_current(m["meta"]):
            st.warning(f"Model #{m['id']} is in use but was trained on outdated data (before the data fixes of 2026-09-14). Its backtest is not valid and is hidden.",
                       icon=":material/warning:")
        elif m:
            meta = m["meta"]; wf = meta.get("walk_forward") or {}; rb = meta.get("random_baseline") or {}; run = m.get("run") or {}
            st.caption(f"Model #{m['id']} · gradient-boosted selector · trained on {meta.get('first_day', '—')} → {meta.get('trained_through', '—')} · saved {ago(m['ts'])} · "
                       f"at work since {ago(m['since'])}")
            with st.container(horizontal=True):
                mi = meta.get("model") or {}
                st.metric("Buys when predicted return ≥", pct(float(run.get('threshold') or meta.get('threshold') or 0), 1),
                          help=f"Each minute the model predicts every eligible token's net return over the hold after fees and price impact. It buys tokens at least "
                               f"{mi.get('min_age_h', '—')} h past graduation (or older than the archive) whose prediction clears this line.")
                st.metric("Holds for", f"{run.get('horizon_min') or meta.get('horizon_min') or '—'} min", help="Each position is sold this long after the buy.")
                st.metric("Test trades", wf.get("n", "—"), help=f"Trades on the backtest's evaluation half ({wf.get('days', '—')} days), each traded by a model that never saw "
                                                                "that day, at a buy line chosen on the earlier selection half.")
                st.metric("Average per trade", pct(wf.get("mean")), delta=(f"random {pct(rb['mean'])}" if rb.get("mean") is not None else None), delta_color="off",
                          help="Net return per trade on the test days, after fees and price impact. 'random' is the same number of random buys under the same rules.")
                st.metric("Winning trades", pct(wf.get("win"), 0, False), help="Share of test trades that made money.")
                st.metric("Profit factor", _pf(wf.get("pf")), help="Total gains divided by total losses on the test trades. Above 1 makes money.")
                st.metric("Profitable test days", f"{wf.get('days_positive', '—')} of {wf.get('days', '—')}", help="Test days whose average trade made money after costs.")
            st.caption(backtest_line(meta))
            sizing_panel(meta)
            stack_panel(meta)
        elif not latest:
            st.info("No model has been trained on the current data yet, so nothing is trading. The automatic retraining below runs as soon as the data is ready.",
                    icon=":material/info:")
        else:
            st.info(f"No model is at work. The latest, #{latest['id']}, was not put to work: {latest['meta'].get('deploy_reason') or 'it did not qualify'}.",
                    icon=":material/block:")
        if m and latest and latest["id"] > m["id"]:
            if is_deployable(latest["meta"]):
                st.info(f"Model #{latest['id']} qualified and replaces #{m['id']} within 10 minutes, without a restart.", icon=":material/upgrade:")
            else:
                st.caption(f"Model #{latest['id']} ({ago(latest['ts'])}) was not put to work: {latest['meta'].get('deploy_reason') or 'it did not qualify'}.")


@st.fragment(run_every="3s")
def retraining() -> None:
    s = system_state(); sup = get_supervisor(); tr = s["train_status"]; ps = pipeline.state(); alive = sup.alive("train")
    running = alive and str(ps.get("stage", "")).startswith("training")
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/autorenew: **Automatic retraining** · every {pipeline.INTERVAL_DAYS} days" if not s["handover"] else ":material/autorenew: **Retraining** · on request only: the fly holds the seat")
            st.space("stretch")
            if st.button("Retrain now", icon=":material/play_arrow:", type="primary", key="retrain_now", disabled=running,
                         help="Runs the whole pipeline now; the 7-day schedule restarts from this run."):
                pipeline.request_run()
                if not alive:
                    try:
                        sup.start("train", started_by="console")
                    except RuntimeError as e:
                        st.error(str(e))
                st.rerun()
            if running and st.button("Stop", icon=":material/stop:", key="train_stop", help="Stops the current run; the schedule resumes when the trainer is started again."):
                sup.request_stop("train"); st.rerun()
        st.caption("1 · The selector retrains on every day of history. Every day after the first 21 is backtested by a model that never saw it, against random picks, "
                   "with real fees and price impact.  \n2 · It goes to work only if that backtest made money and beat random picks; the trading engine switches to it "
                   "without a restart.  \n3 · The fly is taught by the selector once — when none exists for the current definitions, or on request — "
                   "and from then on learns from the market; weekly retraining stops when the fly takes the selector's seat.")
        if not alive:
            st.warning("Automatic retraining is off because the Trainer is not running. Click Retrain now, or start the Trainer on the Processes page.", icon=":material/warning:")
        last = ps.get("last_run_at"); nxt = ps.get("next_run_at")
        sel = latest_snapshot("selector")
        with st.container(horizontal=True):
            st.metric("Status", (ps.get("stage") or "idle").capitalize() if alive else "Off")
            st.metric("Last run", ago(last) if last else "never")
            st.metric("Next run", _until(nxt) if nxt else ("as soon as the data is ready" if alive else "—"))
            st.metric("Latest selector", f"#{sel['id']}" if sel else "—", delta=("at work" if sel and is_deployable(sel["meta"]) else "not at work" if sel else None),
                      delta_color="off", help=backtest_line(sel["meta"]) if sel else None)
        if running and tr.get("step") is not None and tr.get("total"):
            st.progress(min(1.0, float(tr["step"]) / max(float(tr["total"]), 1.0)), text=f"{tr.get('stage', '')}" + (f" · about {float(tr['eta_s']) / 60:.0f} min left" if tr.get("eta_s") else ""))
        if ps.get("stage") == "waiting for data" and ps.get("waiting"):
            st.caption(f"Waiting for the training data: {ps['waiting']}. Checking again {_until(ps.get('retry_after'))}.")
        if ps.get("stage") == "failed" and ps.get("last_error"):
            st.error(f"The last run failed: {ps['last_error']}. Retrying {_until(ps.get('retry_after'))}.")


@st.fragment(run_every="15s")
def results() -> None:
    wfd = q("SELECT detail FROM events WHERE source = 'selector' AND message LIKE 'walk-forward%%' ORDER BY id")
    with st.container(border=True):
        st.markdown("**Latest selector backtest, day by day**")
        if wfd and not system_state()["latest"]:
            st.caption("The last backtest ran on outdated data and is hidden.")
        elif wfd:
            df = pd.DataFrame([{"day": d.get("day"), "half": d.get("half"), "IC": d.get("ic"), "trades": d.get("n"), "model": (d["mean"] * 100) if d.get("mean") is not None else None,
                                "random": (d["random_mean"] * 100) if d.get("random_mean") is not None else None, "winners": (d["win"] * 100) if d.get("win") is not None else None,
                                "profit factor": d.get("pf")} for d in (jv(r["detail"]) for r in wfd)])
            st.bar_chart(df, x="day", y=[c for c in ("model", "random") if df[c].notna().any()], stack=False, x_label="", y_label="% per trade", height=220)
            st.dataframe(df, hide_index=True, column_config={"IC": st.column_config.NumberColumn(format="%.3f", help="Rank correlation of the predictions with the realised returns (0 = no skill)."),
                                                             "half": st.column_config.TextColumn(help="selection: the days that chose the buy line; evaluation: the days that judge it."),
                                                             "model": st.column_config.NumberColumn("model % / trade", format="%+.2f"),
                                                             "random": st.column_config.NumberColumn("random % / trade", format="%+.2f", help="Random picks, same count, costs and fills."),
                                                             "winners": st.column_config.NumberColumn("winners %", format="%.0f"),
                                                             "profit factor": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Each day is traded by a model trained only on days at least two days earlier, with the paper broker's fees and price impact. "
                       "The buy line is chosen on the selection half and judged on the evaluation half.")
        else:
            st.caption("No selector backtest on the current data yet.")


def _fly_bootstrap() -> dict | None:
    for r in q("SELECT id, ts, note FROM brain_snapshots WHERE kind = 'fly_selector' ORDER BY id DESC LIMIT 10"):
        m = parse_note(r["note"])
        if fly_selector.is_current(m):
            return {**r, "meta": m}
    return None


@st.fragment(run_every="15s")
def plastic_fly() -> None:
    s = system_state(); fly = s["fly"]; rp = s["replay"]; sup = get_supervisor()
    running = bool(fly.get("stage")) and fly.get("stage") != "not trading"; frozen = bool(fly.get("learning_frozen"))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":material/neurology: **The plastic fly** · taught once by the selector, then learning from the market")
            st.space("stretch")
            if st.button("Resume learning" if frozen else "Pause learning", key="fly_learn", disabled=not running,
                         help="Pause: the fly keeps trading with what it has learned; its mushroom body stops changing."):
                fly_session.send_command("resume" if frozen else "pause"); st.toast("Sent: applied at the next minute")
            if st.button("Roll back", key="fly_rollback", disabled=not running,
                         help="Restore the newest good hourly snapshot at least a day old (or the bootstrap). Does not count toward the automatic re-bootstrap."):
                fly_session.send_command("rollback"); st.toast("Sent: applied at the next minute")
            if st.button("Re-bootstrap", key="fly_reboot", help="The selector retrains and teaches a new fly; it trades at once if it qualifies."):
                pipeline.request_run(fly=True, reason="console"); st.toast("Re-bootstrap requested")
            with st.popover("Reset fly", disabled=sup.alive("runner"), help="Stop the trading engine first." if sup.alive("runner") else None):
                st.markdown("Archive and clear the fly's paper book and everything it learned? It restarts from its bootstrap and the race starts over.")
                if st.button("Reset", type="primary", key="fly_reset_confirm"):
                    st.success(f"Archived to {reset_fly(reason='console button')['archived_to']}")
        if not rp:
            st.info("The replay has not run on the current definitions. It is the proof required before the fly trades: `fly-trader fly-replay` "
                    "bootstraps a fly, lets it learn over months of history without retraining, and judges it.", icon=":material/info:")
        else:
            ev, rnd, fr = rp.get("evaluation") or {}, rp.get("random") or {}, rp.get("frozen") or {}
            (st.success if rp.get("passed") else st.warning)(f"Replay {'passed' if rp.get('passed') else 'failed'}: {rp.get('reason')}",
                                                            icon=":material/verified:" if rp.get("passed") else ":material/block:")
            with st.container(horizontal=True):
                st.metric("Replay trades", ev.get("n", "—"), border=True, help="Trades on the evaluation half of the replay.")
                st.metric("Average per trade", pct(ev.get("mean")), delta=(f"random {pct(rnd.get('mean'))}" if rnd.get("mean") is not None else None), delta_color="off", border=True)
                st.metric("Frozen fly", pct(fr.get("mean")), border=True, help="The same bootstrap without plasticity, on the same days.")
                st.metric("Profitable days", f"{ev.get('days_positive', '—')} of {ev.get('days', '—')}", border=True)
                st.metric("Learning rate α", f"{rp.get('alpha') or 0:g}", border=True)
                st.metric("Forgetting half-life", f"{rp['half_life_days']:g} days" if rp.get("half_life_days") else "none", border=True)
            st.caption(f"Bootstrapped for {rp.get('S')} and never retrained: every scored minute taught its mushroom body with the realized return two hours later. "
                       f"Configuration chosen on {' → '.join(rp.get('selection_days') or ['—'])}, judged on {' → '.join(rp.get('evaluation_days') or ['—'])}.")
        if running:
            ch = fly.get("checks") or {}; sh = ch.get("shadow") or {}
            with st.container(horizontal=True):
                st.metric("Buy line", pct(fly.get("line")), delta=f"shadow {pct(fly.get('frozen_line'))}", delta_color="off",
                          help="Recalibrated every day from its own scores and the realized returns of the last 7 days.")
                st.metric("Drift", pct(fly.get("drift"), 1, False), help="Size of the learned KC→MBON change relative to the bootstrap weights; rolled back above 50 %.")
                st.metric("Labels pending", fly.get("pending", "—"), help="Scored minutes waiting for their two-hour outcome.")
                st.metric("24 h IC", f"{ch['ic']:.3f}" if ch.get("ic") is not None else "—", help="Rank correlation of its scores with the realized returns; rolled back below 0.")
                st.metric("Picks vs shadow (3 d)", pct(sh.get("mean_plastic")), delta=f"shadow {pct(sh.get('mean_frozen'))}", delta_color="off",
                          help="Average realized return of its picks vs the frozen bootstrap's picks; rolled back when it trails by more than one standard error.")
                st.metric("Learning", "frozen" if frozen else "on")
                st.metric("Device", fly.get("device") or "—", help="Where the fly runs: DEVICE in .env (auto: CUDA, else Apple MPS, else CPU).")
            upd = q("SELECT hour, n, mean_abs_delta, drift, ic FROM fly_updates ORDER BY hour DESC LIMIT 72")
            if upd:
                st.line_chart(pd.DataFrame(upd), x="hour", y=["drift", "ic"], x_label="", height=200)
            rb = q("SELECT ts, reason, from_snapshot, to_snapshot FROM fly_rollbacks ORDER BY id DESC LIMIT 20")
            if rb:
                with st.expander(f"Rollbacks ({len(rb)})", icon=":material/history:"):
                    st.dataframe(pd.DataFrame(rb), hide_index=True, column_config={"ts": st.column_config.DatetimeColumn("when", format="MMM D HH:mm")})
        else:
            st.caption(f"Not trading: {fly.get('detail') or 'the trading engine is stopped'}.")
        b = _fly_bootstrap()
        if b:
            d = b["meta"].get("diagnostics") or {}; ok, why = fly_selector.deployable(b["meta"])
            st.caption(f"Bootstrap #{b['id']} ({ago(b['ts'])}): the mushroom body carries {pct(d.get('mbon_share'), 0, False)} of the score, "
                       f"{pct(d.get('mbon_saturated'), 0, False)} of MBONs saturated, {pct(d.get('kc_active'), 0, False)} of Kenyon cells active, "
                       f"slope {d.get('slope') or 0:.2f} against its teacher. " + ("May trade: " if ok else "May not trade: ") + why + ".")


def history() -> None:
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('selector', 'fly_selector') ORDER BY id DESC LIMIT 30")
    s = system_state(); in_use = (s["model"] or {}).get("id")
    with st.expander("Saved models", icon=":material/folder_open:"):
        rows = []
        for r in snaps:
            m = parse_note(r["note"]); wf = m.get("walk_forward") or m.get("fly") or {}; rb = m.get("random_baseline") or m.get("random") or {}
            sel = r["kind"] == "selector"; ok = is_current(m) if sel else fly_selector.is_current(m)
            if not sel:
                wf, rb = m.get("calibration") or {}, {}
                wf = {"n": wf.get("trades"), "mean": wf.get("mean")}
            rows.append({"model": f"#{r['id']}", "type": "selector" if sel else "fly bootstrap", "saved": r["ts"], "data": "current" if ok else "outdated",
                         "trades": wf.get("n") if ok else None, "% / trade": (wf["mean"] * 100) if ok and wf.get("mean") is not None else None,
                         "random %": (rb["mean"] * 100) if ok and rb.get("mean") is not None else None, "profit factor": wf.get("pf") if ok else None,
                         "put to work": (bool(m.get("deployable")) if sel else fly_selector.deployable(m)[0]) if ok else None, "in use": r["id"] == in_use})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"saved": st.column_config.DatetimeColumn(format="MMM D HH:mm"),
                                                                             "% / trade": st.column_config.NumberColumn(format="%+.2f"),
                                                                             "random %": st.column_config.NumberColumn(format="%+.2f"),
                                                                             "profit factor": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Outdated models were trained before the data fixes of 2026-09-14; their numbers are not valid, are hidden, and they are never loaded.")
        else:
            st.caption("No saved models.")
    with st.expander("Trainer log", icon=":material/terminal:"):
        st.code(get_supervisor().log_tail("train", 40) or "(no log yet)", language="json")


loaded()
retraining()
results()
plastic_fly()
history()
