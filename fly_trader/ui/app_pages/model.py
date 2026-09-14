"""Model & training: which model trades and how it tested against random picks, the automatic retraining pipeline
(selector, then the fly imitating it, every 7 days), the latest results, and every saved model."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader.db.queries import q, q1
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.train import pipeline
from fly_trader.train.selector import is_current, is_deployable
from fly_trader.ui.common import ago, backtest_line, jv, latest_snapshot, parse_note, pct, system_state


def _pf(v) -> str:
    return "∞" if v is None else f"{float(v):.2f}"


def _until(iso: str | None) -> str:
    if not iso:
        return "—"
    s = (datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()
    return "due now" if s <= 0 else f"in {s / 3600:.0f} h" if s < 172800 else f"in {s / 86400:.1f} days"


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
                st.metric("Buys when score ≥", f"{float(run.get('threshold') or meta.get('threshold') or 0):.3f}",
                          help="The model scores every eligible token each minute (0–1). It buys only above this line: the top 1 % of scores seen in training.")
                st.metric("Holds for", f"{run.get('horizon_min') or meta.get('horizon_min') or 30} min", help="Each position is sold this long after the buy.")
                st.metric("Test trades", wf.get("n", "—"), help=f"Trades the model would have made on {wf.get('days', '—')} test days, each traded by a model that never saw that day.")
                st.metric("Average per trade", pct(wf.get("mean")), delta=(f"random {pct(rb['mean'])}" if rb.get("mean") is not None else None), delta_color="off",
                          help="Net return per trade on the test days, after fees and price impact. 'random' is the same number of random buys under the same rules.")
                st.metric("Winning trades", pct(wf.get("win"), 0, False), help="Share of test trades that made money.")
                st.metric("Profit factor", _pf(wf.get("pf")), help="Total gains divided by total losses on the test trades. Above 1 makes money.")
                st.metric("Profitable test days", f"{wf.get('days_positive', '—')} of {wf.get('days', '—')}", help="Test days whose average trade made money after costs.")
            st.caption(backtest_line(meta))
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
            st.markdown(f":material/autorenew: **Automatic retraining** · every {pipeline.INTERVAL_DAYS} days")
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
                   "without a restart.  \n3 · The fly then learns to imitate the selector and is tested on the last 21 days.")
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
            df = pd.DataFrame([{"day": d.get("day"), "AUC": d.get("auc"), "trades": d.get("n"), "model": (d["mean"] * 100) if d.get("mean") is not None else None,
                                "random": (d["random_mean"] * 100) if d.get("random_mean") is not None else None, "winners": (d["win"] * 100) if d.get("win") is not None else None,
                                "profit factor": d.get("pf")} for d in (jv(r["detail"]) for r in wfd)])
            st.bar_chart(df, x="day", y=[c for c in ("model", "random") if df[c].notna().any()], stack=False, x_label="", y_label="% per trade", height=220)
            st.dataframe(df, hide_index=True, column_config={"AUC": st.column_config.NumberColumn(format="%.3f", help="Ranking quality of the scores (0.5 = chance)."),
                                                             "model": st.column_config.NumberColumn("model % / trade", format="%+.2f"),
                                                             "random": st.column_config.NumberColumn("random % / trade", format="%+.2f", help="Random picks, same count, costs and fills."),
                                                             "winners": st.column_config.NumberColumn("winners %", format="%.0f"),
                                                             "profit factor": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Each day is traded by a model trained only on days at least two days earlier, with the paper broker's fees and price impact.")
        else:
            st.caption("No selector backtest on the current data yet.")
    fv = q1("SELECT ts, detail FROM events WHERE source = 'fly_selector' AND message LIKE 'fly vs gbm%%' ORDER BY id DESC LIMIT 1")
    if fv and latest_snapshot("fly_selector"):
        d = jv(fv["detail"]); ag = d.get("agreement") or {}
        with st.container(border=True):
            st.markdown("**The fly, imitating the selector** (same test days)")
            rows = [(n, d.get(k) or {}) for n, k in (("Fly (connectome)", "fly"), ("Selector (its teacher)", "gbm"), ("Random picks", "random"))]
            st.dataframe(pd.DataFrame([{"model": n, "trades": v.get("n"), "% / trade": (v["mean"] * 100) if v.get("mean") is not None else None,
                                        "median %": (v["median"] * 100) if v.get("median") is not None else None, "winners %": (v["win"] * 100) if v.get("win") is not None else None,
                                        "profit factor": v.get("pf"), "AUC": v.get("auc")} for n, v in rows]), hide_index=True,
                         column_config={"% / trade": st.column_config.NumberColumn(format="%+.2f"), "median %": st.column_config.NumberColumn(format="%+.2f"),
                                        "winners %": st.column_config.NumberColumn(format="%.0f"), "profit factor": st.column_config.NumberColumn(format="%.2f"),
                                        "AUC": st.column_config.NumberColumn(format="%.3f")})
            parts = [f"run {ago(fv['ts'])}"]
            if ag.get("pick_overlap") is not None:
                parts.append(f"the fly picks {ag['pick_overlap'] * 100:.0f}% of what the selector picks")
            if ag.get("rank_corr") is not None:
                parts.append(f"score agreement {ag['rank_corr']:.2f} (1 = identical ranking)")
            st.caption(" · ".join(parts))


def history() -> None:
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('selector', 'fly_selector') ORDER BY id DESC LIMIT 30")
    s = system_state(); in_use = (s["model"] or {}).get("id")
    with st.expander("Saved models", icon=":material/folder_open:"):
        rows = []
        for r in snaps:
            m = parse_note(r["note"]); wf = m.get("walk_forward") or m.get("fly") or {}; rb = m.get("random_baseline") or m.get("random") or {}
            ok = is_current(m)
            rows.append({"model": f"#{r['id']}", "type": "selector" if r["kind"] == "selector" else "fly", "saved": r["ts"], "data": "current" if ok else "outdated",
                         "trades": wf.get("n") if ok else None, "% / trade": (wf["mean"] * 100) if ok and wf.get("mean") is not None else None,
                         "random %": (rb["mean"] * 100) if ok and rb.get("mean") is not None else None, "profit factor": wf.get("pf") if ok else None,
                         "put to work": bool(m.get("deployable")) if ok and r["kind"] == "selector" else None, "in use": r["id"] == in_use})
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
history()
