"""Model & training: which model trades and how it tested against random picks, training controls and progress,
the latest results, and every saved model."""
import json

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.db.queries import q, q1
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago, backtest_line, jv, parse_note, pct, restart_runner, setting, system_state


def _pf(v) -> str:
    return "∞" if v is None else f"{float(v):.2f}"


@st.fragment(run_every="10s")
def loaded() -> None:
    s = system_state(); m, latest = s["model"], s["latest"]
    with st.container(border=True):
        st.markdown("**Model in use**")
        if m:
            meta = m["meta"]; wf = meta.get("walk_forward") or {}; rb = meta.get("random_baseline") or {}; run = m.get("run") or {}
            st.caption(f"Model #{m['id']} · gradient-boosted selector · trained through {meta.get('trained_through', '—')} · saved {ago(m['ts'])} · loaded {ago(m['since'])}")
            with st.container(horizontal=True):
                st.metric("Buy when score ≥", f"{float(run.get('threshold') or meta.get('threshold') or 0):.3f}", help="The top 1 % of scores on the training days.")
                st.metric("Hold", f"{run.get('horizon_min') or meta.get('horizon_min') or 30} min")
                st.metric("Backtest trades", wf.get("n", "—"))
                st.metric("Per trade", pct(wf.get("mean")), delta=(f"random {pct(rb['mean'])}" if rb.get("mean") is not None else None), delta_color="off")
                st.metric("Winners", pct(wf.get("win"), 0, False))
                st.metric("Profit factor", _pf(wf.get("pf")))
                st.metric("Days positive", f"{wf.get('days_positive', '—')}/{wf.get('days', '—')}")
            st.caption(backtest_line(meta))
        else:
            st.caption("No model in use: the trading engine is stopped.")
        if latest and (not m or latest["id"] > m["id"]):
            st.info(f"Model #{latest['id']} was saved {ago(latest['ts'])}. {backtest_line(latest['meta'])}", icon=":material/upgrade:")
            if st.button(f"Load model #{latest['id']}", icon=":material/restart_alt:", type="primary", key="load_latest",
                         help="Restarts the trading engine with the newest model. A restart archives and clears the paper book so its numbers belong to one model."):
                with st.spinner("Restarting the trading engine…"):
                    try:
                        restart_runner()
                    except RuntimeError as e:
                        st.error(str(e))
                st.rerun()


@st.fragment(run_every="3s")
def trainer() -> None:
    s = system_state(); sup = get_supervisor(); tr = s["train_status"]
    with st.container(border=True):
        st.markdown("**Training**")
        if s["training"]:
            step, total = tr.get("step"), tr.get("total")
            st.caption(f"{tr.get('stage', '')} · updated {ago(s['train_at'])}" + (" · started outside the console (CLI)" if s["train_external"] else "")
                       + (f" · about {float(tr['eta_s']) / 60:.0f} min left" if tr.get("eta_s") else ""))
            if step is not None and total:
                st.progress(min(1.0, float(step) / max(float(total), 1.0)))
            if sup.alive("train") and st.button("Stop training", icon=":material/stop:", key="train_stop"):
                sup.request_stop("train"); st.rerun()
        else:
            st.caption("Not training." + (f" Last run: {tr.get('stage')} ({ago(s['train_at'])})." if tr.get("stage") else ""))
        err = s["workers"]["train"].get("error")
        if err:
            st.error(err.splitlines()[0][:300])


def start_form() -> None:
    sup = get_supervisor()
    p = q1("SELECT value FROM ui_settings WHERE key = 'training_params'"); pv = jv(p["value"]) if p else {}
    names = {"selector": "Selector (gradient-boosted)", "fly": "Fly (connectome)"}
    with st.form("train_form", border=True):
        st.markdown("**Start a training run**")
        regimen = st.segmented_control("Model to train", list(names), format_func=names.get, default=pv.get("regimen") if pv.get("regimen") in names else "selector",
                                       help="Selector: backtested day by day, then saved as the newest model. Fly: the connectome trained on the same data and scored against the selector.")
        with st.container(horizontal=True):
            days = st.number_input("Days of history", 5, 200, int(pv.get("days", 45)))
            test_days = st.number_input("Backtest days", 2, 30, int(pv.get("test_days", 9)), help="The last days, each tested with a model trained only on earlier days.")
            top_frac = st.number_input("Buy the top fraction", 0.001, 0.2, float(pv.get("top_frac", 0.01)), step=0.005, format="%.3f")
            epochs = st.number_input("Fly epochs", 1, 20, int(pv.get("epochs", 2)), help="Fly only.")
        go = st.form_submit_button("Start training", icon=":material/play_arrow:", type="primary", disabled=sup.alive("train"))
    if go:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('training_params', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps({**pv, "regimen": regimen or "selector", "days": int(days), "test_days": int(test_days), "top_frac": float(top_frac), "epochs": int(epochs)}),))
        try:
            sup.start("train", started_by="console")
        except RuntimeError as e:
            st.error(str(e))
        st.rerun()


@st.fragment(run_every="15s")
def results() -> None:
    wfd = q("SELECT detail FROM events WHERE source = 'selector' AND message LIKE 'walk-forward%%' ORDER BY id")
    with st.container(border=True):
        st.markdown("**Latest selector backtest, day by day**")
        if wfd:
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
            st.caption("No selector backtest recorded since the last training reset.")
    fv = q1("SELECT ts, detail FROM events WHERE source = 'fly_selector' AND message LIKE 'fly vs gbm%%' ORDER BY id DESC LIMIT 1")
    if fv:
        d = jv(fv["detail"])
        with st.container(border=True):
            st.markdown("**Fly vs selector vs random** (same test days)")
            rows = [(n, d.get(k) or {}) for n, k in (("Fly (connectome)", "fly"), ("Selector", "gbm"), ("Random picks", "random"))]
            st.dataframe(pd.DataFrame([{"model": n, "trades": v.get("n"), "% / trade": (v["mean"] * 100) if v.get("mean") is not None else None,
                                        "median %": (v["median"] * 100) if v.get("median") is not None else None, "winners %": (v["win"] * 100) if v.get("win") is not None else None,
                                        "profit factor": v.get("pf"), "AUC": v.get("auc")} for n, v in rows]), hide_index=True,
                         column_config={"% / trade": st.column_config.NumberColumn(format="%+.2f"), "median %": st.column_config.NumberColumn(format="%+.2f"),
                                        "winners %": st.column_config.NumberColumn(format="%.0f"), "profit factor": st.column_config.NumberColumn(format="%.2f"),
                                        "AUC": st.column_config.NumberColumn(format="%.3f")})
            st.caption(f"Run {ago(fv['ts'])} · " + ("the fly beat the selector." if d.get("fly_beats_gbm") else "the selector beat the fly."))


def history() -> None:
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('selector', 'fly_selector') ORDER BY id DESC LIMIT 30")
    s = system_state(); in_use = (s["model"] or {}).get("id")
    with st.expander("Saved models", icon=":material/folder_open:"):
        rows = []
        for r in snaps:
            m = parse_note(r["note"]); wf = m.get("walk_forward") or m.get("fly") or {}; rb = m.get("random_baseline") or m.get("random") or {}
            rows.append({"model": f"#{r['id']}", "type": "selector" if r["kind"] == "selector" else "fly", "saved": r["ts"], "trained through": m.get("trained_through"),
                         "trades": wf.get("n"), "% / trade": (wf["mean"] * 100) if wf.get("mean") is not None else None,
                         "random %": (rb["mean"] * 100) if rb.get("mean") is not None else None, "profit factor": wf.get("pf"),
                         "costs": "real" if (m.get("costs") or rb) else "flat 0.6 %", "in use": r["id"] == in_use})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"saved": st.column_config.DatetimeColumn(format="MMM D HH:mm"),
                                                                             "% / trade": st.column_config.NumberColumn(format="%+.2f"),
                                                                             "random %": st.column_config.NumberColumn(format="%+.2f"),
                                                                             "profit factor": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Models trained before the look-ahead and cost fixes of 2026-09-14 report optimistic numbers.")
        else:
            st.caption("No saved models.")
    with st.expander("Trainer log", icon=":material/terminal:"):
        st.code(get_supervisor().log_tail("train", 40) or "(no log yet)", language="json")


loaded()
left, right = st.columns([3, 2])
with left:
    trainer()
with right:
    start_form()
results()
history()
