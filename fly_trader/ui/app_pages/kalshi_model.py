"""Kalshi model & training: the strategy stack the visual fly was taught by, the automatic pipeline, the replay verdict,
the plastic visual fly's controls and health, and every saved Kalshi model."""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q
from fly_trader.kalshi import fly as KF, fly_session, pipeline, selector as kselector
from fly_trader.ops.reset import reset_kalshi_fly
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago, jv, kalshi_state, parse_note, pct, usd


def _until(iso: str | None) -> str:
    if not iso:
        return "—"
    s = (datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()
    return "due now" if s <= 0 else f"in {s / 3600:.0f} h" if s < 172800 else f"in {s / 86400:.1f} days"


def _half(x, k):
    return None if not x or x.get(k) is None else float(x[k])


@st.fragment(run_every="15s")
def stack() -> None:
    k = kalshi_state(); sel = k["selector"]
    with st.container(border=True):
        st.markdown("**The teacher: the Kalshi strategy stack**")
        if not sel:
            st.info("No Kalshi selector has been fitted on the current definitions yet. The pipeline below fits one as soon as the corpus is built.", icon=":material/info:")
            return
        meta = sel["meta"]; wf = meta.get("walk_forward") or {}; rb = meta.get("random_baseline") or {}
        st.caption(f"Selector #{sel['id']} · fitted walk-forward by settlement day on {meta.get('first_day', '—')} → {meta.get('last_day', '—')} ({meta.get('rows', '—'):,} rows) · saved {ago(sel['ts'])} · "
                   + ("put to work: " if meta.get("deployable") else "not put to work: ") + str(meta.get("deploy_reason") or ""))
        with st.container(horizontal=True):
            st.metric("Evaluation trades", wf.get("n", "—"), help="Positions on the evaluation half, each scored by a classifier that never saw that settlement day.")
            st.metric("Average per trade", pct(wf.get("mean")), delta=(f"random {pct(rb['mean'])}" if rb.get("mean") is not None else None), delta_color="off",
                      help="Net return per dollar of payout at Kalshi's fees; 'random' is the same number of random rows bought at the ask.")
            st.metric("Winning", pct(wf.get("win"), 0, False)); st.metric("Profit factor", f"{wf['pf']:.2f}" if wf.get("pf") else "—")
            st.metric("Profitable days", f"{wf.get('days_positive', '—')} of {wf.get('days', '—')}")
        comps = meta.get("components") or []
        if comps:
            st.dataframe(pd.DataFrame([{"component": c["name"], "kept": c["passed"], "why": c["reason"], "winning (evaluation)": (_half(c.get("evaluation"), "win") or 0) * 100 if _half(c.get("evaluation"), "win") is not None else None,
                                        "profit factor": _half(c.get("evaluation"), "pf"), "trades": _half(c.get("evaluation"), "n"), "settings tried": c.get("trials")} for c in comps]), hide_index=True,
                         column_config={"winning (evaluation)": st.column_config.NumberColumn(format="%.0f%%"), "profit factor": st.column_config.NumberColumn(format="%.2f")})
        strat = meta.get("strategies") or {}
        if strat:
            st.dataframe(pd.DataFrame([{"strategy": name, "arm": v.get("hold_min"), "edge line": v.get("line"), "trigger": json.dumps(v.get("thr") or {}, default=str),
                                        "trades (evaluation)": _half(v.get("evaluation"), "n"), "winning": (_half(v.get("evaluation"), "win") or 0) * 100 if _half(v.get("evaluation"), "win") is not None else None,
                                        "profit factor": _half(v.get("evaluation"), "pf")} for name, v in strat.items()]), hide_index=True,
                         column_config={"edge line": st.column_config.NumberColumn(format="%.3f", help="Minimum p̂ − fee-inclusive price to trade."), "winning": st.column_config.NumberColumn(format="%.0f%%"),
                                        "profit factor": st.column_config.NumberColumn(format="%.2f")})


@st.fragment(run_every="3s")
def retraining() -> None:
    k = kalshi_state(); sup = get_supervisor(); ps = pipeline.state(); alive = sup.alive("kalshi_train"); pr = k["train_progress"]
    running = alive and str(ps.get("stage", "")).startswith(("training", "bootstrapping"))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/autorenew: **Kalshi training pipeline** · every {pipeline.INTERVAL_DAYS} days until the visual fly trades, then on request")
            st.space("stretch")
            if st.button("Retrain now", icon=":material/play_arrow:", type="primary", key="kretrain_now", disabled=running):
                pipeline.request_run()
                if not alive:
                    try:
                        sup.start("kalshi_train", started_by="console")
                    except RuntimeError as e:
                        st.error(str(e))
                st.rerun()
            if running and st.button("Stop", icon=":material/stop:", key="ktrain_stop"):
                sup.request_stop("kalshi_train"); st.rerun()
        st.caption("1 · The Kalshi strategy stack (favorite, momentum, constraint, ev; taker and maker arms) is fitted walk-forward by settlement day at Kalshi's fees and judged against random picks.  \n"
                   "2 · The visual fly is distilled from it once — when none exists for the current definitions, or on request — with its features entering the photoreceptors; "
                   "then it learns from every settlement.  \n3 · The replay (`fly-trader kalshi-fly-replay`) must pass before it trades.")
        if not alive:
            st.warning("The Kalshi trainer is not running: start it on the Processes page or click Retrain now.", icon=":material/warning:")
        with st.container(horizontal=True):
            st.metric("Status", (ps.get("stage") or "idle").capitalize() if alive else "Off")
            st.metric("Last run", ago(ps["last_run_at"]) if ps.get("last_run_at") else "never")
            st.metric("Next run", _until(ps.get("next_run_at")) if ps.get("next_run_at") else ("as soon as the corpus is built" if alive else "—"))
            st.metric("Latest selector", f"#{k['selector']['id']}" if k["selector"] else "—")
            st.metric("Latest bootstrap", f"#{k['bootstrap']['id']}" if k["bootstrap"] else "—", delta=("may trade" if k["deployable"] else "may not trade") if k["bootstrap"] else None, delta_color="off")
        if running and pr.get("total"):
            st.progress(min(1.0, float(pr.get("step") or 0) / max(float(pr["total"]), 1.0)), text=f"{pr.get('stage', '')}" + (f" · about {float(pr['eta_s']) / 60:.0f} min left" if pr.get("eta_s") else ""))
        if ps.get("stage") == "waiting for data" and ps.get("waiting"):
            st.caption(f"Waiting for the corpus: {ps['waiting']}. Checking again {_until(ps.get('retry_after'))}.")
        if ps.get("stage") == "failed" and ps.get("last_error"):
            st.error(f"The last run failed: {ps['last_error']}. Retrying {_until(ps.get('retry_after'))}.")


@st.fragment(run_every="15s")
def visual_fly() -> None:
    k = kalshi_state(); fly = k["fly"]; rp = k["replay"]; sup = get_supervisor()
    running = bool(fly.get("stage")) and fly.get("stage") != "not trading"; frozen = bool(fly.get("learning_frozen"))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":material/visibility: **The visual fly** · the optic lobes taught once by the Kalshi stack, then learning from every settlement")
            st.space("stretch")
            if st.button("Resume learning" if frozen else "Pause learning", key="kfly_learn", disabled=not running):
                fly_session.send_command("resume" if frozen else "pause"); st.toast("Sent: applied at the next minute")
            if st.button("Roll back", key="kfly_rollback", disabled=not running, help="Restore the newest good hourly snapshot at least a day old (or the bootstrap)."):
                fly_session.send_command("rollback"); st.toast("Sent: applied at the next minute")
            if st.button("Re-bootstrap", key="kfly_reboot", help="The Kalshi stack refits and teaches a new visual fly."):
                pipeline.request_run(fly=True, reason="console"); st.toast("Re-bootstrap requested")
            with st.popover("Reset fly", disabled=sup.alive("kalshi_runner"), help="Stop the Kalshi engine first." if sup.alive("kalshi_runner") else None):
                st.markdown("Archive and clear the visual fly's paper books, orders and everything it learned?")
                if st.button("Reset", type="primary", key="kfly_reset_confirm"):
                    st.success(f"Archived to {reset_kalshi_fly(reason='console button')['archived_to']}")
        if not rp:
            st.info("The Kalshi replay has not run on the current definitions: `fly-trader kalshi-fly-replay` bootstraps a visual fly, lets it learn from months of settlements "
                    "without retraining, and judges it — the proof required before it trades.", icon=":material/info:")
        else:
            ev, rnd, fr = rp.get("evaluation") or {}, rp.get("random") or {}, rp.get("frozen") or {}
            (st.success if rp.get("passed") else st.warning)(f"Replay {'passed' if rp.get('passed') else 'failed'}: {rp.get('reason')}", icon=":material/verified:" if rp.get("passed") else ":material/block:")
            with st.container(horizontal=True):
                st.metric("Replay trades", ev.get("n", "—"), border=True)
                st.metric("Average per trade", pct(ev.get("mean")), delta=(f"random {pct(rnd.get('mean'))}" if rnd.get("mean") is not None else None), delta_color="off", border=True)
                st.metric("Frozen fly", pct(fr.get("mean")), border=True, help="The same bootstrap without plasticity.")
                st.metric("Profitable days", f"{ev.get('days_positive', '—')} of {ev.get('days', '—')}", border=True)
                st.metric("Learning rate α", f"{rp.get('alpha') or 0:g}", border=True)
                st.metric("Forgetting half-life", f"{rp['half_life_days']:g} days" if rp.get("half_life_days") else "none", border=True)
            per = rp.get("per_strategy") or {}
            if per:
                st.caption("Per strategy: " + " · ".join(f"{s} ({v.get('arm')}): α {v.get('alpha')}, τ½ {v.get('half_life_days') or '∞'}" for s, v in per.items()))
        if running:
            ch = fly.get("checks") or {}; lines = fly.get("lines") or {}; fl = fly.get("frozen_lines") or {}; arms = fly.get("arms") or {}
            st.dataframe(pd.DataFrame([{"strategy": s, "arm": arms.get(s), "edge line": float(lines[s]), "shadow line": float(fl.get(s, lines[s])), "drift": float((fly.get("drift_by_strategy") or {}).get(s) or 0) * 100,
                                        "24 h IC": (ch.get(s) or {}).get("ic"), "checks": ", ".join((ch.get(s) or {}).get("triggers") or []) or "ok"} for s in lines]), hide_index=True,
                         column_config={"edge line": st.column_config.NumberColumn(format="%.3f"), "shadow line": st.column_config.NumberColumn(format="%.3f"),
                                        "drift": st.column_config.NumberColumn(format="%.2f%%"), "24 h IC": st.column_config.NumberColumn(format="%.3f")})
            with st.container(horizontal=True):
                st.metric("Settlements pending", fly.get("pending", "—"), help="Scored rows waiting for their market to resolve.")
                st.metric("Learned this minute", (fly.get("learned") or {}).get("n", 0))
                st.metric("Learning", "frozen" if frozen else "on"); st.metric("Device", fly.get("device") or "—")
            upd = q("SELECT hour, n, mean_abs_delta, drift, ic FROM kalshi_fly_updates ORDER BY hour DESC LIMIT 72")
            if upd:
                st.line_chart(pd.DataFrame(upd), x="hour", y=["drift", "ic"], x_label="", height=200)
            rb = q("SELECT ts, reason, from_snapshot, to_snapshot FROM kalshi_fly_rollbacks ORDER BY id DESC LIMIT 20")
            if rb:
                with st.expander(f"Rollbacks ({len(rb)})", icon=":material/history:"):
                    st.dataframe(pd.DataFrame(rb), hide_index=True)
        else:
            st.caption(f"Not trading: {fly.get('detail') or k['trading_why']}")
        b = k["bootstrap"]
        if b:
            m = b["meta"]; d = m.get("diagnostics") or {}; ok, why = KF.deployable(m)
            st.caption(f"Bootstrap #{b['id']} ({ago(b['ts'])}): {m.get('photoreceptors', '?'):,} photoreceptors carry the inputs into {m.get('neurons', '?'):,} neurons; the mushroom body carries "
                       f"{pct(d.get('mbon_share'), 0, False)} of the score, {pct(d.get('mbon_saturated'), 0, False)} of MBONs saturated, slope {d.get('slope') or 0:.2f} against its teacher. "
                       + ("May trade: " if ok else "May not trade: ") + why + ".")


def history() -> None:
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind IN ('kalshi_selector', 'kalshi_fly_selector') ORDER BY id DESC LIMIT 30")
    with st.expander("Saved Kalshi models", icon=":material/folder_open:"):
        rows = []
        for r in snaps:
            m = parse_note(r["note"]); sel = r["kind"] == "kalshi_selector"
            ok = kselector.is_current(m) if sel else KF.is_current(m)
            wf = (m.get("walk_forward") or {}) if sel else {"n": (m.get("calibration") or {}).get("trades"), "mean": (m.get("calibration") or {}).get("mean")}
            rows.append({"model": f"#{r['id']}", "type": "stack" if sel else "visual fly bootstrap", "saved": r["ts"], "data": "current" if ok else "outdated",
                         "trades": wf.get("n") if ok else None, "% / trade": (wf["mean"] * 100) if ok and wf.get("mean") is not None else None,
                         "put to work": (bool(m.get("deployable")) if sel else KF.deployable(m)[0]) if ok else None})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"saved": st.column_config.DatetimeColumn(format="MMM D HH:mm"), "% / trade": st.column_config.NumberColumn(format="%+.2f")})
        else:
            st.caption("No saved Kalshi models.")
    with st.expander("Kalshi trainer log", icon=":material/terminal:"):
        st.code(get_supervisor().log_tail("kalshi_train", 40) or "(no log yet)", language="json")


stack()
retraining()
visual_fly()
history()
