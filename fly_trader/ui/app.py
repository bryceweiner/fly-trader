"""fly-trader operator console (Streamlit). One place to start/stop workers and watch everything.
Never runs trading logic; every control writes a Postgres row or signals a worker process."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.ops import procs
from fly_trader.ops.supervisor import WORKERS, get_supervisor

st.set_page_config(page_title="fly-trader", page_icon="🪰", layout="wide")


def _df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fmt_sol(x):
    return "—" if x is None else f"{float(x):.4f} SOL"


def _age(ts):
    if ts is None:
        return "—"
    s = (datetime.now(timezone.utc) - ts).total_seconds()
    return f"{s:.0f}s" if s < 120 else f"{s/60:.0f}m" if s < 7200 else f"{s/3600:.1f}h"


def _pf(v) -> str:
    """Profit factor for display: None means no losing trade (infinite)."""
    return "∞" if v is None else f"{float(v):.2f}"


@st.fragment(run_every="3s")
def overview():
    wealth = q("""SELECT DISTINCT ON (book) book, ts, wealth, sol_free, positions_value, exposure, n_open, peak, drawdown
                  FROM wealth_marks ORDER BY book, ts DESC""")
    circuit = q1("SELECT * FROM circuit_state WHERE id = 1") or {}
    cap = q1("SELECT * FROM capture_status") or {}
    beat = q1("SELECT ts, total_ms, gpu_ms, n_slots_active, notes FROM beats ORDER BY id DESC LIMIT 1") or {}
    tape = q1("SELECT count(*) AS n FROM swap_tape WHERE ts > now() - interval '1 minute'") or {}
    cols = st.columns(4)
    for i, w in enumerate(wealth[:4]):
        with cols[i % 4]:
            st.metric(f"{w['book']} capital", _fmt_sol(w["wealth"]), delta=f"peak {float(w['peak'] or 0):.4f}",
                      help=f"free {_fmt_sol(w['sol_free'])}, positions {_fmt_sol(w['positions_value'])}, open {w['n_open']}")
    sel = q1("SELECT value, updated_at FROM ui_settings WHERE key = 'selector_status'")
    if sel:
        sv = sel["value"] if isinstance(sel["value"], dict) else json.loads(sel["value"] or "{}")
        st.caption(f"selector · minute {str(sv.get('minute', '—'))[11:16]} UTC · {sv.get('mints_traded', 0)} tokens traded · {sv.get('eligible', 0)} eligible · "
                   f"{sv.get('picks', 0)} picks (threshold {sv.get('threshold', 0):.3f}, p99 score {sv.get('score_p99') or 0:.3f}) · entered {sv.get('entered', 0)} · exited {sv.get('exited', 0)} · "
                   f"open {sv.get('open', 0)} · wealth {_fmt_sol(sv.get('wealth'))} · updated {_age(sel['updated_at'])} ago")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("kill switch", "TRIPPED" if circuit.get("kill_switch") else "armed")
    c2.metric("circuit", "TRIPPED" if circuit.get("tripped") else f"ok ({circuit.get('fail_count', 0)} fails)")
    c3.metric("entries", "PAUSED" if circuit.get("entries_paused") else "open")
    c4.metric("last beat", _age(beat.get("ts")), delta=f"{beat.get('total_ms', 0)} ms" if beat else None)
    c5.metric("tape rows / min", int(tape.get("n") or 0), delta=f"feed age {_age(cap.get('last_swap_ts'))}")
    if beat.get("notes"):
        n = beat["notes"] if isinstance(beat["notes"], dict) else json.loads(beat["notes"])
        if n.get("blocks"):
            st.warning("live entries blocked by: " + ", ".join(n["blocks"]))
    hist = q("""SELECT b.ts, w.book, w.wealth FROM wealth_marks w JOIN beats b ON b.id = w.beat_id
                WHERE b.ts > now() - interval '24 hours' AND b.sim_ts IS NULL ORDER BY b.ts""")
    if hist:
        df = _df(hist).pivot_table(index="ts", columns="book", values="wealth")
        st.line_chart(df, height=220)


@st.fragment(run_every="5s")
def positions():
    rows = q("""SELECT p.book, p.mint, t.symbol, p.opened_at, p.cost_sol, p.entry_price, p.last_mark_price,
                       CASE WHEN p.entry_price > 0 THEN (p.last_mark_price / p.entry_price - 1) * 100 END AS unreal_pct,
                       p.qty, p.forced_exit_kind
                FROM positions p LEFT JOIN tokens t ON t.mint = p.mint WHERE p.status = 'open' ORDER BY p.book, p.opened_at DESC""")
    st.dataframe(_df(rows), width="stretch", hide_index=True)
    closed = q("""SELECT p.book, count(*) AS n, sum(realized_sol) AS realized, sum(CASE WHEN realized_sol > 0 THEN 1 ELSE 0 END) AS winners,
                         sum(CASE WHEN forced_exit_kind IS NOT NULL THEN 1 ELSE 0 END) AS forced
                  FROM positions p WHERE status = 'closed' AND opened_at > now() - interval '7 days' GROUP BY 1 ORDER BY 1""")
    st.caption("closed positions, last 7 days (counted from rows)")
    st.dataframe(_df(closed), width="stretch", hide_index=True)


@st.fragment(run_every="5s")
def decisions():
    rows = q("""SELECT d.ts, d.kind, d.rail, t.symbol, d.mint, d.m_hat, d.size_sol, d.forced, d.book_targets, d.reason
                FROM decisions d LEFT JOIN tokens t ON t.mint = d.mint ORDER BY d.id DESC LIMIT 60""")
    st.dataframe(_df(rows), width="stretch", hide_index=True)
    st.caption("fills")
    fills = q("""SELECT f.ts, f.book, f.side, t.symbol, f.token_delta, f.sol_delta_lamports / 1e9 AS sol_delta, f.price_sol, f.verified_by, f.signature
                 FROM fills f LEFT JOIN tokens t ON t.mint = f.mint ORDER BY f.id DESC LIMIT 40""")
    st.dataframe(_df(fills), width="stretch", hide_index=True)
    errs = q("SELECT ts, service, endpoint, status, error FROM api_calls WHERE NOT ok ORDER BY id DESC LIMIT 20")
    if errs:
        st.caption("recent API errors")
        st.dataframe(_df(errs), width="stretch", hide_index=True)


@st.fragment(run_every="5s")
def brain():
    st.caption(f"brain mode: {config.BRAIN_MODE}")
    if config.BRAIN_MODE == "policy":
        pol = q("SELECT id, ts, note FROM brain_snapshots WHERE kind = 'policy' ORDER BY id DESC LIMIT 20")
        rows = []
        for r in pol:
            try:
                m = json.loads(r["note"] or "{}")
            except Exception:
                m = {}
            rows.append({"id": r["id"], "ts": r["ts"], "iter": m.get("iter"), "eval_net_sol": m.get("eval_net_sol"), "eval_trades": m.get("eval_trades"),
                         "train_net_sol": m.get("train_net_sol"), "trades": m.get("trades"), "entropy": m.get("entropy")})
        bs0 = q1("SELECT live_snapshot_id, pending_snapshot_id FROM brain_state") or {}
        st.caption(f"policy in use: {bs0.get('live_snapshot_id')} · pending: {bs0.get('pending_snapshot_id')} (promote from Controls; applies at the next runner start)")
        st.dataframe(_df(rows), width="stretch", hide_index=True)
        hist = q("SELECT ts, (detail->>'iter')::int AS iter, (detail->>'train_net_sol')::float AS train_net_sol, (detail->>'eval_net_sol')::float AS eval_net_sol, "
                 "(detail->>'trades')::int AS trades, (detail->>'entropy')::float AS entropy FROM events WHERE source = 'ppo' AND message LIKE 'iteration%%' ORDER BY id")
        if hist:
            dfh = _df(hist).set_index("iter")
            c1, c2 = st.columns(2)
            c1.line_chart(dfh[["train_net_sol", "eval_net_sol"]], height=200)
            c2.line_chart(dfh[["trades"]], height=200)
        last = q1("SELECT id FROM beats WHERE sim_ts IS NULL ORDER BY id DESC LIMIT 1")
        if last:
            tg = q("SELECT s.m_hat AS target, s.rho_app AS value, t.symbol, s.mint, s.portfolio[1] AS pos_frac FROM beat_slots s LEFT JOIN tokens t ON t.mint = s.mint WHERE s.beat_id = %s ORDER BY s.m_hat DESC", (last["id"],))
            if tg:
                dft = _df(tg)
                c1, c2 = st.columns([1, 2])
                c1.bar_chart(dft["target"].round(2).value_counts(bins=10).sort_index(), height=200)
                c2.dataframe(dft.head(20), width="stretch", hide_index=True)
        return
    act = q("""SELECT b.ts, a.kc_sparsity, a.mbon_app_rate, a.mbon_av_rate, a.dan_rew_rate, a.dan_pun_rate, a.total_spikes, a.nan_flag
               FROM brain_activity a JOIN beats b ON b.id = a.beat_id WHERE b.sim_ts IS NULL ORDER BY b.id DESC LIMIT 400""")
    if act:
        df = _df(act).set_index("ts").sort_index()
        c1, c2 = st.columns(2)
        c1.line_chart(df[["mbon_app_rate", "mbon_av_rate"]], height=200)
        c2.line_chart(df[["dan_rew_rate", "dan_pun_rate", "kc_sparsity"]], height=200)
    last = q1("SELECT id FROM beats WHERE sim_ts IS NULL ORDER BY id DESC LIMIT 1")
    if last:
        mh = q("SELECT s.m_hat, t.symbol, s.mint, s.danger, s.dwell_beats FROM beat_slots s LEFT JOIN tokens t ON t.mint = s.mint WHERE s.beat_id = %s ORDER BY s.m_hat DESC", (last["id"],))
        if mh:
            df = _df(mh)
            c1, c2 = st.columns([1, 2])
            c1.bar_chart(df["m_hat"].round(3).value_counts(bins=20).sort_index(), height=200)
            c2.dataframe(df.head(20), width="stretch", hide_index=True)
    syn = q("""SELECT ts, frob, frob_capped, n_slots, w_mean, w_max FROM synapse_updates ORDER BY beat_id DESC LIMIT 300""")
    if syn:
        st.line_chart(_df(syn).set_index("ts").sort_index()[["frob", "w_mean"]], height=180)
    snaps = q("SELECT id, ts, kind, beat_id, promoted_at, note FROM brain_snapshots ORDER BY id DESC LIMIT 15")
    bs = q1("SELECT * FROM brain_state") or {}
    st.caption(f"live snapshot: {bs.get('live_snapshot_id')} · pending: {bs.get('pending_snapshot_id')}")
    st.dataframe(_df(snaps), width="stretch", hide_index=True)


@st.fragment(run_every="10s")
def capture():
    cs = q1("SELECT * FROM capture_status") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("pools subscribed", cs.get("pools_subscribed") or 0)
    c2.metric("rows total", cs.get("rows_total") or 0)
    c3.metric("reconnects", cs.get("reconnects") or 0)
    c4.metric("decode failures", cs.get("decode_failures") or 0)
    wp = q1("SELECT count(*) FILTER (WHERE active) AS active, count(*) FILTER (WHERE active AND base_vault IS NOT NULL) AS learned, count(*) AS total FROM watch_pools") or {}
    st.caption(f"watch pools: {wp.get('active')} active, {wp.get('learned')} with vaults learned, {wp.get('total')} total")
    rows = q("""SELECT date_trunc('minute', ts) AS minute, count(*) AS swaps, count(DISTINCT pool) AS pools
                FROM swap_tape WHERE ts > now() - interval '2 hours' GROUP BY 1 ORDER BY 1""")
    if rows:
        st.line_chart(_df(rows).set_index("minute"), height=200)
    tk = q("SELECT watch_status, count(*) AS n FROM tokens GROUP BY 1 ORDER BY 1")
    st.dataframe(_df(tk), hide_index=True)


@st.fragment(run_every="10s")
def wallet():
    ev = q("SELECT ts, kind, pubkey, detail FROM wallet_events ORDER BY id DESC LIMIT 20")
    pub = ev[0]["pubkey"] if ev else None
    st.code(pub or "no wallet yet — run `fly-trader wallet new`")
    st.caption(f"LIVE_ENABLED={'1' if config.LIVE_ENABLED else '0'} · cluster {config.SOLANA_CLUSTER} · capital {config.CAPITAL_SOL} SOL · "
               f"max position {config.MAX_POSITION_SOL} SOL · reserve {config.GAS_RESERVE_SOL} SOL")
    st.dataframe(_df(ev), width="stretch", hide_index=True)


@st.fragment(run_every="5s")
def workers():
    sup = get_supervisor()
    st.caption(f"all workers are threads inside this console process (pid {sup.pid}); nothing runs outside it")
    ext = sup.external()
    if ext:
        st.error("worker processes are running OUTSIDE the console (started from the CLI). Stop them to keep everything contained.")
        for name, a in ext.items():
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"external **{name}** pid {a['pid']} since {_age(a['started_at'])} ago")
            if c2.button("stop external", key=f"ext_stop_{name}"):
                procs.stop(name); st.rerun()
    status = sup.status()
    for name in WORKERS:
        stt = status[name]
        c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
        c1.markdown(f"**{name}**")
        if stt["alive"]:
            c2.markdown(f"🟢 thread alive since {_age(stt['started_at'])} ago" + (" · stopping…" if stt["stopping"] else ""))
        else:
            c2.markdown("⚪ stopped" + (f" · last error: `{stt['error'].splitlines()[0][:120]}`" if stt.get("error") else ""))
        if c3.button("start", key=f"start_{name}", disabled=stt["alive"] or name in ext):
            try:
                sup.start(name, started_by="console")
            except RuntimeError as e:
                st.error(str(e))
            st.rerun()
        if c4.button("stop", key=f"stop_{name}", disabled=not stt["alive"]):
            sup.request_stop(name); st.rerun()
    name = st.selectbox("log", list(WORKERS), key="logsel")
    st.code(sup.log_tail(name, 40) or "(no log yet)", language="json")
    err = status[name].get("error")
    if err:
        st.code(err, language="text")



def _light(stt: dict) -> tuple[str, str]:
    if stt["alive"]:
        return "🟢", "running"
    if stt.get("error"):
        return "🔴", "crashed"
    return "⚪", "stopped"


def _close_dialog():
    st.session_state["light_open"] = None


@st.dialog("thread activity", width="large", on_dismiss=_close_dialog)
def thread_dialog(name: str):
    @st.fragment(run_every="2s")
    def body():
        sup = get_supervisor()
        stt = sup.status()[name]
        dot, word = _light(stt)
        st.markdown(f"### {dot} {name} · {word}")
        meta = []
        if stt.get("started_at"):
            meta.append(f"started {_age(stt['started_at'])} ago")
        if stt.get("stopped_at") and not stt["alive"]:
            meta.append(f"stopped {_age(stt['stopped_at'])} ago")
        if meta:
            st.caption(" · ".join(meta))
        if stt.get("error"):
            st.error(stt["error"].splitlines()[0][:300])
        lines = [ln for ln in sup.log_tail(name, 200).splitlines() if ln.strip()][-10:]
        rows = []
        for ln in lines:
            try:
                d = json.loads(ln)
                rows.append({"time": d.get("ts", "")[11:19], "level": d.get("level", ""), "what": d.get("msg", "")[:400]})
            except Exception:
                rows.append({"time": "", "level": "", "what": ln[:400]})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)
        else:
            st.info("nothing logged yet")
        if stt.get("error"):
            with st.expander("traceback"):
                st.code(stt["error"], language="text")
    body()


@st.fragment(run_every="3s")
def lights():
    sup = get_supervisor()
    status = sup.status()
    with st.container(horizontal=True):
        for name in WORKERS:
            dot, word = _light(status[name])
            if st.button(f"{dot} {name}", key=f"light_{name}", help=f"{word} · click for the last 10 things this thread did"):
                st.session_state["light_open"] = name
                st.rerun()



@st.fragment(run_every="3s")
def training():
    sup = get_supervisor()
    alive = sup.alive("train")
    ts = q1("SELECT value, updated_at FROM ui_settings WHERE key = 'training_status'")
    st_ = (ts["value"] if ts and isinstance(ts["value"], dict) else json.loads((ts or {}).get("value") or "{}")) if ts else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("trainer", "running" if alive else "stopped")
    c2.metric("stage", st_.get("stage", "—"))
    step, total = st_.get("step"), st_.get("total")
    c3.metric("progress", f"{step}/{total}" if step is not None and total else "—", delta=(f"ETA {st_.get('eta_s', 0)/60:.0f} min" if st_.get("eta_s") else None))
    c4.metric("last update", _age(ts["updated_at"]) + " ago" if ts else "—")
    if step is not None and total:
        st.progress(min(1.0, step / max(total, 1)))
    if st_.get("rows"):
        st.caption(f"decision points: {st_['rows']:,} eligible minutes · {st_.get('tokens', 0):,} tokens · {st_.get('days')} days · base rate {st_.get('base_rate', 0)*100:.1f}% · universe mean {st_.get('universe_mean', 0)*100:+.2f}%")
    wf = st_.get("walk_forward")
    if wf:
        st.success(f"selector walk-forward (top 1%, pessimistic fills): {wf.get('n')} trades · mean {(wf.get('mean') or 0)*100:+.2f}% · median {(wf.get('median') or 0)*100:+.2f}% · "
                   f"win {(wf.get('win') or 0)*100:.0f}% · PF {_pf(wf.get('pf'))} · days positive {wf.get('days_positive')}/{wf.get('days')}")
    if st_.get("reference_gbm"):
        rg = st_["reference_gbm"]
        st.caption(f"reference selector on this split: AUC {rg.get('auc', 0):.3f} · n {rg.get('n')} · mean {(rg.get('mean') or 0)*100:+.2f}% · PF {_pf(rg.get('pf'))}")
    if st_.get("verdict"):
        v = st_["verdict"]; f, g = v.get("fly", {}), v.get("gbm", {})
        (st.success if v.get("fly_beats_gbm") else st.warning)(f"fly {(f.get('mean') or 0)*100:+.2f}%/trade (AUC {f.get('auc', 0):.3f}, n {f.get('n')}, PF {_pf(f.get('pf'))}) vs selector "
                                                                f"{(g.get('mean') or 0)*100:+.2f}%/trade (AUC {g.get('auc', 0):.3f}, n {g.get('n')}, PF {_pf(g.get('pf'))}) → "
                                                                f"{'the fly takes the seat' if v.get('fly_beats_gbm') else 'the selector keeps the seat'}")
    if st_.get("graph"):
        g = st_["graph"]
        st.caption(f"graph: {g.get('N')} neurons · {g.get('edges')} synapses · {g.get('params')} trainable params" + (f" · dataset {st_.get('dataset','')} · train beats {st_.get('train_beats')} · eval beats {st_.get('eval_beats')}" if st_.get('dataset') else ""))
    if st_.get("expert_train_net") is not None:
        st.caption(f"expert baseline: train {st_['expert_train_net']:+.3f} SOL ({st_.get('expert_train_trades')} trades)")
    if st_.get("last_imitation"):
        li = st_["last_imitation"]
        st.info(f"imitation epoch {li.get('epoch')}: accuracy {li.get('acc', 0)*100:.1f}% · student held-out {li.get('student_eval_net', 0):+.3f} SOL ({li.get('student_eval_trades')} trades) · expert held-out {li.get('expert_eval_net', 0):+.3f} SOL")
    if st_.get("last_iteration"):
        lit = st_["last_iteration"]
        st.info(f"PPO iteration {lit.get('iter')}: train {lit.get('train_net_sol', 0):+.3f} SOL over {lit.get('trades')} trades · entropy {lit.get('entropy', 0):.2f}" +
                (f" · held-out {lit['eval_net_sol']:+.3f} SOL ({lit.get('eval_trades')} trades)" if lit.get("eval_net_sol") is not None else ""))
    st.divider()
    with st.container(horizontal=True):
        p = q1("SELECT value FROM ui_settings WHERE key = 'training_params'")
        pv = (p["value"] if p and isinstance(p["value"], dict) else json.loads((p or {}).get("value") or "{}")) if p else {}
        regimen = st.segmented_control("regimen", ["selector", "fly", "ppo"], default=pv.get("regimen", "selector"), key="tp_regimen",
                                       help="selector: gradient-boosted selector, walk-forward then deployable fit · fly: connectome trained on the same decision points, scored against the selector · ppo: legacy")
        days = st.number_input("corpus days", 5, 200, int(pv.get("days", 45)), key="tp_days")
        test_days = st.number_input("test days", 2, 30, int(pv.get("test_days", 9)), key="tp_test_days")
        top_frac = st.number_input("top fraction", 0.001, 0.2, float(pv.get("top_frac", 0.01)), step=0.005, format="%.3f", key="tp_top")
        epochs = st.number_input("fly epochs", 1, 20, int(pv.get("epochs", 2)), key="tp_epochs")
        imitate = int(pv.get("imitate", 4)); iters = int(pv.get("iterations", 24)); window = int(pv.get("window", 400)); eval_every = int(pv.get("eval_every", 4))
        if st.button("start training", disabled=alive, key="train_start"):
            from fly_trader.db.connection import transaction
            with transaction() as conn:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES ('training_params', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (json.dumps({"regimen": regimen or "selector", "days": int(days), "test_days": int(test_days), "top_frac": float(top_frac), "epochs": int(epochs),
                                          "imitate": imitate, "iterations": iters, "window": window, "eval_every": eval_every}),))
            try:
                sup.start("train", started_by="console")
            except RuntimeError as e:
                st.error(str(e))
            st.rerun()
        if st.button("stop training", disabled=not alive, key="train_stop"):
            sup.request_stop("train"); st.rerun()
    err = sup.status()["train"].get("error")
    if err:
        st.error(err.splitlines()[0][:300])
    st.divider()
    wfd = q("SELECT ts, detail->>'day' AS day, round((detail->>'auc')::numeric, 3) AS auc, (detail->>'n')::int AS trades, round((detail->>'mean')::numeric * 100, 2) AS mean_pct, "
            "round((detail->>'median')::numeric * 100, 2) AS median_pct, round((detail->>'win')::numeric * 100) AS win_pct, round((detail->>'pf')::numeric, 2) AS pf "
            "FROM events WHERE source = 'selector' AND message LIKE 'walk-forward%%' ORDER BY id DESC LIMIT 30")
    if wfd:
        st.caption("selector walk-forward days (newest first)")
        st.dataframe(_df(wfd), width="stretch", hide_index=True)
    fv = q("SELECT ts, detail FROM events WHERE source = 'fly_selector' AND message LIKE 'fly vs gbm%%' ORDER BY id DESC LIMIT 5")
    if fv:
        st.caption("fly vs selector verdicts (newest first)")
        st.dataframe(_df(fv), width="stretch", hide_index=True)
    im = q("SELECT ts, (detail->>'epoch')::int AS epoch, (detail->>'acc')::float AS acc, (detail->>'bce')::float AS bce, (detail->>'student_eval_net')::float AS student_eval_net, "
           "(detail->>'student_eval_trades')::int AS student_trades, (detail->>'expert_eval_net')::float AS expert_eval_net FROM events WHERE source = 'ppo' AND message LIKE 'imitation epoch%%' ORDER BY id DESC LIMIT 20")
    if im:
        st.caption("imitation epochs (newest first)")
        st.dataframe(_df(im), width="stretch", hide_index=True)
    hist = q("SELECT ts, (detail->>'iter')::int AS iter, (detail->>'train_net_sol')::float AS train_net_sol, (detail->>'eval_net_sol')::float AS eval_net_sol, "
             "(detail->>'trades')::int AS trades, (detail->>'entropy')::float AS entropy, (detail->>'clipfrac')::float AS clipfrac, (detail->>'secs')::float AS secs "
             "FROM events WHERE source = 'ppo' AND message LIKE 'iteration%%' ORDER BY id DESC LIMIT 200")
    if hist:
        st.caption("PPO iterations (newest first)")
        dfh = _df(hist)
        c1, c2 = st.columns(2)
        c1.line_chart(dfh.set_index("iter").sort_index()[["train_net_sol", "eval_net_sol"]], height=200)
        c2.line_chart(dfh.set_index("iter").sort_index()[["trades"]], height=200)
        st.dataframe(dfh, width="stretch", hide_index=True)
    snaps = q("SELECT id, ts, note FROM brain_snapshots WHERE kind = 'policy' ORDER BY id DESC LIMIT 20")
    rows = []
    for r in snaps:
        try:
            m = json.loads(r["note"] or "{}")
        except Exception:
            m = {}
        rows.append({"id": r["id"], "ts": r["ts"], "stage": m.get("stage", "ppo"), "iter": m.get("iter"), "eval_net_sol": m.get("eval_net_sol", m.get("student_eval_net")),
                     "eval_trades": m.get("eval_trades", m.get("student_eval_trades")), "train_net_sol": m.get("train_net_sol")})
    st.caption("policy checkpoints (promote from Controls)")
    st.dataframe(_df(rows), width="stretch", hide_index=True)
    st.caption("trainer log (last 30 lines)")
    st.code(sup.log_tail("train", 30) or sup.log_tail("ppo", 30) or "(no log yet)", language="json")


def controls():
    from fly_trader.agent import rails
    from fly_trader.brain import checkpoint
    from fly_trader.db.apilog import record_event
    st.subheader("Controls")
    c1, c2, c3 = st.columns(3)
    if c1.button("reset circuit"):
        rails.reset_circuit(kill=False); st.success("circuit reset")
    if c2.button("reset circuit + kill switch (re-base peak)"):
        rails.reset_circuit(kill=True); st.success("kill switch cleared")
    circuit = q1("SELECT entries_paused FROM circuit_state WHERE id = 1") or {}
    if c3.button("resume entries" if circuit.get("entries_paused") else "pause entries"):
        rails.set_entries_paused(not circuit.get("entries_paused")); st.rerun()
    st.divider()
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots ORDER BY id DESC LIMIT 30")
    if snaps:
        sid = st.selectbox("snapshot", [f"{s['id']} · {s['kind']} · {s['ts']:%m-%d %H:%M} · {s['note'] or ''}" for s in snaps])
        if st.button("promote (applies at next runner start)"):
            checkpoint.promote(int(sid.split(" ·")[0]), hot=False); st.success("pending promotion recorded")
    st.divider()
    sup = get_supervisor()
    if sup.alive("runner"):
        st.caption("in selector mode the runner keeps its history; the legacy policy/lif modes reset training state on runner start (RESET_ON_START)")
    elif st.button("reset training state now (archives, then wipes paper/replay history and the brain)"):
        from fly_trader.ops.reset import reset_training_state
        st.json(reset_training_state(reason="console button"))
    kind = st.text_input("snapshot kind", "manual")
    if st.button("request snapshot now"):
        st.info(checkpoint.snapshot_from_live(kind))
    st.divider()
    auto = q1("SELECT value FROM ui_settings WHERE key = 'autostart'") or {"value": {}}
    val = auto["value"] if isinstance(auto["value"], dict) else json.loads(auto["value"] or "{}")
    cols = st.columns(len(WORKERS))
    changed = False
    for i, name in enumerate(WORKERS):
        v = cols[i].checkbox(f"autostart {name}", value=bool(val.get(name)), key=f"auto_{name}")
        if v != bool(val.get(name)):
            val[name] = v; changed = True
    if changed:
        from fly_trader.db.connection import transaction
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('autostart', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (json.dumps(val),))
        record_event("info", "console", "autostart changed", val)


def _autostart_once():
    if get_supervisor().autostarted or st.session_state.get("_autostarted"):
        return
    st.session_state["_autostarted"] = True
    try:
        get_supervisor().autostart()
    except Exception as e:  # never block the page on a worker failing to start
        st.warning(f"autostart: {e}")


@st.fragment(run_every="5s")
def corpus_panel():
    """Historical corpus puller: every pump.fun graduation, whole-life candles + first-hours trades."""
    sup = get_supervisor()
    alive = sup.alive("corpus")
    cs = q1("SELECT value, updated_at FROM ui_settings WHERE key = 'corpus_status'")
    s = (cs["value"] if cs and isinstance(cs["value"], dict) else json.loads((cs or {}).get("value") or "{}")) if cs else {}
    cnt = {r["status"]: int(r["n"]) for r in q("SELECT status, count(*) AS n FROM corpus_tokens GROUP BY status")}
    total = sum(cnt.values())
    c = st.columns(6)
    src = {r["source"]: int(r["n"]) for r in q("SELECT source, count(*) AS n FROM corpus_tokens WHERE status = 'done' GROUP BY source")}
    c[0].metric("tokens ready", f"{cnt.get('done', 0):,}", help="graduated tokens with candle/trade files on disk")
    c[1].metric("from replay archive", f"{src.get('replay', 0):,}", help="pumpapi.io replay: all trades, real reserves, since 2026-04-18")
    c[2].metric("from swap-api pull", f"{src.get('migration_wallet', 0):,}", help="pre-April graduations only (CORPUS_PULL_BEFORE)")
    c[3].metric("pending (swap-api)", f"{cnt.get('pending', 0):,}")
    c[4].metric("with trade rows", f"{int((q1('SELECT count(*) AS n FROM corpus_tokens WHERE trade_path IS NOT NULL') or {}).get('n') or 0):,}")
    eta = s.get("eta_h")
    c[5].metric("swap-api puller", ("idle" if s.get("stage") == "idle" else s.get("stage", "—")) if alive else "stopped",
                delta=(f"{s.get('tokens_per_h', 0):.0f} tokens/h · ETA {eta:.1f} h" if eta and s.get("tokens_per_h") else None))
    if total:
        st.progress(min(1.0, (cnt.get("done", 0) + cnt.get("empty", 0) + cnt.get("error", 0)) / total))
    st.caption(f"stage {s.get('stage', '—')} · backfill through {str(s.get('backfill_through', '—'))[:16]} (target {s.get('backfill_target', '—')}"
               f"{', done' if s.get('backfill_done') else ''}) · candle rows {s.get('candles_rows', 0):,} · trade rows {s.get('trades_rows', 0):,} · "
               f"pace {s.get('pace_s', '—')} s · throttled {s.get('req_429', 0)}× · errors {cnt.get('error', 0)} · last {str(s.get('last_mint', '—'))[:8]} "
               f"(graduated {str(s.get('last_graduated', '—'))[:16]}) · updated {_age(cs['updated_at']) + ' ago' if cs else '—'}")
    rs = q1("SELECT value, updated_at FROM ui_settings WHERE key = 'replay_status'")
    rv = (rs["value"] if rs and isinstance(rs["value"], dict) else json.loads((rs or {}).get("value") or "{}")) if rs else {}
    if rv:
        st.caption(f"replay archive (pumpapi.io, since {rv.get('started_at', '')[:10] and '2026-04-18'}): {rv.get('hours_done', 0):,}/{rv.get('hours_total', 0):,} hours · "
                   f"{rv.get('trades', 0):,} trade rows · {rv.get('events', 0):,} lifecycle rows · {rv.get('mb_s', 0):.1f} MB/s · {rv.get('hours_per_h', 0):.0f} hours/h"
                   + (f" · ETA {rv['eta_h']:.1f} h" if rv.get('eta_h') else "") + f" · last {str(rv.get('last_hour', '—'))[:13]} · errors {rv.get('errors', 0)} · "
                   f"{'running' if sup.alive('replay') else 'stopped'} · updated {_age(rs['updated_at']) + ' ago' if rs else '—'}")
    ps = q1("SELECT value, updated_at FROM ui_settings WHERE key = 'pumpstream_status'")
    if ps:
        pv = ps["value"] if isinstance(ps["value"], dict) else json.loads(ps["value"] or "{}")
        st.caption(f"live stream (pumpapi.io): {'connected' if pv.get('connected') else 'disconnected'} · {pv.get('events_per_s', 0):.0f} events/s · {pv.get('trades', 0):,} PumpSwap trades · "
                   f"{pv.get('flushed_rows', 0):,} minute rows · {pv.get('migrates', 0)} graduations seen · reconnects {pv.get('reconnects', 0)} · updated {_age(ps['updated_at'])} ago")
    fs = q1("SELECT value FROM ui_settings WHERE key = 'corpus_features_status'")
    fv = (fs["value"] if fs and isinstance(fs["value"], dict) else json.loads((fs or {}).get("value") or "{}")) if fs else {}
    fc = q1("SELECT count(*) AS n, coalesce(sum(rows), 0) AS rows, count(*) FILTER (WHERE has_trades) AS with_trades FROM corpus_features") or {}
    st.caption(f"features built: {int(fc.get('n') or 0):,} tokens · {int(fc.get('rows') or 0):,} minute rows · {int(fc.get('with_trades') or 0):,} with trade-path rows · "
               f"builder {fv.get('stage', '—')}" + (f" · {fv['last_error']}" if fv.get('last_error') else ""))
    with st.container(horizontal=True):
        if st.button("start corpus pull", disabled=alive, key="corpus_start"):
            try:
                sup.start("corpus", started_by="console")
            except RuntimeError as e:
                st.error(str(e))
            st.rerun()
        if st.button("stop corpus pull", disabled=not alive, key="corpus_stop"):
            sup.request_stop("corpus"); st.rerun()
    err = sup.status()["corpus"].get("error")
    if err:
        st.error(err.splitlines()[0][:300])
    recent = q("SELECT mint, graduated_at, status, round(life_h::numeric, 1) AS life_h, candles_1m, candles_5m, trades, last_error FROM corpus_tokens "
               "WHERE status <> 'pending' ORDER BY updated_at DESC LIMIT 15")
    if recent:
        st.caption("latest pulled tokens")
        st.dataframe(_df(recent), width="stretch", hide_index=True)


def main():
    st.title("🪰 fly-trader")
    try:
        _autostart_once()
        lights()
        if st.session_state.get("light_open"):
            thread_dialog(st.session_state["light_open"])
        tabs = st.tabs(["Overview", "Training", "Corpus", "Positions", "Decisions & fills", "Brain", "Capture", "Wallet", "Workers", "Controls", "Events"])
        with tabs[0]:
            overview()
        with tabs[1]:
            training()
        with tabs[2]:
            corpus_panel()
        with tabs[3]:
            positions()
        with tabs[4]:
            decisions()
        with tabs[5]:
            brain()
        with tabs[6]:
            capture()
        with tabs[7]:
            wallet()
        with tabs[8]:
            workers()
        with tabs[9]:
            controls()
        with tabs[10]:
            st.dataframe(_df(q("SELECT ts, level, source, message, detail FROM events ORDER BY id DESC LIMIT 100")),
                         width="stretch", hide_index=True)
    except Exception as e:  # keep the console alive on any query error
        st.error(f"{type(e).__name__}: {e}")


main()
