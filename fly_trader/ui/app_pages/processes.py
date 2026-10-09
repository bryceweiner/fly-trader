"""Processes: every worker thread with what it does, its state, start/stop, autostart and recent activity; the event log."""
import json

import pandas as pd
import streamlit as st

from fly_trader.db.apilog import record_event
from fly_trader.db.connection import transaction
from fly_trader.db.queries import q, q1
from fly_trader.ops import procs
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ORDER, ROLE_LABEL, WORKER_INFO, WORKER_TYPE, ago, jv


def _set_autostart(name: str) -> None:
    r = q1("SELECT value FROM ui_settings WHERE key = 'autostart'"); val = jv(r["value"]) if r else {}
    val[name] = bool(st.session_state[f"auto_{name}"])
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('autostart', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (json.dumps(val),))
    record_event("info", "console", "autostart changed", val)


def _recent(name: str) -> pd.DataFrame:
    rows = []
    for ln in [ln for ln in get_supervisor().log_tail(name, 200).splitlines() if ln.strip()][-12:]:
        try:
            d = json.loads(ln); rows.append({"time": d.get("ts", "")[11:19], "level": d.get("level", ""), "what": d.get("msg", "")[:300]})
        except ValueError:
            rows.append({"time": "", "level": "", "what": ln[:300]})
    return pd.DataFrame(rows[::-1])


@st.fragment(run_every="3s")
def workers() -> None:
    sup = get_supervisor(); status = sup.status(); ext = sup.external()
    auto = jv((q1("SELECT value FROM ui_settings WHERE key = 'autostart'") or {}).get("value"))
    if ext:
        st.error("Worker processes are running outside the console (started from the command line). Stop them to keep everything in one place.", icon=":material/warning:")
        for name, a in ext.items():
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"External **{name}** · pid {a['pid']} · started {ago(a['started_at'])}")
                if st.button("Stop external", key=f"ext_stop_{name}"):
                    procs.stop(name); st.rerun()
    shown = None
    for name in ORDER:
        if WORKER_TYPE.get(name) != shown:
            shown = WORKER_TYPE.get(name); st.markdown(f"#### {shown.capitalize()}")
        title, icon, desc, role = WORKER_INFO[name]; stt = status[name]
        state = "stopping" if stt["stopping"] else "running" if stt["alive"] else "crashed" if stt.get("error") else "stopped"
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"{icon} **{title}**")
                st.badge(state, color={"running": "green", "stopping": "orange", "crashed": "red", "stopped": "gray"}[state])
                st.badge(ROLE_LABEL[role], color="gray")
                st.space("stretch")
                if st.button("Start", icon=":material/play_arrow:", key=f"start_{name}", disabled=stt["alive"] or name in ext):
                    try:
                        sup.start(name, started_by="console")
                    except RuntimeError as e:
                        st.error(str(e))
                    st.rerun()
                if st.button("Stop", icon=":material/stop:", key=f"stop_{name}", disabled=not stt["alive"] or stt["stopping"]):
                    sup.request_stop(name); st.rerun()
                st.toggle("Autostart", value=bool(auto.get(name)), key=f"auto_{name}", on_change=_set_autostart, args=(name,),
                          help="Start this worker whenever the console starts.")
                with st.popover("Activity", icon=":material/list:"):
                    df = _recent(name)
                    if len(df):
                        st.dataframe(df, hide_index=True, width=720)
                    else:
                        st.caption("Nothing logged yet.")
            since = f"running since {ago(stt['started_at'])}" if stt["alive"] and stt.get("started_at") else (f"stopped {ago(stt['stopped_at'])}" if stt.get("stopped_at") else "not started")
            st.caption(f"{desc} · {since} · thread `{name}`")
            if stt.get("error") and not stt["alive"]:
                st.error(stt["error"].splitlines()[0][:300])


@st.fragment(run_every="10s")
def event_log() -> None:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":material/history_edu: **Event log**")
            st.space("stretch")
            levels = st.pills("Levels", ["info", "warning", "error"], selection_mode="multi", default=["warning", "error"], key="ev_levels", label_visibility="collapsed")
        rows = q("SELECT ts, level, source, message FROM events WHERE level = ANY(%s) ORDER BY id DESC LIMIT 200", (levels or ["error"],))
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=320, column_config={"ts": st.column_config.DatetimeColumn("time", format="MMM D HH:mm:ss")})
        else:
            st.caption("Nothing at these levels.")


st.caption(f"All workers are threads inside this console process (pid {get_supervisor().pid}); stopping one waits for it to finish its current step.")
workers()
event_log()
