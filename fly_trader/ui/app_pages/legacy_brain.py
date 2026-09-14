"""Brain (legacy policy / lif modes only): network activity, the latest beat's targets and snapshot promotion."""
import json

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1


@st.fragment(run_every="5s")
def brain() -> None:
    st.caption(f"Brain mode: {config.BRAIN_MODE}")
    last = q1("SELECT id FROM beats WHERE sim_ts IS NULL ORDER BY id DESC LIMIT 1")
    if config.BRAIN_MODE == "policy":
        bs = q1("SELECT live_snapshot_id, pending_snapshot_id FROM brain_state") or {}
        st.caption(f"Policy in use: {bs.get('live_snapshot_id')} · pending: {bs.get('pending_snapshot_id')} (applies at the next runner start)")
        if last:
            tg = q("SELECT s.m_hat AS target, s.rho_app AS value, t.symbol, s.mint FROM beat_slots s LEFT JOIN tokens t ON t.mint = s.mint WHERE s.beat_id = %s ORDER BY s.m_hat DESC", (last["id"],))
            if tg:
                st.dataframe(pd.DataFrame(tg).head(20), hide_index=True)
    else:
        act = q("""SELECT b.ts, a.mbon_app_rate, a.mbon_av_rate, a.dan_rew_rate, a.dan_pun_rate, a.kc_sparsity FROM brain_activity a JOIN beats b ON b.id = a.beat_id
                   WHERE b.sim_ts IS NULL ORDER BY b.id DESC LIMIT 400""")
        if act:
            df = pd.DataFrame(act).set_index("ts").sort_index()
            c1, c2 = st.columns(2)
            c1.line_chart(df[["mbon_app_rate", "mbon_av_rate"]], height=200)
            c2.line_chart(df[["dan_rew_rate", "dan_pun_rate", "kc_sparsity"]], height=200)
        if last:
            mh = q("SELECT s.m_hat, t.symbol, s.mint, s.danger, s.dwell_beats FROM beat_slots s LEFT JOIN tokens t ON t.mint = s.mint WHERE s.beat_id = %s ORDER BY s.m_hat DESC", (last["id"],))
            if mh:
                st.dataframe(pd.DataFrame(mh).head(20), hide_index=True)


def promote() -> None:
    from fly_trader.brain import checkpoint
    snaps = q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind NOT IN ('selector', 'fly_selector') ORDER BY id DESC LIMIT 30")
    with st.container(border=True):
        st.markdown("**Promote a snapshot** (applies at the next runner start)")
        if snaps:
            sid = st.selectbox("Snapshot", [f"{s['id']} · {s['kind']} · {s['ts']:%m-%d %H:%M}" for s in snaps])
            if st.button("Promote"):
                checkpoint.promote(int(sid.split(" ·")[0]), hot=False); st.success("Pending promotion recorded")
        else:
            st.caption("No legacy snapshots.")


brain()
promote()
