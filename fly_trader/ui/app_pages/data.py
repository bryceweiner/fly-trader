"""Data pipelines: is each source current, and how far along are the historical builds."""
import glob
from datetime import datetime, timezone

import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q1
from fly_trader.market.features import FEATURE_VERSION
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago, setting


def _state_badge(alive: bool, ok: bool = True) -> None:
    st.badge("running" if alive and ok else "stale" if alive else "stopped", color="green" if alive and ok else "orange" if alive else "gray")


@st.cache_data(ttl=60, show_spinner=False)
def feature_parts() -> dict:
    from fly_trader.train.mature import part_version
    files = sorted(glob.glob(str(config.CORPUS_DIR / "features_mature" / "*" / "part.parquet")))
    days = [f.split("/")[-2] for f in files]
    return {"n": len(files), "first": days[0] if days else None, "last": days[-1] if days else None,
            "current": sum(1 for f in files if part_version(f) == FEATURE_VERSION)}


@st.fragment(run_every="10s")
def pipelines() -> None:
    sup = get_supervisor(); ws = sup.status(); now = datetime.now(timezone.utc)
    ps, ps_at = setting("pumpstream_status"); rs, rs_at = setting("replay_status")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(":material/sensors: **Market feed** (live)")
                ft = ps.get("flushed_through"); age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None
                _state_badge(ws["pumpstream"]["alive"], age is not None and age < 180)
            with st.container(horizontal=True):
                st.metric("Events / s", f"{float(ps.get('events_per_s') or 0):.0f}")
                st.metric("Newest minute", f"{age:.0f} s old" if age is not None else "—", help="Seconds since the newest complete minute closed.")
                st.metric("Graduations seen", int(ps.get("migrates") or 0))
                st.metric("Reconnects", int(ps.get("reconnects") or 0))
            st.caption(f"PumpSwap trades → 1-minute candles (pump_minutes) · dropped as outliers {int(ps.get('dropped_band') or 0):,} · late {int(ps.get('dropped_late') or 0):,} · updated {ago(ps_at)}")
    with b:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(":material/history: **History archive** (training data)")
                _state_badge(ws["replay"]["alive"])
            done, total = int(rs.get("hours_done") or 0), int(rs.get("hours_total") or 0)
            days = q1("SELECT count(*) AS n, max(day) AS last FROM replay_days") or {}
            with st.container(horizontal=True):
                st.metric("Hours ingested", f"{done:,}/{total:,}" if total else f"{done:,}")
                st.metric("Days assembled", int(days.get("n") or 0), delta=f"newest {days.get('last')}" if days.get("last") else None, delta_color="off")
                st.metric("Errors", int(rs.get("errors") or 0))
            if total:
                st.progress(min(1.0, done / total))
            st.caption(f"pumpapi.io hourly archive since {config.REPLAY_START}, newest first · last hour {str(rs.get('last_hour') or '—')[:13]}"
                       + (f" · about {float(rs['eta_h']):.1f} h left" if rs.get("eta_h") else "") + f" · updated {ago(rs_at)}")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(":material/table_chart: **Training features**")
            fp = feature_parts()
            with st.container(horizontal=True):
                st.metric("Days built", fp["n"])
                st.metric("Range", f"{fp['first'] or '—'} → {fp['last'] or '—'}")
                st.metric("Current version", f"{fp['current']}/{fp['n']}", help="Days built with the live feature engine's current definitions; training refuses a mix.")
            st.caption("One row per token-minute of every PumpSwap token, fed through the live feature engine. Built by the history archive worker.")
    with b:
        with st.container(border=True):
            st.markdown(":material/badge: **Token metadata**")
            m = q1("SELECT count(*) AS n, count(creator) AS creators, count(own_dd60) AS outcomes FROM corpus_meta") or {}
            cr = q1("SELECT count(*) AS n FROM pump_events WHERE action = 'create'") or {}
            with st.container(horizontal=True):
                st.metric("Graduations", f"{int(m.get('n') or 0):,}")
                st.metric("With creator", f"{int(m.get('creators') or 0):,}")
                st.metric("Token launches", f"{int(cr.get('n') or 0):,}", help="Creates in pump_events, used for each creator's launch history.")
            st.caption("Graduation time, creation facts and each creator's track record, shared by training and live scoring.")
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":material/query_stats: **Token stats** (Jupiter)")
            _state_badge(ws["discover"]["alive"])
        cov = token_stats_coverage()
        with st.container(horizontal=True):
            st.metric("Tradable tokens with stats", f"{cov['covered']:,}/{cov['universe']:,}",
                      help="Tokens that traded in the last hour with a pool above the selector's gate, and how many have Jupiter stats from the last hour.")
            st.metric("History since", str(cov["first"])[:10] if cov["first"] else "—")
        st.caption("Holders, organic score and top-holder share, recorded every 10 minutes for every token the selector can trade. "
                   "Not a model input yet: Jupiter serves only current values, so these need weeks of recorded history first.")


@st.cache_data(ttl=60, show_spinner=False)
def token_stats_coverage() -> dict:
    from fly_trader.train.decisions import MIN_RESQ_SOL
    c = q1("""SELECT count(*) AS universe, count(*) FILTER (WHERE EXISTS (SELECT 1 FROM token_stats s WHERE s.mint = u.mint AND s.ts > now() - interval '1 hour')) AS covered
              FROM (SELECT DISTINCT mint FROM pump_minutes WHERE ts > now() - interval '1 hour' AND resq_sol >= %s) u""", (MIN_RESQ_SOL,)) or {}
    first = q1("SELECT min(ts) AS first FROM token_stats") or {}
    return {"universe": int(c.get("universe") or 0), "covered": int(c.get("covered") or 0), "first": first.get("first")}


pipelines()
