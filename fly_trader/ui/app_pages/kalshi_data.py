"""Kalshi data: the live feed, the history corpus (dataset seed, candles, feature builds) and the catalogue's coverage."""
import glob
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.kalshi.features import KALSHI_FEATURE_VERSION
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago, setting


def _state_badge(alive: bool, ok: bool = True) -> None:
    st.badge("running" if alive and ok else "stale" if alive else "stopped", color="green" if alive and ok else "orange" if alive else "gray")


@st.cache_data(ttl=60, show_spinner=False)
def feature_parts() -> dict:
    from fly_trader.kalshi.mature import part_version
    files = sorted(glob.glob(str(config.KALSHI_DIR / "features" / "*" / "part-*.parquet")))
    days = sorted({f.split("/")[-2] for f in files})
    return {"n": len(files), "days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
            "current": sum(1 for f in files if part_version(f) == KALSHI_FEATURE_VERSION)}


@st.cache_data(ttl=60, show_spinner=False)
def corpus_counts() -> dict:
    c = {r["status"]: int(r["n"]) for r in q("SELECT status, count(*) AS n FROM kalshi_corpus GROUP BY status")}
    m = q1("SELECT count(*) AS n, count(*) FILTER (WHERE result IN ('yes','no')) AS settled, count(*) FILTER (WHERE result IS NULL AND close_time > now()) AS open, min(close_time) AS first FROM kalshi_markets") or {}
    seed = q1("SELECT value FROM ui_settings WHERE key = 'kalshi_dataset_seed'")
    cats = q("""SELECT COALESCE(s.category, e.category, 'unknown') AS category, count(*) AS markets FROM kalshi_markets m LEFT JOIN kalshi_events e USING (event_ticker)
                LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker WHERE m.result IN ('yes','no') GROUP BY 1 ORDER BY 2 DESC LIMIT 15""")
    return {"corpus": c, "markets": m, "seed": seed["value"] if seed else None, "categories": cats}


@st.fragment(run_every="10s")
def pipelines() -> None:
    ws = get_supervisor().status(); now = datetime.now(timezone.utc)
    ks, ks_at = setting("kalshi_stream_status"); hs, hs_at = setting("kalshi_history_status")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(":material/sensors: **Kalshi feed** (live)")
                ft = ks.get("flushed_through"); age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None
                _state_badge(ws["kalshi_stream"]["alive"], age is not None and age < 180)
            with st.container(horizontal=True):
                st.metric("Messages / s", f"{float(ks.get('events_per_s') or 0):.0f}")
                st.metric("Newest minute", f"{age:.0f} s old" if age is not None else "—")
                st.metric("Markets quoted", f"{int(ks.get('tickers_quoted') or 0):,}", help="Distinct markets with a quote since the feed started (combos excluded).")
                st.metric("Settlements seen", int(ks.get("settled") or 0)); st.metric("Reconnects", int(ks.get("reconnects") or 0))
            st.caption(f"ticker, trade and lifecycle channels → kalshi_minutes / kalshi_quotes · {'private channels on' if ks.get('private') else 'public channels only'} · "
                       f"late {int(ks.get('dropped_late') or 0):,} · updated {ago(ks_at)}")
    with b:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(":material/history: **Kalshi history** (training data)")
                _state_badge(ws["kalshi_history"]["alive"])
            cc = corpus_counts(); c = cc["corpus"]; total = sum(c.values())
            with st.container(horizontal=True):
                st.metric("Settled markets", f"{int((cc['markets'] or {}).get('settled') or 0):,}")
                st.metric("Candles filled", f"{c.get('done', 0) + c.get('built', 0):,}/{total:,}")
                st.metric("Built", f"{c.get('built', 0):,}"); st.metric("Errors", f"{c.get('error', 0):,}")
            if total:
                st.progress(min(1.0, (c.get("done", 0) + c.get("built", 0)) / total))
            st.caption(f"Since {config.KALSHI_HISTORY_START} · stage {hs.get('stage') or '—'}" + (f" · about {float(hs['eta_h']):.1f} h left" if hs.get("eta_h") else "")
                       + f" · dataset seed {'done' if cc['seed'] else 'not yet'} · updated {ago(hs_at)}")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(":material/table_chart: **Kalshi features**")
            fp = feature_parts()
            with st.container(horizontal=True):
                st.metric("Days built", fp["days"]); st.metric("Range", f"{fp['first'] or '—'} → {fp['last'] or '—'}")
                st.metric("Current version", f"{fp['current']}/{fp['n']}", help="Parts built with the feature engine's current definitions; training refuses a mix.")
            st.caption("One row per (market, minute, side) every 5 minutes inside the entry window, through the same feature engine the live engine runs.")
    with b:
        with st.container(border=True):
            st.markdown(":material/category: **Coverage by category**")
            cats = corpus_counts()["categories"]
            if cats:
                st.dataframe(pd.DataFrame(cats), hide_index=True, height=260)
            else:
                st.caption("No settled markets in the catalogue yet.")


pipelines()
