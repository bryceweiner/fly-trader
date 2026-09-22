"""fly-trader operator console (Streamlit): one place to see what the system is doing and to control each process.
Never runs trading logic; every control writes a Postgres row or signals a worker thread. Pages live in ``app_pages/``;
the status strip under the navigation answers, on every page: paper or live, trading or not, which model, training or
not, and whether the market feed is current."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="fly-trader", page_icon="🪰", layout="wide")

from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import status_strip


def _autostart_once() -> None:
    if get_supervisor().autostarted or st.session_state.get("_autostarted"):
        return
    st.session_state["_autostarted"] = True
    try:
        get_supervisor().autostart()
    except Exception as e:  # never block the page on a worker failing to start
        st.warning(f"autostart: {e}")


_autostart_once()
pages = {
    "Memecoins": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/trades.py", title="Trades", icon=":material/swap_horiz:"),
        st.Page("app_pages/model.py", title="Model & training", icon=":material/psychology:"),
        st.Page("app_pages/data.py", title="Data pipelines", icon=":material/database:"),
    ],
    "Prediction markets": [
        st.Page("app_pages/kalshi_overview.py", title="Kalshi overview", icon=":material/visibility:", url_path="kalshi"),
        st.Page("app_pages/kalshi_trades.py", title="Kalshi trades", icon=":material/receipt_long:", url_path="kalshi-trades"),
        st.Page("app_pages/kalshi_model.py", title="Kalshi model & training", icon=":material/neurology:", url_path="kalshi-model"),
        st.Page("app_pages/kalshi_data.py", title="Kalshi data", icon=":material/database:", url_path="kalshi-data"),
    ],
    "System": [
        st.Page("app_pages/processes.py", title="Processes", icon=":material/settings_applications:"),
        st.Page("app_pages/safety.py", title="Safety & wallet", icon=":material/shield:"),
    ],
}
page = st.navigation(pages, position="top")
status_strip()
try:
    page.run()
except Exception as e:  # keep the console alive on any query error
    st.error(f"{type(e).__name__}: {e}")
