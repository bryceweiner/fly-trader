"""Safety & wallet: the trading rails and their controls, the configured mode and limits, the bot wallet, maintenance."""
import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.agent import rails
from fly_trader.db.queries import q, q1
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago


@st.fragment(run_every="5s")
def rails_panel() -> None:
    c = q1("SELECT * FROM circuit_state WHERE id = 1") or {}
    with st.container(border=True):
        st.markdown(":material/shield: **Trading rails**")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Kill switch tripped" if c.get("kill_switch") else "Kill switch armed", color="red" if c.get("kill_switch") else "green")
            st.caption("Trips when the live book falls too far below its peak wealth; blocks new entries until cleared."
                       + (f" Reason: {c['kill_reason']}." if c.get("kill_reason") else ""))
            st.space("stretch")
            if st.button("Clear kill switch", key="kill_clear", disabled=not c.get("kill_switch"), help="Clears the switch and re-bases the peak at current wealth."):
                rails.reset_circuit(kill=True); st.toast("Kill switch cleared"); st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"Circuit tripped ({c.get('fail_count', 0)} failures)" if c.get("tripped") else f"Circuit ok ({c.get('fail_count', 0)} failures)",
                     color="red" if c.get("tripped") else "green")
            st.caption("Trips after repeated failed or unverifiable live orders; blocks new live entries.")
            st.space("stretch")
            if st.button("Reset circuit", key="circuit_reset", disabled=not c.get("tripped") and not c.get("fail_count")):
                rails.reset_circuit(kill=False); st.toast("Circuit reset"); st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("New entries paused" if c.get("entries_paused") else "New entries allowed", color="orange" if c.get("entries_paused") else "green")
            st.caption("Operator switch: no new positions while paused; open positions still exit on schedule.")
            st.space("stretch")
            if st.button("Resume entries" if c.get("entries_paused") else "Pause entries", key="entries_toggle"):
                rails.set_entries_paused(not c.get("entries_paused")); st.rerun()
        for b in q("SELECT book, halted, reason FROM book_state WHERE halted ORDER BY book"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(f"{b['book']} halted", color="red")
                st.caption(f"The paper book fell too far below its own peak ({b['reason']}); its entries are blocked, the other book trades on.")
                st.space("stretch")
                if st.button("Clear", key=f"halt_{b['book']}"):
                    rails.clear_book_halt(b["book"]); st.rerun()


def settings_panel() -> None:
    with st.container(border=True):
        st.markdown(":material/tune: **Mode and limits**")
        rows = [("Money", "paper, then live once the fly holds the seat", "The race runs on paper books; after the handover the fly also trades the bot wallet, its paper book as the mirror."),
                ("LIVE_ENABLED", "1" if config.LIVE_ENABLED else "0", "Allows signing real transactions: the fly trades the bot wallet once it holds the selector's seat."),
                ("Cluster", config.SOLANA_CLUSTER, "Solana network for live orders."),
                ("Starting capital", f"{config.CAPITAL_SOL:g} SOL", "Paper books start here."),
                ("Position sizing", f"{config.KELLY_FRACTION:g} × Kelly", "Each buy is sized from how certain the model is: this share of the growth-optimal bet for its score band."),
                ("Largest position", f"{config.MAX_POSITION_FRACTION:.0%} of the bankroll", "Bankroll = wealth at cost minus the gas reserve."),
                ("Pool share cap", f"{config.MAX_POOL_SHARE:.0%} of the pool", "Bounds the price impact of larger buys."),
                ("Smallest position", f"{config.MIN_POSITION_SOL:g} SOL", "Smaller sized buys are skipped."),
                ("Fixed size (older models)", f"{config.MAX_POSITION_SOL:g} SOL", "Models trained before sizing trade this size."),
                ("Gas reserve", f"{config.GAS_RESERVE_SOL:g} SOL", "Never spent on positions."),
                ("Reset on start", "yes" if config.RESET_ON_START else "no", "Starting the trading engine archives and clears books other than the race books and the live book.")]
        st.dataframe(pd.DataFrame(rows, columns=["setting", "value", "meaning"]), hide_index=True)
        st.caption("Set in the project .env; changes apply after the console restarts.")


def wallet_panel() -> None:
    ev = q("SELECT ts, kind, pubkey FROM wallet_events ORDER BY id DESC LIMIT 20")
    with st.container(border=True):
        st.markdown(":material/account_balance_wallet: **Bot wallet**")
        st.code(ev[0]["pubkey"] if ev else "No wallet yet: run `fly-trader wallet new`.", language=None)
        if ev:
            with st.expander("Wallet events"):
                st.dataframe(pd.DataFrame(ev), hide_index=True)


def maintenance_panel() -> None:
    sup = get_supervisor()
    with st.container(border=True):
        st.markdown(":material/build: **Maintenance**")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("Archive, then clear old books, their decisions and wealth marks — never the race books (selector, fly), the live book, the fly's learning or the models. Happens automatically when the trading engine starts.")
            st.space("stretch")
            with st.popover("Reset paper history", disabled=sup.alive("runner"), help="Stop the trading engine first." if sup.alive("runner") else None):
                st.markdown("Archive and clear the old books now? The race and live books are kept.")
                if st.button("Reset", type="primary", key="reset_confirm"):
                    from fly_trader.ops.reset import reset_training_state
                    out = reset_training_state(reason="console button")
                    st.success(f"Archived to {out['archived_to']}")
        errs = q("SELECT ts, service, endpoint, status, error FROM api_calls WHERE NOT ok ORDER BY id DESC LIMIT 30")
        if errs:
            with st.expander(f"Recent API errors ({len(errs)})"):
                st.dataframe(pd.DataFrame(errs), hide_index=True)


rails_panel()
left, right = st.columns([3, 2])
with left:
    settings_panel()
with right:
    wallet_panel()
maintenance_panel()
