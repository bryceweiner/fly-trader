"""Safety & wallet: the trading rails and their controls, the configured mode and limits, the bot wallet, maintenance."""
import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.agent import rails
from fly_trader.db.queries import q, q1
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago


@st.fragment(run_every="5s")
def rails_panel(circuit_id: int = 1, title: str = "Trading rails", prefix: str = "") -> None:
    c = q1("SELECT * FROM circuit_state WHERE id = %s", (circuit_id,)) or {}
    with st.container(border=True):
        st.markdown(f":material/shield: **{title}**")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Kill switch tripped" if c.get("kill_switch") else "Kill switch armed", color="red" if c.get("kill_switch") else "green")
            st.caption("Trips when the live book falls too far below its peak wealth; blocks new entries until cleared."
                       + (f" Reason: {c['kill_reason']}." if c.get("kill_reason") else ""))
            st.space("stretch")
            if st.button("Clear kill switch", key=f"{prefix}kill_clear", disabled=not c.get("kill_switch"), help="Clears the switch and re-bases the peak at current wealth."):
                rails.reset_circuit(kill=True, circuit_id=circuit_id); st.toast("Kill switch cleared"); st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"Circuit tripped ({c.get('fail_count', 0)} failures)" if c.get("tripped") else f"Circuit ok ({c.get('fail_count', 0)} failures)",
                     color="red" if c.get("tripped") else "green")
            st.caption("Trips after repeated failed or unverifiable live orders; blocks new live entries.")
            st.space("stretch")
            if st.button("Reset circuit", key=f"{prefix}circuit_reset", disabled=not c.get("tripped") and not c.get("fail_count")):
                rails.reset_circuit(kill=False, circuit_id=circuit_id); st.toast("Circuit reset"); st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("New entries paused" if c.get("entries_paused") else "New entries allowed", color="orange" if c.get("entries_paused") else "green")
            st.caption("Operator switch: no new positions while paused; open positions still exit on schedule.")
            st.space("stretch")
            if st.button("Resume entries" if c.get("entries_paused") else "Pause entries", key=f"{prefix}entries_toggle"):
                rails.set_entries_paused(not c.get("entries_paused"), circuit_id=circuit_id); st.rerun()
        for b in q("SELECT book, halted, reason FROM book_state WHERE halted AND (book LIKE '%%kalshi%%') = %s ORDER BY book", (circuit_id == rails.KALSHI_CIRCUIT,)):
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


def kalshi_settings_panel() -> None:
    missing = config.kalshi_live_prerequisites_missing()
    with st.container(border=True):
        st.markdown(":material/tune: **Kalshi mode and limits**")
        rows = [("Money", "paper; live mirror " + ("on" if not missing else "gated: " + ", ".join(missing)), "Both arms always trade paper books; with KALSHI_LIVE_ENABLED=1 and the prerequisites they mirror onto the subaccount."),
                ("KALSHI_LIVE_ENABLED", "1" if config.KALSHI_LIVE_ENABLED else "0", "Allows real orders on the subaccount."),
                ("Taker / maker live", f"{'on' if config.KALSHI_TAKER_LIVE else 'off'} / {'on' if config.KALSHI_MAKER_LIVE else 'off'}", "Per-arm live switches."),
                ("Subaccount", str(config.KALSHI_SUBACCOUNT) if config.KALSHI_SUBACCOUNT else "none (fly-trader kalshi-subaccount create)", "The dedicated subaccount; the primary never trades."),
                ("Capital cap", f"${config.KALSHI_CAPITAL_USD:g}", "Paper books start here; live never sizes above it. Position ≤ 10 %, slate ≤ 50 % of it."),
                ("Cash floor", f"${config.KALSHI_CASH_FLOOR_USD:g}", "Never spent."),
                ("Kill switch", f"{config.KALSHI_KILL_SWITCH_DRAWDOWN:.0%} drawdown", "On the combined live wealth (circuit 2); paper books halt on their own peaks."),
                ("Entry window", f"{config.KALSHI_MIN_MINUTES_TO_CLOSE:g} min … {config.KALSHI_MAX_DAYS_TO_CLOSE:g} days to close", "Rows outside it are never scored."),
                ("Gates", f"spread ≤ {config.KALSHI_MAX_SPREAD_CENTS}c, OI ≥ {config.KALSHI_MIN_OPEN_INTEREST}, 24 h volume ≥ {config.KALSHI_MIN_VOLUME_24H}", "Eligibility of a (market, side, minute)."),
                ("Maker", f"quiet {config.KALSHI_MAKER_QUIET_MIN:g} min before close · TTL {config.KALSHI_MAKER_TTL_H:g} h · ≤ {config.KALSHI_MAKER_MAX_RESTING} resting", "Resting bids one tick inside the ask, post-only."),
                ("Fees", f"taker {config.KALSHI_FEE_FACTOR:g} · maker {config.KALSHI_MAKER_FEE_FACTOR:g} (× series multiplier × p(1−p))", "The vendored fee model.")]
        st.dataframe(pd.DataFrame(rows, columns=["setting", "value", "meaning"]), hide_index=True)


@st.fragment(run_every="30s")
def kalshi_account_panel() -> None:
    with st.container(border=True):
        st.markdown(":material/account_balance: **Kalshi subaccount**")
        if not config.KALSHI_API_KEY_ID:
            st.caption("No Kalshi key configured (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH)."); return
        st.caption(f"Key {config.KALSHI_API_KEY_ID[:8]}… · subaccount {config.KALSHI_SUBACCOUNT or 'none'} · fund it with `fly-trader kalshi-fund --usd N`; `fly-trader kalshi-status` prints balances, orders and settlements.")
        resting = q1("SELECT count(*) AS n, COALESCE(sum(price_cents * count), 0) AS c FROM kalshi_orders WHERE book LIKE 'live%%' AND status IN ('resting','submitted')") or {}
        opens = q1("SELECT count(*) AS n, COALESCE(sum(cost_cents + fee_cents), 0) AS c FROM kalshi_positions WHERE book LIKE 'live%%' AND status = 'open'") or {}
        w = q1("SELECT wealth, sol_free AS cash, ts FROM wealth_marks WHERE book = 'live_kalshi' ORDER BY ts DESC LIMIT 1") or {}
        with st.container(horizontal=True):
            st.metric("Live wealth", f"${float(w['wealth']):.2f}" if w.get("wealth") is not None else "—", help=f"as of {ago(w.get('ts'))}" if w else None)
            st.metric("Cash", f"${float(w['cash']):.2f}" if w.get("cash") is not None else "—")
            st.metric("Open live positions", int(opens.get("n") or 0), delta=f"${float(opens.get('c') or 0) / 100:.2f} at cost", delta_color="off")
            st.metric("Resting live orders", int(resting.get("n") or 0), delta=f"${float(resting.get('c') or 0) / 100:.2f} collateral", delta_color="off")


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
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("Archive, then clear the visual fly's paper books, orders, scored rows and learning; it restarts from its bootstrap.")
            st.space("stretch")
            with st.popover("Reset Kalshi fly", disabled=sup.alive("kalshi_runner"), help="Stop the Kalshi engine first." if sup.alive("kalshi_runner") else None):
                st.markdown("Archive and clear the Kalshi fly's books and learning now?")
                if st.button("Reset", type="primary", key="kalshi_reset_confirm"):
                    from fly_trader.ops.reset import reset_kalshi_fly
                    st.success(f"Archived to {reset_kalshi_fly(reason='console button')['archived_to']}")
        errs = q("SELECT ts, service, endpoint, status, error FROM api_calls WHERE NOT ok ORDER BY id DESC LIMIT 30")
        if errs:
            with st.expander(f"Recent API errors ({len(errs)})"):
                st.dataframe(pd.DataFrame(errs), hide_index=True)


st.markdown("#### Memecoins")
rails_panel()
left, right = st.columns([3, 2])
with left:
    settings_panel()
with right:
    wallet_panel()
st.markdown("#### Prediction markets")
rails_panel(rails.KALSHI_CIRCUIT, "Kalshi trading rails", prefix="k_")
left, right = st.columns([3, 2])
with left:
    kalshi_settings_panel()
with right:
    kalshi_account_panel()
st.markdown("#### Maintenance")
maintenance_panel()
