"""Safety & wallet: both chains' bot wallets (balance in USD, withdrawals to your own wallets), the trading rails and their
controls, the configured mode and limits, maintenance."""
import pandas as pd
import streamlit as st

from fly_trader import config, markets
from fly_trader.agent import rails
from fly_trader.db.queries import q, q1
from fly_trader.ops import wallets
from fly_trader.ops.supervisor import get_supervisor
from fly_trader.ui.common import ago, amount, usd

BASE_UNITS = {"sol": config.LAMPORTS_PER_SOL, "rh": 10 ** 18}
PRICE_SOURCE = {"sol": "Jupiter", "rh": "KyberSwap ETH→USDG"}
NO_WALLET = {"sol": "No wallet yet: run `fly-trader wallet new`, then fund it with SOL.",
             "rh": "No wallet yet: run `fly-trader rh-wallet new`, then fund it with ETH on Robinhood Chain."}


@st.cache_data(ttl="15s", show_spinner=False)
def wallet_summary(chain: str) -> dict:
    return wallets.summary(chain)


@st.cache_data(ttl="10s", show_spinner=False)
def withdraw_limits(chain: str, to: str) -> dict:
    return wallets.limits(chain, to)


def short(a: str | None) -> str:
    return f"{a[:6]}…{a[-4:]}" if a else "—"


@st.fragment(run_every="30s")
def wallet_balances(chain: str) -> None:
    spec, w = markets.MARKETS[chain], wallet_summary(chain)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f":material/account_balance_wallet: **{spec.name} bot wallet**")
        st.space("stretch")
        if w["address"]:
            st.link_button("Explorer", wallets.explorer(chain, "address", w["address"]), icon=":material/open_in_new:", type="tertiary")
    if not w["address"]:
        st.caption(NO_WALLET[chain])
        return
    st.code(w["address"], language=None)
    with st.container(horizontal=True):
        st.metric("Balance", usd(w["native_usd"]), delta=amount(w["native"], w["unit"]), delta_color="off", delta_arrow="off",
                  help="What the wallet holds in its own coin; this is what can be withdrawn.")
        st.metric("In live positions", usd(w["positions_usd"]), delta=f"{amount(w['positions'], w['unit'])} · {w['n_open']} open", delta_color="off",
                  delta_arrow="off", help="The live book's open positions at its last mark, net of exit cost"
                  + (f" (marked {ago(w['marked_at'])})." if w["marked_at"] else "; at cost before the first mark."))
        st.metric("Total", usd(w["total_usd"]), delta=amount(None if w["native"] is None else w["native"] + w["positions"], w["unit"]),
                  delta_color="off", delta_arrow="off")
    st.caption(f"1 {w['unit']} = {usd(w['price_usd'])} · {PRICE_SOURCE[chain]}, {ago(w['price_at'])}" if w["price_usd"] else f"No {w['unit']}/USD price yet.")
    if w["error"]:
        st.warning(f"Balance unavailable: {w['error']}")


def send(chain: str, amt: float | None, to: str) -> dict:
    try:
        return wallets.withdraw(chain, amt, to)
    except wallets.REFUSALS as e:
        return {"status": "refused", "error": str(e)}
    except Exception as e:                                              # noqa: BLE001 — shown; the outcome is unknown
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def open_withdraw(chain: str) -> None:
    for k in ("result", "amt", "all"):                                  # a fresh form each time it opens
        st.session_state.pop(f"{chain}_wd_{k}", None)
    st.session_state["withdraw_open"] = chain


def close_withdraw() -> None:
    st.session_state.pop("withdraw_open", None)


def show_result(chain: str, r: dict) -> None:
    link = f" · [transaction]({wallets.explorer(chain, 'tx', r['tx'])})" if r.get("tx") else ""
    what = f"{amount(r.get('amount'), markets.MARKETS[chain].unit, 6)} to `{short(r.get('to'))}`"
    if r["status"] == "confirmed":
        st.success(f"Sent {what}{link}", icon=":material/check_circle:")
    elif r["status"] == "pending":
        st.info(f"Sent {what}, no receipt yet: it is booked when it lands. Look at the transaction before sending again{link}")
    elif r["status"] == "expired":
        st.warning("The transaction expired before it landed: nothing moved. You can try again.")
    elif r["status"] == "failed":
        st.error(f"The transaction failed on chain: only the fee moved{link}")
    elif r["status"] == "refused":
        st.error(f"Nothing was sent: {r['error']}")
    elif r["status"] == "sending":
        st.info("The withdrawal was still running when this window refreshed: its outcome is in the wallet's withdrawals.")
    else:
        st.error(f"{r['error']}. Whether it went out is unknown: look at the wallet on the explorer before trying again.")
    if st.button("Close", key=f"{chain}_wd_close", on_click=close_withdraw):
        st.rerun()


@st.dialog("Withdraw", width="medium", on_dismiss=close_withdraw)
def withdraw_dialog(chain: str) -> None:
    key, unit, base = f"{chain}_wd", markets.MARKETS[chain].unit, BASE_UNITS[chain]
    if (done := st.session_state.get(f"{key}_result")) is not None:   # one withdrawal per opening: a second click never sends again
        show_result(chain, done)
        return
    w = wallet_summary(chain)
    to = st.selectbox("To", wallets.destinations(chain), key=f"{key}_to", help=f"Only your own wallets: {wallets.SETTINGS[chain]} in .env.")
    if to is None:
        st.warning(f"None of your own wallets is listed: add yours to {wallets.SETTINGS[chain]} in .env and restart the console.")
        return
    try:
        lim = withdraw_limits(chain, to)
    except Exception as e:                                              # noqa: BLE001
        st.error(f"Cannot read the wallet: {type(e).__name__}: {e}")
        return
    everything = st.toggle(f"Everything withdrawable: {amount(lim['max'] / base, unit, 6)}", key=f"{key}_all")
    amt = None if everything else st.number_input(f"Amount ({unit})", min_value=0.0, value=None, step=0.01 if chain == "sol" else 0.001,
                                                  format="%.6f", placeholder=f"at most {lim['max'] / base:.6f}", key=f"{key}_amt")
    value, why = wallets.plan(chain, lim, amt) if everything or amt else (0, "enter an amount")
    fee = f"fee ≈ {lim['fee'] / base:.6f} {unit}" if chain == "sol" else f"gas at most {lim['fee'] / base:.6f} {unit}"
    keep = f" {lim['keep'] / base:g} {unit} stays for the exits of {lim['n_open']} open live position(s)." if lim["keep"] else ""
    if why and (everything or amt):
        st.warning(why)
    elif not why:
        px = w.get("price_usd")
        st.markdown(f"Sends **{amount(value / base, unit, 6)}**" + (f" ({usd(value / base * px)})" if px else "") + f" to `{short(to)}`; {fee}.{keep}")
    st.caption("This waits for the network's confirmation, up to about a minute.")
    if st.button(f"Withdraw {amount(value / base, unit, 6)}" if not why else "Withdraw", type="primary", disabled=bool(why), icon=":material/send:", key=f"{key}_go"):
        with st.spinner("Signing, sending, waiting for confirmation…"):
            st.session_state[f"{key}_result"] = {"status": "sending"}
            st.session_state[f"{key}_result"] = send(chain, amt, to)
        wallet_summary.clear()
        withdraw_limits.clear()
        st.rerun()                                                      # the dialog stays open on its result; the page refreshes


def wallet_panel(chain: str) -> None:
    with st.container(border=True):
        wallet_balances(chain)
        addr = wallets.address(chain)
        if not addr:
            return
        blockers = wallets.withdraw_blockers(chain)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("Withdraw: " + "; ".join(blockers) + "." if blockers else f"Withdrawals go only to your own wallets ({wallets.SETTINGS[chain]} in .env).")
            st.space("stretch")
            st.button("Withdraw", key=f"{chain}_withdraw", icon=":material/move_up:", disabled=bool(blockers), on_click=open_withdraw, args=(chain,))
        hist = wallets.history(chain, addr)
        if hist:
            with st.expander(f"Withdrawals ({len(hist)})"):
                df = pd.DataFrame(hist).assign(tx=lambda d: [wallets.explorer(chain, "tx", t) if t else None for t in d["tx"]])
                st.dataframe(df, hide_index=True, column_config={"amount": st.column_config.NumberColumn(markets.MARKETS[chain].unit, format="%.6f"),
                                                                  "tx": st.column_config.LinkColumn("transaction", display_text=":material/open_in_new:")})
        if chain == "sol":
            ev = q("SELECT ts, kind, pubkey FROM wallet_events ORDER BY id DESC LIMIT 20")
            if ev:
                with st.expander("Wallet events"):
                    st.dataframe(pd.DataFrame(ev), hide_index=True)


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
        from fly_trader.markets import MARKETS
        rh_books = [b for m in MARKETS.values() if m.circuit_id == rails.RH_CIRCUIT for b in m.books]
        halted = q("SELECT book, halted, reason FROM book_state WHERE halted ORDER BY book")
        mine = [b for b in halted if ("kalshi" in b["book"]) == (circuit_id == rails.KALSHI_CIRCUIT) and (b["book"] in rh_books) == (circuit_id == rails.RH_CIRCUIT)]
        for b in mine:
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


def rh_settings_panel() -> None:
    missing = config.rh_live_prerequisites_missing()
    with st.container(border=True):
        st.markdown(":material/tune: **Robinhood Chain mode and limits**")
        rows = [("Money", "paper; live mirror " + ("on" if not missing else "gated: " + ", ".join(missing)), "The RH race runs on paper books in ETH; after the RH handover the fly also trades the ETH wallet."),
                ("RH_LIVE_ENABLED", "1" if config.RH_LIVE_ENABLED else "0", "Allows signing real Robinhood Chain transactions (live_rh)."),
                ("Chain", f"{config.RH_EXPECTED_CHAIN_ID}" + (" (testnet)" if config.RH_TESTNET else ""), "The guard refuses any other chain id."),
                ("Bot address", config.RH_BOT_ADDRESS or "not set", "RH_BOT_PRIVATE_KEY must derive to it."),
                ("Starting capital", f"{config.RH_CAPITAL_ETH:g} ETH", "The USD value of 5 SOL at build time."),
                ("Largest buy", f"{config.RH_LABEL_SIZE_ETH:g} ETH", "The label size (0.5 SOL in USD at build time): no RH buy is bigger."),
                ("Smallest position", f"{config.RH_MIN_POSITION_ETH:g} ETH", "Smaller sized buys are skipped."),
                ("Gas reserve", f"{config.RH_GAS_RESERVE_ETH:g} ETH", "Never spent on positions."),
                ("Fee cap", f"{config.RH_MAX_FEE_GWEI:g} gwei", "A transaction never offers more per gas; a spike refuses instead."),
                ("Kill switch", f"{config.RH_KILL_SWITCH_DRAWDOWN:.0%} drawdown", "On the live RH wealth (circuit 3)."),
                ("Routes", "KyberSwap atomic, Uniswap v4 fallback, two legs through the base last", "Every round trip returns to ETH; no base inventory is kept.")]
        st.dataframe(pd.DataFrame(rows, columns=["setting", "value", "meaning"]), hide_index=True)


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


st.markdown("#### Wallets")
chains = [m.chain for m in markets.enabled()]
for col, chain in zip(st.columns(len(chains)), chains):
    with col:
        wallet_panel(chain)
if st.session_state.get("withdraw_open") in chains:                   # open until dismissed or closed, through full reruns
    withdraw_dialog(st.session_state["withdraw_open"])
st.markdown("#### Memecoins")
rails_panel()
settings_panel()
if config.RH_ENABLED:
    st.markdown("#### Robinhood Chain memecoins")
    rails_panel(rails.RH_CIRCUIT, "Robinhood Chain trading rails", prefix="rh_")
    rh_settings_panel()
st.markdown("#### Prediction markets")
rails_panel(rails.KALSHI_CIRCUIT, "Kalshi trading rails", prefix="k_")
left, right = st.columns([3, 2])
with left:
    kalshi_settings_panel()
with right:
    kalshi_account_panel()
st.markdown("#### Maintenance")
maintenance_panel()
