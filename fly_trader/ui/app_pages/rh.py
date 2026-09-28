"""Robinhood Chain: the Pons indexer, the quote assets' marks, the per-chain verdict and seat, and the ETH wallet's
intents, transactions, legs and base lots. Rails (circuit 3) and limits live on the Safety page."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader import config, markets
from fly_trader.db.queries import q, q1
from fly_trader.ui.common import ago, amount, labels, pct, setting, system_state

if not config.RH_ENABLED:
    st.info("Robinhood Chain is off (RH_ENABLED=0).", icon=":material/info:")
    st.stop()

RH = markets.RH


@st.fragment(run_every="10s")
def indexer() -> None:
    st_, st_at = setting("rh_stream_status")
    with st.container(border=True):
        st.markdown(":material/sensors: **Indexer** · Pons launches, graduations and Uniswap v4 swaps from the chain into 1-minute rows in ETH")
        if not st_:
            st.caption("The indexer has not run yet: start the Robinhood Chain feed on the Processes page."); return
        with st.container(horizontal=True):
            st.metric("Mode", str(st_.get("mode") or "—"), border=True)
            st.metric("Blocks behind", f"{int(st_.get('behind_blocks') or 0):,}", border=True, help=f"head {st_.get('head')}, indexed through {st_.get('index_through')}")
            mt = st_.get("minutes_through")
            st.metric("Minutes through", datetime.fromtimestamp(int(mt), timezone.utc).strftime("%b %d %H:%M") if mt else "—", border=True,
                      help="The engine trades only minutes the indexer has written.")
            st.metric("Updated", ago(st_at), border=True)
        if st_.get("error"):
            st.warning(f"Last error: {st_['error']}", icon=":material/warning:")
        rows = {r["t"]: r["n"] for r in q("SELECT 'launches' AS t, count(*) AS n FROM rh_tokens UNION ALL SELECT 'graduated', count(*) FROM rh_tokens WHERE status = 'graduated' "
                                          "UNION ALL SELECT 'swaps', count(*) FROM rh_swaps UNION ALL SELECT 'minutes', count(*) FROM rh_minutes")}
        st.caption(" · ".join(f"{k}: {int(v):,}" for k, v in rows.items()))


@st.fragment(run_every="30s")
def assets() -> None:
    rows = q("""SELECT a.symbol, a.class, a.decimals, a.asset, p.ts, p.price_eth, p.stale FROM rh_assets a
                LEFT JOIN LATERAL (SELECT ts, price_eth, stale FROM rh_base_prices b WHERE b.asset = a.asset ORDER BY ts DESC LIMIT 1) p ON true
                WHERE a.approved ORDER BY a.class, a.symbol""")
    with st.container(border=True):
        st.markdown(":material/currency_exchange: **Quote assets** · a coin quoted in one of these is bought ETH → base → coin in one transaction and sold back to ETH at once")
        if not rows:
            st.caption("No approved quote assets indexed yet."); return
        now = datetime.now(timezone.utc)
        df = pd.DataFrame([{"asset": r["symbol"] or r["asset"][:10], "class": r["class"], "ETH per unit": 1.0 if r["class"] == "eth" else r["price_eth"],
                            "marked": "native" if r["class"] == "eth" else (ago(r["ts"]) if r["ts"] else "never"),
                            "entries": "open" if r["class"] == "eth" or (r["ts"] and not r["stale"] and (now - r["ts"]).total_seconds() < 600) else "blocked: mark stale"}
                           for r in rows])
        st.dataframe(df, hide_index=True, column_config={"ETH per unit": st.column_config.NumberColumn(format="%.6g")})


@st.fragment(run_every="30s")
def verdict_and_seat() -> None:
    v, v_at = setting("fly_replay"); ho, _ = setting(RH.handover_key); s = system_state(RH)
    with st.container(border=True):
        st.markdown(":material/gavel: **Verdict and seat** · one combined replay judges each chain on its own trades")
        chains = (v or {}).get("chains") or {}
        if not chains:
            st.caption("No combined replay verdict yet: the RH books wait for one (the Robinhood corpus must be built and the models trained on both chains).")
        else:
            with st.container(horizontal=True):
                for c, r in chains.items():
                    ev = r.get("evaluation") or {}
                    st.metric(markets.for_chain(c).name, "passed" if r.get("passed") else "failed", border=True,
                              delta=f"{int(ev.get('n') or 0)} trades · mean {pct(ev.get('mean'))}" if ev else None, delta_color="off", help=r.get("reason"))
            st.caption(f"Replay {ago(v_at)}.")
        if ho:
            st.success(f"The RH fly holds the seat since {ago(ho.get('at'))}." + (" It trades the ETH wallet." if s["live_money"] else
                       " Live trading waits for RH_LIVE_ENABLED=1, the key, the pinned address and a funded wallet."), icon=":material/swap_horiz:")
        else:
            st.caption("The RH fly takes the seat when its realized P&L over 14 days is at least the RH selector's, with at least 30 trades.")


@st.fragment(run_every="15s")
def wallet() -> None:
    wm = q1("SELECT ts, native_wei, consistent FROM rh_wallet_marks ORDER BY ts DESC LIMIT 1") or {}
    w = q1("SELECT wealth, exposure, n_open, drawdown FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (RH.live_book,)) or {}
    lots = q("SELECT asset, qty_raw, basis_eth, source, position_id, opened_at FROM rh_base_lots WHERE status = 'open' ORDER BY id")
    with st.container(border=True):
        st.markdown(":material/account_balance_wallet: **ETH wallet** · book live_rh")
        st.code(config.RH_BOT_ADDRESS or "RH_BOT_ADDRESS is not set", language=None)
        with st.container(horizontal=True):
            st.metric("Native ETH", amount(int(wm["native_wei"]) / 1e18 if wm.get("native_wei") is not None else None, "ETH", 6), border=True,
                      help=f"at the last reconciliation, {ago(wm.get('ts'))}" if wm else "not reconciled yet")
            st.metric("Live wealth", amount(w.get("wealth"), "ETH", 6), border=True)
            st.metric("Open positions", int(w.get("n_open") or 0), border=True)
            st.metric("Books match chain", "yes" if wm.get("consistent", True) else "NO", border=True,
                      help="Token balances vs open positions and lots; native ETH vs every booked flow since the last check.")
        if lots:
            st.markdown("Base lots (two-leg routes that stopped half way; sold back to ETH hourly)")
            st.dataframe(pd.DataFrame(lots), hide_index=True)


@st.fragment(run_every="15s")
def activity() -> None:
    with st.container(border=True):
        st.markdown(":material/receipt_long: **Live activity**")
        view = st.segmented_control("Show", ["Intents", "Transactions", "Legs"], default="Intents", key="rh_activity") or "Intents"
        if view == "Intents":
            rows = q("SELECT created_at, kind, token, route, state, attempts, error FROM rh_intents ORDER BY id DESC LIMIT 100")
            if rows:
                names = labels([r["token"] for r in rows if r["token"]])
                for r in rows:
                    r["token"] = names.get(r["token"], (r["token"] or "")[:10])
        elif view == "Transactions":
            rows = q("SELECT created_at, kind, nonce, status, gas_used, fee_wei::float / 1e18 AS fee_eth, router, hash FROM rh_txs ORDER BY id DESC LIMIT 100")
        else:
            rows = q("SELECT ts, leg_no, asset_in, asset_out, amount_in_raw::text AS amount_in, amount_out_raw::text AS amount_out, eth_in, eth_out, verified_by "
                     "FROM rh_legs ORDER BY id DESC LIMIT 100")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("Nothing yet: the live RH book has not traded.")


indexer()
left, right = st.columns([3, 2])
with left:
    verdict_and_seat()
    assets()
with right:
    wallet()
activity()
