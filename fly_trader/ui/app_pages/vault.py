"""$FLY vault: the trading wallet's ledger (deposits, profit, claims), settlements, the claims queue, halts and the
Robinhood Chain index. Only meaningful on the hosted vault fly (VAULT_ENABLED)."""
import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.vault import flows, state

SOL = config.LAMPORTS_PER_SOL

st.title("$FLY vault")
if not config.VAULT_ENABLED:
    st.info("The vault is off on this machine (VAULT_ENABLED is not set). It runs on the hosted vault fly.")
    st.stop()


@st.fragment(run_every="15s")
def summary() -> None:
    h = state.halted()
    if h:
        st.error("Vault halted: " + "; ".join(h["reasons"]) + ". Claims and entries are stopped.")
        if st.button("Resume vault (I checked the reason)"):
            state.resume(); st.rerun()
    f = q1("SELECT COALESCE(sum(lamports) FILTER (WHERE kind='deposit'),0) d, COALESCE(sum(lamports+fee_lamports) FILTER (WHERE kind='withdrawal'),0) w, "
           "COALESCE(sum(lamports) FILTER (WHERE kind='claim'),0) p, COALESCE(sum(lamports) FILTER (WHERE kind='profit'),0) g FROM vault_flows") or {}
    a = q1("SELECT COALESCE(sum(allocated),0) a FROM vault_settlements WHERE status='allocated'") or {}
    nav = q1("SELECT * FROM vault_nav ORDER BY ts DESC LIMIT 1") or {}
    c = st.columns(6)
    c[0].metric("Wallet NAV", f"{int(nav.get('wealth') or 0) / SOL:.4f} SOL")
    c[1].metric("Deposits", f"{int(f.get('d') or 0) / SOL:.4f}")
    c[2].metric("Withdrawn", f"{int(f.get('w') or 0) / SOL:.4f}")
    c[3].metric("Allocated to lockers", f"{int(a.get('a') or 0) / SOL:.4f}")
    c[4].metric("Claims paid", f"{int(f.get('p') or 0) / SOL:.4f}")
    c[5].metric("Owed (reserved)", f"{max(0, int(a.get('a') or 0) - int(f.get('p') or 0)) / SOL:.4f}")
    scan = flows.cursor()
    ix = q1("SELECT slot, detail, updated_at FROM vault_scan WHERE name='rh_vault'") or {}
    st.caption(f"Wallet scan through slot {scan.get('slot')} · Robinhood Chain indexed through block {ix.get('slot')} "
               f"(updated {ix.get('updated_at')}) · implementation {(ix.get('detail') or {}).get('impl')}")


summary()

st.subheader("Settlements")
st.dataframe(pd.DataFrame(q("SELECT id, period_end, status, realized/1e9 AS realized_sol, pot/1e9 AS pot_sol, allocated/1e9 AS allocated_sol, "
                           "carried/1e9 AS carried_sol, earners, inputs_sha256 FROM vault_settlements ORDER BY period_end DESC LIMIT 52")), hide_index=True)

st.subheader("Claims")
st.dataframe(pd.DataFrame(q("SELECT id, relay_id, evm, sol, status, reason, lamports/1e9 AS sol_amount, tx_signature, updated_at "
                           "FROM vault_claims ORDER BY id DESC LIMIT 100")), hide_index=True)

st.subheader("Wallet flows")
fl = q("SELECT id, block_time, kind, direction, counterparty, lamports/1e9 AS sol_amount, classified_by, signature, note FROM vault_flows ORDER BY id DESC LIMIT 200")
st.dataframe(pd.DataFrame(fl), hide_index=True)
with st.form("reclassify"):
    st.caption("Relabel a transaction (e.g. a deposit you sent from a new address). A change after its week settled only moves future weeks.")
    sig = st.text_input("Transaction signature")
    kind = st.selectbox("Label", ["deposit", "profit", "internal"])
    if st.form_submit_button("Reclassify") and sig:
        try:
            st.success(f"{flows.reclassify(sig.strip(), kind)} row(s) changed")
        except Exception as e:
            st.error(str(e))

st.subheader("Holders")
st.dataframe(pd.DataFrame(q("SELECT evm, allocated/1e9 AS allocated_sol, claimed/1e9 AS claimed_sol, owed/1e9 AS owed_sol FROM vault_accounts ORDER BY owed DESC")),
             hide_index=True)
st.caption("Principal withdrawals: `fly-trader vault withdraw --sol X --to <funding address>` (refuses owed SOL, open positions and the gas reserve).")
