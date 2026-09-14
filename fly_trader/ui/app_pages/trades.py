"""Trades: closed trades with their results, the decision log (every buy, sell and pick not taken) and fills."""
import pandas as pd
import streamlit as st

from fly_trader.db.queries import q, q1
from fly_trader.ui.common import BOOK, labels, pct, sol

books = [r["book"] for r in q("SELECT DISTINCT book FROM positions UNION SELECT DISTINCT book FROM fills ORDER BY 1")] or [BOOK]
with st.container(horizontal=True, vertical_alignment="bottom"):
    view = st.segmented_control("View", ["Closed trades", "Decision log", "Fills"], default="Closed trades", key="trades_view")
    book = st.selectbox("Book", books, index=books.index(BOOK) if BOOK in books else 0, key="trades_book", width=220,
                        help="paper_selector is the selector's paper book; live is the bot wallet.")

if view == "Closed trades":
    c = q1("SELECT count(*) AS n, sum(realized_sol) AS pnl, avg(realized_sol / NULLIF(cost_sol, 0)) AS avg_ret, "
           "count(*) FILTER (WHERE realized_sol > 0) AS wins FROM positions WHERE book = %s AND status = 'closed'", (book,)) or {}
    n = int(c.get("n") or 0)
    with st.container(horizontal=True):
        st.metric("Trades", n, border=True)
        st.metric("Realized P&L", sol(c.get("pnl"), signed=True), border=True)
        st.metric("Average return", pct(c.get("avg_ret")), border=True, help="Per trade, after fees and price impact.")
        st.metric("Winners", f"{int(c.get('wins') or 0) / n:.0%}" if n else "—", border=True)
    rows = q("SELECT mint, opened_at, closed_at, cost_sol, realized_sol, forced_exit_kind FROM positions WHERE book = %s AND status = 'closed' ORDER BY closed_at DESC LIMIT 300", (book,))
    if rows:
        names = labels([r["mint"] for r in rows])
        df = pd.DataFrame([{"token": names.get(r["mint"], r["mint"][:6]), "opened": r["opened_at"], "closed": r["closed_at"],
                            "held (min)": (r["closed_at"] - r["opened_at"]).total_seconds() / 60 if r["closed_at"] else None, "cost (SOL)": float(r["cost_sol"] or 0),
                            "P&L (SOL)": float(r["realized_sol"] or 0), "return": float(r["realized_sol"] or 0) / float(r["cost_sol"]) * 100 if r["cost_sol"] else None,
                            "exit": r["forced_exit_kind"] or "time"} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"opened": st.column_config.DatetimeColumn(format="MMM D HH:mm"), "closed": st.column_config.DatetimeColumn(format="MMM D HH:mm"),
                                                         "held (min)": st.column_config.NumberColumn(format="%.0f"), "cost (SOL)": st.column_config.NumberColumn(format="%.3f"),
                                                         "P&L (SOL)": st.column_config.NumberColumn(format="%+.4f"), "return": st.column_config.NumberColumn(format="%+.2f%%")})
    else:
        st.caption("No closed trades in this book yet.")

elif view == "Decision log":
    kinds = {"Bought": ("selector_enter", "enter"), "Sold": ("selector_exit", "exit"), "Not taken": ("blocked",)}
    pick = st.pills("Show", list(kinds), selection_mode="multi", default=list(kinds), key="dec_kinds")
    want = [k for p in (pick or []) for k in kinds[p]]
    rows = q("SELECT ts, kind, mint, m_hat, size_sol, rail, reason FROM decisions WHERE kind = ANY(%s) ORDER BY id DESC LIMIT 300", (want,)) if want else []
    if rows:
        names = labels([r["mint"] for r in rows]); action = {"selector_enter": "Bought", "enter": "Bought", "selector_exit": "Sold", "exit": "Sold", "blocked": "Not taken"}
        df = pd.DataFrame([{"time": r["ts"], "action": action.get(r["kind"], r["kind"]), "token": names.get(r["mint"], (r["mint"] or "")[:6]),
                            "score": float(r["m_hat"]) if r["m_hat"] is not None else None, "size (SOL)": float(r["size_sol"] or 0),
                            "why": (f"{r['rail']}: " if r["rail"] else "") + (r["reason"] or "")} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"time": st.column_config.DatetimeColumn(format="MMM D HH:mm:ss"), "score": st.column_config.NumberColumn(format="%.3f"),
                                                         "size (SOL)": st.column_config.NumberColumn(format="%.3f")})
    else:
        st.caption("No decisions recorded yet.")

else:
    rows = q("SELECT ts, side, mint, sol_delta_lamports, price_sol, fee_lamports, verified_by, signature FROM fills WHERE book = %s ORDER BY id DESC LIMIT 300", (book,))
    if rows:
        names = labels([r["mint"] for r in rows])
        df = pd.DataFrame([{"time": r["ts"], "side": r["side"], "token": names.get(r["mint"], r["mint"][:6]), "SOL": (r["sol_delta_lamports"] or 0) / 1e9,
                            "price (SOL)": float(r["price_sol"]) if r["price_sol"] is not None else None, "fee (SOL)": (r["fee_lamports"] or 0) / 1e9,
                            "source": "simulated" if r["verified_by"] == "model" else r["verified_by"], "signature": r["signature"]} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"time": st.column_config.DatetimeColumn(format="MMM D HH:mm:ss"), "SOL": st.column_config.NumberColumn(format="%+.4f"),
                                                         "price (SOL)": st.column_config.NumberColumn(format="%.3e"), "fee (SOL)": st.column_config.NumberColumn(format="%.5f")})
    else:
        st.caption("No fills in this book yet.")
