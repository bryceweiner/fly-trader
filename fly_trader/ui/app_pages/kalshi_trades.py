"""Kalshi trades: settled positions with their results, resting and recent maker orders, exchange fills and the decision log, per book."""
import pandas as pd
import streamlit as st

from fly_trader.db.queries import q, q1
from fly_trader.ui.common import KALSHI_BOOKS, pct, usd

with st.container(horizontal=True, vertical_alignment="bottom"):
    view = st.segmented_control("View", ["Settled positions", "Orders", "Fills", "Decision log"], default="Settled positions", key="ktrades_view")
    book = st.selectbox("Book", list(KALSHI_BOOKS), format_func=lambda b: KALSHI_BOOKS[b], key="ktrades_book", width=220,
                        help="Paper books simulate at the quoted book with real fees; live books are the subaccount's own orders and fills.")

if view == "Settled positions":
    c = q1("SELECT count(*) AS n, sum(realized_cents) AS pnl, avg(realized_cents / NULLIF(cost_cents + fee_cents, 0)) AS avg_ret, count(*) FILTER (WHERE realized_cents > 0) AS wins, "
           "sum(fee_cents) AS fees FROM kalshi_positions WHERE book = %s AND status <> 'open'", (book,)) or {}
    n = int(c.get("n") or 0)
    with st.container(horizontal=True):
        st.metric("Positions", n, border=True)
        st.metric("Realized P&L", usd((c.get("pnl") or 0) / 100.0, signed=True), border=True)
        st.metric("Average return", pct(c.get("avg_ret")), border=True, help="Per position, on the dollars put down (cost plus fee).")
        st.metric("Winners", f"{int(c.get('wins') or 0) / n:.0%}" if n else "—", border=True)
        st.metric("Fees paid", usd((c.get("fees") or 0) / 100.0), border=True)
    rows = q("""SELECT p.ticker, p.side, p.contracts, p.avg_price_cents, p.fee_cents, p.result, p.payout_cents, p.realized_cents, p.cost_cents, p.strategy, p.opened_at, p.closed_at, p.status, m.title
                FROM kalshi_positions p LEFT JOIN kalshi_markets m USING (ticker) WHERE p.book = %s AND p.status <> 'open' ORDER BY p.closed_at DESC LIMIT 300""", (book,))
    if rows:
        df = pd.DataFrame([{"market": (r["title"] or r["ticker"])[:70], "side": r["side"], "strategy": r["strategy"] or "—", "contracts": float(r["contracts"]),
                            "paid (c)": float(r["avg_price_cents"] or 0), "fee ($)": float(r["fee_cents"] or 0) / 100, "result": r["result"] or r["status"],
                            "payout ($)": float(r["payout_cents"] or 0) / 100, "P&L ($)": float(r["realized_cents"] or 0) / 100,
                            "return": float(r["realized_cents"] or 0) / (float(r["cost_cents"] or 0) + float(r["fee_cents"] or 0)) * 100 if r["cost_cents"] else None,
                            "opened": r["opened_at"], "settled": r["closed_at"]} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"opened": st.column_config.DatetimeColumn(format="MMM D HH:mm"), "settled": st.column_config.DatetimeColumn(format="MMM D HH:mm"),
                                                         "paid (c)": st.column_config.NumberColumn(format="%.1f"), "fee ($)": st.column_config.NumberColumn(format="%.2f"),
                                                         "payout ($)": st.column_config.NumberColumn(format="%.2f"), "P&L ($)": st.column_config.NumberColumn(format="%+.2f"),
                                                         "return": st.column_config.NumberColumn(format="%+.1f%%")})
    else:
        st.caption("No settled positions in this book yet.")

elif view == "Orders":
    rows = q("""SELECT o.ts, o.status, o.ticker, o.side, o.price_cents, o.count, o.fill_count, o.tif, o.post_only, o.expiration_ts, o.strategy, o.order_id, o.error, m.title
                FROM kalshi_orders o LEFT JOIN kalshi_markets m USING (ticker) WHERE o.book = %s ORDER BY o.id DESC LIMIT 300""", (book,))
    resting = sum(1 for r in rows if r["status"] == "resting")
    st.caption(f"{resting} resting · the maker arm rests a post-only bid one tick inside the ask and cancels or replaces it when the ask moves; a taker order is an IOC at the walk's worst level.")
    if rows:
        df = pd.DataFrame([{"time": r["ts"], "status": r["status"], "market": (r["title"] or r["ticker"])[:60], "side": r["side"], "price (c)": int(r["price_cents"]), "count": float(r["count"]),
                            "filled": float(r["fill_count"] or 0), "type": ("post-only " if r["post_only"] else "") + r["tif"].replace("_", " "), "expires": r["expiration_ts"],
                            "strategy": r["strategy"] or "—", "exchange id": r["order_id"] or "", "error": r["error"] or ""} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"time": st.column_config.DatetimeColumn(format="MMM D HH:mm:ss"), "expires": st.column_config.DatetimeColumn(format="MMM D HH:mm")})
    else:
        st.caption("No orders in this book yet.")

elif view == "Fills":
    rows = q("SELECT ts, ticker, side, action, price_cents, count, fee_cents, is_taker, order_id, trade_id FROM kalshi_fills WHERE book = %s OR (%s LIKE 'live%%' AND book IS NULL) ORDER BY ts DESC LIMIT 300",
             (book, book))
    if rows:
        df = pd.DataFrame([{"time": r["ts"], "market": r["ticker"], "side": r["side"], "action": r["action"], "price (c)": float(r["price_cents"]), "count": float(r["count"]),
                            "fee ($)": float(r["fee_cents"] or 0) / 100, "taker": bool(r["is_taker"]), "order": r["order_id"], "trade": r["trade_id"]} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"time": st.column_config.DatetimeColumn(format="MMM D HH:mm:ss")})
    else:
        st.caption("No exchange fills for this book (paper books have none: their fills are the positions themselves).")

else:
    kinds = {"Taker bought": ("kalshi_taker_enter",), "Maker posted": ("kalshi_maker_post",), "Maker filled": ("kalshi_maker_fill",), "Maker cancelled": ("kalshi_maker_cancel",),
             "Settled": ("kalshi_settle",), "Exits": ("kalshi_exit",), "Not taken": ("blocked",)}
    pick = st.pills("Show", list(kinds), selection_mode="multi", default=["Taker bought", "Maker posted", "Maker filled", "Settled"], key="kdec_kinds")
    want = [k for p in (pick or []) for k in kinds[p]]
    rows = q("SELECT ts, kind, mint, pool, m_hat, size_sol, rail, reason, detail FROM decisions WHERE kind = ANY(%s) AND detail->>'book' = %s ORDER BY id DESC LIMIT 300", (want, book)) if want else []
    if rows:
        action = {v[0]: k for k, v in kinds.items()}
        df = pd.DataFrame([{"time": r["ts"], "action": action.get(r["kind"], r["kind"]), "market": r["mint"], "side": r["pool"], "edge": float(r["m_hat"]) if r["m_hat"] is not None else None,
                            "dollars": float(r["size_sol"] or 0), "why": (f"{r['rail']}: " if r["rail"] else "") + (r["reason"] or "")} for r in rows])
        st.dataframe(df, hide_index=True, column_config={"time": st.column_config.DatetimeColumn(format="MMM D HH:mm:ss"), "edge": st.column_config.NumberColumn(format="%.3f"),
                                                         "dollars": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.caption("No decisions recorded for this book yet.")
