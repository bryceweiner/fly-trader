"""Overview: what the system is doing in plain words, the paper book, this minute's scoring and open positions."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.ui.common import BOOK, ago, backtest_line, labels, pct, sol, system_state


def narrative(s: dict) -> str:
    m = s["model"]; lines = []
    if m:
        run = m.get("run") or {}; meta = m["meta"]
        money = "No real money is at risk." if not s["live_money"] else "Trades spend real SOL from the bot wallet."
        lines.append(f"**{s['trading_why']}** The selector is paper trading with model **#{m['id']}** (trained through {meta.get('trained_through', '—')}). "
                     f"Every minute it predicts the net {meta.get('horizon_min', '—')}-minute return of each PumpSwap token that traded, at least {(meta.get('model') or {}).get('min_age_h', '—')} hours "
                     f"past graduation, with a pool of at least {(meta.get('model') or {}).get('min_resq_sol', '—')} SOL and {(meta.get('model') or {}).get('min_vol_15m_sol', '—')} SOL traded in the last 15 minutes, and buys those predicted to make at least "
                     f"**{pct(float(run.get('threshold') or meta.get('threshold') or 0), 1)}** — sized by how certain the "
                     f"model is (up to {config.MAX_POSITION_FRACTION:.0%} of the bankroll, never the {config.GAS_RESERVE_SOL:g} SOL gas reserve) — "
                     f"and sells it **{run.get('horizon_min') or meta.get('horizon_min') or 30} minutes** later. {money}")
        lines.append(backtest_line(meta))
    fly = s["fly"]
    if fly.get("stage") and fly.get("stage") != "not trading":
        lines.append("**The plastic fly** " + ("holds the seat" if s["handover"] else "races the selector on its own paper book")
                     + f": it buys at a predicted {pct(fly.get('line'), 1)} and learns every minute from each scored token's realized return, two hours later.")
    else:
        lines.append(f"**{s['trading_why']}** Start the trading engine on the Processes page to trade.")
    if s["training"]:
        tr = s["train_status"]; step, total = tr.get("step"), tr.get("total")
        lines.append(f"Training is running: {tr.get('stage', '')}" + (f" ({step}/{total})" if step is not None and total else "") + ".")
    return "\n\n".join(lines)


@st.fragment(run_every="5s")
def summary() -> None:
    s = system_state()
    with st.container(border=True):
        st.markdown(narrative(s))
    w = q1("SELECT wealth, exposure, n_open, drawdown FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (BOOK,)) or {}
    c = q1("SELECT count(*) AS n, sum(realized_sol) AS pnl, count(*) FILTER (WHERE realized_sol > 0) AS wins FROM positions WHERE book = %s AND status = 'closed'", (BOOK,)) or {}
    spark = [float(r["wealth"]) for r in q("SELECT wealth FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 180", (BOOK,))][::-1]
    n = int(c.get("n") or 0)
    with st.container(horizontal=True):
        st.metric("Paper wealth", sol(w.get("wealth")), delta=(f"{float(w['wealth']) - config.CAPITAL_SOL:+.4f} SOL since start" if w.get("wealth") is not None else None),
                  border=True, chart_data=spark or None, help=f"Started at {config.CAPITAL_SOL:g} SOL; open positions valued net of exit costs.")
        st.metric("Realized P&L", sol(c.get("pnl"), signed=True), border=True, help="Closed trades, after fees and price impact.")
        st.metric("Closed trades", n, delta=(f"{int(c.get('wins') or 0) / n:.0%} winners" if n else None), delta_color="off", border=True)
        st.metric("Open positions", int(w.get("n_open") or 0), delta=(f"{float(w.get('exposure') or 0):.2f} SOL at cost" if w else None), delta_color="off", border=True)
        st.metric("Drawdown", pct(w.get("drawdown"), signed=False), border=True, help="Below the book's peak wealth.")
    sel = s["selector"]
    with st.container(border=True):
        st.markdown("**This minute**")
        if not sel.get("minute"):
            st.caption("No minute scored yet." if s["runner"] else "The trading engine is stopped.")
        else:
            thr = float(sel.get("threshold") or 0)
            with st.container(horizontal=True):
                st.metric("Minute (UTC)", str(sel["minute"])[11:16])
                st.metric("Tokens traded", int(sel.get("mints_traded") or 0))
                st.metric("Eligible", int(sel.get("eligible") or 0), help="Pool ≥ 20 SOL and ≥ 5 SOL traded in the last 15 minutes.")
                st.metric("Best score", f"{float(sel.get('score_max') or 0):.3f}", delta=f"needs {thr:.3f}", delta_color="off")
                st.metric("Picks", int(sel.get("picks") or 0))
                st.metric("Bought / sold", f"{int(sel.get('entered') or 0)} / {int(sel.get('exited') or 0)}")
            top = sel.get("top_scores") or []
            if top:
                names = labels([t[0] for t in top])
                st.dataframe(pd.DataFrame({"token": [names.get(t[0], t[0][:6]) for t in top], "score": [float(t[1]) for t in top]}), hide_index=True,
                             column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f")})
            lat = sel.get("latency_s")
            st.caption(f"Scored {ago(s['selector_at'])}" + (f" · {float(lat):.0f} s after the minute closed" if lat is not None else "") +
                       (f" · {s['trading_why']}" if s["trading"] != "trading" else ""))


@st.fragment(run_every="15s")
def book() -> None:
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Paper wealth**")
            hist = q("SELECT ts, wealth FROM wealth_marks WHERE book = %s ORDER BY ts", (BOOK,))
            if hist:
                st.line_chart(pd.DataFrame(hist), x="ts", y="wealth", x_label="", y_label="SOL", height=240)
            else:
                st.caption("No marks yet.")
    with right:
        with st.container(border=True):
            st.markdown("**Open positions**")
            rows = q("SELECT mint, opened_at, cost_sol, entry_price, last_mark_price FROM positions WHERE book = %s AND status = 'open' ORDER BY opened_at", (BOOK,))
            if not rows:
                st.caption("None open.")
            else:
                hold = int((system_state().get("model") or {}).get("run", {}).get("horizon_min") or 30)
                now = datetime.now(timezone.utc); names = labels([r["mint"] for r in rows])
                df = pd.DataFrame([{"token": names.get(r["mint"], r["mint"][:6]), "held (min)": (now - r["opened_at"]).total_seconds() / 60,
                                    "exits in (min)": max(0.0, hold - (now - r["opened_at"]).total_seconds() / 60), "cost (SOL)": float(r["cost_sol"]),
                                    "return": (float(r["last_mark_price"]) / float(r["entry_price"]) - 1) * 100 if r["entry_price"] and r["last_mark_price"] else None}
                                   for r in rows])
                st.dataframe(df, hide_index=True, column_config={"held (min)": st.column_config.NumberColumn(format="%.0f"),
                                                                 "exits in (min)": st.column_config.NumberColumn(format="%.0f"),
                                                                 "cost (SOL)": st.column_config.NumberColumn(format="%.3f"),
                                                                 "return": st.column_config.NumberColumn("return (gross)", format="%+.2f%%")})


@st.fragment(run_every="30s")
def race() -> None:
    s = system_state(); ho = s["handover"]; fly = s["fly"]
    marks = q("SELECT ts, book, wealth FROM wealth_marks WHERE book IN ('paper_selector', 'paper_fly', 'live') AND ts > now() - interval '30 days' ORDER BY ts")
    if not marks and not (fly.get("stage") and fly.get("stage") != "not trading"):
        return
    since = datetime.now(timezone.utc) - timedelta(days=14)
    pnl = {r["book"]: r for r in q("SELECT book, COALESCE(sum(realized_sol), 0) AS pnl, count(*) AS n FROM positions WHERE book IN ('paper_selector', 'paper_fly', 'live') "
                                   "AND status = 'closed' AND closed_at >= %s GROUP BY book", (since,))}
    start = (q1("SELECT min(ts) AS t FROM wealth_marks WHERE book = 'paper_fly'") or {}).get("t")
    with st.container(border=True):
        st.markdown(":material/sports_score: **The race: selector vs plastic fly**")
        if ho:
            st.success(f"The fly took the selector's seat {ago(ho.get('at'))}: {float(ho.get('fly_pnl_sol') or 0):+.3f} SOL vs {float(ho.get('selector_pnl_sol') or 0):+.3f} SOL "
                       f"over {ho.get('days')} days. " + ("It trades the bot wallet; its paper book is the mirror." if s["live_money"]
                                                          else "Live trading waits for LIVE_ENABLED=1, the live prerequisites and a funded wallet."), icon=":material/swap_horiz:")
        else:
            days = (datetime.now(timezone.utc) - start).total_seconds() / 86400 if start else 0.0
            st.caption("The fly takes the seat when its realized P&L over the last 14 days is at least the selector's, with at least 30 trades"
                       + (f" · racing for {days:.1f} days." if start else " · the race starts when the fly first trades."))
        with st.container(horizontal=True):
            for book, title in (("paper_selector", "Selector · 14 days"), ("paper_fly", "Fly · 14 days")) + ((("live", "Live · 14 days"),) if ho else ()):
                r = pnl.get(book) or {}
                st.metric(title, sol(r.get("pnl"), signed=True), delta=f"{int(r.get('n') or 0)} trades", delta_color="off", border=True, help="Realized P&L of closed trades.")
        if marks:
            df = pd.DataFrame(marks).pivot_table(index="ts", columns="book", values="wealth")
            st.line_chart(df, x_label="", y_label="SOL", height=240)


summary()
race()
book()
