"""Kalshi overview: what the visual fly is doing in plain words, both arms' books, this minute's scoring, the shared
brain with both flies lit, and the open positions by category."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.ui import brain3d
from fly_trader.ui.brain3d import feed
from fly_trader.ui.common import KALSHI_BOOKS, ago, kalshi_state, pct, system_state, usd


def narrative(k: dict) -> str:
    fly = k["fly"]; lines = []
    if fly.get("stage") and fly.get("stage") != "not trading":
        arms = fly.get("arms") or {}; ln = fly.get("lines") or {}
        parts = ", ".join(f"**{s}** ({arms.get(s, '?')}, edge ≥ {float(ln.get(s, 0)):.3f})" for s in ln)
        money = ("Both arms also trade Kalshi subaccount " + f"{config.KALSHI_SUBACCOUNT} with real dollars (cap ${config.KALSHI_CAPITAL_USD:g})." if k["live_money"]
                 else "No real money is at risk" + (": live mirroring is off (KALSHI_LIVE_ENABLED=0)." if not config.KALSHI_LIVE_ENABLED else f" yet: {', '.join(k['prerequisites_missing'])}."))
        lines.append(f"**{k['trading_why']}** The visual fly (bootstrap **#{fly.get('bootstrap')}**) scores both sides of every quoted Kalshi market inside "
                     f"{config.KALSHI_MAX_DAYS_TO_CLOSE} days of its close, every minute: it predicts the probability the side pays and trades the **edge** — that probability "
                     f"minus the fee-inclusive price. Strategies: {parts}. The **taker arm** buys at the ask (an IOC at the book walk's worst level); the **maker arm** rests a "
                     f"bid one tick inside the ask and cancels or replaces it as the quote moves. Every position runs to settlement, which is also what teaches the fly. {money}")
    else:
        lines.append(f"**{k['trading_why']}**" + ("" if k["runner"] else " Start the Kalshi trading engine on the Processes page."))
    if k["training"]:
        tr = k["train"]; pr = k["train_progress"]
        lines.append(f"Kalshi training is running: {tr.get('stage', '')}" + (f" · {pr.get('stage', '')} ({pr.get('step')}/{pr.get('total')})" if pr.get("total") else "") + ".")
    return "\n\n".join(lines)


@st.fragment(run_every="5s")
def brain() -> None:
    """The whole brain: the memecoin fly's central-brain neurons and the Kalshi fly's optic lobes, each lit by its own minute."""
    s = system_state(); k = kalshi_state(); mfly, kfly = s["fly"], k["fly"]
    with st.container(border=True):
        st.markdown(":material/neurology: **One brain, two flies** · the optic lobes belong to the Kalshi fly (its inputs enter the photoreceptors), the antennal lobe "
                    "and the other senses to the memecoin fly; the central brain is shared and shows the fly chosen below")
        try:
            meta = brain3d.geometry_full()
        except FileNotFoundError as e:
            st.caption(f"No brain to draw: {e}"); return
        st.session_state.setdefault("kbrain_live", True)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.toggle("Follow live", key="kbrain_live", help="Show each fly's newest minute as it arrives; moving the slider turns this off.")
            owner = st.segmented_control("Central brain shows", ["kalshi", "memecoin"], key="kbrain_owner", default="kalshi") or "kalshi"
            mode = st.segmented_control("Learned pathways", list(feed.MODES), key="kbrain_pmode", default="since bootstrap") or "since bootstrap"
        flies = []
        for name, fly, d, sdir in (("memecoin", mfly, feed.activity_dir(), feed.fly_state_dir()), ("kalshi", kfly, feed.kalshi_activity_dir(), feed.kalshi_fly_state_dir())):
            running = bool(fly.get("stage")) and fly.get("stage") != "not trading"
            minutes = feed.minutes(feed.activity_mtime(d), str(d)); act = None
            if minutes:
                key = f"kbrain_minute_{name}"
                if st.session_state["kbrain_live"] or st.session_state.get(key) not in minutes:
                    st.session_state[key] = minutes[-1]
                stamp = st.select_slider(f"{name} minute (UTC)", options=minutes, key=key, format_func=feed.label, on_change=lambda: st.session_state.__setitem__("kbrain_live", False))
                act = feed.activity_at(stamp, str(d))
            pth = feed.pathways(mode, fly, meta, fly_name=name) if running else {"mode": feed.MODES[mode], "key": "none", "reference": None, "items": [], "note": None, "changed": 0}
            flies.append(brain3d.fly_payload(name, meta, act, pth, list((fly.get("channels") or {}).keys())))
        msg = None if any(f["activity_b64"] for f in flies) else "Neither fly has scored a minute yet: this is the resting brain."
        brain3d.brain_view(brain3d.payload_flies(meta, flies, owner, message=msg), key="kbrain3d", height=600)
        st.caption(" · ".join(f"{f['name']}: " + (f"{f['minute'][11:16]} UTC, {f['candidates']} candidates of {f['rows']} rows" if f["minute"] else "no minute") for f in flies))


@st.fragment(run_every="5s")
def summary() -> None:
    k = kalshi_state()
    with st.container(border=True):
        st.markdown(narrative(k))
    cols = st.columns(2)
    for col, (arm, book) in zip(cols, (("taker", "paper_kalshi_taker"), ("maker", "paper_kalshi_maker"))):
        w = k["books"].get(book) or {}
        c = q1("SELECT count(*) AS n, sum(realized_cents) AS pnl, count(*) FILTER (WHERE realized_cents > 0) AS wins FROM kalshi_positions WHERE book = %s AND status <> 'open'", (book,)) or {}
        spark = [float(r["wealth"]) for r in q("SELECT wealth FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 180", (book,))][::-1]
        n = int(c.get("n") or 0)
        with col:
            with st.container(border=True):
                st.markdown(f"**{KALSHI_BOOKS[book]}**")
                with st.container(horizontal=True):
                    st.metric("Wealth", usd(w.get("wealth")), delta=(f"{float(w['wealth']) - config.KALSHI_CAPITAL_USD:+.2f} $ since start" if w.get("wealth") is not None else None),
                              chart_data=spark or None, help=f"Started at ${config.KALSHI_CAPITAL_USD:g}; open positions at the bid net of the exit fee, resting bids at their collateral.")
                    st.metric("Settled P&L", usd((c.get("pnl") or 0) / 100.0, signed=True), help="Settled and closed positions, after fees.")
                    st.metric("Settled", n, delta=(f"{int(c.get('wins') or 0) / n:.0%} winners" if n else None), delta_color="off")
                    st.metric("Open", int(w.get("n_open") or 0), delta=(f"${float(w.get('exposure') or 0):.2f} at cost" if w else None), delta_color="off")
                    st.metric("Drawdown", pct(w.get("drawdown"), signed=False))
    if k["live_money"] or config.KALSHI_LIVE_ENABLED:
        lv = k["fly"].get("live") or {}
        with st.container(border=True):
            st.markdown("**Live mirror** · " + (lv.get("stage") or "not running"))
            if lv.get("wealth") is not None:
                with st.container(horizontal=True):
                    st.metric("Live wealth", usd(lv.get("wealth"))); st.metric("Cash", usd(lv.get("cash")))
                    st.metric("Taker", usd(lv.get("wealth_taker"))); st.metric("Maker", usd(lv.get("wealth_maker")))
                    g = lv.get("gap") or {}
                    st.metric("Gap · taker", pct((g.get("taker") or {}).get("live_mean")), delta=f"mirror {pct((g.get('taker') or {}).get('mirror_mean'))}", delta_color="off")
                    st.metric("Gap · maker", pct((g.get("maker") or {}).get("live_mean")), delta=f"mirror {pct((g.get('maker') or {}).get('mirror_mean'))}", delta_color="off")
            if lv.get("error"):
                st.error(lv["error"])
    fly = k["fly"]
    with st.container(border=True):
        st.markdown("**This minute**")
        if not fly.get("minute"):
            st.caption("No minute scored yet." if k["runner"] else "The Kalshi engine is stopped.")
        else:
            b = fly.get("books") or {}
            with st.container(horizontal=True):
                st.metric("Minute (UTC)", str(fly["minute"])[11:16])
                st.metric("Markets quoted", int(fly.get("markets_active") or 0), help="Markets with a message this minute inside the entry window.")
                st.metric("Eligible rows", int(fly.get("eligible") or 0), help="Both sides of each market passing the spread, depth, volume and time gates.")
                st.metric("Picks", int(fly.get("picks") or 0))
                st.metric("Taker bought", int((b.get("taker") or {}).get("entered") or 0))
                st.metric("Maker posted / resting", f"{int((b.get('maker') or {}).get('posted') or 0)} / {int((b.get('maker') or {}).get('resting') or 0)}")
                st.metric("Settled", int((b.get("taker") or {}).get("settled") or 0) + int((b.get("maker") or {}).get("settled") or 0))
            st.caption(f"Scored {ago(k['fly_at'])} · {fly.get('stage')}" + (f" · {k['trading_why']}" if k["trading"] != "trading" else ""))


@st.fragment(run_every="15s")
def positions() -> None:
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Wealth by book**")
            marks = q("SELECT ts, book, wealth FROM wealth_marks WHERE book = ANY(%s) AND ts > now() - interval '30 days' ORDER BY ts", (list(KALSHI_BOOKS),))
            if marks:
                st.line_chart(pd.DataFrame(marks).pivot_table(index="ts", columns="book", values="wealth"), x_label="", y_label="USD", height=240)
            else:
                st.caption("No marks yet.")
    with right:
        with st.container(border=True):
            st.markdown("**Open positions**")
            rows = q("""SELECT p.book, p.ticker, p.side, p.contracts, p.cost_cents, p.fee_cents, p.avg_price_cents, p.last_mark_cents, p.strategy, p.opened_at, m.close_time, m.title,
                               COALESCE(s.category, e.category) AS category
                        FROM kalshi_positions p LEFT JOIN kalshi_markets m USING (ticker) LEFT JOIN kalshi_events e USING (event_ticker) LEFT JOIN kalshi_series s ON s.ticker = e.series_ticker
                        WHERE p.book = ANY(%s) AND p.status = 'open' ORDER BY p.opened_at DESC""", (list(KALSHI_BOOKS),))
            if not rows:
                st.caption("None open.")
            else:
                now = datetime.now(timezone.utc)
                df = pd.DataFrame([{"book": KALSHI_BOOKS.get(r["book"], r["book"]), "market": (r["title"] or r["ticker"])[:60], "side": r["side"], "strategy": r["strategy"] or "—",
                                    "contracts": float(r["contracts"]), "paid (c)": float(r["avg_price_cents"] or 0), "now (c)": float(r["last_mark_cents"]) if r["last_mark_cents"] is not None else None,
                                    "closes in (h)": (r["close_time"] - now).total_seconds() / 3600 if r["close_time"] else None, "category": r["category"] or "—"} for r in rows])
                st.dataframe(df, hide_index=True, column_config={"paid (c)": st.column_config.NumberColumn(format="%.1f"), "now (c)": st.column_config.NumberColumn(format="%.0f"),
                                                                 "closes in (h)": st.column_config.NumberColumn(format="%.1f")})
                cat = df.groupby("category")["contracts"].count().sort_values(ascending=False)
                st.caption("By category: " + ", ".join(f"{c} {n}" for c, n in cat.items()))


brain()
summary()
positions()
