"""Shared reads and renderers for the operator console: one place that answers what the system is doing — trading or
not, training or not, paper or live money, which model is loaded — so every page says it the same way."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import streamlit as st

from fly_trader import config
from fly_trader.db.queries import q, q1
from fly_trader.ops.supervisor import WORKERS, get_supervisor
from fly_trader.train.selector import DATA_VERSION, is_current, is_deployable

BOOK = "paper_selector"
STALE_FEED_S = 180            # agent/minute_engine.py: no entries or exits when the newest complete minute is older

WORKER_INFO = {   # name → (title, icon, what it does, role)
    "runner": ("Trading engine", ":material/candlestick_chart:", "Scores every traded token once a minute with the selector and the plastic fly, trades their books and lets the fly learn.", "trade"),
    "pumpstream": ("Market feed", ":material/sensors:", "Streams every PumpSwap trade from pumpapi.io into 1-minute candles. The trading engine reads these.", "trade"),
    "train": ("Trainer", ":material/model_training:", "On demand: builds decision points from the archive, backtests day by day and saves a new model.", "train"),
    "replay": ("History archive", ":material/history:", "Downloads the hourly trade archive and builds the training feature set.", "train"),
    "discover": ("Token stats", ":material/query_stats:", "Records Jupiter stats (holders, organic score, top-holder share) every 10 minutes for every token the selector can trade: history for future model inputs.", "optional"),
    "kalshi_runner": ("Kalshi trading engine", ":material/candlestick_chart:", "Scores every quoted Kalshi market once a minute with the visual fly, trades its taker and maker paper books (and the subaccount when live) and lets it learn from settlements.", "trade"),
    "kalshi_stream": ("Kalshi market feed", ":material/sensors:", "Streams every Kalshi quote, trade and market lifecycle event into 1-minute rows. The Kalshi engine reads these.", "trade"),
    "kalshi_train": ("Kalshi trainer", ":material/model_training:", "Fits the Kalshi strategy stack walk-forward by settlement day and bootstraps the visual fly when none exists.", "train"),
    "kalshi_history": ("Kalshi history", ":material/history:", "Seeds settled markets and trades from the open dataset, fills 1-minute candles from the API and builds the feature rows.", "train"),
}
WORKER_TYPE = {"runner": "memecoins", "pumpstream": "memecoins", "train": "memecoins", "replay": "memecoins", "discover": "memecoins",
               "kalshi_runner": "prediction markets", "kalshi_stream": "prediction markets", "kalshi_train": "prediction markets", "kalshi_history": "prediction markets"}
ROLE_LABEL = {"trade": "needed to trade", "train": "needed to train", "optional": "optional"}
ORDER = [n for n in ("runner", "pumpstream", "train", "replay", "discover", "kalshi_runner", "kalshi_stream", "kalshi_train", "kalshi_history") if n in WORKERS]
KALSHI_BOOKS = {"paper_kalshi_taker": "Taker · paper", "paper_kalshi_maker": "Maker · paper", "live_kalshi_taker": "Taker · live", "live_kalshi_maker": "Maker · live"}


# ---------------------------------------------------------------- formatting
def jv(v) -> dict:
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v or "{}")
    except (TypeError, ValueError):
        return {}


def setting(key: str) -> tuple[dict, datetime | None]:
    r = q1("SELECT value, updated_at FROM ui_settings WHERE key = %s", (key,))
    return (jv(r["value"]), r["updated_at"]) if r else ({}, None)


def ago(ts) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    s = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
    return f"{s:.0f} s ago" if s < 120 else f"{s / 60:.0f} min ago" if s < 7200 else f"{s / 3600:.1f} h ago" if s < 172800 else f"{s / 86400:.0f} d ago"


def sol(x, nd: int = 4, signed: bool = False) -> str:
    return "—" if x is None else f"{float(x):{'+' if signed else ''},.{nd}f} SOL"


def pct(x, nd: int = 2, signed: bool = True) -> str:
    return "—" if x is None else f"{float(x) * 100:{'+' if signed else ''}.{nd}f}%"


def usd(x, nd: int = 2, signed: bool = False) -> str:
    return "—" if x is None else f"{'+' if signed and float(x) >= 0 else ''}{'-' if float(x) < 0 else ''}${abs(float(x)):,.{nd}f}"


def cents(x, nd: int = 0) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}c"


# ---------------------------------------------------------------- models
_NOTE_OBJ = re.compile(r'"(walk_forward|random_baseline|fly|gbm|random)": (\{[^{}]*\})')
_NOTE_SCALAR = re.compile(r'"(threshold|trained_through|horizon_min|line|costs|rows|days|fly_beats_gbm)": ("[^"]*"|[-0-9.eE]+|true|false)')


def parse_note(note) -> dict:
    """A snapshot's note (JSON; older notes were cut at 900 characters, so fall back to the fields that survive)."""
    try:
        return json.loads(note or "{}")
    except (TypeError, ValueError):
        out = {k: json.loads(v) for k, v in _NOTE_SCALAR.findall(note or "")}
        for k, v in _NOTE_OBJ.findall(note or ""):
            try:
                out[k] = json.loads(v)
            except ValueError:
                pass
        return out


def _snap(r) -> dict | None:
    return {**r, "meta": parse_note(r["note"])} if r else None


def snapshot(sid: int) -> dict | None:
    return _snap(q1("SELECT id, ts, kind, note FROM brain_snapshots WHERE id = %s", (sid,)))


def latest_snapshot(kind: str = "selector", deployable: bool = False) -> dict | None:
    """The newest snapshot of this kind trained on the current data definitions (older ones are never loaded);
    with ``deployable`` only one whose backtest qualified it to trade."""
    for r in q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC LIMIT 50", (kind,)):
        s = _snap(r)
        if (is_deployable if deployable else is_current)(s["meta"]):
            return s
    return None


def loaded_model() -> dict | None:
    """The snapshot the running selector session loaded (its runs row), with its backtest."""
    r = q1("SELECT brain_snapshot_id, config, started_at FROM runs WHERE kind = 'selector' AND status = 'running' ORDER BY started_at DESC LIMIT 1")
    if not r or not r["brain_snapshot_id"]:
        return None
    s = snapshot(int(r["brain_snapshot_id"]))
    return {**s, "run": jv(r["config"]), "since": r["started_at"]} if s else None


def outdated_line(meta: dict) -> str:
    """Which definitions changed since this model was trained (its backtest is not comparable to a current model's)."""
    d = (meta or {}).get("data") or {}
    diff = [f"{k} {d.get(k)} → {v}" for k, v in DATA_VERSION.items() if d.get(k) != v and (k != "cols" or "cols" in d)]
    return ("Trained on earlier definitions (" + "; ".join(diff) + "), so its backtest is not comparable to a current model's and it is loaded only when pinned."
            if diff else "Trained before the definitions were recorded, so its backtest is not comparable and it is loaded only when pinned.")


def backtest_line(meta: dict) -> str:
    if not is_current(meta):
        return outdated_line(meta)
    wf, rb = meta.get("walk_forward") or {}, meta.get("random_baseline") or {}
    if not wf:
        return "No backtest recorded."
    s = (f"Backtest on {wf.get('days')} held-out days: {pct(wf.get('mean'))} average per trade over {wf.get('n')} trades, "
         f"profitable on {wf.get('days_positive')} of those {wf.get('days')} days")
    s += (f"; random picks made {pct(rb['mean'])} per trade under the same costs." if rb.get("mean") is not None else ".")
    if meta.get("deploy_reason"):
        s += (" Put to work: " if meta.get("deployable") else " Not put to work: ") + meta["deploy_reason"] + "."
    return s


def labels(mints) -> dict[str, str]:
    """Readable token names: symbol from corpus_meta or the watch list, else the short mint."""
    mints = [m for m in set(mints) if m]
    if not mints:
        return {}
    rows = q("""SELECT m.mint, COALESCE(NULLIF(cm.symbol, ''), NULLIF(t.symbol, '')) AS sym FROM unnest(%s::text[]) AS m(mint)
                LEFT JOIN corpus_meta cm ON cm.mint = m.mint LEFT JOIN tokens t ON t.mint = m.mint""", (mints,))
    return {r["mint"]: (f"{r['sym']} · {r['mint'][:4]}…" if r["sym"] else f"{r['mint'][:6]}…") for r in rows}


# ---------------------------------------------------------------- system state
def system_state() -> dict:
    sup = get_supervisor(); ws = sup.status(); now = datetime.now(timezone.utc)
    sel, sel_at = setting("selector_status"); tr, tr_at = setting("training_status"); ps, _ = setting("pumpstream_status")
    fly, fly_at = setting("fly_status"); ho, _ = setting("handover"); rp, _ = setting("fly_replay")
    circuit = q1("SELECT kill_switch, kill_reason, tripped, fail_count, entries_paused FROM circuit_state WHERE id = 1") or {}
    ft = ps.get("flushed_through")
    feed_age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None     # since the newest complete minute closed
    feed_ok = ws["pumpstream"]["alive"] and feed_age is not None and feed_age < STALE_FEED_S
    runner = ws["runner"]["alive"]
    stage = str(tr.get("stage") or "")
    training = ws["train"]["alive"] or (tr_at is not None and (now - tr_at).total_seconds() < 180 and not stage.endswith(("saved", "done")))
    if not runner:
        err = (ws["runner"].get("error") or "").splitlines()
        trading, why = "stopped", ("The trading engine is stopped: " + err[0].split(": ", 1)[-1]) if err else "The trading engine is stopped."
    elif str(sel.get("stage", "")).startswith("waiting for a model") and fly.get("stage") != "trading":
        trading, why = "waiting", "No model has qualified to trade yet: a model trades only if its backtest made money after costs and beat random picks. It starts as soon as one does."
    elif circuit.get("kill_switch"):
        trading, why = "blocked", "Kill switch tripped" + (f" ({circuit['kill_reason']})" if circuit.get("kill_reason") else "") + ": no new entries."
    elif circuit.get("entries_paused"):
        trading, why = "paused", "New entries are paused by the operator; open positions still exit on time."
    elif str(sel.get("stage", "")).startswith("stream stale") or not feed_ok:
        trading, why = "holding", "The market feed is stale: no entries or exits until it recovers."
    elif not sel and not fly:
        trading, why = "starting", "Warming up on the last 24 hours of market minutes."
    else:
        trading, why = "trading", "Trading every minute."
    from fly_trader.chain.cluster_guard import signing_allowed
    live_money = bool(ho) and runner and fly.get("stage") == "trading" and signing_allowed()      # the fly holds the seat and may sign (agent/fly_live.py)
    return {"workers": ws, "runner": runner, "trading": trading, "trading_why": why, "live_money": live_money,
            "fly": fly, "fly_at": fly_at, "handover": ho or None, "replay": rp or None,
            "model": loaded_model() if runner else None, "latest": latest_snapshot("selector"), "deployable": latest_snapshot("selector", deployable=True),
            "training": training, "train_status": tr, "train_at": tr_at,
            "train_external": training and not ws["train"]["alive"], "selector": sel, "selector_at": sel_at, "feed": ps, "feed_ok": feed_ok, "feed_age": feed_age,
            "circuit": circuit}


# ---------------------------------------------------------------- prediction markets (Kalshi)
def kalshi_state() -> dict:
    """What the Kalshi side is doing: feed, engine, the visual fly, training, history, rails (circuit 2) and books."""
    from fly_trader.kalshi import fly as KF, selector as kselector
    sup = get_supervisor(); ws = sup.status(); now = datetime.now(timezone.utc)
    fly, fly_at = setting("kalshi_fly_status"); st_, _ = setting("kalshi_stream_status"); tr, tr_at = setting("kalshi_training_pipeline")
    rp, _ = setting("kalshi_fly_replay"); hs, hs_at = setting("kalshi_history_status"); ts_, ts_at = setting("training_status")
    circuit = q1("SELECT kill_switch, kill_reason, tripped, fail_count, entries_paused FROM circuit_state WHERE id = 2") or {}
    ft = st_.get("flushed_through")
    feed_age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None
    feed_ok = ws["kalshi_stream"]["alive"] and feed_age is not None and feed_age < STALE_FEED_S
    runner = ws["kalshi_runner"]["alive"]
    stage = str(fly.get("stage") or "")
    if not runner:
        err = (ws["kalshi_runner"].get("error") or "").splitlines()
        trading, why = "stopped", ("The Kalshi engine is stopped: " + err[0].split(": ", 1)[-1]) if err else "The Kalshi engine is stopped."
    elif stage in ("", "not trading"):
        trading, why = "waiting", fly.get("detail") or "The visual fly has not qualified yet: it trades once its replay passed and a deployable bootstrap exists."
    elif circuit.get("kill_switch"):
        trading, why = "blocked", "Kalshi kill switch tripped" + (f" ({circuit['kill_reason']})" if circuit.get("kill_reason") else "") + ": no new entries."
    elif circuit.get("entries_paused"):
        trading, why = "paused", "New Kalshi entries are paused by the operator; open positions still run to settlement."
    elif not feed_ok:
        trading, why = "holding", "The Kalshi feed is stale: no entries until it recovers (the fly keeps learning)."
    elif stage.startswith("catching"):
        trading, why = "starting", "Catching up on the minutes the feed wrote while the engine was away."
    else:
        trading, why = "trading", "Trading both arms every minute: IOC entries and resting bids."
    live = bool(config.KALSHI_LIVE_ENABLED) and not config.kalshi_live_prerequisites_missing() and runner and stage == "trading"
    books = {b: q1("SELECT wealth, exposure, n_open, drawdown, ts FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (b,)) for b in KALSHI_BOOKS}
    with __import__("fly_trader.db.connection", fromlist=["transaction"]).transaction() as conn:
        sel = kselector.latest_current(conn); boot = KF.latest_current(conn); dep = KF.latest_deployable(conn)
    training = ws["kalshi_train"]["alive"] and str(tr.get("stage") or "").startswith(("training", "bootstrapping"))
    return {"workers": ws, "runner": runner, "trading": trading, "trading_why": why, "live_money": live, "fly": fly, "fly_at": fly_at, "replay": rp or None,
            "feed": st_, "feed_ok": feed_ok, "feed_age": feed_age, "train": tr, "train_at": tr_at, "training": training, "train_progress": ts_ if training else {},
            "history": hs, "history_at": hs_at, "circuit": circuit, "books": books, "selector": sel, "bootstrap": boot, "deployable": dep,
            "prerequisites_missing": config.kalshi_live_prerequisites_missing()}


KALSHI_BADGE = {"trading": ("Kalshi trading", "green"), "holding": ("Kalshi holding: feed stale", "orange"), "paused": ("Kalshi entries paused", "orange"),
                "blocked": ("Kalshi kill switch", "red"), "starting": ("Kalshi starting", "blue"), "stopped": ("Kalshi not trading", "gray"),
                "waiting": ("Kalshi waiting for the fly", "gray")}


def kalshi_strip() -> None:
    """The prediction-markets row of the status strip."""
    k = kalshi_state(); fly = k["fly"]
    with st.container(horizontal=True, gap="small"):
        if k["live_money"]:
            st.badge("Kalshi live money", icon=":material/payments:", color="red", help=f"Both arms mirror onto Kalshi subaccount {config.KALSHI_SUBACCOUNT} (cap ${config.KALSHI_CAPITAL_USD:g}).")
        else:
            st.badge("Kalshi paper", icon=":material/receipt:", color="blue", help="Prediction-market trades are simulated at the quoted book with real fees; "
                     + ("live mirroring is off (KALSHI_LIVE_ENABLED=0)." if not config.KALSHI_LIVE_ENABLED else "live gated: " + ", ".join(k["prerequisites_missing"])))
        text, color = KALSHI_BADGE[k["trading"]]
        st.badge(text, icon=":material/candlestick_chart:", color=color, help=k["trading_why"])
        if fly.get("stage") and fly.get("stage") != "not trading":
            st.badge(f"Visual fly learning · drift {float(fly.get('drift') or 0):.1%}" if not fly.get("learning_frozen") else "Visual fly: learning frozen", icon=":material/visibility:",
                     color="violet" if not fly.get("learning_frozen") else "orange",
                     help=f"The Kalshi fly on {fly.get('device') or '?'}: {fly.get('pending', 0)} settlements pending; edge lines " + ", ".join(f"{a} {float(v):.3f}" for a, v in (fly.get("lines") or {}).items()))
        else:
            st.badge("Visual fly not trading", icon=":material/visibility:", color="gray", help=fly.get("detail") or k["trading_why"])
        tr = k["train"]
        st.badge(f"Kalshi training · {tr.get('stage', '')}" if k["training"] else "Kalshi not training", icon=":material/model_training:", color="blue" if k["training"] else "gray")
        hs = k["history"]
        if k["workers"]["kalshi_history"]["alive"]:
            st.badge(f"Kalshi history · {hs.get('stage', '')}", icon=":material/history:", color="blue", help=f"{hs.get('done', '?')} of {hs.get('total', '?')} markets filled")
        st.badge("Kalshi feed live" if k["feed_ok"] else "Kalshi feed stale", icon=":material/sensors:", color="green" if k["feed_ok"] else "red",
                 help=f"newest complete minute closed {k['feed_age']:.0f} s ago" if k["feed_age"] is not None else "no minute written yet")


TRADING_BADGE = {"trading": ("Trading", "green"), "holding": ("Holding: feed stale", "orange"), "paused": ("Entries paused", "orange"),
                 "blocked": ("Kill switch tripped", "red"), "starting": ("Starting", "blue"), "stopped": ("Not trading", "gray"),
                 "waiting": ("Waiting for a qualified model", "gray")}


@st.fragment(run_every="5s")
def status_strip() -> None:
    memecoin_strip()
    try:
        kalshi_strip()
    except Exception as e:                     # the Kalshi side must never take the memecoin console down
        st.caption(f"Kalshi status unavailable: {type(e).__name__}: {e}")


def memecoin_strip() -> None:
    s = system_state()
    with st.container(horizontal=True, gap="small"):
        if s["live_money"]:
            st.badge("Live money", icon=":material/payments:", color="red", help="Trades spend real SOL from the bot wallet.")
        else:
            st.badge("Paper — no real SOL", icon=":material/receipt:", color="blue", help="Trades are simulated at market prices with real fees; no wallet funds are used.")
        text, color = TRADING_BADGE[s["trading"]]
        st.badge(text, icon=":material/candlestick_chart:", color=color, help=s["trading_why"])
        m, latest, dep = s["model"], s["latest"], s["deployable"]
        st.badge(f"Selector #{m['id']}" if m else "No selector at work", icon=":material/psychology:", color="violet" if m else "gray",
                 help=backtest_line(m["meta"]) if m else ("No model has been trained on the current data yet." if not latest
                                                         else "No model has earned a place yet: " + (latest["meta"].get("deploy_reason") or "none qualified") if not dep
                                                         else "The trading engine is stopped, so no model is in use."))
        if latest and not is_deployable(latest["meta"]) and (not m or latest["id"] > m["id"]):
            st.badge(f"Selector #{latest['id']} not put to work", icon=":material/block:", color="orange", help=backtest_line(latest["meta"]))
        if m and dep and dep["id"] > m["id"]:
            st.badge(f"Switching to selector #{dep['id']}", icon=":material/upgrade:", color="blue", help="The trading engine picks it up within 10 minutes.")
        fly = s["fly"]
        if fly.get("bootstrap"):          # the fly's own model, beside the selector's: they are separate snapshots
            st.badge(f"Fly #{fly['bootstrap']}", icon=":material/neurology:", color="violet",
                     help="The fly bootstrap trading the fly book (taught once by the selector, then learning on its own).")
        if fly.get("stage") and fly.get("stage") != "not trading":
            st.badge(f"Fly learning · drift {float(fly.get('drift') or 0):.1%}" if not fly.get("learning_frozen") else "Fly: learning frozen", icon=":material/neurology:",
                     color="violet" if not fly.get("learning_frozen") else "orange",
                     help=f"Plastic fly on {fly.get('device') or '?'}: buy line {pct(fly.get('line'))}, {fly.get('pending', 0)} labels pending.")
        else:
            st.badge("Fly not trading", icon=":material/neurology:", color="gray", help=fly.get("detail") or "The trading engine is stopped.")
        if s["handover"]:
            st.badge("Fly holds the seat", icon=":material/swap_horiz:", color="green", help="The fly won the race; the selector no longer trades while the fly can.")
        tr = s["train_status"]
        st.badge(f"Training · {tr.get('stage', '')}" if s["training"] else "Not training", icon=":material/model_training:", color="blue" if s["training"] else "gray")
        st.badge("Market feed live" if s["feed_ok"] else "Market feed stale", icon=":material/sensors:", color="green" if s["feed_ok"] else "red",
                 help=f"newest complete minute closed {s['feed_age']:.0f} s ago" if s["feed_age"] is not None else "no minute written yet")


def restart_runner() -> None:
    """Stop the trading engine and start it again (it loads the newest model; RESET_ON_START clears the paper book)."""
    sup = get_supervisor()
    if sup.alive("runner"):
        sup.stop("runner", grace_s=60.0)
    sup.start("runner", started_by="console")
