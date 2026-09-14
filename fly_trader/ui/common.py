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
from fly_trader.train.selector import is_current

BOOK = "paper_selector"
STALE_FEED_S = 180            # agent/selector_session.py: no entries or exits when the newest complete minute is older

WORKER_INFO = {   # name → (title, icon, what it does, role)
    "runner": ("Trading engine", ":material/candlestick_chart:", "Scores every traded token once a minute with the loaded model and trades the paper book.", "trade"),
    "pumpstream": ("Market feed", ":material/sensors:", "Streams every PumpSwap trade from pumpapi.io into 1-minute candles. The trading engine reads these.", "trade"),
    "train": ("Trainer", ":material/model_training:", "On demand: builds decision points from the archive, backtests day by day and saves a new model.", "train"),
    "replay": ("History archive", ":material/history:", "Downloads the hourly trade archive and builds the training feature set.", "train"),
    "corpus": ("Pre-April corpus", ":material/inventory_2:", "Pulls candles for graduations older than the archive. Not used by the selector.", "optional"),
    "discover": ("Watch list", ":material/travel_explore:", "Polls Jupiter for graduated tokens and their stats. Used by the legacy brain modes.", "optional"),
    "capture": ("Swap tape", ":material/receipt_long:", "Helius websocket swaps for watched pools. Used by the legacy brain modes.", "optional"),
}
ROLE_LABEL = {"trade": "needed to trade", "train": "needed to train", "optional": "optional"}
ORDER = [n for n in ("runner", "pumpstream", "train", "replay", "corpus", "discover", "capture") if n in WORKERS]


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


# ---------------------------------------------------------------- models
_NOTE_OBJ = re.compile(r'"(walk_forward|random_baseline|fly|gbm|random)": (\{[^{}]*\})')
_NOTE_SCALAR = re.compile(r'"(threshold|trained_through|horizon_min|top_frac|costs|rows|days|fly_beats_gbm)": ("[^"]*"|[-0-9.eE]+|true|false)')


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


def latest_snapshot(kind: str = "selector") -> dict | None:
    """The newest snapshot of this kind trained on the current data definitions (older ones are never loaded)."""
    for r in q("SELECT id, ts, kind, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC LIMIT 50", (kind,)):
        s = _snap(r)
        if is_current(s["meta"]):
            return s
    return None


def loaded_model() -> dict | None:
    """The snapshot the running selector session loaded (its runs row), with its backtest."""
    r = q1("SELECT brain_snapshot_id, config, started_at FROM runs WHERE kind = 'selector' AND status = 'running' ORDER BY started_at DESC LIMIT 1")
    if not r or not r["brain_snapshot_id"]:
        return None
    s = snapshot(int(r["brain_snapshot_id"]))
    return {**s, "run": jv(r["config"]), "since": r["started_at"]} if s else None


OUTDATED = "Trained on outdated data (before the data fixes of 2026-09-14), so its backtest is not valid and it will not be loaded."


def backtest_line(meta: dict) -> str:
    if not is_current(meta):
        return OUTDATED
    wf, rb = meta.get("walk_forward") or {}, meta.get("random_baseline") or {}
    if not wf:
        return "No backtest recorded."
    s = (f"Backtest on {wf.get('days')} held-out days: {pct(wf.get('mean'))} average per trade over {wf.get('n')} trades, "
         f"profitable on {wf.get('days_positive')} of those {wf.get('days')} days")
    return s + (f"; random picks made {pct(rb['mean'])} per trade under the same costs." if rb.get("mean") is not None else ".")


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
    elif circuit.get("kill_switch"):
        trading, why = "blocked", "Kill switch tripped" + (f" ({circuit['kill_reason']})" if circuit.get("kill_reason") else "") + ": no new entries."
    elif circuit.get("entries_paused"):
        trading, why = "paused", "New entries are paused by the operator; open positions still exit on time."
    elif str(sel.get("stage", "")).startswith("stream stale") or not feed_ok:
        trading, why = "holding", "The market feed is stale: no entries or exits until it recovers."
    elif not sel:
        trading, why = "starting", "Warming up on the last 24 hours of market minutes."
    else:
        trading, why = "trading", "Trading every minute."
    return {"workers": ws, "runner": runner, "trading": trading, "trading_why": why, "live_money": bool(config.LIVE_ENABLED and config.BRAIN_MODE != "selector"),
            "model": loaded_model() if runner else None, "latest": latest_snapshot("selector"), "training": training, "train_status": tr, "train_at": tr_at,
            "train_external": training and not ws["train"]["alive"], "selector": sel, "selector_at": sel_at, "feed": ps, "feed_ok": feed_ok, "feed_age": feed_age,
            "circuit": circuit}


TRADING_BADGE = {"trading": ("Trading", "green"), "holding": ("Holding: feed stale", "orange"), "paused": ("Entries paused", "orange"),
                 "blocked": ("Kill switch tripped", "red"), "starting": ("Starting", "blue"), "stopped": ("Not trading", "gray")}


@st.fragment(run_every="5s")
def status_strip() -> None:
    s = system_state()
    with st.container(horizontal=True, gap="small"):
        if s["live_money"]:
            st.badge("Live money", icon=":material/payments:", color="red", help="Trades spend real SOL from the bot wallet.")
        else:
            st.badge("Paper — no real SOL", icon=":material/receipt:", color="blue", help="Trades are simulated at market prices with real fees; no wallet funds are used.")
        text, color = TRADING_BADGE[s["trading"]]
        st.badge(text, icon=":material/candlestick_chart:", color=color, help=s["trading_why"])
        m, latest = s["model"], s["latest"]
        st.badge(f"Model #{m['id']}" if m else "No model loaded", icon=":material/psychology:", color="violet" if m else "gray",
                 help=backtest_line(m["meta"]) if m else ("No model has been trained on the current data yet." if not latest else "The trading engine is stopped, so no model is in use."))
        if m and latest and latest["id"] > m["id"]:
            st.badge(f"Newer model #{latest['id']} saved", icon=":material/upgrade:", color="orange", help="Load it from Model & training.")
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
