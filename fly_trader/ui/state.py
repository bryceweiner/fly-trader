"""Shared reads for the operator console: one place that answers what the system is doing — trading or not, training or
not, paper or live money, which model is loaded — so every page says it the same way. Pure functions returning data;
the React pages render it (fly_trader/ui/api/* exposes them as topics)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .. import config, markets
from ..db.queries import q, q1
from ..train.selector import DATA_VERSION, is_current, is_deployable
from . import runtime
from .cache import ttl_cache

STALE_FEED_S = 180            # agent/minute_engine.py: no entries or exits when the newest complete minute is older

WORKER_INFO = {   # name → (title, icon, what it does, role)
    "runner": ("Trading engine", "candlestick", "Scores every traded token once a minute with the selector and the plastic fly, trades their books and lets the fly learn.", "trade"),
    "pumpstream": ("Market feed", "sensors", "Streams every PumpSwap trade from pumpapi.io into 1-minute candles. The trading engine reads these.", "trade"),
    "train": ("Trainer", "model_training", "On demand: builds decision points from the archive, backtests day by day and saves a new model.", "train"),
    "replay": ("History archive", "history", "Downloads the hourly trade archive and builds the training feature set.", "train"),
    "discover": ("Token stats", "query_stats", "Records Jupiter stats (holders, organic score, top-holder share) every 10 minutes for every token the selector can trade: history for future model inputs.", "optional"),
    "rh_stream": ("Robinhood Chain feed", "sensors", "Indexes Pons launches, graduations and Uniswap v4 swaps from Robinhood Chain into 1-minute rows in ETH (backfill from Pons V2, then live). The engine's Robinhood books read these.", "trade"),
    "vault": ("$FLY vault", "savings", "Runs the vault's jobs: scans the wallets' flows, pays claims, keeps the trading float, marks NAV, settles weekly profit to lockers and publishes the public stats. Signs nothing itself: every signature comes from the signer.", "vault"),
    "kalshi_runner": ("Kalshi trading engine", "candlestick", "Scores every quoted Kalshi market once a minute with the visual fly, trades its taker and maker paper books (and the subaccount when live) and lets it learn from settlements.", "trade"),
    "kalshi_stream": ("Kalshi market feed", "sensors", "Streams every Kalshi quote, trade and market lifecycle event into 1-minute rows. The Kalshi engine reads these.", "trade"),
    "kalshi_train": ("Kalshi trainer", "model_training", "Fits the Kalshi strategy stack walk-forward by settlement day and bootstraps the visual fly when none exists.", "train"),
    "kalshi_history": ("Kalshi history", "history", "Seeds settled markets and trades from the open dataset, fills 1-minute candles from the API and builds the feature rows.", "train"),
}
WORKER_TYPE = {"runner": "memecoins", "pumpstream": "memecoins", "train": "memecoins", "replay": "memecoins", "discover": "memecoins", "rh_stream": "memecoins",
               "vault": "system", "kalshi_runner": "prediction markets", "kalshi_stream": "prediction markets", "kalshi_train": "prediction markets",
               "kalshi_history": "prediction markets"}
ROLE_LABEL = {"trade": "needed to trade", "train": "needed to train", "optional": "optional", "vault": "runs the vault"}
KALSHI_WORKERS = ("kalshi_runner", "kalshi_stream", "kalshi_train", "kalshi_history")
KALSHI_BOOKS = {"paper_kalshi_taker": "Taker · paper", "paper_kalshi_maker": "Maker · paper", "live_kalshi_taker": "Taker · live", "live_kalshi_maker": "Maker · live"}


def kalshi_enabled() -> bool:
    """The distribution ships Kalshi dormant: its pages, status row and worker cards appear once it is set up."""
    return bool(config.KALSHI_API_KEY_ID)


def worker_order() -> list[str]:
    from ..ops.supervisor import WORKERS
    names = ("runner", "pumpstream", "rh_stream", "train", "replay", "discover", "vault", "kalshi_runner", "kalshi_stream", "kalshi_train", "kalshi_history")
    return [n for n in names if n in WORKERS and (kalshi_enabled() or n not in KALSHI_WORKERS)]


# ---------------------------------------------------------------- formatting (server-side sentences)
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


def until(iso: str | None) -> str:
    if not iso:
        return "—"
    s = (datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()
    return "due now" if s <= 0 else f"in {s / 3600:.0f} h" if s < 172800 else f"in {s / 86400:.1f} days"


def amount(x, unit: str = "SOL", nd: int = 4, signed: bool = False) -> str:
    """A book amount in its market's unit (SOL on Solana, ETH on Robinhood Chain)."""
    return "—" if x is None else f"{float(x):{'+' if signed else ''},.{nd}f} {unit}"


def pct(x, nd: int = 2, signed: bool = True) -> str:
    return "—" if x is None else f"{float(x) * 100:{'+' if signed else ''}.{nd}f}%"


def usd(x, nd: int = 2, signed: bool = False) -> str:
    return "—" if x is None else f"{'+' if signed and float(x) >= 0 else ''}{'-' if float(x) < 0 else ''}${abs(float(x)):,.{nd}f}"


def market_param(params: dict) -> markets.MarketSpec:
    """The memecoin chain a topic is asked about (``chain``: sol | rh); Solana by default and when RH is off."""
    chain = params.get("chain") or "sol"
    for m in markets.enabled():
        if m.chain == chain:
            return m
    return markets.SOL


def market_info(m: markets.MarketSpec) -> dict:
    return {"chain": m.chain, "name": m.name, "unit": m.unit, "selector_book": m.selector_book, "fly_book": m.fly_book, "live_book": m.live_book,
            "books": list(m.books), "capital": m.capital()}


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
    if wf.get("days"):
        s = (f"Backtest on {wf.get('days')} held-out days: {pct(wf.get('mean'))} average per trade over {wf.get('n')} trades, "
             f"profitable on {wf.get('days_positive')} of those {wf.get('days')} days")
    elif wf.get("weeks"):                  # the strategy-stack selectors judge their evaluation half by week (train/strategies.py)
        s = (f"Backtest on {wf.get('weeks')} held-out weeks: {pct(wf.get('mean'))} average per trade over {wf.get('n')} trades, "
             f"profitable in {wf.get('weeks_pos')} of those {wf.get('weeks')} weeks")
    else:
        s = f"Backtest: {pct(wf.get('mean'))} average per trade over {wf.get('n')} held-out trades"
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
@ttl_cache(5)
def external_workers() -> frozenset[str]:
    """Workers running as CLI processes (`fly-trader kalshi-stream` and friends), read without touching their rows."""
    from ..ops.procs import _is_ours
    return frozenset(r["name"] for r in q("SELECT name, pid FROM processes WHERE stopped_at IS NULL AND cmd[1] <> 'thread'") if _is_ours(r["pid"], r["name"]))


ENGINE_LOCKS = {"runner": ("agent.runner", "RUNNER_LOCK_KEY"), "kalshi_runner": ("kalshi.engine", "KALSHI_LOCK_KEY")}


def runs_elsewhere(name: str) -> str | None:
    """Why ``name`` counts as running outside THIS console, read fresh (the guards of resets and withdrawals use it): a CLI
    process of it, another live console's thread, or — for an engine — its one-per-database advisory lock held while this
    console does not run it (a console still winding down, a second `fly-trader ui`)."""
    import importlib
    import os
    import psutil
    from ..ops.procs import _is_ours
    me = os.getpid()
    for r in q("SELECT pid, cmd FROM processes WHERE name = %s AND stopped_at IS NULL", (name,)):
        pid = int(r["pid"])
        if pid == me:
            continue
        if (r["cmd"] or [None])[0] == "thread":
            if psutil.pid_exists(pid):
                return f"another console (pid {pid}) runs it"
        elif _is_ours(pid, name):
            return f"it runs outside the console (pid {pid})"
    lock = ENGINE_LOCKS.get(name)
    if lock and not runtime.sup().alive(name):
        key = int(getattr(importlib.import_module(f"fly_trader.{lock[0]}"), lock[1]))
        held = q1("SELECT 1 AS held FROM pg_locks WHERE locktype = 'advisory' AND granted AND objsubid = 1 AND classid = %s::oid AND objid = %s::oid "
                  "AND database = (SELECT oid FROM pg_database WHERE datname = current_database())",     # pg_locks spans every database
                  (key >> 32, key & 0xFFFFFFFF))
        if held:
            return "its database lock is held (another console is still running or winding it down)"
    return None


def running(ws: dict, name: str) -> bool:
    """A worker runs when this console's thread does or a CLI process of it does (a feed is a feed either way)."""
    return bool(ws.get(name, {}).get("alive")) or name in external_workers()


def system_state(market: markets.MarketSpec | None = None) -> dict:
    """What one memecoin market (default Solana) is doing: its books' status keys, its feed, its circuit, its seat."""
    m = market or markets.SOL; rh = m.chain != "sol"
    ws = runtime.sup().status(); now = datetime.now(timezone.utc)
    sel, sel_at = setting(m.selector_status_key); tr, tr_at = setting("training_status"); ps, _ = setting(m.stream_status_key)
    fly, fly_at = setting(m.fly_status_key); ho, _ = setting(m.handover_key); rp, _ = setting("fly_replay")
    circuit = q1("SELECT kill_switch, kill_reason, tripped, fail_count, entries_paused FROM circuit_state WHERE id = %s", (m.circuit_id,)) or {}
    ft = ps.get("flushed_through")
    feed_age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None     # since the newest complete minute closed
    feed_worker = "rh_stream" if rh else "pumpstream"
    feed_ok = running(ws, feed_worker) and feed_age is not None and feed_age < STALE_FEED_S
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
    if rh:
        signing = not config.rh_live_prerequisites_missing()
    else:
        from ..chain.cluster_guard import signing_allowed
        signing = signing_allowed()
    live_money = bool(ho) and runner and fly.get("stage") == "trading" and signing      # the fly holds the seat and may sign (agent/fly_live.py, rh/live.py)
    return {"market": m, "workers": ws, "runner": runner, "trading": trading, "trading_why": why, "live_money": live_money,
            "fly": fly, "fly_at": fly_at, "handover": ho or None, "replay": rp or None,
            "model": loaded_model() if runner else None, "latest": latest_snapshot("selector"), "deployable": latest_snapshot("selector", deployable=True),
            "training": training, "train_status": tr, "train_at": tr_at,
            "train_external": training and not ws["train"]["alive"], "selector": sel, "selector_at": sel_at, "feed": ps, "feed_ok": feed_ok, "feed_age": feed_age,
            "circuit": circuit}


def kalshi_state() -> dict:
    """What the Kalshi side is doing: feed, engine, the visual fly, training, history, rails (circuit 2) and books."""
    from ..db.queries import read_connection
    from ..kalshi import fly as KF, selector as kselector
    ws = runtime.sup().status(); now = datetime.now(timezone.utc)
    fly, fly_at = setting("kalshi_fly_status"); st_, _ = setting("kalshi_stream_status"); tr, tr_at = setting("kalshi_training_pipeline")
    rp, _ = setting("kalshi_fly_replay"); hs, hs_at = setting("kalshi_history_status"); ts_, _ = setting("training_status")
    circuit = q1("SELECT kill_switch, kill_reason, tripped, fail_count, entries_paused FROM circuit_state WHERE id = 2") or {}
    ft = st_.get("flushed_through")
    feed_age = (now - datetime.fromisoformat(ft)).total_seconds() - 60 if ft else None
    feed_ok = running(ws, "kalshi_stream") and feed_age is not None and feed_age < STALE_FEED_S
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
    with read_connection() as conn:
        sel = kselector.latest_current(conn); boot = KF.latest_current(conn); dep = KF.latest_deployable(conn)
    training = ws["kalshi_train"]["alive"] and str(tr.get("stage") or "").startswith(("training", "bootstrapping"))
    return {"workers": ws, "runner": runner, "trading": trading, "trading_why": why, "live_money": live, "fly": fly, "fly_at": fly_at, "replay": rp or None,
            "feed": st_, "feed_ok": feed_ok, "feed_age": feed_age, "train": tr, "train_at": tr_at, "training": training, "train_progress": ts_ if training else {},
            "history": hs, "history_at": hs_at, "circuit": circuit, "books": books, "selector": sel, "bootstrap": boot, "deployable": dep,
            "prerequisites_missing": config.kalshi_live_prerequisites_missing()}


# ---------------------------------------------------------------- the status widgets (one row of colour widgets per market)
def widget(key: str, label: str, value: str, color: str, icon: str, sub: str | None = None, help: str | None = None) -> dict:
    """One colour static widget: what (label), its state (value), a detail line (sub); colour green/orange/red/blue/violet/gray."""
    return {"key": key, "label": label, "value": value, "sub": sub, "color": color, "icon": icon, "help": help}


TRADING_STATE = {"trading": ("Trading", "green"), "holding": ("Holding", "orange"), "paused": ("Entries paused", "orange"),
                 "blocked": ("Kill switch", "red"), "starting": ("Starting", "blue"), "stopped": ("Not trading", "gray"),
                 "waiting": ("Waiting", "gray")}
TRADING_SUB = {"holding": "the market feed is stale", "waiting": "for a qualified model", "blocked": "no new entries", "paused": "exits still run"}


def _feed_help(age) -> str:
    return f"newest complete minute closed {age:.0f} s ago" if age is not None else "no minute written yet"


def _feed_sub(age) -> str:
    return f"newest minute {age:.0f} s ago" if age is not None else "no minute yet"


def memecoin_widgets() -> list[dict]:
    s = system_state(); out = []
    if s["live_money"]:
        out.append(widget("money", "Money", "Live", "red", "payments", "real SOL from the bot wallet", "Trades spend real SOL from the bot wallet."))
    else:
        out.append(widget("money", "Money", "Paper", "blue", "receipt", "no real SOL", "Trades are simulated at market prices with real fees; no wallet funds are used."))
    text, color = TRADING_STATE[s["trading"]]
    out.append(widget("engine", "Trading engine", text, color, "candlestick", TRADING_SUB.get(s["trading"], "every minute" if s["trading"] == "trading" else None), s["trading_why"]))
    m, latest, dep = s["model"], s["latest"], s["deployable"]
    sub = None; color = "violet" if m else "gray"
    if m and dep and dep["id"] > m["id"]:
        sub = f"switching to #{dep['id']}"
    elif latest and not is_deployable(latest["meta"]) and (not m or latest["id"] > m["id"]):
        sub = f"#{latest['id']} not put to work"; color = "orange" if not m else color
    elif m:
        sub = "at work"
    out.append(widget("selector", "Selector", f"#{m['id']}" if m else "None at work", color, "psychology", sub,
                      backtest_line(m["meta"]) if m else ("No model has been trained on the current data yet." if not latest
                                                          else "No model has earned a place yet: " + (latest["meta"].get("deploy_reason") or "none qualified") if not dep
                                                          else "The trading engine is stopped, so no model is in use.")))
    fly = s["fly"]
    if fly.get("stage") and fly.get("stage") != "not trading":
        frozen = bool(fly.get("learning_frozen"))
        out.append(widget("fly", "Plastic fly", f"#{fly['bootstrap']}" if fly.get("bootstrap") else "Running", "orange" if frozen else "violet", "neurology",
                          "learning frozen" if frozen else f"learning · drift {float(fly.get('drift') or 0):.1%}",
                          f"Plastic fly on {fly.get('device') or '?'}: buy line {pct(fly.get('line'))}, {fly.get('pending', 0)} labels pending."
                          + (" Taught once by the selector, then learning on its own." if fly.get("bootstrap") else "")))
    else:
        out.append(widget("fly", "Plastic fly", "Not trading", "gray", "neurology", f"bootstrap #{fly['bootstrap']}" if fly.get("bootstrap") else None,
                          fly.get("detail") or "The trading engine is stopped."))
    if s["handover"]:
        out.append(widget("seat", "Seat", "Fly holds it", "green", "swap", "the selector stands aside", "The fly won the race; the selector no longer trades while the fly can."))
    tr = s["train_status"]
    out.append(widget("training", "Training", "Running" if s["training"] else "Idle", "blue" if s["training"] else "gray", "model_training",
                      str(tr.get("stage") or "") if s["training"] else "not training", None))
    out.append(widget("feed", "Market feed", "Live" if s["feed_ok"] else "Stale", "green" if s["feed_ok"] else "red", "sensors", _feed_sub(s["feed_age"]), _feed_help(s["feed_age"])))
    return out


def rh_widgets() -> list[dict]:
    """The Robinhood Chain row: money, trading, the RH fly, its seat and the indexer."""
    s = system_state(markets.RH); fly = s["fly"]; ix = s["feed"]; out = []
    if s["live_money"]:
        out.append(widget("money", "Money", "Live", "red", "payments", "real ETH from the RH bot wallet", "Trades spend real ETH from the Robinhood Chain bot wallet."))
    else:
        miss = config.rh_live_prerequisites_missing()
        out.append(widget("money", "Money", "Paper", "blue", "receipt", "no real ETH",
                          "Pons trades are simulated at market prices with the hook fee and impact" + (f"; live gated: {', '.join(miss)}." if miss else ".")))
    text, color = TRADING_STATE[s["trading"]]
    out.append(widget("engine", "Trading engine", text, color, "candlestick", TRADING_SUB.get(s["trading"], "every minute" if s["trading"] == "trading" else None), s["trading_why"]))
    if fly.get("stage") and fly.get("stage") != "not trading":
        out.append(widget("fly", "RH fly", "Trading", "violet", "neurology", "one brain with the Solana fly", "The fly on Robinhood Chain rows (one brain with the Solana fly)."))
    else:
        out.append(widget("fly", "RH fly", "Not trading", "gray", "neurology", None, fly.get("detail") or s["trading_why"]))
    if s["handover"]:
        out.append(widget("seat", "Seat", "Fly holds it", "green", "swap", "the selector stands aside"))
    if ix.get("mode") == "backfill":
        out.append(widget("feed", "RH indexer", "Backfilling", "blue", "history", f"{int(ix.get('behind_blocks') or 0):,} blocks behind"))
    else:
        out.append(widget("feed", "RH feed", "Live" if s["feed_ok"] else "Stale", "green" if s["feed_ok"] else "red", "sensors", _feed_sub(s["feed_age"]), _feed_help(s["feed_age"])))
    return out


def kalshi_widgets() -> list[dict]:
    k = kalshi_state(); fly = k["fly"]; out = []
    if k["live_money"]:
        out.append(widget("money", "Money", "Live", "red", "payments", f"subaccount {config.KALSHI_SUBACCOUNT}",
                          f"Both arms mirror onto Kalshi subaccount {config.KALSHI_SUBACCOUNT} (cap ${config.KALSHI_CAPITAL_USD:g})."))
    else:
        out.append(widget("money", "Money", "Paper", "blue", "receipt", "simulated at the quoted book", "Prediction-market trades are simulated at the quoted book with real fees; "
                          + ("live mirroring is off (KALSHI_LIVE_ENABLED=0)." if not config.KALSHI_LIVE_ENABLED else "live gated: " + ", ".join(k["prerequisites_missing"]))))
    text, color = TRADING_STATE[k["trading"]]
    sub = {"waiting": "for the visual fly", "holding": "the Kalshi feed is stale", "starting": "catching up"}.get(k["trading"], TRADING_SUB.get(k["trading"]))
    out.append(widget("engine", "Kalshi engine", text, color, "candlestick", sub or ("both arms, every minute" if k["trading"] == "trading" else None), k["trading_why"]))
    if fly.get("stage") and fly.get("stage") != "not trading":
        frozen = bool(fly.get("learning_frozen"))
        out.append(widget("fly", "Visual fly", f"#{fly['bootstrap']}" if fly.get("bootstrap") else "Running", "orange" if frozen else "violet", "visibility",
                          "learning frozen" if frozen else f"learning · drift {float(fly.get('drift') or 0):.1%}",
                          f"The Kalshi fly on {fly.get('device') or '?'}: {fly.get('pending', 0)} settlements pending; edge lines "
                          + ", ".join(f"{a} {float(v):.3f}" for a, v in (fly.get("lines") or {}).items())))
    else:
        out.append(widget("fly", "Visual fly", "Not trading", "gray", "visibility", None, fly.get("detail") or k["trading_why"]))
    tr = k["train"]
    out.append(widget("training", "Kalshi training", "Running" if k["training"] else "Idle", "blue" if k["training"] else "gray", "model_training",
                      str(tr.get("stage") or "") if k["training"] else "not training"))
    hs = k["history"]
    if running(k["workers"], "kalshi_history"):
        out.append(widget("history", "Kalshi history", str(hs.get("stage") or "running").capitalize(), "blue", "history",
                          f"{hs.get('done', '?')} of {hs.get('total', '?')} markets filled"))
    out.append(widget("feed", "Kalshi feed", "Live" if k["feed_ok"] else "Stale", "green" if k["feed_ok"] else "red", "sensors", _feed_sub(k["feed_age"]), _feed_help(k["feed_age"])))
    return out


def status_rows() -> list[dict]:
    """Every market's widgets; a market whose state cannot be read says why instead of taking the others down."""
    rows = []
    for key, label, fn, on in (("sol", "Solana", memecoin_widgets, True), ("rh", "Robinhood Chain", rh_widgets, config.RH_ENABLED),
                               ("kalshi", "Kalshi", kalshi_widgets, kalshi_enabled())):
        if not on:
            continue
        try:
            rows.append({"key": key, "label": label, "widgets": fn(), "error": None})
        except Exception as e:
            rows.append({"key": key, "label": label, "widgets": [], "error": f"{label} status unavailable: {type(e).__name__}: {e}"})
    return rows
