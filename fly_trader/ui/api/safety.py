"""Safety & wallet: both chains' bot wallets (balance in USD, withdrawals to your own wallets), the trading rails and their
controls, the configured mode and limits, maintenance.

Real money leaves only through ``safety.withdraw``, and in THIS process: the Solana withdrawal takes the live broker's
in-process lock (execution/broker_live._INFLIGHT) so a swap's balance delta never contains a withdrawal — which is why the
console hosts the trading engine's thread. Nothing the browser sends is trusted: the blockers, the destination and the plan
are checked again before anything is signed, one withdrawal per chain is in flight at a time, and a dialog opening sends at
most once. Every other guard the page shows by disabling a button (a tripped switch, a stopped engine) is enforced here too."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from contextlib import contextmanager

from ... import config, markets
from ...agent import rails
from ...db.queries import q, q1
from ...logging_setup import scrub
from ...ops import wallets
from .. import runtime
from ..cache import ttl_cache
from ..hub import ActionError, TopicError, action, topic
from ..state import amount, kalshi_enabled, runs_elsewhere, usd

BASE_UNITS = {"sol": config.LAMPORTS_PER_SOL, "rh": 10 ** 18}
PRICE_SOURCE = {"sol": "Jupiter", "rh": "KyberSwap ETH→USDG"}
NO_WALLET = {"sol": "No wallet yet: run `fly-trader wallet new`, then fund it with SOL.",
             "rh": "No wallet yet: run `fly-trader rh-wallet new`, then fund it with ETH on Robinhood Chain."}
CIRCUITS = {   # id → (section, card title, what it guards)
    rails.SOLANA_CIRCUIT: ("Memecoins", "Trading rails", "Solana memecoin"),
    rails.RH_CIRCUIT: ("Robinhood Chain memecoins", "Robinhood Chain trading rails", "Robinhood Chain memecoin"),
    rails.KALSHI_CIRCUIT: ("Prediction markets", "Kalshi trading rails", "Kalshi"),
}
_WD = threading.Lock()                     # guards the two sets below
_WD_INFLIGHT: set[str] = set()             # chains with a withdrawal being sent right now
_WD_USED: OrderedDict[str, None] = OrderedDict()     # dialog openings that already sent (one withdrawal per opening)
_RAILS = threading.Lock()                  # a rails check and its write happen together
_RESET = threading.Lock()                  # one reset at a time, and never while an engine starts (processes.start)


@ttl_cache(15)
def wallet_summary(chain: str) -> dict:
    return wallets.summary(chain)


@ttl_cache(10)
def withdraw_limits(chain: str, to: str) -> dict:
    return wallets.limits(chain, to)


def short(a: str | None) -> str:
    return f"{a[:6]}…{a[-4:]}" if a else "—"


def _chains() -> list[str]:
    return [m.chain for m in markets.enabled()]


def _chain(params: dict, in_action: bool = False) -> str:
    c = params.get("chain")
    if c not in _chains():
        raise ActionError(f"no wallet {c!r} on this install", 400) if in_action else TopicError(f"no wallet {c!r} on this install")
    return c


def _amount(v) -> float | None:
    """The amount typed in the form, in the chain's coin, at the 6 decimals the form shows (what the button names is what
    goes out); None when empty. Raises ValueError for anything but a finite number ≥ 0."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError(v)
    return round(float(v), 6)


def _blockers(chain: str) -> list[str]:
    """Why this console cannot withdraw from ``chain`` (wallets.withdraw_blockers, plus: an observer never signs, and a Solana
    withdrawal must share the trading engine's process, whose swaps hold the lock it takes)."""
    try:
        out = list(wallets.withdraw_blockers(chain))
    except Exception as e:                                             # noqa: BLE001 — never let a key error's text out
        out = [f"cannot check the wallet's key ({type(e).__name__})"]
    sup = runtime.sup()
    if isinstance(sup, runtime.Observer):
        out.append("this console only observes (fly-trader ui --observe): withdraw from the console that hosts the trading engine")
    elif chain == "sol" and (why := runs_elsewhere("runner")):
        out.append(f"the trading engine runs outside this console ({why}), so its swaps would not share the withdrawal's lock")
    return out


# ---------------------------------------------------------------- wallets
@topic("safety.wallets", every=30)
def wallets_panel(params: dict) -> dict:
    """One column per enabled chain: the balance in USD (live RPC reads, cached 15 s), what blocks a withdrawal, the history."""
    out = []
    for m in markets.enabled():
        c = m.chain; w = wallet_summary(c); addr = wallets.address(c)
        row = {"chain": c, "name": m.name, "unit": m.unit, "price_source": PRICE_SOURCE[c], "no_wallet": NO_WALLET[c], "setting": wallets.SETTINGS[c],
               "summary": w, "explorer": wallets.explorer(c, "address", w["address"]) if w.get("address") else None, "address": addr,
               "blockers": [], "history": [], "events": []}
        if addr:
            row["blockers"] = _blockers(c)
            row["history"] = [{**h, "tx_url": wallets.explorer(c, "tx", h["tx"]) if h.get("tx") else None} for h in wallets.history(c, addr)]
            if c == "sol":
                row["events"] = q("SELECT ts, kind, pubkey FROM wallet_events ORDER BY id DESC LIMIT 20")
        out.append(row)
    return {"chains": out}


@topic("safety.withdraw_plan", every=10)
def withdraw_plan(params: dict) -> dict:
    """The withdraw dialog's form, read-only (it signs nothing): the destinations, fresh limits (cached 10 s), and what a click
    would send — or why not. ``to`` defaults to the first destination; ``all`` is everything withdrawable, else ``amount``."""
    c = _chain(params); unit, base = markets.MARKETS[c].unit, BASE_UNITS[c]
    dests = wallets.destinations(c)
    out = {"chain": c, "unit": unit, "setting": wallets.SETTINGS[c], "destinations": dests, "blockers": _blockers(c),
           "step": 0.01 if c == "sol" else 0.001, "to": None, "error": None}
    to = params.get("to") or (dests[0] if dests else None)
    if to is None:
        return out
    if to not in dests:
        raise TopicError(f"{to!r} is not one of your own wallets ({wallets.SETTINGS[c]})")
    everything = params.get("all") is True
    try:
        amt = None if everything else _amount(params.get("amount"))
    except ValueError:
        raise TopicError("amount must be a number ≥ 0") from None
    out.update(to=to, all=everything, amount=amt)
    try:
        lim = withdraw_limits(c, to)
    except Exception as e:                                             # noqa: BLE001 — shown in the dialog
        return {**out, "error": f"Cannot read the wallet: {type(e).__name__}: {scrub(str(e))[:300]}"}
    value, why = wallets.plan(c, lim, amt) if everything or amt else (0, "enter an amount")
    px = wallet_summary(c).get("price_usd")
    fee = f"fee ≈ {lim['fee'] / base:.6f} {unit}" if c == "sol" else f"gas at most {lim['fee'] / base:.6f} {unit}"
    keep = f" {lim['keep'] / base:g} {unit} stays for the exits of {lim['n_open']} open live position(s)." if lim["keep"] else ""
    send = amount(value / base, unit, 6)
    return {**out, "max": lim["max"] / base, "everything_label": f"Everything withdrawable: {amount(lim['max'] / base, unit, 6)}",
            "placeholder": f"at most {lim['max'] / base:.6f}", "value": value / base, "value_usd": value / base * px if px else None,
            "why": why, "show_why": bool(why and (everything or amt)),
            "summary": None if why else f"Sends **{send}**" + (f" ({usd(value / base * px)})" if px else "") + f" to `{short(to)}`; {fee}.{keep}",
            "button": f"Withdraw {send}" if not why else "Withdraw", "send": send}


def _send(chain: str, amt: float | None, to: str) -> dict:
    """wallets.withdraw, its refusals (nothing was sent) and any other failure (the outcome is unknown) as a result."""
    try:
        return wallets.withdraw(chain, amt, to)
    except wallets.REFUSALS as e:
        return {"status": "refused", "error": scrub(str(e))[:500]}
    except Exception as e:                                             # noqa: BLE001 — shown; whether it went out is unknown
        return {"status": "error", "error": scrub(f"{type(e).__name__}: {e}")[:500]}


RESULT_MESSAGE = {"confirmed": "Withdrawal confirmed", "pending": "Withdrawal sent, no receipt yet", "expired": "The withdrawal expired: nothing moved",
                  "failed": "The withdrawal failed on chain: only the fee moved", "refused": "Nothing was sent"}


@action("safety.withdraw")
def withdraw(body: dict) -> dict:
    """Send real money from a bot wallet to one of the operator's own wallets: ``{chain, to, amount}`` or ``{chain, to, all: true}``,
    with ``opening`` (the dialog's id: one withdrawal per opening). Refusals before anything is signed are 409s; once
    wallets.withdraw is called, every outcome is a result (confirmed / pending / expired / failed / refused / error)."""
    c = _chain(body, in_action=True); spec = markets.MARKETS[c]
    opening = body.get("opening")
    if not isinstance(opening, str) or not 8 <= len(opening) <= 80:
        raise ActionError("a withdrawal needs its dialog's opening id", 400)
    everything = body.get("all") is True
    try:
        amt = _amount(body.get("amount"))
    except ValueError:
        raise ActionError("amount must be a number ≥ 0", 400) from None
    if everything and amt is not None:
        raise ActionError("send an amount or everything withdrawable, not both", 400)
    if not everything and not amt:
        raise ActionError("enter an amount", 400)                     # an absent amount never means "everything"
    if blockers := _blockers(c):
        raise ActionError("Withdraw: " + "; ".join(blockers) + ".")
    to = body.get("to")
    if not isinstance(to, str) or to not in wallets.destinations(c):
        raise ActionError(f"{to!r} is not one of your own wallets ({wallets.SETTINGS[c]} in .env).", 400)
    with _WD:
        if opening in _WD_USED:
            raise ActionError("This withdraw window already sent one withdrawal: close it and open a new one to send another.")
        if c in _WD_INFLIGHT:
            raise ActionError(f"A withdrawal from the {spec.name} wallet is already in flight: wait for its outcome before sending another.")
        _WD_INFLIGHT.add(c); _WD_USED[opening] = None
        while len(_WD_USED) > 512:
            _WD_USED.popitem(last=False)
    try:
        try:
            lim = wallets.limits(c, to)                                # fresh: the dialog's plan may be 10 s old
        except Exception as e:                                         # noqa: BLE001
            raise ActionError(f"Cannot read the wallet: {type(e).__name__}: {scrub(str(e))[:300]}") from None
        _, why = wallets.plan(c, lim, None if everything else amt)
        if why:
            raise ActionError(why)
        r = _send(c, None if everything else amt, to)                  # the Solana path takes the broker's in-process lock
    finally:
        with _WD:
            _WD_INFLIGHT.discard(c)
        wallet_summary.cache_clear(); withdraw_limits.cache_clear()
    status = r.get("status") if r.get("status") in ("confirmed", "pending", "expired", "failed", "refused", "error") else "error"
    res = {"status": status, "error": r.get("error") or (None if status != "error" else f"unexpected outcome {r.get('status')!r}"),
           "tx": r.get("tx"), "tx_url": wallets.explorer(c, "tx", r["tx"]) if r.get("tx") else None, "amount": r.get("amount"),
           "to": r.get("to") or to, "to_short": short(r.get("to") or to), "unit": spec.unit}
    return {"message": RESULT_MESSAGE.get(status, "Whether the withdrawal went out is unknown"), "result": res}


# ---------------------------------------------------------------- rails
def _circuits() -> list[int]:
    """The circuits this install trades: Solana always, Robinhood Chain when RH_ENABLED, Kalshi once it is set up."""
    return [rails.SOLANA_CIRCUIT] + ([rails.RH_CIRCUIT] if config.RH_ENABLED else []) + ([rails.KALSHI_CIRCUIT] if kalshi_enabled() else [])


def _in_circuit(book: str, cid: int) -> bool:
    """A halted paper book's circuit: Kalshi books on 2, Robinhood Chain books on 3, every other book on 1."""
    rh_books = [b for m in markets.MARKETS.values() if m.circuit_id == rails.RH_CIRCUIT for b in m.books]
    return ("kalshi" in book) == (cid == rails.KALSHI_CIRCUIT) and (book in rh_books) == (cid == rails.RH_CIRCUIT)


@topic("safety.rails", every=2)
def rails_panel(params: dict) -> dict:
    cids = _circuits()
    rows = {r["id"]: r for r in q("SELECT id, kill_switch, kill_reason, tripped, fail_count, entries_paused FROM circuit_state WHERE id = ANY(%s)", (cids,))}
    halted = q("SELECT book, reason FROM book_state WHERE halted ORDER BY book")
    out = []
    for cid in cids:
        c = rows.get(cid) or {}; section, title, scope = CIRCUITS[cid]
        out.append({"id": cid, "section": section, "title": title, "scope": scope, "kill_switch": bool(c.get("kill_switch")), "kill_reason": c.get("kill_reason"),
                    "tripped": bool(c.get("tripped")), "fail_count": int(c.get("fail_count") or 0), "entries_paused": bool(c.get("entries_paused")),
                    "halted": [b for b in halted if _in_circuit(b["book"], cid)]})
    return {"circuits": out, "observer": _observer()}


LIVE_BOOK = {rails.SOLANA_CIRCUIT: ("live", "SOL"), rails.RH_CIRCUIT: ("live_rh", "ETH"), rails.KALSHI_CIRCUIT: ("live_kalshi", "USD")}


def _kill_limit(cid: int) -> float:
    return {rails.SOLANA_CIRCUIT: config.KILL_SWITCH_DRAWDOWN, rails.RH_CIRCUIT: markets.RH.kill_drawdown(),
            rails.KALSHI_CIRCUIT: config.KALSHI_KILL_SWITCH_DRAWDOWN}[cid]


@topic("safety.live_books", every=30)
def live_books(params: dict) -> dict:
    """Each circuit's live book against its kill switch: the last mark's drawdown from the peak the engine measures (marks
    since the last clear or withdrawal), the limit that trips it, and the last 180 wealth marks."""
    out = []
    for cid in _circuits():
        book, unit = LIVE_BOOK[cid]
        marks = q("SELECT ts, wealth, peak, drawdown, n_open FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 180", (book,))[::-1]
        last = marks[-1] if marks else {}
        note = ("The hosted vault fly's kill switch measures its NAV index, not this book's wealth." if cid == rails.SOLANA_CIRCUIT and config.VAULT_ENABLED
                else None)
        out.append({"id": cid, "book": book, "unit": unit, "limit": float(_kill_limit(cid)), "wealth": last.get("wealth"), "peak": last.get("peak"),
                    "drawdown": last.get("drawdown"), "ts": last.get("ts"), "n_open": last.get("n_open"), "note": note,
                    "series": [[m["ts"], m["wealth"]] for m in marks if m["wealth"] is not None]})
    return {"books": out}


def _observer() -> bool:
    return isinstance(runtime.sup(), runtime.Observer)


def _hosting(what: str) -> None:
    """Every control on this page belongs to the console that hosts the workers: an observer (a preview) changes nothing."""
    if _observer():
        raise runtime.ObserverMode(f"this console only observes (fly-trader ui --observe): {what} from the console that hosts the workers")


def _circuit(body: dict) -> tuple[int, dict]:
    _hosting("change the rails")
    cid = body.get("circuit")
    if isinstance(cid, bool) or not isinstance(cid, int) or cid not in _circuits():
        raise ActionError(f"no circuit {cid!r} on this page", 400)
    c = q1("SELECT * FROM circuit_state WHERE id = %s", (cid,))
    if c is None:
        raise ActionError(f"circuit {cid} has no state yet: the engine creates it")
    return cid, c


@action("safety.kill_clear")
def kill_clear(body: dict) -> dict:
    with _RAILS:
        cid, c = _circuit(body)
        if not c["kill_switch"]:
            raise ActionError("The kill switch is not tripped: there is nothing to clear.")
        rails.reset_circuit(kill=True, circuit_id=cid)
    return {"message": "Kill switch cleared"}


@action("safety.circuit_reset")
def circuit_reset(body: dict) -> dict:
    with _RAILS:
        cid, c = _circuit(body)
        if not c["tripped"] and not c["fail_count"]:
            raise ActionError("The circuit is ok with no failures: there is nothing to reset.")
        rails.reset_circuit(kill=False, circuit_id=cid)
    return {"message": "Circuit reset"}


@action("safety.entries")
def entries(body: dict) -> dict:
    paused = body.get("paused")
    if not isinstance(paused, bool):
        raise ActionError("paused must be true or false", 400)
    with _RAILS:
        cid, c = _circuit(body)
        if bool(c["entries_paused"]) == paused:
            raise ActionError("New entries are already paused." if paused else "New entries are already allowed.")
        rails.set_entries_paused(paused, circuit_id=cid)
    return {"message": "New entries paused" if paused else "New entries resumed"}


@action("safety.clear_halt")
def clear_halt(body: dict) -> dict:
    book = body.get("book")
    with _RAILS:
        cid, _ = _circuit(body)
        if not isinstance(book, str) or not _in_circuit(book, cid):
            raise ActionError(f"{book!r} is not a book of the {CIRCUITS[cid][2]} circuit", 400)
        r = q1("SELECT halted FROM book_state WHERE book = %s", (book,))
        if not r or not r["halted"]:
            raise ActionError(f"{book} is not halted.")
        rails.clear_book_halt(book)
    return {"message": f"{book}: halt cleared"}


# ---------------------------------------------------------------- mode and limits (config, read at start)
def _rows(rows) -> list[dict]:
    return [{"setting": s, "value": v, "meaning": m} for s, v, m in rows]


def _sol_settings() -> list[dict]:
    return _rows([("Money", "paper, then live once the fly holds the seat", "The race runs on paper books; after the handover the fly also trades the bot wallet, its paper book as the mirror."),
                  ("LIVE_ENABLED", "1" if config.LIVE_ENABLED else "0", "Allows signing real transactions: the fly trades the bot wallet once it holds the selector's seat."),
                  ("Cluster", config.SOLANA_CLUSTER, "Solana network for live orders."),
                  ("Starting capital", f"{config.CAPITAL_SOL:g} SOL", "Paper books start here."),
                  ("Position sizing", f"{config.KELLY_FRACTION:g} × Kelly", "Each buy is sized from how certain the model is: this share of the growth-optimal bet for its score band."),
                  ("Largest position", f"{config.MAX_POSITION_FRACTION:.0%} of the bankroll", "Bankroll = wealth at cost minus the gas reserve."),
                  ("Pool share cap", f"{config.MAX_POOL_SHARE:.0%} of the pool", "Bounds the price impact of larger buys."),
                  ("Smallest position", f"{config.MIN_POSITION_SOL:g} SOL", "Smaller sized buys are skipped."),
                  ("Fixed size (older models)", f"{config.MAX_POSITION_SOL:g} SOL", "Models trained before sizing trade this size."),
                  ("Gas reserve", f"{config.GAS_RESERVE_SOL:g} SOL", "Never spent on positions."),
                  ("Reset on start", "yes" if config.RESET_ON_START else "no", "Starting the trading engine archives and clears books other than the race books and the live book.")])


def _rh_settings() -> list[dict]:
    missing = config.rh_live_prerequisites_missing()
    return _rows([("Money", "paper; live mirror " + ("on" if not missing else "gated: " + ", ".join(missing)), "The RH race runs on paper books in ETH; after the RH handover the fly also trades the ETH wallet."),
                  ("RH_LIVE_ENABLED", "1" if config.RH_LIVE_ENABLED else "0", "Allows signing real Robinhood Chain transactions (live_rh)."),
                  ("Chain", f"{config.RH_EXPECTED_CHAIN_ID}" + (" (testnet)" if config.RH_TESTNET else ""), "The guard refuses any other chain id."),
                  ("Bot address", config.RH_BOT_ADDRESS or "not set", "RH_BOT_PRIVATE_KEY must derive to it."),
                  ("Starting capital", f"{config.RH_CAPITAL_ETH:g} ETH", "The USD value of 5 SOL at build time."),
                  ("Largest buy", f"{config.RH_LABEL_SIZE_ETH:g} ETH", "The label size (0.5 SOL in USD at build time): no RH buy is bigger."),
                  ("Smallest position", f"{config.RH_MIN_POSITION_ETH:g} ETH", "Smaller sized buys are skipped."),
                  ("Gas reserve", f"{config.RH_GAS_RESERVE_ETH:g} ETH", "Never spent on positions."),
                  ("Fee cap", f"{config.RH_MAX_FEE_GWEI:g} gwei", "A transaction never offers more per gas; a spike refuses instead."),
                  ("Kill switch", f"{config.RH_KILL_SWITCH_DRAWDOWN:.0%} drawdown", "On the live RH wealth (circuit 3)."),
                  ("Routes", "KyberSwap atomic, Uniswap v4 fallback, two legs through the base last", "Every round trip returns to ETH; no base inventory is kept.")])


def _kalshi_settings() -> list[dict]:
    missing = config.kalshi_live_prerequisites_missing()
    return _rows([("Money", "paper; live mirror " + ("on" if not missing else "gated: " + ", ".join(missing)), "Both arms always trade paper books; with KALSHI_LIVE_ENABLED=1 and the prerequisites they mirror onto the subaccount."),
                  ("KALSHI_LIVE_ENABLED", "1" if config.KALSHI_LIVE_ENABLED else "0", "Allows real orders on the subaccount."),
                  ("Taker / maker live", f"{'on' if config.KALSHI_TAKER_LIVE else 'off'} / {'on' if config.KALSHI_MAKER_LIVE else 'off'}", "Per-arm live switches."),
                  ("Subaccount", str(config.KALSHI_SUBACCOUNT) if config.KALSHI_SUBACCOUNT else "none (fly-trader kalshi-subaccount create)", "The dedicated subaccount; the primary never trades."),
                  ("Capital cap", f"${config.KALSHI_CAPITAL_USD:g}", "Paper books start here; live never sizes above it. Position ≤ 10 %, slate ≤ 50 % of it."),
                  ("Cash floor", f"${config.KALSHI_CASH_FLOOR_USD:g}", "Never spent."),
                  ("Kill switch", f"{config.KALSHI_KILL_SWITCH_DRAWDOWN:.0%} drawdown", "On the combined live wealth (circuit 2); paper books halt on their own peaks."),
                  ("Entry window", f"{config.KALSHI_MIN_MINUTES_TO_CLOSE:g} min … {config.KALSHI_MAX_DAYS_TO_CLOSE:g} days to close", "Rows outside it are never scored."),
                  ("Gates", f"spread ≤ {config.KALSHI_MAX_SPREAD_CENTS}c, OI ≥ {config.KALSHI_MIN_OPEN_INTEREST}, 24 h volume ≥ {config.KALSHI_MIN_VOLUME_24H}", "Eligibility of a (market, side, minute)."),
                  ("Maker", f"quiet {config.KALSHI_MAKER_QUIET_MIN:g} min before close · TTL {config.KALSHI_MAKER_TTL_H:g} h · ≤ {config.KALSHI_MAKER_MAX_RESTING} resting", "Resting bids one tick inside the ask, post-only."),
                  ("Fees", f"taker {config.KALSHI_FEE_FACTOR:g} · maker {config.KALSHI_MAKER_FEE_FACTOR:g} (× series multiplier × p(1−p))", "The vendored fee model.")])


@topic("safety.settings", every=60)
def settings(params: dict) -> dict:
    """The mode and limits from config, per trading type (Robinhood Chain when enabled, Kalshi once set up)."""
    tables = {"sol": {"title": "Mode and limits", "rows": _sol_settings()}}
    if config.RH_ENABLED:
        tables["rh"] = {"title": "Robinhood Chain mode and limits", "rows": _rh_settings()}
    if kalshi_enabled():
        tables["kalshi"] = {"title": "Kalshi mode and limits", "rows": _kalshi_settings()}
    return {"tables": tables, "caption": "Set in the project .env; changes apply after the console restarts."}


@topic("safety.kalshi_account", every=30)
def kalshi_account(params: dict) -> dict:
    """The Kalshi subaccount's live side. KALSHI_API_KEY_ID is a secret (config.SECRET_ENV_NAMES): never any part of it."""
    if not kalshi_enabled():
        return {"enabled": False}
    resting = q1("SELECT count(*) AS n, COALESCE(sum(price_cents * count), 0) AS c FROM kalshi_orders WHERE book LIKE %s AND status IN ('resting','submitted')", ("live%",)) or {}
    opens = q1("SELECT count(*) AS n, COALESCE(sum(cost_cents + fee_cents), 0) AS c FROM kalshi_positions WHERE book LIKE %s AND status = 'open'", ("live%",)) or {}
    w = q1("SELECT wealth, sol_free AS cash, ts FROM wealth_marks WHERE book = 'live_kalshi' ORDER BY ts DESC LIMIT 1") or {}
    return {"enabled": True, "key": "configured", "subaccount": config.KALSHI_SUBACCOUNT or None,
            "wealth": float(w["wealth"]) if w.get("wealth") is not None else None, "cash": float(w["cash"]) if w.get("cash") is not None else None,
            "ts": w.get("ts"), "open_n": int(opens.get("n") or 0), "open_usd": float(opens.get("c") or 0) / 100,
            "resting_n": int(resting.get("n") or 0), "resting_usd": float(resting.get("c") or 0) / 100}


# ---------------------------------------------------------------- maintenance
@contextmanager
def no_reset():
    """Held while an engine starts (processes.start) and while a reset runs: a reset never clears books under a starting engine."""
    if not _RESET.acquire(blocking=False):
        raise ActionError("A reset is archiving and clearing books right now: wait for it to finish.")
    try:
        yield
    finally:
        _RESET.release()


def engine_stopped(name: str, what: str) -> None:
    """Refuse unless ``name`` runs nowhere: not in this console, not as a CLI process, not in another console (resets use it)."""
    _hosting("reset")
    sup = runtime.sup()
    if sup.alive(name):
        raise ActionError(f"Stop the {what} first.")
    if why := runs_elsewhere(name):
        raise ActionError(f"The {what} is running ({why}): stop it first.")


@topic("safety.maintenance", every=3)
def maintenance(params: dict) -> dict:
    sup = runtime.sup(); k = kalshi_enabled()
    return {"runner_alive": sup.alive("runner"), "kalshi": k, "kalshi_runner_alive": bool(k and sup.alive("kalshi_runner")),
            "archive_dir": str(config.PG_ARCHIVE_DIR), "resetting": _RESET.locked(), "observer": isinstance(sup, runtime.Observer)}


@topic("safety.api_errors", every=15)
def api_errors(params: dict) -> dict:
    return {"rows": q("SELECT ts, service, endpoint, status, error FROM api_calls WHERE NOT ok ORDER BY id DESC LIMIT 30")}


@action("safety.reset_training")
def reset_training(body: dict) -> dict:
    """Archive, then clear the old paper books (ops/reset.reset_training_state). Refused while the trading engine runs."""
    engine_stopped("runner", "trading engine")
    with no_reset():
        engine_stopped("runner", "trading engine")
        from ...ops.reset import reset_training_state
        out = reset_training_state(reason="console button")
    return {"message": f"Archived to {out['archived_to']}", "archived_to": out["archived_to"], "rows": out.get("rows")}


@action("safety.reset_kalshi_fly")
def reset_kalshi(body: dict) -> dict:
    """Archive, then clear the visual fly's paper books and learning (ops/reset.reset_kalshi_fly). Refused while the Kalshi engine runs."""
    if not kalshi_enabled():
        raise ActionError("Kalshi is not set up on this install.")
    engine_stopped("kalshi_runner", "Kalshi engine")
    with no_reset():
        engine_stopped("kalshi_runner", "Kalshi engine")
        from ...ops.reset import reset_kalshi_fly
        out = reset_kalshi_fly(reason="console button")
    return {"message": f"Archived to {out['archived_to']}", "archived_to": out["archived_to"], "rows": out.get("rows")}
