"""Rising + clean, the winners study's strategy (2026-10-31, data/scratch/winners_20261031): a selector pick takes twice its
stake when the market is recovering and the coin's launch was clean, Solana only.

- rising: the share of coins up on the hour gained at least ``RISE_6H`` over 6 h (``MinuteEngine.breadth``,
  train/flow.BreadthWindow), read from closed minutes;
- clean: holder concentration at graduation (``corpus_meta.grad_hhi``) at most ``CLEAN_HHI``; unknown is not clean.
Both cuts are the study's, set on the book's trades before 2026-09-12 and frozen: the top third of breadth changes, the
lowest third of concentrations.

The paper test: ``RisingBook`` trades ``paper_rising``, the selector's own rules and picks with these stakes doubled, beside
``paper_selector``. ``ui_settings['rising_clean']`` is the test ({started_at, go_live_at, live_since, held}; no row: no test,
no book), written from the console (ui/api/overview.py), so the go-live moves without a restart or an .env change.

The go-live (Bryce, 2026-11-01): at ``go_live_at`` the book judges the test once (``decide_go_live``). Its doubled trades
closed at a profit: the wallet's stakes on these picks double from then on (``live_since``, ``for_wallet``). Otherwise the
wallet stays as it is, the date is cleared and the reason kept (``held``) until the operator sets a new one. Setting or
clearing the date takes the wallet back to its plain stakes first (``set_go_live``), from its next buy.

Live money first (CLAUDE.md): before a go-live a live minute only counts its breadth (under 0.1 ms), the fly holds
``live_since`` in memory, read after each minute's live orders. After one, a minute with wallet entries reads it once more
before doubling them, so a cleared date binds the next buy; a failed entry is tried again at its plain stake.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from ..db.apilog import record_event
from ..market.exit_cost import label_size
from .selector_session import SelectorBook

log = logging.getLogger(__name__)
KEY, STATUS_KEY = "rising_clean", "rising_status"
BOOK, KIND = "paper_rising", "rising"
RISE_6H, CLEAN_HHI, STAKE_X = 0.027572083, 0.019701237, 2.0
TEST_DAYS = 14


def state(conn, lock: bool = False) -> dict | None:
    """The test (ISO times; ``go_live_at`` None: no date; ``live_since`` None: the wallet is not doubling; ``held``: the last
    go-live refused), None when no test was started. ``lock``: for a change, so the console and the book's judgement of the
    go-live take turns and neither overwrites the other."""
    r = conn.execute("SELECT value FROM ui_settings WHERE key = %s" + (" FOR UPDATE" if lock else ""), (KEY,)).fetchone()
    if not r or not r["value"]:
        return None
    return r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])


def _write(conn, value: dict) -> None:
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                 (KEY, json.dumps(value)))


def start(conn, now: datetime | None = None) -> dict:
    """The test starts: the book joins at the engine's next admission, the go-live ``TEST_DAYS`` from now."""
    now = now or datetime.now(timezone.utc)
    value = {"started_at": now.isoformat(), "go_live_at": (now + timedelta(days=TEST_DAYS)).isoformat(), "live_since": None, "held": None}
    _write(conn, value)
    record_event("info", KIND, f"rising + clean paper test started, go-live {value['go_live_at']}", value)
    return value


def set_go_live(conn, at: datetime | None) -> dict:
    """Moves the go-live, or clears it (``at`` None). Either way the wallet goes back to its plain stakes until a go-live passes."""
    s = state(conn, lock=True)
    if s is None:
        raise ValueError("the rising + clean test has not started")
    value = {**s, "go_live_at": at.isoformat() if at else None, "live_since": None, "held": None}
    _write(conn, value)
    record_event("warning", KIND, f"rising + clean go-live {'moved to ' + value['go_live_at'] if at else 'cleared'}",
                 {"from": s.get("go_live_at"), "to": value["go_live_at"], "was_live_since": s.get("live_since")})
    return value


def wallet_doubles(conn) -> bool:
    """Whether the wallet takes these stakes now."""
    return bool((state(conn) or {}).get("live_since"))


def doubled_result(conn, since: str) -> tuple[int, float]:
    """(closed doubled trades of the test, their realized P&L)."""
    r = conn.execute("""SELECT count(*) AS n, COALESCE(sum(p.realized_sol), 0) AS pnl FROM positions p JOIN decisions d ON d.id = p.entry_decision_id
                        WHERE p.book = %s AND p.status = 'closed' AND p.opened_at >= %s AND (d.detail->>'stake_x')::float > 1""", (BOOK, since)).fetchone()
    return int(r["n"]), float(r["pnl"])


def _due(s: dict | None, now: datetime) -> bool:
    return s is not None and not s.get("live_since") and bool(s.get("go_live_at")) and now >= datetime.fromisoformat(s["go_live_at"])


def decide_go_live(conn, now: datetime) -> str | None:
    """Once ``go_live_at`` has passed, the test is judged: 'live' or 'held' (None: nothing to judge yet)."""
    if not _due(state(conn), now):                     # most minutes: no lock
        return None
    s = state(conn, lock=True)
    if not _due(s, now):                               # the console moved the date just now
        return None
    n, pnl = doubled_result(conn, s["started_at"])
    detail = {"trades": n, "pnl_sol": pnl, "go_live_at": s["go_live_at"], "at": now.isoformat()}
    if pnl > 0:
        _write(conn, {**s, "live_since": now.isoformat(), "held": None})
        record_event("warning", KIND, f"rising + clean goes live: the wallet doubles these stakes ({n} doubled trades made {pnl:+.4f} SOL)", detail)
        return "live"
    _write(conn, {**s, "go_live_at": None, "held": detail})
    record_event("warning", KIND, f"rising + clean stays on paper: {n} doubled trades made {pnl:+.4f} SOL; set a new go-live date in the console", detail)
    return "held"


def qualifies(breadth_chg: float | None, grad_hhi: float | None) -> bool:
    return breadth_chg is not None and grad_hhi is not None and breadth_chg >= RISE_6H and grad_hhi <= CLEAN_HHI


def mark(breadth_chg: float | None, info: dict) -> dict:
    """What a pick of the row ``info`` carries on ``paper_rising``'s decisions (agent/paper_trading.plan_minute)."""
    return {"stake_x": STAKE_X if qualifies(breadth_chg, info.get("grad_hhi")) else 1.0, "breadth_chg": breadth_chg, "grad_hhi": info.get("grad_hhi")}


def for_wallet(ctx, entries: list, broker) -> int:
    """The minute's wallet entries that qualify take ``STAKE_X`` (agent/fly_live.py sizes them), and the most they may pay is
    their paper fill at that size, under the label size. An entry the caps leave no room to double, or that paper would not
    fill at that size, keeps its stake. How many doubled."""
    chg = ctx.engine.breadth.change(ctx.t_start); n = 0
    for e in entries:
        i = e["info"]; plain = float(e["size_sol"])
        if i.get("creator_heavy") or not qualifies(chg, i.get("grad_hhi")):
            continue
        size = min(plain * STAKE_X, label_size(i["resq"]))
        if size <= plain:
            continue
        q = broker.quote_buy(size_sol=size, price=i["price"], res_quote_sol=i["resq"], mcap_sol=i["mcap"], decimals=i["decimals"],
                             program_label=ctx.cost_label(e["mint"], i["program_label"]), pool_fee=ctx.fee(e["mint"]))
        if q.ok:
            e["stake_x"] = STAKE_X; e["ref_price"] = size / (q.token_delta / 10 ** int(i["decimals"])); n += 1
    return n


class RisingBook(SelectorBook):
    """``paper_rising``: a selector of its own (the book outlives ``paper_selector`` at a handover), the same picks, these
    stakes multiplied. It judges the go-live."""

    def __init__(self):
        self.chg: float | None = None
        super().__init__("sol", names=(KIND, BOOK, STATUS_KEY))

    def _handed(self, conn, ctx) -> bool:
        return False

    def _marks(self, ctx) -> list:
        return [mark(self.chg, i) for i in ctx.infos]

    def on_minute(self, ctx) -> dict:
        self.chg = ctx.engine.breadth.change(ctx.t_start)
        out = super().on_minute(ctx)
        if ctx.trade:
            decide_go_live(ctx.conn, ctx.m1)
        return out

    def _write_status(self, conn, status: dict) -> None:
        super()._write_status(conn, {**status, "breadth_chg": self.chg})
