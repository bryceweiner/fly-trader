"""Rules for the market we are in, switched on by a trigger read from closed data (Bryce, 2026-11-03: "there are rules that need
to be applied now that don't need to be applied earlier in the year. Discover those rules and what the proper triggers for them
should be"; the study is data/scratch/regime_20261103).

Quiet bar, old coin (Solana): while few young coins pass the selector's bar, a coin at least ``OLD_H`` hours past its graduation
is not bought. The trigger is the selector's own pass rate: the buys it allowed (``fly_scored.teacher_allow``, one per coin per
240 minutes, as the book trades them) of coins younger than ``OLD_H``, per day over the last ``SHORT_D`` days against the last
``LONG_D``; quiet when log((short + 0.5) / (long + 0.5)) <= ``QUIET_LOG``. Counting young coins only keeps the rule out of its
own trigger. With under ``MIN_LONG_D`` days of history, or a failed read, the market is not quiet and the rule stands aside.
- Measured on #640's out-of-sample Solana book under today's rules, 05-10..11-02, at deployed stakes: the old coins it skips
  averaged -3.5 % a trade against +10.5 % for the rest bought in quiet spells; it helped in 8 of the 9 half-months it acted in.
  Skipping the same coins at every pass rate loses money: the rule belongs to quiet spells. On Robinhood Chain it does not help.
- On the live wallet since 10-17 the same rule (with the study's first cut) would have skipped 6 buys that lost 0.281 SOL,
  both 7rTNgVdQ buys among them.
Live money first (CLAUDE.md): the rate is read in the paper phase, at most every ``REFRESH_S`` (one indexed query, ~0.35 s),
after the wallet's orders; ``MinuteEngine.eligible`` reads the cached flag.

The take-profit its recent trades favour (Robinhood Chain): ``ExitChoice``, below.
"""
from __future__ import annotations

import logging
import math

from ..db.apilog import record_event

log = logging.getLogger(__name__)
OLD_H = 20.4                     # hours past graduation: the rule's old coin, and the young coins its trigger counts are younger
SHORT_D, LONG_D, MIN_LONG_D = 3, 30, 14
QUIET_LOG = -0.230               # the lowest fifth of the study's history before 10-02 (the book's own young picks)
ENTRY_S = 240 * 60.0             # one buy per coin per hold, as the book trades the selector's picks
REFRESH_S = 600


def pass_rates(rows, now: float) -> tuple[float, float, float] | None:
    """(young buys a day over ``SHORT_D`` days, over ``LONG_D`` days or the history there is, the log ratio) from ``rows``
    of (mint, decision time, graduation time) the selector allowed; None with under ``MIN_LONG_D`` days of history."""
    by: dict[str, list[float]] = {}
    for mint, t, g in rows:
        if g is not None and t <= now and (t - g) / 3600.0 < OLD_H:
            by.setdefault(mint, []).append(t)
    if not by:
        return None
    entries = []
    for ts in by.values():
        last = -math.inf
        for t in sorted(ts):
            if t >= last + ENTRY_S:
                entries.append(t); last = t
    days = min(float(LONG_D), (now - min(entries)) / 86400.0)
    if days < MIN_LONG_D:
        return None
    short = sum(1 for t in entries if t >= now - SHORT_D * 86400.0) / SHORT_D
    long_ = sum(1 for t in entries if t >= now - days * 86400.0) / days
    return short, long_, math.log((short + 0.5) / (long_ + 0.5))


class QuietBar:
    """The trigger's last reading; ``blocks`` is the rule."""
    __slots__ = ("short", "long", "ratio", "quiet", "at")

    def __init__(self):
        self.short = self.long = self.ratio = None; self.quiet = False; self.at = -math.inf

    def refresh(self, conn, now: float) -> None:
        self.at = now
        try:
            with conn.transaction():                     # a savepoint: a failed read never aborts the minute's transaction
                rows = conn.execute("""SELECT f.mint, extract(epoch from f.ts)::float8 AS t, extract(epoch from cm.graduated_at)::float8 AS g
                                       FROM fly_scored f JOIN corpus_meta cm ON cm.mint = f.mint
                                       WHERE f.chain = 'sol' AND f.teacher_allow AND f.ts > to_timestamp(%s) AND f.ts <= to_timestamp(%s)""",
                                    (now - (LONG_D + 1) * 86400.0, now)).fetchall()
            r = pass_rates([(x["mint"], x["t"], x["g"]) for x in rows], now)
        except Exception:
            log.exception("quiet-bar trigger: read failed; the rule stands aside")
            r = None
        was = self.quiet
        self.short, self.long, self.ratio = r if r is not None else (None, None, None)
        self.quiet = r is not None and r[2] <= QUIET_LOG
        if self.quiet != was:                            # the console's bell shows each switch
            msg = (f"quiet bar {'on: Solana coins at least ' + format(OLD_H, 'g') + ' h old are not bought' if self.quiet else 'off'}; young buys a day "
                   f"{self.short or 0.0:.2f} over {SHORT_D} d against {self.long or 0.0:.2f} (log ratio {self.ratio if self.ratio is not None else float('nan'):+.3f}, trigger {QUIET_LOG:+.3f})")
            log.info(msg); record_event("info", "regime", msg, self.status())

    def maybe_refresh(self, conn, now: float) -> None:
        if now - self.at >= REFRESH_S:
            self.refresh(conn, now)

    def blocks(self, info: dict) -> bool:
        age = info.get("age_h")
        return self.quiet and age is not None and age >= OLD_H

    def status(self) -> dict:
        return {"quiet": self.quiet, "young_buys_3d": self.short, "young_buys_30d": self.long, "log_ratio": self.ratio, "trigger": QUIET_LOG, "old_h": OLD_H}


# ---------------------------------------------------------------- Robinhood Chain: the take-profit its recent trades favour
# Each new RH buy takes the take-profit line that would have done best on the selector book's trades closed in the last
# ``EXIT_WINDOW_D`` days: the base rule (agent/paper_trading.take_profit_base: x2 on a coin bought far below its high, else
# none) or a sale at +20 / +30 / +50 % on any coin. A line replaces the base rule only when its mean beat the base rule's over
# at least ``EXIT_MIN_TRADES`` trades; each line's return is replayed on the trade's own minute closes (rh_minutes) and keeps
# its real exit's costs as a share, so the line in use never decides its own comparison.
# - Measured on #640's out-of-sample RH book, 08-09..11-02 (data/scratch/regime_20261103, 14 days): +0.136 of the bankroll
#   against the base rule's +1.598, ahead in each period (+0.039 / +0.019 / +0.078) and in every half-month it changed (4 of 4);
#   21 and 28 days ahead too, 7 days behind. On Solana every window loses, so Solana keeps the base rule.
EXIT_WINDOW_D, EXIT_MIN_TRADES, EXIT_LEVELS, EXIT_BOOK = 14, 8, (1.2, 1.3, 1.5), "paper_rh_selector"


def exit_price(path, entry: float, x: float | None, t_end: float) -> float | None:
    """The close a position bought at ``entry`` would have been sold on: the first minute close at or above ``x`` × ``entry``
    (a take-profit, sold on that close as the books do) before ``t_end``, else the last close up to ``t_end`` (its hold);
    None without a close. ``path``: (minute start, close) in time order."""
    last = None
    for t, c in path:
        if t > t_end:
            break
        if c is None or c <= 0:
            continue
        if x and c >= x * entry:
            return c
        last = c
    return last


def exit_means(rows) -> tuple[dict, int]:
    """Each line's mean return over ``rows`` of (realized return, entry price, line used, base line, path, hold end): every
    line replayed on the path, scaled by the real exit (its costs kept as a share)."""
    acc = {"base": [], **{x: [] for x in EXIT_LEVELS}}
    for r, entry, used, base, path, t_end in rows:
        p_used = exit_price(path, entry, used, t_end)
        if not p_used:
            continue
        for k in acc:
            p = exit_price(path, entry, base if k == "base" else k, t_end)
            if p:
                acc[k].append((1.0 + r) * p / p_used - 1.0)
    n = min(len(v) for v in acc.values())
    return {k: sum(v) / len(v) for k, v in acc.items() if v}, n


class ExitChoice:
    """The line new RH buys take (``line``), re-chosen at most every ``REFRESH_S`` in the paper phase."""
    __slots__ = ("x", "means", "n", "at")

    def __init__(self):
        self.x = None; self.means: dict = {}; self.n = 0; self.at = -math.inf

    def refresh(self, conn, now: float) -> None:
        self.at = now; rows = []
        try:
            with conn.transaction():                     # a savepoint: a failed read never aborts the minute's transaction
                pos = conn.execute("""SELECT p.mint, extract(epoch from COALESCE(d.ts, p.opened_at))::float8 AS t0, p.entry_price, p.cost_sol, p.realized_sol,
                                             COALESCE(p.hold_s, (d.detail->>'hold_s')::float, 14400) AS hold_s, (d.detail->>'take_profit_x')::float AS used,
                                             CASE WHEN d.detail ? 'tp_base_x' THEN (d.detail->>'tp_base_x')::float
                                                  ELSE (d.detail->>'take_profit_x')::float END AS base       -- older buys: no choice existed
                                      FROM positions p LEFT JOIN decisions d ON d.id = p.entry_decision_id
                                      WHERE p.book = %s AND p.status = 'closed' AND (p.forced_exit_kind IS NULL OR p.forced_exit_kind = 'take_profit')
                                        AND p.closed_at > to_timestamp(%s) AND p.closed_at <= to_timestamp(%s) AND p.cost_sol > 0 AND p.entry_price > 0
                                        AND p.realized_sol IS NOT NULL""", (EXIT_BOOK, now - EXIT_WINDOW_D * 86400.0, now)).fetchall()
                for p in pos:
                    t_end = p["t0"] + float(p["hold_s"]) - 60.0
                    path = conn.execute("SELECT extract(epoch from ts)::float8 AS t, close FROM rh_minutes WHERE mint = %s AND ts >= to_timestamp(%s) AND ts <= to_timestamp(%s) ORDER BY ts",
                                        (p["mint"], p["t0"], t_end)).fetchall()
                    rows.append((float(p["realized_sol"]) / float(p["cost_sol"]), float(p["entry_price"]), p["used"], p["base"],
                                 [(r["t"], float(r["close"]) if r["close"] is not None else None) for r in path], t_end))
            means, n = exit_means(rows)
        except Exception:
            log.exception("RH take-profit choice: read failed; new buys take the base rule")
            means, n = {}, 0
        was = self.x; self.means, self.n = means, n
        best = max(EXIT_LEVELS, key=lambda k: (round(means.get(k, -math.inf), 9), k)) if means else None   # a tie sells later
        self.x = best if best is not None and n >= EXIT_MIN_TRADES and means.get(best, -math.inf) > means.get("base", math.inf) else None
        if self.x != was:                                # the console's bell shows each change
            msg = (f"RH take-profit: new buys sell at {'+' + format(100 * (self.x - 1), '.0f') + ' % on any coin' if self.x else 'the base rule (x2 on a coin bought far below its high)'}; "
                   f"over the last {EXIT_WINDOW_D} days' {n} trades: " + ", ".join(f"{'base' if k == 'base' else '+' + format(100 * (k - 1), '.0f') + ' %'} {100 * v:+.1f} %" for k, v in means.items()))
            log.info(msg); record_event("info", "regime", msg, self.status())

    def maybe_refresh(self, conn, now: float) -> None:
        if now - self.at >= REFRESH_S:
            self.refresh(conn, now)

    def line(self, base: float | None) -> float | None:
        return self.x if self.x is not None else base

    def status(self) -> dict:
        return {"take_profit_x": self.x, "trades": self.n, "means": {("base" if k == "base" else f"x{k:g}"): v for k, v in self.means.items()}, "window_days": EXIT_WINDOW_D}
