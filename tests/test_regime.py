"""Quiet bar, old coin (agent/regime.py): the trigger counts the selector's young buys (one per coin per hold), switches on
when the last 3 days fall far enough below the last 30, stands aside without history; the rule keeps old Solana coins off the
books' buy rows only while it is on, and never touches Robinhood Chain."""
import math
from datetime import datetime, timezone

from fly_trader.agent import regime

DAY = 86400.0
NOW = 1_800_000_000.0


def _young(mint, t):
    return (mint, t, t - 3600.0)                                 # graduated an hour before the decision


def test_one_buy_per_coin_per_hold_and_only_young_coins_count():
    rows = [_young("A", NOW - 20 * DAY)]                          # history reaches 20 days back
    rows += [_young("B", NOW - 5 * 3600.0), _young("B", NOW - 5 * 3600.0 + 600.0), _young("B", NOW - 3600.0 - 60.0)]   # one buy: inside one hold
    rows += [_young("C", NOW - 10 * 3600.0), _young("C", NOW - 5 * 3600.0)]                                          # two buys: a hold apart
    rows += [("D", NOW - 7200.0, NOW - 30 * 3600.0)]              # 28 h past graduation: an old coin, not counted
    short, long_, ratio = regime.pass_rates(rows, NOW)
    assert short == 3 / regime.SHORT_D
    assert math.isclose(long_, 4 / 20.0) and math.isclose(ratio, math.log((short + 0.5) / (long_ + 0.5)))


def test_too_little_history_stands_aside():
    assert regime.pass_rates([_young("A", NOW - 13 * DAY), _young("B", NOW - 3600.0)], NOW) is None
    assert regime.pass_rates([], NOW) is None
    assert regime.pass_rates([("A", NOW - 3600.0, None)], NOW) is None         # graduation unknown: not a young coin


def test_the_rule_blocks_old_coins_only_while_quiet():
    q = regime.QuietBar()
    assert not q.blocks({"age_h": 100.0})                         # never read: not quiet
    q.quiet = True
    assert q.blocks({"age_h": regime.OLD_H}) and q.blocks({"age_h": 153.6})
    assert not q.blocks({"age_h": regime.OLD_H - 0.01}) and not q.blocks({"age_h": None})


def _scored(conn, rows):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fly_scored WHERE mint LIKE 'QB%%pump'"); cur.execute("DELETE FROM corpus_meta WHERE mint LIKE 'QB%%pump'")
        for mint, t, g in rows:
            cur.execute("INSERT INTO corpus_meta (mint, graduated_at) VALUES (%s, %s) ON CONFLICT (mint) DO NOTHING", (mint, datetime.fromtimestamp(g, timezone.utc)))
            cur.execute("INSERT INTO fly_scored (ts, mint, strategy, teacher_allow, chain) VALUES (%s, %s, 'ev', true, 'sol')", (datetime.fromtimestamp(t, timezone.utc), mint))
    conn.commit()


def test_the_trigger_reads_the_selectors_allowed_buys(db_conn):
    """30 days at four young buys a day, the last three at one a day: quiet. Then busy again: not quiet."""
    rows = [(f"QB{d}x{k}pump", NOW - d * DAY - (k + 1) * 3600.0, NOW - d * DAY - (k + 2) * 3600.0) for d in range(3, 30) for k in range(4)]
    rows += [(f"QB{d}x0pump", NOW - d * DAY - 3600.0, NOW - d * DAY - 7200.0) for d in range(3)]
    _scored(db_conn, rows)
    q = regime.QuietBar()
    try:
        q.refresh(db_conn, NOW)
        assert q.quiet and q.short == 1.0 and q.ratio <= regime.QUIET_LOG
        _scored(db_conn, rows + [(f"QB{d}x{k}pump", NOW - d * DAY - k * 3600.0 - 1800.0, NOW - d * DAY - k * 3600.0 - 5400.0) for d in range(3) for k in range(1, 6)])
        q.refresh(db_conn, NOW)
        assert not q.quiet and q.short == 6.0
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM fly_scored WHERE mint LIKE 'QB%%pump'"); cur.execute("DELETE FROM corpus_meta WHERE mint LIKE 'QB%%pump'")
        db_conn.commit()


def test_the_engine_keeps_old_coins_off_solanas_buy_rows_while_quiet():
    from fly_trader import markets
    from fly_trader.agent import minute_engine as me
    e = me.MinuteEngine(); base = {"broken": False, "resq": 100.0, "logvol_15m": 100.0, "creator_held": 0.0}
    assert e.eligible({**base, "age_h": 30.0})
    e.quiet.quiet = True
    assert not e.eligible({**base, "age_h": 30.0}) and e.eligible({**base, "age_h": 8.0})
    rh = me.MinuteEngine(market=markets.RH)
    assert rh.quiet is None and rh.eligible({**base, "age_h": 30.0, "source": "pons", "seen_h": 10.0})


def test_a_failed_read_stands_aside(db_conn, monkeypatch):
    q = regime.QuietBar(); q.quiet = True
    monkeypatch.setattr(regime, "pass_rates", lambda rows, now: (_ for _ in ()).throw(RuntimeError("boom")))
    q.refresh(db_conn, NOW)
    assert not q.quiet and q.ratio is None


# ---------------------------------------------------------------- Robinhood Chain: the take-profit its recent trades favour
def _path(t0, closes):
    return [(t0 + 60.0 * k, c) for k, c in enumerate(closes)]


def test_a_line_is_sold_on_its_first_close_at_or_above_it_else_the_hold_ends():
    p = _path(0.0, [1.0, 1.25, 1.1, 1.6, 0.9])
    assert regime.exit_price(p, 1.0, 1.2, 1e9) == 1.25 and regime.exit_price(p, 1.0, 1.5, 1e9) == 1.6
    assert regime.exit_price(p, 1.0, None, 1e9) == 0.9 and regime.exit_price(p, 1.0, 2.0, 1e9) == 0.9
    assert regime.exit_price(p, 1.0, 1.5, 120.0) == 1.1                         # the hold ended before the line
    assert regime.exit_price([], 1.0, 1.2, 1e9) is None


def test_every_line_is_replayed_on_the_path_and_keeps_the_real_exits_costs():
    """A trade that rose to +40 % and fell back to -20 %: held it lost 25 % after costs; +20 % and +30 % lines would have sold
    on the way up, +50 % never filled. The line in use only scales the costs: the comparison is the same whichever it was."""
    p = _path(0.0, [1.0, 1.25, 1.4, 1.1, 0.8])
    for used in (None, 1.3):
        r_real = -0.25 if used is None else (1 - 0.25) * 1.4 / 0.8 - 1
        means, n = regime.exit_means([(r_real, 1.0, used, None, p, 1e9)])
        assert n == 1 and math.isclose(means["base"], -0.25) and math.isclose(means[1.2], 0.75 * 1.25 / 0.8 - 1)
        assert math.isclose(means[1.3], 0.75 * 1.4 / 0.8 - 1) and math.isclose(means[1.5], -0.25)


def _rh_trade(conn, mint, t0, closes, realized, used, base="absent"):
    from psycopg.types.json import Jsonb
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rh_minutes WHERE mint = %s", (mint,))
        for k, c in enumerate(closes):
            cur.execute("INSERT INTO rh_minutes (mint, ts, close) VALUES (%s, %s, %s)", (mint, datetime.fromtimestamp(t0 + 60.0 * k, timezone.utc), c))
        d = cur.execute("INSERT INTO decisions (ts, kind, mint, detail) VALUES (%s, 'selector_enter', %s, %s) RETURNING id",
                        (datetime.fromtimestamp(t0, timezone.utc), mint, Jsonb({"hold_s": 14400.0, "take_profit_x": used, **({"tp_base_x": base} if base != "absent" else {})}))).fetchone()["id"]
        cur.execute("""INSERT INTO positions (book, mint, opened_at, closed_at, qty, cost_sol, entry_price, realized_sol, status, chain, entry_decision_id, hold_s)
                       VALUES ('paper_rh_selector', %s, %s, %s, 1, 0.01, 1.0, %s, 'closed', 'rh', %s, 14400)""",
                    (mint, datetime.fromtimestamp(t0, timezone.utc), datetime.fromtimestamp(t0 + 14400.0, timezone.utc), 0.01 * realized, d))
    conn.commit()


def _rh_clear(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE book = 'paper_rh_selector' AND mint LIKE '0xRG%%'")
        cur.execute("DELETE FROM decisions WHERE mint LIKE '0xRG%%'"); cur.execute("DELETE FROM rh_minutes WHERE mint LIKE '0xRG%%'")
    conn.commit()


def test_new_rh_buys_take_the_line_that_beat_the_base_rule(db_conn):
    """Ten pops that faded (up through +25 / +35 / +60 %, then -30 %), each sale costing 2 %: +50 % beat holding. With seven
    trades nothing changes; the line the book used (the chosen one, for the last three) never biases the next choice."""
    _rh_clear(db_conn); q = regime.ExitChoice(); closes = [1.0] + [1.1] * 30 + [1.25] * 5 + [1.35] * 5 + [1.6] + [1.2] * 30 + [0.7] * 168
    try:
        for k in range(7):
            _rh_trade(db_conn, f"0xRG{k}", NOW - (k + 1) * DAY, closes, realized=0.7 * 0.98 - 1, used=None)
        q.refresh(db_conn, NOW)
        assert q.x is None and q.n == 7 and q.line(2.0) == 2.0
        for k in range(7, 10):
            _rh_trade(db_conn, f"0xRG{k}", NOW - (k + 1) * DAY, closes, realized=1.6 * 0.98 - 1, used=1.5, base=None)
        q.refresh(db_conn, NOW)
        assert q.n == 10 and q.x == 1.5 and q.line(None) == 1.5 and q.line(2.0) == 1.5
        assert math.isclose(q.means[1.5], 1.6 * 0.98 - 1) and math.isclose(q.means[1.2], 1.25 * 0.98 - 1) and math.isclose(q.means["base"], 0.7 * 0.98 - 1)
    finally:
        _rh_clear(db_conn)


def test_the_take_profit_line_is_the_choice_on_rh_and_the_base_rule_on_solana():
    from types import SimpleNamespace
    from fly_trader import markets
    from fly_trader.agent import minute_engine as me, paper_trading as pt
    rh = me.MinuteEngine(market=markets.RH); sol = me.MinuteEngine()
    assert sol.exits is None and rh.exits is not None
    ctx = lambda e, hi: SimpleNamespace(engine=e, high=lambda m: hi)
    assert pt.take_profit_x(ctx(rh, 3.0), markets.RH, "m", 1.0) == 2.0 and pt.take_profit_x(ctx(rh, 1.2), markets.RH, "m", 1.0) is None
    rh.exits.x = 1.3
    assert pt.take_profit_x(ctx(rh, 1.2), markets.RH, "m", 1.0) == 1.3 and pt.take_profit_base(ctx(rh, 1.2), markets.RH, "m", 1.0) is None
    assert pt.take_profit_x(ctx(sol, 3.0), markets.SOL, "m", 1.0) == 2.0
