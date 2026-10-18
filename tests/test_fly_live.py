"""The fly on the bot wallet: entries mirrored from the paper book and sized from the wallet, exits retried then written off
as dead-bags, live wealth marked, and entries paused when live trails its paper mirror. Live keeps paper's clock and
decisions: holds count from the decision's minute, a failed entry is retried while paper holds it, a buy queues behind
the sale of its token, a tripped breaker lets entries try again after its cooldown, and every paper entry the wallet did
not take is a decisions row of the live book. Live money first: the sales due go out at the minute boundary, once, and
the mark, the write-offs and the gap check come after the minute's orders."""
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fly_trader import config
from fly_trader.agent import fly_live, live_first, sizing
from fly_trader.chain.balances import Snapshot
from fly_trader.db.connection import transaction
from fly_trader.execution.worker import ExecResult


class _Worker:
    """execution/worker.ExecutionWorker's queue rules, nothing executed: one request per mint, a buy behind a lone sale."""

    def __init__(self):
        self.submitted, self.inflight, self.results, self.busy = [], {}, [], threading.Lock()

    def submit(self, req):
        sides = self.inflight.get(req.mint)
        if sides and not (req.side == "buy" and sides == ["sell"]):
            return False
        self.submitted.append(req); self.inflight.setdefault(req.mint, []).append(req.side); return True

    def pending(self):
        return set(self.inflight)

    def selling(self):
        return {m for m, sides in self.inflight.items() if sides == ["sell"]}

    def drain(self):
        out, self.results = self.results, []
        return out

    def stop(self):
        pass


class _Rpc:
    def __init__(self, lamports, tokens=None):
        self.lamports, self.tokens = lamports, tokens or {}

    def get_balance(self, pk):
        return self.lamports

    def get_token_accounts_by_owner(self, pk):
        return [{"mint": m, "amount": a, "decimals": 6} for m, a in self.tokens.items()]


def _ctx(conn, m1, lag_s=0.0):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), prices={}, resqs={}, fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: None,
                           fee=lambda m: None, lag_s=lag_s)


def _mirror(w):
    return fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)


@pytest.fixture()
def clean():
    def wipe():
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint LIKE 'LIVE_%%'"); conn.execute("DELETE FROM decisions WHERE mint LIKE 'LIVE_%%'")
            conn.execute("DELETE FROM wealth_marks WHERE book = 'live'")
            conn.execute("UPDATE circuit_state SET kill_switch = false, tripped = false, entries_paused = false, fail_count = 0 WHERE id = 1")
    wipe(); yield; wipe()


def test_entries_are_sized_from_the_wallet_and_exits_retry_then_dead_bag(db_conn, clean, monkeypatch):
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3)
    w = _Worker(); mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    m1 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc); table = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.5, "kelly": 0.4}]
    with transaction() as conn:
        d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', 'LIVE_A') RETURNING id").fetchone()["id"]
        for mint, age in (("LIVE_X", 7200 + 60), ("LIVE_D", 7200 + fly_live.DEAD_BAG_S + 60), ("LIVE_Y", 600)):
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status) VALUES ('live', %s, 'P', %s, 1000000, 0.2, 0.0000002, 'open')",
                         (mint, m1 - timedelta(seconds=age)))
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]; run = str(uuid.uuid4())
        out = mirror.minute(_ctx(conn, m1), run_id=run, beat_id=beat, line=0.01, table=table,
                            entries=[{"mint": "LIVE_A", "decision_id": d, "score": 0.03, "info": {"resq": 1000.0, "pool": "PA", "decimals": 6},
                                      "hold_s": 14400.0, "strategy": "capitulation"}])
        unmarked = conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()["n"]
        after = mirror.after(conn, _ctx(conn, m1), run_id=run, beat_id=beat)
        mirror.mark_due(conn)
        dead = conn.execute("SELECT status, forced_exit_kind, realized_sol FROM positions WHERE book = 'live' AND mint = 'LIVE_D'").fetchone()
        mark = conn.execute("SELECT wealth FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()
        targets = conn.execute("SELECT book_targets FROM decisions WHERE id = %s", (d,)).fetchone()["book_targets"]
    sells = [r for r in w.submitted if r.side == "sell"]; buys = [r for r in w.submitted if r.side == "buy"]
    assert [r.mint for r in sells] == ["LIVE_X"] and sells[0].slippage_bps == config.SLIPPAGE_EXIT_BPS and sells[0].amount_in == 1000000
    assert dead["status"] == "closed" and dead["forced_exit_kind"] == "dead_bag" and dead["realized_sol"] == pytest.approx(-0.2)
    bankroll = 5.0 + 0.2 + 0.2                                                     # wallet SOL + open live positions at cost (not the dead-bag)
    expect, _ = sizing.size_position(0.03, 0.01, table, bankroll, 5.0, 1000.0)
    assert len(buys) == 1 and buys[0].mint == "LIVE_A" and buys[0].amount_in == int(expect * config.LAMPORTS_PER_SOL) and buys[0].decision_id == d
    assert (buys[0].strategy, buys[0].hold_s) == ("capitulation", 14400.0)           # booked on the live position (execution/worker.py)
    assert "live" in targets and out["entered"] == 1 and after["dead_bags"] == 1 and mark is not None
    assert unmarked == 0                                                            # the mark is bookkeeping: never in front of the orders


def test_blocked_entries_and_the_gap_pause(db_conn, clean):
    w = _Worker(); mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    m1 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        for i in range(fly_live.GAP_TRADES):
            d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', %s) RETURNING id", (f"LIVE_G{i}",)).fetchone()["id"]
            for book, r in (("live", -0.04), ("paper_fly", 0.0)):
                conn.execute("INSERT INTO positions (book, mint, entry_decision_id, qty, cost_sol, realized_sol, status, closed_at) VALUES (%s,%s,%s,0,0.1,%s,'closed',%s)",
                             (book, f"LIVE_G{i}", d, 0.1 * r, m1 - timedelta(minutes=i)))
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]; run = str(uuid.uuid4())
        mirror.minute(_ctx(conn, m1), run_id=run, beat_id=beat, line=0.01, table=[{"lo": 0.0, "kelly": 0.4, "n": 1, "mean": 0.1, "win": 1}], entries=[])
        gap = mirror.after(conn, _ctx(conn, m1), run_id=run, beat_id=beat)["gap"]
        paused = conn.execute("SELECT entries_paused FROM circuit_state WHERE id = 1").fetchone()["entries_paused"]
        assert gap["paused"] and paused
        d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', 'LIVE_B') RETURNING id").fetchone()["id"]
        out2 = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1000000, line=0.01,
                             table=[{"lo": 0.0, "kelly": 0.4, "n": 1, "mean": 0.1, "win": 1}],
                             entries=[{"mint": "LIVE_B", "decision_id": d, "score": 0.05, "info": {"resq": 1000.0, "pool": "PB", "decimals": 6}}])
        skip = conn.execute("SELECT rail, detail FROM decisions WHERE kind = 'blocked' AND mint = 'LIVE_B'").fetchall()
    assert out2["entered"] == 0 and out2["blocked"] == "paused" and not [r for r in w.submitted if r.side == "buy"]
    assert [(r["rail"], r["detail"]["book"], r["detail"]["of_decision"]) for r in skip] == [("paused", "live", d)]     # the divergence is on record


def test_a_tripped_kill_switch_liquidates_when_asked_and_only_then(db_conn, clean, monkeypatch):
    """The kill switch blocked entries and kept holding. With KILL_SWITCH_LIQUIDATE every open live position is sold at the
    forced slippage from the boundary after the mark tripped it, retried while it stays on; a position already in flight
    is left to the worker."""
    w = _Worker(); mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    m1 = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        for mint, age in (("LIVE_K1", 600), ("LIVE_K2", 7200 + 60)):                    # one young, one due for its scheduled exit
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status) VALUES ('live', %s, 'P', %s, 1000000, 0.2, 0.0000002, 'open')",
                         (mint, m1 - timedelta(seconds=age)))
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) "
                     "VALUES (%s, 'live', %s, 100, 0, 0, 100, 100, 0, 0, 0)", (beat + 500000, m1 - timedelta(minutes=5)))   # a peak far above today's 5 SOL
        monkeypatch.setattr(config, "KILL_SWITCH_LIQUIDATE", False)
        out = mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=[], entries=[])
        assert mirror.mark_due(conn)["killed"]                                     # the minute's mark, after its orders, trips it
        killed = conn.execute("SELECT kill_switch FROM circuit_state WHERE id = 1").fetchone()["kill_switch"]
    assert killed and out["liquidated"] == 0 and [r.mint for r in w.submitted] == ["LIVE_K2"]      # default: halt only, the scheduled exit alone
    w2 = _Worker(); mirror.worker = w2
    with transaction() as conn:
        monkeypatch.setattr(config, "KILL_SWITCH_LIQUIDATE", True)
        out = mirror.exits(conn, m1=m1 + timedelta(minutes=1), run_id=str(uuid.uuid4()))       # the next boundary
        reasons = {r["mint"]: r["reason"] for r in conn.execute("SELECT mint, reason FROM decisions WHERE mint LIKE 'LIVE_K%%' AND kind = 'fly_exit'").fetchall()}
    sells = {r.mint: r for r in w2.submitted}
    assert out["liquidated"] == 1 and set(sells) == {"LIVE_K1", "LIVE_K2"}
    assert sells["LIVE_K1"].slippage_bps == config.SLIPPAGE_FORCED_BPS and reasons["LIVE_K1"] == "kill switch: liquidating"
    assert sells["LIVE_K2"].slippage_bps == config.SLIPPAGE_EXIT_BPS                                  # the scheduled exit went first; not re-submitted


def test_the_handover_gate_reads_its_thresholds_from_config(db_conn, monkeypatch):
    """HANDOVER_DAYS=0 counts the whole race (a zero-length window would count nothing), HANDOVER_MIN_TRADES is the trade
    count, and HANDOVER_BEAT_SELECTOR=0 drops the race against the paper selector."""
    from fly_trader.agent import fly_session, handover
    now = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc); race = now - timedelta(days=2)
    from fly_trader import markets
    stub = SimpleNamespace(race_started_at=race.timestamp(), boot_id=62, chain="sol", market=markets.SOL, BOOK="paper_fly")
    with transaction() as conn:
        conn.execute("DELETE FROM ui_settings WHERE key = %s", (handover.KEY,)); conn.execute("DELETE FROM positions WHERE mint LIKE 'HAND_%%'")
        for i in range(20):
            conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, realized_sol, status, closed_at) VALUES (%s, %s, 0, 0.1, -0.01, 'closed', %s)",
                         (fly_session.BOOK, f"HAND_{i}", race + timedelta(hours=i)))                   # 20 losing trades: no selector to beat
        try:
            monkeypatch.setattr(config, "HANDOVER_DAYS", 14); monkeypatch.setattr(config, "HANDOVER_MIN_TRADES", 20); monkeypatch.setattr(config, "HANDOVER_BEAT_SELECTOR", True)
            fly_session.FlyBook._handover_check(stub, conn, now.timestamp())
            assert handover.state(conn) is None                                                     # two days into a 14-day race
            monkeypatch.setattr(config, "HANDOVER_DAYS", 0)
            fly_session.FlyBook._handover_check(stub, conn, now.timestamp())
            assert handover.state(conn) is None                                                     # loses to a flat selector
            monkeypatch.setattr(config, "HANDOVER_BEAT_SELECTOR", False)
            fly_session.FlyBook._handover_check(stub, conn, now.timestamp())
            st = handover.state(conn)
            assert st and st["fly_trades"] == 20 and st["days"] == 0 and st["beat_selector"] is False
        finally:
            conn.execute("DELETE FROM ui_settings WHERE key = %s", (handover.KEY,)); conn.execute("DELETE FROM positions WHERE mint LIKE 'HAND_%%'")


def test_live_calibration_spends_the_teachers_trade_budget(monkeypatch):
    """Live, the match rule sizes each day's line to the trades the teacher would have taken over the same resolved rows;
    rows scored before the teacher's decision was recorded cannot count."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import numpy as np
    from fly_trader.agent import fly_session as fs
    rng = np.random.default_rng(0); n = 400; t0 = 1_790_000_000.0
    rows = [{"ts": datetime.fromtimestamp(t0 + 60.0 * i, timezone.utc), "mint": f"m{i % 150}", "score": float(s), "frozen_score": float(s),
             "label": float(rng.normal(0, 0.1)), "teacher_allow": (None if i < 50 else bool(s > 1.5))} for i, s in enumerate(rng.normal(size=n))]
    calls = []
    real = fs.fly_calibrate.calibrate
    monkeypatch.setattr(fs.fly_calibrate, "RULE", "match")
    monkeypatch.setattr(fs.fly_calibrate, "calibrate", lambda *a, **kw: calls.append((len(a[0]), kw)) or real(*a, **kw))
    monkeypatch.setattr(fs, "record_event", lambda *a, **k: None)
    conn = SimpleNamespace(execute=lambda *a, **k: None)
    me = SimpleNamespace(chain="sol", names=["capitulation"], holds={"capitulation": 4 * 3600.0}, lines={"plastic": {"capitulation": 0.02}, "frozen": {"capitulation": 0.02}},
                         sizing={"plastic": {"capitulation": []}, "frozen": {"capitulation": []}}, _resolved=lambda conn, since, cols, name: rows)
    fs.FlyBook._calibrate(me, conn, t0 + 86400.0)
    known = [r for r in rows if r["teacher_allow"] is not None]
    ts = np.array([r["ts"].timestamp() for r in known]); mint = np.array([r["mint"] for r in known])
    want = len(fs.fly_calibrate.taken_idx(ts, mint, 4 * 3600.0, np.flatnonzero([r["teacher_allow"] for r in known])))
    assert calls and all(c[0] == len(known) and c[1]["rule"] == "match" and c[1]["target_trades"] == want for c in calls)
    assert me.lines["plastic"]["capitulation"] > 1.0                                    # as selective as the teacher (scores above 1.5)


def test_the_live_fly_reads_the_stack_that_taught_it(monkeypatch):
    from types import SimpleNamespace

    from fly_trader.agent import fly_session as fs
    ev = {"cols": ["age_h"] if "age_h" in fs.X_COLS else [fs.X_COLS[0]], "line": 0.1, "hold_min": 240, "thr": {}, "high": None}
    deployed, blind = {"strategies": {"capitulation": ev}}, {"strategies": {"capitulation": {**ev, "line": 0.2}}}
    monkeypatch.setattr(fs.selector, "load_snapshot", lambda i: SimpleNamespace(stack=deployed, blind={"models": blind}))
    me = SimpleNamespace(boot_id=133)
    pick = lambda note, sid=129: fs.FlyBook._teacher_models(me, {"meta": {"teacher": note, "teacher_snapshot": sid}})
    assert pick("deployed selector #129: capitulation, refit through 2026-09-03 (blind to the calibration week from 2026-09-18)") is blind
    assert pick("deployed selector #100: capitulation (line 0.1157, 240 min)") is deployed
    assert pick("a stack refit on the fly's training days", None) is None


def _entry(mint, did, pool="P", ref_price=None):
    return {"mint": mint, "decision_id": did, "score": 0.03, "info": {"resq": 1000.0, "pool": pool, "decimals": 6}, "ref_price": ref_price}


TABLE = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.5, "kelly": 0.4}]


def test_paper_sells_and_buys_back_in_one_minute_and_so_does_live(db_conn, clean):
    """The live position filled 15 s after its decision: its hold runs from the decision's minute, so it is sold in the
    minute paper sells its twin; paper buys the token back in that minute, and the live buy queues behind the sale."""
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        d0 = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_Q', %s) RETURNING id", (m1 - timedelta(seconds=7200),)).fetchone()["id"]
        conn.execute("INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, status) "
                     "VALUES ('live', 'LIVE_Q', 'PQ', %s, %s, 1000000, 0.2, 0.0000002, 'open')", (m1 - timedelta(seconds=7185), d0))
        d1 = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_Q', %s) RETURNING id", (m1,)).fetchone()["id"]
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        out = mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[_entry("LIVE_Q", d1, "PQ", 2.1e-7)])
    assert [(r.side, r.mint) for r in w.submitted] == [("sell", "LIVE_Q"), ("buy", "LIVE_Q")] and out["entered"] == 1 and out["exited"] == 1
    assert w.submitted[1].decision_id == d1 and w.submitted[1].ref_price == 2.1e-7                      # the paper fill it may not chase past


def test_a_failed_entry_is_retried_while_paper_holds_it_then_given_up(db_conn, clean, monkeypatch):
    monkeypatch.setattr(config, "ENTRY_RETRY_MIN", 2.0)
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc); run = str(uuid.uuid4())
    with transaction() as conn:
        d = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_R', %s) RETURNING id", (m1,)).fetchone()["id"]
        conn.execute("INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, status) "
                     "VALUES ('paper_fly', 'LIVE_R', 'PR', %s, %s, 1000000, 0.2, 0.0000002, 'open')", (m1, d))
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        mirror.minute(_ctx(conn, m1), run_id=run, beat_id=beat, line=0.01, table=TABLE, entries=[_entry("LIVE_R", d, "PR", 2.1e-7)])
        first = w.submitted[-1]
        w.inflight.clear(); w.results.append(ExecResult(first, False, error="price ran: quoted 2.3e-07 SOL per token", retryable=True))
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=run, beat_id=beat + 1000001, line=0.01, table=TABLE, entries=[])
        again = w.submitted[-1]
        assert again is not first and (again.side, again.decision_id, again.ref_price) == ("buy", d, 2.1e-7) and out["retried"] == 1
        w.inflight.clear(); w.results.append(ExecResult(again, False, error="Insufficient funds", retryable=True))
        conn.execute("UPDATE circuit_state SET kill_switch = true WHERE id = 1")             # every rail still applies to a retry
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=2)), run_id=run, beat_id=beat + 1000002, line=0.01, table=TABLE, entries=[])
        assert out["retried"] == 0 and w.submitted[-1] is again
        conn.execute("UPDATE circuit_state SET kill_switch = false WHERE id = 1")
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=3)), run_id=run, beat_id=beat + 1000003, line=0.01, table=TABLE, entries=[])
        rails = [r["rail"] for r in conn.execute("SELECT rail FROM decisions WHERE kind = 'blocked' AND mint = 'LIVE_R' ORDER BY id").fetchall()]
    assert out["retried"] == 0 and w.submitted[-1] is again and len(mirror.failed) == 0
    assert rails == ["price ran", "entry failed", "kill switch", "retries exhausted"]


def test_a_failed_entry_whose_paper_position_closed_is_not_retried(db_conn, clean):
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        d = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_S', %s) RETURNING id", (m1,)).fetchone()["id"]
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[_entry("LIVE_S", d)])
        w.inflight.clear(); w.results.append(ExecResult(w.submitted[-1], False, error="order_error", retryable=True))
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1000001, line=0.01, table=TABLE, entries=[])
        why = conn.execute("SELECT reason FROM decisions WHERE kind = 'blocked' AND mint = 'LIVE_S' AND rail = 'retries exhausted'").fetchone()["reason"]
    assert out["retried"] == 0 and len(w.submitted) == 1 and why.startswith("its paper position is closed")


def test_a_tripped_breaker_lets_entries_try_again_after_its_cooldown(db_conn, clean, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_COOLDOWN_S", 900.0)
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        d = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_T', %s) RETURNING id", (m1,)).fetchone()["id"]
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        conn.execute("UPDATE circuit_state SET tripped = true, fail_count = 3, last_failure_ts = now() - interval '1 minute' WHERE id = 1")
        out = mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[_entry("LIVE_T", d)])
        assert out["blocked"] == "circuit tripped" and out["entered"] == 0
        conn.execute("UPDATE circuit_state SET last_failure_ts = now() - interval '20 minutes' WHERE id = 1")
        d2 = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'LIVE_T2', %s) RETURNING id", (m1,)).fetchone()["id"]
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1000001, line=0.01, table=TABLE, entries=[_entry("LIVE_T2", d2)])
        conn.execute("UPDATE circuit_state SET last_failure_ts = NULL WHERE id = 1")
    assert out["blocked"] is None and out["entered"] == 1 and w.submitted[-1].mint == "LIVE_T2"


def test_the_wealth_mark_waits_for_an_idle_worker_and_never_reads_beside_a_booking(db_conn, clean, monkeypatch):
    """The mark is bookkeeping: taken between minutes once no order is queued or in flight. A wallet read beside a booking
    in progress counts that trade twice or not at all: that minute goes unmarked."""
    monkeypatch.setattr(fly_live, "MARK_WAIT_S", 0.01)
    monkeypatch.setattr(live_first, "may_mark", lambda now=None: True); monkeypatch.setattr(live_first, "mid_minute", lambda now=None: False)
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[])
        w.busy.acquire()
        try:
            assert mirror.mark_due(conn) is None and mirror.due_mark is None
        finally:
            w.busy.release()
        none = conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()["n"]
        mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1000001, line=0.01, table=TABLE, entries=[])
        w.inflight["LIVE_M"] = ["sell"]
        mirror.idle()                                                              # an order in flight: no wallet read yet
        waiting = mirror.due_mark is not None
        w.inflight.clear()
        mark = mirror.mark_due(conn)
    assert none == 0 and waiting and mark["wealth"] == pytest.approx(5.0) and mirror.last_mark == mark


def test_the_sales_due_go_out_at_the_boundary_once(db_conn, clean):
    """Live money first: the clock sends the sales due at the boundary, before the stream has written the minute; the
    minute's own pass seconds later sends none twice; a dead-bag is left to the bookkeeping; a sale that failed is sent
    again at the next boundary."""
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc); run = str(uuid.uuid4())
    with transaction() as conn:
        for mint, age in (("LIVE_C1", 7200), ("LIVE_C2", 600), ("LIVE_C3", 7200 + fly_live.DEAD_BAG_S + 60)):
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status) VALUES ('live', %s, 'P', %s, 1000000, 0.2, 0.0000002, 'open')",
                         (mint, m1 - timedelta(seconds=age)))
        first = mirror.exits(conn, m1=m1, run_id=run)
        w.inflight.clear()                                                         # its sale failed and left the queue
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        out = mirror.minute(_ctx(conn, m1), run_id=run, beat_id=beat, line=0.01, table=TABLE, entries=[])
        dead = mirror.after(conn, _ctx(conn, m1), run_id=run, beat_id=beat)["dead_bags"]
        again = mirror.exits(conn, m1=m1 + timedelta(minutes=1), run_id=run)
        rows = conn.execute("SELECT mint, beat_id FROM decisions WHERE mint LIKE 'LIVE_C%%' AND kind = 'fly_exit' ORDER BY id").fetchall()
    assert first == {"exited": 1, "liquidated": 0} and out["exited"] == 0 and dead == 1 and again["exited"] == 1
    assert [r.mint for r in w.submitted] == ["LIVE_C1", "LIVE_C1"]
    assert [(r["mint"], r["beat_id"]) for r in rows] == [("LIVE_C1", None), ("LIVE_C3", beat), ("LIVE_C1", None)]


def test_a_chains_boundary_sales_go_out_only_while_its_wallet_trades(db_conn, monkeypatch):
    """The clock's sales need no fly book: the chain's mirror is made on demand (its book may still be loading)."""
    calls = []
    stub = SimpleNamespace(exits=lambda conn, m1, run_id: calls.append((m1, run_id)) or {"exited": 1})
    monkeypatch.setattr(live_first, "mirror", lambda chain, horizon_s=None: stub)
    monkeypatch.setattr(live_first, "signing_refused", lambda chain: None); monkeypatch.setattr(live_first, "trades_live", lambda conn, chain, live: True)
    assert live_first.chain_exits("sol", 1_800_000_000.0, True, "R") == {"exited": 1}
    assert calls == [(datetime.fromtimestamp(1_800_000_000.0, timezone.utc), "R")]
    assert live_first.chain_exits("sol", 1_800_000_060.0, False) is None                          # the runner may not sign
    monkeypatch.setattr(live_first, "trades_live", lambda conn, chain, live: False)
    assert live_first.chain_exits("sol", 1_800_000_060.0, True) is None                           # no seat, no LIVE_WITHOUT_HANDOVER
    monkeypatch.setattr(live_first, "signing_refused", lambda chain: "LIVE_ENABLED=1")
    assert live_first.chain_exits("rh", 1_800_000_060.0, True) is None and len(calls) == 1


def test_the_clock_sends_every_live_chain_and_survives_a_failure(monkeypatch):
    monkeypatch.setattr(live_first, "record_event", lambda *a, **k: None)
    calls = []

    def exits(chain, m1, live, run_id=None):
        calls.append((chain, m1, live, run_id))
        if chain == "rh":
            raise RuntimeError("rpc down")
        return {"exited": 1}
    monkeypatch.setattr(live_first, "chain_exits", exits)
    books = [SimpleNamespace(name="selector", chain="sol", run_id="S"), SimpleNamespace(name="fly", LIVE=True, chain="sol", run_id="F")]
    live_first.LiveClock(books, ["rh", "sol"], True).tick(1_800_000_000.0)                    # the RH fly's book has not loaded yet
    assert calls == [("rh", 1_800_000_000.0, True, None), ("sol", 1_800_000_000.0, True, "F")]
    assert live_first.may_mark(600.0 + 10) and not live_first.may_mark(600.0 + 58)
    assert live_first.mid_minute(600.0 + 25) and not live_first.mid_minute(600.0 + 5) and not live_first.mid_minute(600.0 + 45)


def test_one_mirror_per_chain_whoever_makes_it_first_and_the_runner_stops_it(monkeypatch):
    monkeypatch.setattr(live_first, "record_event", lambda *a, **k: None)
    made = []

    class Fake:
        def __init__(self, horizon_s=None):
            made.append(self); self.H, self.stopped = horizon_s, False

        def stop(self):
            self.stopped = True
    monkeypatch.setattr(fly_live, "LiveMirror", Fake)
    live_first.stop_mirrors()
    got, go = [], threading.Event()

    def make():
        go.wait(5); got.append(live_first.mirror("sol"))
    ts = [threading.Thread(target=make) for _ in range(4)]
    for t in ts:
        t.start()
    go.set()
    for t in ts:
        t.join(5)
    assert len(made) == 1 and all(g is made[0] for g in got) and made[0].H is None          # the clock's: no hold yet
    assert live_first.mirror("sol", 14400.0) is made[0] and made[0].H == 14400.0            # the fly's first live minute sets it
    live_first.stop_mirrors()
    assert made[0].stopped and live_first.mirror_of("sol") is None


def test_before_the_fly_runs_only_positions_with_their_own_hold_are_sold(db_conn, clean):
    w = _Worker(); m1 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
    mirror = fly_live.LiveMirror(None, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    with transaction() as conn:
        for mint, hold in (("LIVE_H1", 600.0), ("LIVE_H2", None)):
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status, hold_s) VALUES ('live', %s, 'P', %s, 1000000, 0.2, 0.0000002, 'open', %s)",
                         (mint, m1 - timedelta(seconds=7200), hold))
        first = mirror.exits(conn, m1=m1, run_id=None)
        mirror.H = 7200.0
        second = mirror.exits(conn, m1=m1 + timedelta(minutes=1), run_id=None)
    assert first["exited"] == 1 and second["exited"] == 1 and [r.mint for r in w.submitted] == ["LIVE_H1", "LIVE_H2"]


def test_live_wealth_counts_the_token_accounts_deposits_until_their_close_returns_them(db_conn, clean, monkeypatch):
    monkeypatch.setattr(live_first, "may_mark", lambda now=None: True)
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status, deposit_sol) VALUES ('live', 'LIVE_DEP', 'P', %s, 0, 0.1, 0.0000002, 'closed', 0.0015)",
                     (m1,))                                                      # sold; its token account not closed yet
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[])
        mark = mirror.mark_due(conn)
        row = conn.execute("SELECT sol_free, positions_value FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()
    assert mark["wealth"] == pytest.approx(5.0 + 0.0015) and row["sol_free"] == pytest.approx(5.0) and row["positions_value"] == pytest.approx(0.0015)


def test_a_booked_order_marks_the_wallet_at_once_and_tells_the_console(db_conn, clean, monkeypatch):
    """The live stats come before anything else: once an order is booked and none waits, the latest minute is marked
    again on the wallet as it now stands (its row updated) and the listeners -- the console's panels -- are told."""
    monkeypatch.setattr(live_first, "may_mark", lambda now=None: True)
    w = _Worker(); mirror = _mirror(w); m1 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc); told = []
    assert w.on_done == mirror._after_order
    live_first.on_change(lambda: told.append(1))
    try:
        with transaction() as conn:
            beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
            mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[])
        w.inflight["LIVE_W"] = ["buy"]                                                # an order still queued: no mark yet
        mirror._after_order(None)
        with transaction() as conn:
            none = conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()["n"]
        w.inflight.clear(); mirror.broker.rpc.lamports = 5_200_000_000                # the sale landed: 0.2 SOL more in the wallet
        mirror._after_order(None)
        with transaction() as conn:
            first = conn.execute("SELECT wealth FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()["wealth"]
        mirror.broker.rpc.lamports = 5_300_000_000
        mirror._after_order(None)                                                      # the same minute, marked again
        with transaction() as conn:
            rows = conn.execute("SELECT wealth FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchall()
    finally:
        live_first._LISTENERS.clear()
    assert none == 0 and first == pytest.approx(5.2) and [r["wealth"] for r in rows] == [pytest.approx(5.3)] and len(told) == 3

