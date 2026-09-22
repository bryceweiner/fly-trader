"""The fly on the bot wallet: entries mirrored from the paper book and sized from the wallet, exits retried then written off
as dead-bags, live wealth marked, and entries paused when live trails its paper mirror."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fly_trader import config
from fly_trader.agent import fly_live, sizing
from fly_trader.chain.balances import Snapshot
from fly_trader.db.connection import transaction


class _Worker:
    def __init__(self):
        self.submitted, self.inflight = [], set()

    def submit(self, req):
        self.submitted.append(req); self.inflight.add(req.mint); return True

    def pending(self):
        return set(self.inflight)

    def drain(self):
        return []

    def stop(self):
        pass


class _Rpc:
    def __init__(self, lamports, tokens=None):
        self.lamports, self.tokens = lamports, tokens or {}

    def get_balance(self, pk):
        return self.lamports

    def get_token_accounts_by_owner(self, pk):
        return [{"mint": m, "amount": a, "decimals": 6} for m, a in self.tokens.items()]


def _ctx(conn, m1):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), prices={}, resqs={}, fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: None)


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
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        out = mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=table,
                            entries=[{"mint": "LIVE_A", "decision_id": d, "score": 0.03, "info": {"resq": 1000.0, "pool": "PA", "decimals": 6}}])
        dead = conn.execute("SELECT status, forced_exit_kind, realized_sol FROM positions WHERE book = 'live' AND mint = 'LIVE_D'").fetchone()
        mark = conn.execute("SELECT wealth FROM wealth_marks WHERE book = 'live' AND beat_id = %s", (beat,)).fetchone()
        targets = conn.execute("SELECT book_targets FROM decisions WHERE id = %s", (d,)).fetchone()["book_targets"]
    sells = [r for r in w.submitted if r.side == "sell"]; buys = [r for r in w.submitted if r.side == "buy"]
    assert [r.mint for r in sells] == ["LIVE_X"] and sells[0].slippage_bps == config.SLIPPAGE_EXIT_BPS and sells[0].amount_in == 1000000
    assert dead["status"] == "closed" and dead["forced_exit_kind"] == "dead_bag" and dead["realized_sol"] == pytest.approx(-0.2)
    bankroll = 5.0 + 0.2 + 0.2                                                     # wallet SOL + open live positions at cost (the dead-bag is gone)
    expect, _ = sizing.size_position(0.03, 0.01, table, bankroll, 5.0, 1000.0)
    assert len(buys) == 1 and buys[0].mint == "LIVE_A" and buys[0].amount_in == int(expect * config.LAMPORTS_PER_SOL) and buys[0].decision_id == d
    assert "live" in targets and out["entered"] == 1 and out["dead_bags"] == 1 and mark is not None


def test_blocked_entries_and_the_gap_pause(db_conn, clean):
    w = _Worker(); mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    m1 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        for i in range(fly_live.GAP_TRADES):
            d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', %s) RETURNING id", (f"LIVE_G{i}",)).fetchone()["id"]
            for book, r in (("live", -0.04), ("paper_fly", 0.0)):
                conn.execute("INSERT INTO positions (book, mint, entry_decision_id, qty, cost_sol, realized_sol, status, closed_at) VALUES (%s,%s,%s,0,0.1,%s,'closed',%s)",
                             (book, f"LIVE_G{i}", d, 0.1 * r, m1 - timedelta(minutes=i)))
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        out = mirror.minute(_ctx(conn, m1), run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=[{"lo": 0.0, "kelly": 0.4, "n": 1, "mean": 0.1, "win": 1}], entries=[])
        paused = conn.execute("SELECT entries_paused FROM circuit_state WHERE id = 1").fetchone()["entries_paused"]
        assert out["gap"]["paused"] and paused
        d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', 'LIVE_B') RETURNING id").fetchone()["id"]
        out2 = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1000000, line=0.01,
                             table=[{"lo": 0.0, "kelly": 0.4, "n": 1, "mean": 0.1, "win": 1}],
                             entries=[{"mint": "LIVE_B", "decision_id": d, "score": 0.05, "info": {"resq": 1000.0, "pool": "PB", "decimals": 6}}])
    assert out2["entered"] == 0 and out2["blocked"] == "paused" and not [r for r in w.submitted if r.side == "buy"]


def test_a_tripped_kill_switch_liquidates_when_asked_and_only_then(db_conn, clean, monkeypatch):
    """The kill switch blocked entries and kept holding. With KILL_SWITCH_LIQUIDATE every open live position is sold at the
    forced slippage the minute it trips, retried while it stays on; a position already in flight is left to the worker."""
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
        killed = conn.execute("SELECT kill_switch FROM circuit_state WHERE id = 1").fetchone()["kill_switch"]
    assert killed and out["liquidated"] == 0 and [r.mint for r in w.submitted] == ["LIVE_K2"]      # default: halt only, the scheduled exit alone
    w2 = _Worker(); mirror.worker = w2
    with transaction() as conn:
        monkeypatch.setattr(config, "KILL_SWITCH_LIQUIDATE", True)
        out = mirror.minute(_ctx(conn, m1 + timedelta(minutes=1)), run_id=str(uuid.uuid4()), beat_id=beat + 1, line=0.01, table=[], entries=[])
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
    stub = SimpleNamespace(race_started_at=race.timestamp(), boot_id=62)
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
