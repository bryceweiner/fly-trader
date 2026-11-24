"""Bear-market changes of 2026-11-18 (data/scratch/bear_20261118): Robinhood Chain's own Kelly fraction; the re-pick add
(a held coin the book picks again 10 % or more under its entry, 5+ minutes in, is bought once more into the same position,
which is then held its hold from the add; paper and the Solana wallet alike); and no rebuy within an hour of a crash sale."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import crash_exit, fly_live, paper_trading, sizing
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker
from fly_trader.execution.worker import ExecRequest, ExecutionWorker
from fly_trader.markets import RH
from fly_trader.execution.broker_live import FillResult

BOOK = "replay_repick_add_test"
TABLE = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
PAPER_TABLE = [{**TABLE[0], "kelly": 0.2}]       # a first buy under the bankroll cap, so an add has room
M0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
H = 14400.0


def _ctx(conn, m1, prices, lag_s=0.0):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in prices}, prices=dict(prices), resqs={m: 1000.0 for m in prices},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=lag_s,
                           high=lambda m: None)


def _info(m, price):
    return {"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": price, "mcap": 1e6, "decimals": 6, "program_label": None, "fee_rate": None}


def _trade(conn, broker, run, beat, m1, prices, buy=(), kind="selector"):
    buy = list(buy)
    return paper_trading.trade_minute(_ctx(conn, m1, prices), book=BOOK, run_id=run, beat_no=beat, broker=broker, kind=kind, mints=buy,
                                      infos=[_info(m, prices[m]) for m in buy], scores=np.full(len(buy), 0.05), threshold=0.01, table=PAPER_TABLE, horizon_s=H)


def _pos(conn, mint):
    return next((p for p in ledger.open_positions(conn, BOOK) if p["mint"] == mint), None)


@pytest.fixture()
def book(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0)
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3)

    def wipe():
        with transaction() as conn:
            for t in ("positions", "wealth_marks", "fills"):
                conn.execute(f"DELETE FROM {t} WHERE book = %s", (BOOK,))
            conn.execute("DELETE FROM positions WHERE book = 'live' AND mint LIKE 'RPA%%'")
            conn.execute("DELETE FROM decisions WHERE detail->>'book' = %s OR mint LIKE 'RPA%%'", (BOOK,))
            conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
    wipe(); yield PaperBroker(BOOK); wipe()


# ---------------------------------------------------------------- sizing
def test_rh_buys_take_their_own_kelly_fraction_and_an_add_counts_the_position_against_the_caps(monkeypatch):
    monkeypatch.setattr(config, "KELLY_FRACTION", 0.125); monkeypatch.setattr(config, "RH_KELLY_FRACTION", 0.25)
    sol, why_s = sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0)
    rh_table = [{**TABLE[0], "kelly": 0.2}]                       # under the RH label size (0.0223 ETH) and bankroll cap
    rh, why_r = sizing.size_position(0.05, 0.01, rh_table, 0.25, 0.25, 50.0, market=RH)
    assert sol == pytest.approx(0.125 * 0.4 * (5.0 - config.GAS_RESERVE_SOL)) and "× 0.125" in why_s
    assert rh == pytest.approx(0.25 * 0.2 * (0.25 - RH.gas_reserve())) and "× 0.25" in why_r and "capped" not in why_r
    # an add: the position's cost comes off the bankroll cap and the label size
    monkeypatch.setattr(config, "MAX_POSITION_FRACTION", 0.06)
    held = 0.2; cap = 0.06 * (5.0 - config.GAS_RESERVE_SOL)
    add, why = sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0, held_cost=held)
    assert add == pytest.approx(cap - held) and "capped by bankroll cap" in why
    assert sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0, held_cost=cap)[0] == 0.0       # a full position gets no add
    monkeypatch.setattr(config, "MAX_POSITION_FRACTION", 0.5)
    add, why = sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0, held_cost=0.45)              # label size 0.5 SOL (2 % of 1000 = 20)
    assert add == pytest.approx(config.LABEL_SIZE_SOL - 0.45) and "capped by label size" in why


# ---------------------------------------------------------------- the re-pick add, paper
def test_a_held_coin_picked_again_under_its_entry_is_added_once_and_held_from_the_add(book):
    run = str(uuid.uuid4()); m = "RPAONEpump"
    with transaction() as conn:
        _trade(conn, book, run, 1, M0, {m: 1.0}, buy=[m])
        p0 = _pos(conn, m); e0, c0, q0 = float(p0["entry_price"]), float(p0["cost_sol"]), int(p0["qty"])
        _trade(conn, book, run, 2, M0 + timedelta(minutes=3), {m: 0.80 * e0}, buy=[m])              # deep enough, too early
        _trade(conn, book, run, 3, M0 + timedelta(minutes=6), {m: 0.95 * e0}, buy=[m])              # late enough, not deep enough
        early = (int(_pos(conn, m)["qty"]), float(_pos(conn, m)["cost_sol"]))
        out = _trade(conn, book, run, 4, M0 + timedelta(minutes=7), {m: 0.88 * e0}, buy=[m])        # the add
        p1 = _pos(conn, m)
        _trade(conn, book, run, 5, M0 + timedelta(minutes=8), {m: 0.70 * e0}, buy=[m])              # once per position
        p2 = _pos(conn, m)
        adds = conn.execute("SELECT detail FROM decisions WHERE kind = 'selector_add' AND mint = %s", (m,)).fetchall()
        held_rows = conn.execute("SELECT count(*) AS n FROM decisions WHERE kind = 'blocked' AND rail = 'held' AND mint = %s", (m,)).fetchone()["n"]
        # held from the add: not sold at the first buy's 240 minutes, sold at the add's
        _trade(conn, book, run, 6, M0 + timedelta(seconds=H), {m: 0.9 * e0})
        still = _pos(conn, m) is not None
        _trade(conn, book, run, 7, M0 + timedelta(minutes=7) + timedelta(seconds=H), {m: 0.9 * e0})
        sold = conn.execute("SELECT status FROM positions WHERE book = %s AND mint = %s", (BOOK, m)).fetchone()["status"]
    assert early == (q0, pytest.approx(c0)) and out["added"] == 1 and out["entered"] == 0
    assert int(p1["qty"]) > q0 and float(p1["cost_sol"]) > c0 and float(p1["entry_price"]) < e0              # averaged down
    assert float(p1["hold_s"]) == pytest.approx(7 * 60 + H)                                                   # its hold now runs 240 min past the add
    assert (int(p2["qty"]), float(p2["cost_sol"])) == (int(p1["qty"]), pytest.approx(float(p1["cost_sol"])))
    assert len(adds) == 1 and adds[0]["detail"]["add_to"] == int(p0["id"]) and adds[0]["detail"]["of_entry"] == p0["entry_decision_id"]
    assert adds[0]["detail"]["dip"] == pytest.approx(-0.12, abs=0.01) and held_rows == 3
    assert still and sold == "closed"


def test_the_add_follows_its_switches_and_never_on_a_blocked_minute(book, monkeypatch):
    run = str(uuid.uuid4())
    with transaction() as conn:
        for k, m in enumerate(("RPAOFFpump", "RPACHNpump", "RPAKILLpump")):
            _trade(conn, book, run, 1 + k, M0, {m: 1.0}, buy=[m])
        monkeypatch.setattr(config, "REPICK_ADD_DIP", 0.0)
        _trade(conn, book, run, 10, M0 + timedelta(minutes=10), {"RPAOFFpump": 0.5}, buy=["RPAOFFpump"])
        monkeypatch.setattr(config, "REPICK_ADD_DIP", 0.10); monkeypatch.setattr(config, "REPICK_ADD_CHAINS", ["rh"])
        _trade(conn, book, run, 11, M0 + timedelta(minutes=10), {"RPACHNpump": 0.5}, buy=["RPACHNpump"])
        monkeypatch.setattr(config, "REPICK_ADD_CHAINS", ["sol", "rh"])
        conn.execute("UPDATE circuit_state SET kill_switch = true WHERE id = 1")
        _trade(conn, book, run, 12, M0 + timedelta(minutes=10), {"RPAKILLpump": 0.5}, buy=["RPAKILLpump"])
        conn.execute("UPDATE circuit_state SET kill_switch = false WHERE id = 1")
        n = conn.execute("SELECT count(*) AS n FROM decisions WHERE kind = 'selector_add' AND detail->>'book' = %s", (BOOK,)).fetchone()["n"]
    assert n == 0


# ---------------------------------------------------------------- the crash cooldown, paper
def test_a_coin_sold_in_a_crash_is_not_bought_again_for_an_hour(book, monkeypatch):
    run = str(uuid.uuid4()); m = "RPACRASHpump"
    with transaction() as conn:
        conn.execute("""INSERT INTO positions (book, mint, pool, opened_at, closed_at, qty, cost_sol, entry_price, exit_price, realized_sol, status, forced_exit_kind)
                        VALUES (%s, %s, 'P', %s, %s, 0, 0.2, 1.0, 0.7, -0.06, 'closed', 'crash')""", (BOOK, m, M0 - timedelta(minutes=30), M0 - timedelta(minutes=10)))
        a = _trade(conn, book, run, 1, M0, {m: 1.0}, buy=[m])                                         # 10 minutes after the crash sale
        rail = conn.execute("SELECT rail FROM decisions WHERE kind = 'blocked' AND mint = %s ORDER BY id DESC LIMIT 1", (m,)).fetchone()["rail"]
        b = _trade(conn, book, run, 2, M0 + timedelta(minutes=51), {m: 1.0}, buy=[m])                # 61 minutes after it
    assert a["entered"] == 0 and rail == "crashed" and b["entered"] == 1


def test_the_crash_cooldown_covers_this_minutes_own_crash_and_turns_off(book, monkeypatch):
    run = str(uuid.uuid4()); m, n_ = "RPACR2pump", "RPACR3pump"
    with transaction() as conn:
        _trade(conn, book, run, 1, M0, {m: 1.0}, buy=[m])
        monkeypatch.setattr(crash_exit.WATCH, "crashed", lambda mint, since: (M0.timestamp() + 600, 0.7) if mint == m else None)
        out = _trade(conn, book, run, 2, M0 + timedelta(minutes=11), {m: 0.7}, buy=[m])               # sold in a crash this minute, picked again
        sold = conn.execute("SELECT forced_exit_kind FROM positions WHERE book = %s AND mint = %s", (BOOK, m)).fetchone()["forced_exit_kind"]
        monkeypatch.setattr(crash_exit.WATCH, "crashed", lambda mint, since: None)
        conn.execute("""INSERT INTO positions (book, mint, pool, opened_at, closed_at, qty, cost_sol, entry_price, exit_price, realized_sol, status, forced_exit_kind)
                        VALUES (%s, %s, 'P', %s, %s, 0, 0.2, 1.0, 0.7, -0.06, 'closed', 'crash')""", (BOOK, n_, M0, M0 + timedelta(minutes=5)))
        on = _trade(conn, book, run, 3, M0 + timedelta(minutes=20), {n_: 1.0}, buy=[n_])
        monkeypatch.setattr(config, "CRASH_REBUY_COOLDOWN_MIN", 0.0)
        off = _trade(conn, book, run, 4, M0 + timedelta(minutes=21), {n_: 1.0}, buy=[n_])
    assert sold == "crash" and out["entered"] == 0 and on["entered"] == 0 and off["entered"] == 1


def test_only_the_fly_book_counts_the_wallets_crash_sales_and_one_query_reads_both_cooldowns(book):
    """The wallet mirrors the fly's book, so a crash sale of the wallet's own cools the fly's book; the race books (the
    selector's, rising) count only their own. The take-profit cooldown is still the book's own."""
    from fly_trader.markets import SOL
    m1 = M0 + timedelta(minutes=20)
    with transaction() as conn:
        conn.execute("""INSERT INTO positions (book, mint, pool, opened_at, closed_at, qty, cost_sol, entry_price, exit_price, realized_sol, status, forced_exit_kind)
                        VALUES ('live', 'RPALIVECR', 'P', %s, %s, 0, 0.2, 1.0, 0.7, -0.06, 'closed', 'crash')""", (M0, M0 + timedelta(minutes=5)))
        conn.execute("""INSERT INTO positions (book, mint, pool, opened_at, closed_at, qty, cost_sol, entry_price, exit_price, realized_sol, status, forced_exit_kind, hold_s)
                        VALUES (%s, 'RPATP', 'P', %s, %s, 0, 0.2, 1.0, 2.0, 0.2, 'closed', 'take_profit', %s)""", (BOOK, M0, M0 + timedelta(minutes=10), H))
        fly_tp, fly_cr = paper_trading.cooling(conn, SOL.fly_book, SOL, m1, H)
        race_tp, race_cr = paper_trading.cooling(conn, BOOK, SOL, m1, H)
        conn.execute("DELETE FROM positions WHERE book = 'live' AND mint = 'RPALIVECR'")
    assert "RPALIVECR" in fly_cr and "RPALIVECR" not in race_cr
    assert "RPATP" in race_tp and "RPATP" not in fly_tp


# ---------------------------------------------------------------- the Solana wallet
class _Worker:
    def __init__(self):
        self.submitted, self.inflight, self.results, self.emptied = [], {}, [], set()

    def submit(self, req):
        if self.inflight.get(req.mint):
            return False
        self.submitted.append(req); self.inflight.setdefault(req.mint, []).append(req.side); return True

    def pending(self):
        return set(self.inflight)

    def selling(self):
        return {m for m, s in self.inflight.items() if s == ["sell"]}

    def drain(self):
        out, self.results = self.results, []
        return out


def test_the_wallet_adds_only_to_its_own_position_of_the_same_entry_and_never_retries_an_add(book, monkeypatch):
    w = _Worker(); rpc = SimpleNamespace(get_balance=lambda pk: 5 * config.LAMPORTS_PER_SOL)
    mirror = fly_live.LiveMirror(H, broker=SimpleNamespace(rpc=rpc, pubkey="PK"), worker=w)
    m1 = M0 + timedelta(minutes=30)
    with transaction() as conn:
        d_in = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'RPALIVE', %s) RETURNING id", (M0,)).fetchone()["id"]
        d_other = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_enter', 'RPAOTHER', %s) RETURNING id", (M0,)).fetchone()["id"]
        conn.execute("INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, status) VALUES ('live','RPALIVE','P',%s,%s,1000000,0.2,0.0000002,'open')",
                     (M0, d_in))
        conn.execute("INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, status) VALUES ('live','RPAOTHER','P',%s,%s,1000000,0.2,0.0000002,'open')",
                     (M0, d_other))
        add_a = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_add', 'RPALIVE', %s) RETURNING id", (m1,)).fetchone()["id"]
        add_b = conn.execute("INSERT INTO decisions (kind, mint, ts) VALUES ('fly_add', 'RPAOTHER', %s) RETURNING id", (m1,)).fetchone()["id"]
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]; run = str(uuid.uuid4())
        new_hold = 30 * 60 + H
        info = {"resq": 1000.0, "pool": "P", "decimals": 6}
        entries = [{"mint": "RPALIVE", "decision_id": add_a, "score": 0.05, "threshold": 0.01, "table": TABLE, "info": info, "hold_s": new_hold, "strategy": "ev",
                    "add_to": 1, "of_entry": d_in, "ref_price": 2e-7},
                   {"mint": "RPAOTHER", "decision_id": add_b, "score": 0.05, "threshold": 0.01, "table": TABLE, "info": info, "hold_s": new_hold, "strategy": "ev",
                    "add_to": 2, "of_entry": d_other + 999, "ref_price": 2e-7}]                                 # a different entry than the wallet's
        ctx = SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), prices={}, resqs={}, fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: None,
                              fee=lambda m: None, lag_s=0.0)
        out = mirror.minute(ctx, run_id=run, beat_id=beat, entries=entries, line=0.01, table=TABLE)
        blocked = conn.execute("SELECT mint, rail FROM decisions WHERE kind = 'blocked' AND mint LIKE 'RPA%%' AND detail->>'book' = 'live'").fetchall()
        bankroll = 5.0 + 0.2 + 0.2
        expect, _ = sizing.size_position(0.05, 0.01, TABLE, bankroll, 5.0, 1000.0, held_cost=0.2)
        req = w.submitted[0]
        # a failed add is recorded, not kept for another try
        w.results.append(SimpleNamespace(request=req, ok=False, error="swap failed", retryable=True))
        w.inflight.clear(); mirror.sent[add_a] = mirror.sent.get(add_a) or {"e": entries[0], "ts": m1, "tries": 1}
        mirror.minute(ctx, run_id=run, beat_id=beat, entries=[], line=0.01, table=TABLE)
        conn.execute("DELETE FROM beats WHERE id = %s", (beat,))
    assert out["added"] == 1 and len(w.submitted) == 1 and req.mint == "RPALIVE" and req.add and req.hold_s == pytest.approx(new_hold)
    assert req.amount_in == int(expect * config.LAMPORTS_PER_SOL) and req.decision_id == add_a
    assert [(r["mint"], r["rail"]) for r in blocked] == [("RPAOTHER", "add: no such position")]
    assert add_a not in mirror.failed


# ---------------------------------------------------------------- the worker
class _ScriptBroker:
    def __init__(self, fills):
        self.fills = list(fills)

    def swap(self, conn, **kw):
        return self.fills.pop(0)


def _fill(token_delta, lamports_delta, price=1e-7):
    return FillResult(True, None, None, "sig", "confirmed", 0, None, token_delta, lamports_delta, price, 1)


@pytest.fixture
def circuit1():
    with transaction() as conn:
        before = dict(conn.execute("SELECT fail_count, tripped, last_failure_ts FROM circuit_state WHERE id=1").fetchone())
        conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false WHERE id = 1")
    yield
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET fail_count=%(fail_count)s, tripped=%(tripped)s, last_failure_ts=%(last_failure_ts)s WHERE id=1", before)


def test_the_worker_books_an_add_into_the_open_position_and_refuses_one_with_nothing_to_add_to(circuit1):
    mint = f"RPAW{uuid.uuid4().hex[:8]}"
    with transaction() as conn:
        pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=0.2, entry_price=2e-7, decision_id=None, fees_sol=0.0, hold_s=H)
    w = ExecutionWorker(_ScriptBroker([_fill(1_000_000, -100_000_000, 1e-7)]))
    req = lambda add, **kw: ExecRequest(decision_id=None, mint=mint, pool=None, side="buy", amount_in=100_000_000, slippage_bps=150, max_slippage_bps=300,
                                        decimals=6, add=add, **kw)
    try:
        plain = w._execute(req(False, hold_s=H))                                   # a plain buy of a held token is still refused
        r = w._execute(req(True, hold_s=H + 1800.0))
        with transaction() as conn:
            p = dict(conn.execute("SELECT * FROM positions WHERE id = %s", (pid,)).fetchone())
            conn.execute("UPDATE positions SET status = 'closed', closed_at = now() WHERE id = %s", (pid,))
        gone = w._execute(req(True, hold_s=H + 1800.0))
    finally:
        w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))
    assert not plain.ok and plain.error.startswith("held:") and plain.retryable
    assert r.ok and r.position_id == pid and int(p["qty"]) == 2_000_000 and float(p["cost_sol"]) == pytest.approx(0.3)
    assert float(p["entry_price"]) == pytest.approx(1.5e-7) and float(p["hold_s"]) == pytest.approx(H + 1800.0)
    assert not gone.ok and not gone.retryable and "closed" in gone.error


def test_an_add_never_takes_a_creator_heavy_coin_past_its_cap_and_the_ticker_shows_it_as_a_buy():
    from fly_trader.ui.api import ticker
    cap = config.CREATOR_HEAVY_CAP_FRACTION * 5.0
    first, _ = sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0, creator_heavy=True)
    add, why = sizing.size_position(0.05, 0.01, TABLE, 5.0, 5.0, 1000.0, creator_heavy=True, held_cost=first)
    assert first == pytest.approx(cap) and add == 0.0 and "below" in why
    assert ticker.action_of("fly_add") == "added" and ticker.action_of("fly_rh_add") == "added" and "fly_add" in ticker.FLY_KINDS


def test_the_gap_check_leaves_out_a_trade_paper_added_to(db_conn, book):
    w = _Worker(); mirror = fly_live.LiveMirror(H, broker=SimpleNamespace(rpc=None, pubkey="PK"), worker=w)
    with transaction() as conn:
        ids = []
        for i in range(3):
            d = conn.execute("INSERT INTO decisions (kind, mint) VALUES ('fly_enter', %s) RETURNING id", (f"RPAGAP{i}",)).fetchone()["id"]; ids.append(d)
            for bk, r in (("live", -0.5), ("paper_fly", 0.3)):
                conn.execute("INSERT INTO positions (book, mint, entry_decision_id, qty, cost_sol, realized_sol, status, closed_at) VALUES (%s,%s,%s,0,0.1,%s,'closed',now())",
                             (bk, f"RPAGAP{i}", d, 0.1 * r))
        before = mirror.gap(conn)["trades"]
        conn.execute("INSERT INTO decisions (kind, mint, detail) VALUES ('fly_add', 'RPAGAP0', %s)", (f'{{"of_entry": {ids[0]}}}',))
        after = mirror.gap(conn)["trades"]
        conn.execute("DELETE FROM positions WHERE mint LIKE 'RPAGAP%%'")
    assert after == before - 1
