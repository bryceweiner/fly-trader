"""The Kalshi paper books: sizing from the certainty bands under the cap's limits, the taker arm's book walk and one
position per market, settlement at face value, USD wealth marks with the book's own halt, and the maker arm's plan
(post, keep, replace, cancel), paper adjudication against the minute's extremes and expiry."""
import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.kalshi import maker, paper as P
from fly_trader.kalshi.features import MarketMeta

TABLE = [{"lo": 0.0, "n": 50, "mean": 0.05, "win": 0.7, "kelly": 0.10}]


def _beat(conn):
    run = str(uuid.uuid4()); conn.execute("INSERT INTO runs (run_id, kind, status) VALUES (%s, 'kalshi_fly', 'running')", (run,))
    m1 = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,1,0) RETURNING id", (run, m1)).fetchone()["id"]
    return run, beat, m1


@pytest.fixture(autouse=True)
def fresh(db_conn):
    db_conn.execute("DELETE FROM wealth_marks WHERE book LIKE '%kalshi%'"); db_conn.execute("DELETE FROM kalshi_positions"); db_conn.execute("DELETE FROM kalshi_orders")
    db_conn.execute("DELETE FROM book_state WHERE book LIKE '%kalshi%'")
    yield


def test_budget_respects_band_kelly_position_cap_slate_and_cash_floor(monkeypatch):
    monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0)
    size, why = P.budget_cents(0.05, 0.02, TABLE, bankroll=10_000, cash=10_000, deployed=0)
    assert size == 1000.0 and "kelly 0.10" in why                                   # 10 % of a $100 bankroll = the position cap
    assert P.budget_cents(0.05, 0.02, TABLE, 20_000, 20_000, 0)[0] == 1000.0         # capped at 10 % of the cap, not of the bankroll
    assert P.budget_cents(0.05, 0.02, TABLE, 10_000, 10_000, 4_800)[0] == 200.0      # the slate: 50 % of the cap across open positions
    assert P.budget_cents(0.05, 0.02, TABLE, 10_000, 600, 0)[0] == 100.0             # the cash floor ($5) is never spent
    assert P.budget_cents(0.05, 0.02, [], 10_000, 10_000, 0)[0] == 0.0 and P.budget_cents(0.01, 0.02, TABLE, 10_000, 10_000, 0)[0] == 0.0   # below the line: no band


def test_taker_arm_walks_the_top_of_book_opens_one_position_per_market_and_settles(db_conn, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0)
    run, beat, m1 = _beat(db_conn); book = P.BOOKS["taker"]
    quotes = {"T1": (70.0, 72.0, 40.0, 50.0), "T2": (30.0, 33.0, 10.0, 10.0)}
    picks = [{"ticker": "T1", "side": "yes", "p": 0.80, "edge": 0.06, "line": 0.02, "table": TABLE, "strategy": "favorite", "hold_s": 3600.0},
             {"ticker": "T1", "side": "no", "p": 0.35, "edge": 0.05, "line": 0.02, "table": TABLE, "strategy": "ev", "hold_s": 3600.0},
             {"ticker": "T2", "side": "yes", "p": 0.30, "edge": 0.03, "line": 0.02, "table": TABLE, "strategy": "ev", "hold_s": 3600.0}]   # p below the ask: refused by the walk
    out = P.taker_entries(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, picks=picks, quotes=quotes)
    assert out["entered"] == 1 and out["entries"][0]["ticker"] == "T1" and out["entries"][0]["limit_price_cents"] == 72
    pos = P.open_positions(db_conn, book); assert len(pos) == 1
    p = pos[0]; assert p["side"] == "yes" and p["contracts"] == 13 and abs(p["avg_price_cents"] - 72.0) < 1e-9 and 0 < p["fee_cents"] < 13 * 2
    kinds = [r["kind"] for r in db_conn.execute("SELECT kind FROM decisions WHERE beat_id = %s ORDER BY id", (beat,)).fetchall()]
    assert kinds == ["kalshi_taker_enter", "blocked", "blocked"]
    rails = [r["rail"] for r in db_conn.execute("SELECT rail FROM decisions WHERE beat_id = %s AND kind = 'blocked' ORDER BY id", (beat,)).fetchall()]
    assert rails == ["other side held", "paper_fill"]
    assert abs(P.cash_cents(db_conn, book) - (10_000 - p["cost_cents"] - p["fee_cents"])) < 1e-6
    m = P.mark(db_conn, book=book, quotes=quotes, ts=m1, beat_id=beat)
    assert 90 < m["wealth"] < 100 and m["open"] == 1 and not m["halted"]                 # marked at the bid net of the exit fee
    settled = P.settle(db_conn, book=book, results={"T1": "yes"}, ts=m1 + timedelta(hours=1), run_id=run, beat_id=beat)
    assert len(settled) == 1 and settled[0]["payout_cents"] == 1300 and abs(settled[0]["realized_cents"] - (1300 - p["cost_cents"] - p["fee_cents"])) < 1e-6
    assert P.open_positions(db_conn, book) == [] and P.cash_cents(db_conn, book) > 10_000
    lost = P.settle(db_conn, book=book, results={"T1": "no"}, ts=m1)                       # nothing open: nothing to settle
    assert lost == []


def test_book_halt_on_its_own_drawdown(db_conn, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0); monkeypatch.setattr(config, "KALSHI_KILL_SWITCH_DRAWDOWN", 0.30)
    run, beat, m1 = _beat(db_conn); book = P.BOOKS["taker"]
    P.open_position(db_conn, book=book, ticker="T9", side="yes", contracts=100, cost_cents=5000, fee_cents=100, avg_price=50, decision_id=None, order_id=None, strategy="ev", arm="taker",
                    score=0.1, line=0.0, ts=m1)
    P.mark(db_conn, book=book, quotes={"T9": (50.0, 52.0)}, ts=m1, beat_id=beat)         # ≈ $98
    beat2 = db_conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,2,0) RETURNING id", (run, m1)).fetchone()["id"]
    m = P.mark(db_conn, book=book, quotes={"T9": (5.0, 7.0)}, ts=m1, beat_id=beat2)         # the position collapsed: $49 + $5 = well below 70 % of the peak
    assert m["halted"] and P.blocked_reason(db_conn, book) == "book halted"


def test_maker_arm_posts_keeps_replaces_cancels_adjudicates_and_expires(db_conn, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0); monkeypatch.setattr(config, "KALSHI_MAKER_QUIET_MIN", 30.0); monkeypatch.setattr(config, "KALSHI_MAKER_TTL_H", 6.0)
    run, beat, m1 = _beat(db_conn); book = P.BOOKS["maker"]; now = m1.timestamp()
    metas = {"M1": MarketMeta("M1", close_ts=now + 2 * 86400, maker_fee=0.0), "M2": MarketMeta("M2", close_ts=now + 20 * 60, maker_fee=1.0)}
    quotes = {"M1": (70.0, 72.0), "M2": (40.0, 42.0)}
    pick = lambda tk, side, edge: {"ticker": tk, "side": side, "p": 0.8, "edge": edge, "line": 0.02, "table": TABLE, "strategy": "favorite"}
    out = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now, picks=[pick("M1", "yes", 0.06), pick("M2", "yes", 0.06)], quotes=quotes, metas=metas)
    assert len(out["posted"]) == 1 and out["posted"][0]["price_cents"] == 71 and out["resting"] == 1          # M2 is inside the quiet margin
    o = out["posted"][0]; assert o["count"] == 14 and abs(o["expiration_ts"] - (now + 6 * 3600)) < 1e-6 and o["decision_id"]
    row = maker.resting(db_conn)[0]; assert row["post_only"] and row["tif"] == "good_till_canceled" and row["status"] == "resting"
    out2 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 60, picks=[pick("M1", "yes", 0.06)], quotes=quotes, metas=metas)
    assert out2["posted"] == [] and out2["replaced"] == 0 and out2["resting"] == 1                             # unchanged ask: the order keeps resting
    out3 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 120, picks=[pick("M1", "yes", 0.06)], quotes={"M1": (73.0, 75.0)}, metas=metas)
    assert out3["replaced"] == 1 and len(out3["posted"]) == 1 and out3["posted"][0]["price_cents"] == 74       # the ask moved: cancel/replace
    assert [r["status"] for r in db_conn.execute("SELECT status FROM kalshi_orders WHERE book = %s ORDER BY id", (book,)).fetchall()] == ["replaced", "resting"]
    adj = maker.adjudicate(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 180, extremes={"M1": (75.0, 73.0)}, metas=metas)
    assert adj["filled"] == [] and adj["expired"] == 0                                                          # the ask never reached 74
    adj = maker.adjudicate(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 240, extremes={"M1": (74.0, 72.0)}, metas=metas)
    assert len(adj["filled"]) == 1 and adj["filled"][0]["price_cents"] == 74
    pos = P.open_positions(db_conn, book); assert len(pos) == 1 and pos[0]["arm"] == "maker" and pos[0]["fee_cents"] == 0.0 and pos[0]["cost_cents"] == 74 * pos[0]["contracts"]
    out4 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 300, picks=[pick("M1", "yes", 0.06)], quotes={"M1": (73.0, 75.0)}, metas=metas)
    assert out4["posted"] == [] and out4["resting"] == 0                                                        # held: no second order on the market
    # a NO bid fills when 100 − the minute's highest YES bid reaches it; a fee is charged where the series charges makers
    metas["M3"] = MarketMeta("M3", close_ts=now + 86400, maker_fee=1.0)
    out5 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 360, picks=[pick("M3", "no", 0.06)], quotes={"M3": (30.0, 33.0)}, metas=metas)
    assert out5["posted"][0]["price_cents"] == 69                                                               # NO ask = 100 − 30 = 70; rest one tick inside
    adj = maker.adjudicate(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 420, extremes={"M3": (33.0, 31.0)}, metas=metas)
    assert len(adj["filled"]) == 1 and adj["filled"][0]["side"] == "no"
    fee = [p for p in P.open_positions(db_conn, book) if p["ticker"] == "M3"][0]["fee_cents"]; assert fee > 0
    # cancel when no longer picked; expire past the expiration
    out6 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 480, picks=[pick("M1", "no", 0.06), pick("M4", "yes", 0.06)],
                      quotes={"M1": (73.0, 75.0), "M4": (50.0, 52.0)}, metas={**metas, "M4": MarketMeta("M4", close_ts=now + 3600, maker_fee=0.0)})
    assert len(out6["posted"]) == 1 and out6["posted"][0]["ticker"] == "M4" and abs(out6["posted"][0]["expiration_ts"] - (now + 3600 - 1800)) < 1e-6
    out7 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 540, picks=[], quotes={"M4": (50.0, 52.0)}, metas=metas)
    assert out7["canceled"] == 1 and maker.resting(db_conn) == []
    out8 = maker.plan(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 600, picks=[pick("M4", "yes", 0.06)], quotes={"M4": (50.0, 52.0)},
                      metas={"M4": MarketMeta("M4", close_ts=now + 3600, maker_fee=0.0)})
    assert len(out8["posted"]) == 1
    adj = maker.adjudicate(db_conn, book=book, run_id=run, beat_id=beat, ts=m1, now=now + 3600, extremes={}, metas=metas)
    assert adj["expired"] == 1 and maker.resting(db_conn) == []
