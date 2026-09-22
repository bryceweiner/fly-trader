"""The Kalshi live mirror with a fake exchange: IOC orders at the paper walk's limit carrying the subaccount and a
decision-keyed client order id, GTC post-only maker orders with the paper expiry, cancels for replaced paper orders,
settlement booking from the exchange's rows, USD wealth marks, and the kill switch cancelling and liquidating."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fly_trader import config
from fly_trader.agent import rails
from fly_trader.kalshi import live as L, paper as P


class FakeRest:
    def __init__(self):
        self.calls = []; self.n = 0; self.balance_v = "100.00"; self.settle_rows = []; self.resting = []

    def balance(self, sub=None):
        self.calls.append(("balance", sub)); return {"balance_dollars": self.balance_v}

    def create_order(self, ticker, side, price_cents, count, *, tif, post_only, expiration_ts, client_order_id, subaccount, action="buy"):
        self.n += 1; oid = f"ord{self.n}"
        self.calls.append(("create", dict(ticker=ticker, side=side, price=price_cents, count=count, tif=tif, post_only=post_only, expiration_ts=expiration_ts, coid=client_order_id,
                                          subaccount=subaccount, action=action)))
        if tif == "immediate_or_cancel":
            return {"order_id": oid, "status": "executed", "fill_count_fp": f"{count:.2f}", "remaining_count_fp": "0.00",
                    "taker_fill_cost_dollars": f"{count * price_cents / 100:.2f}", "taker_fees_dollars": f"{count * 0.014:.2f}", "side": side, "yes_price_dollars": f"{price_cents / 100:.4f}"}
        self.resting.append({"order_id": oid, "status": "resting", "remaining_count_fp": f"{count:.2f}", "fill_count_fp": "0.00"})
        return {"order_id": oid, "status": "resting", "fill_count_fp": "0.00", "remaining_count_fp": f"{count:.2f}"}

    def cancel_order(self, order_id, sub=None):
        self.calls.append(("cancel", order_id)); self.resting = [o for o in self.resting if o["order_id"] != order_id]; return {"order_id": order_id, "status": "canceled"}

    def orders(self, status=None, subaccount=None, **kw):
        return list(self.resting)

    def order(self, oid):
        return {"order": {"order_id": oid, "status": "canceled"}}

    def settlements(self, sub=None, **kw):
        return list(self.settle_rows)


def _ctx(conn, m1, quotes, results=None):
    res = dict(results or {})
    return SimpleNamespace(conn=conn, m0=m1 - timedelta(minutes=1), m1=m1, m1_epoch=m1.timestamp(), quotes=quotes, settled=lambda tks: {t: res[t] for t in tks if t in res})


@pytest.fixture
def setup(db_conn, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_SUBACCOUNT", 7); monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0); monkeypatch.setattr(config, "KALSHI_CASH_FLOOR_USD", 5.0)
    monkeypatch.setattr(config, "KALSHI_TAKER_LIVE", True); monkeypatch.setattr(config, "KALSHI_MAKER_LIVE", True); monkeypatch.setattr(config, "KILL_SWITCH_LIQUIDATE", True)
    for t in ("kalshi_positions", "kalshi_orders", "kalshi_fills"):
        db_conn.execute(f"DELETE FROM {t}")
    db_conn.execute("DELETE FROM wealth_marks WHERE book LIKE '%kalshi%'"); db_conn.execute("DELETE FROM book_state WHERE book LIKE '%kalshi%'")
    db_conn.execute("UPDATE circuit_state SET kill_switch = false, tripped = false, entries_paused = false, fail_count = 0 WHERE id = %s", (rails.KALSHI_CIRCUIT,))
    run = str(uuid.uuid4()); db_conn.execute("INSERT INTO runs (run_id, kind, status) VALUES (%s, 'kalshi_fly', 'running')", (run,))
    m1 = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    beat = db_conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,1,0) RETURNING id", (run, m1)).fetchone()["id"]
    did = db_conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced) VALUES (%s,%s,%s,'T1','yes','kalshi_taker_enter',3.6,false) RETURNING id", (beat, run, m1)).fetchone()["id"]
    did2 = db_conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced) VALUES (%s,%s,%s,'T2','yes','kalshi_maker_post',2.1,false) RETURNING id", (beat, run, m1)).fetchone()["id"]
    return SimpleNamespace(conn=db_conn, run=run, beat=beat, m1=m1, did=did, did2=did2, rest=FakeRest())


def test_taker_ioc_and_maker_gtc_orders_carry_subaccount_and_decision_ids_and_book_positions(setup):
    s = setup; mirror = L.KalshiLiveMirror(rest=s.rest)
    quotes = {"T1": (70.0, 72.0, 50.0, 50.0), "T2": (68.0, 70.0, 50.0, 50.0)}
    entries = [{"ticker": "T1", "side": "yes", "contracts": 5, "limit_price_cents": 72, "vwap_cents": 73.4, "cost_cents": 367, "decision_id": s.did, "edge": 0.05, "line": 0.02, "strategy": "favorite"}]
    plan = {"posted": [{"order_row": 0, "decision_id": s.did2, "ticker": "T2", "side": "yes", "price_cents": 69, "count": 3, "expiration_ts": s.m1.timestamp() + 3600, "strategy": "ev"}],
            "canceled": 0, "replaced": 0}
    out = mirror.minute(_ctx(s.conn, s.m1, quotes), run_id=s.run, beat_id=s.beat, taker_entries=entries, maker_plan=plan)
    creates = [c[1] for c in s.rest.calls if c[0] == "create"]
    assert len(creates) == 2 and all(c["subaccount"] == 7 for c in creates)
    assert creates[0] == dict(ticker="T1", side="yes", price=72, count=5, tif="immediate_or_cancel", post_only=False, expiration_ts=None, coid=f"fly-{s.did}", subaccount=7, action="buy")
    assert creates[1]["tif"] == "good_till_canceled" and creates[1]["post_only"] and creates[1]["expiration_ts"] == int(s.m1.timestamp() + 3600) and creates[1]["coid"] == f"fly-{s.did2}"
    pos = P.open_positions(s.conn, L.BOOKS["taker"]); assert len(pos) == 1 and pos[0]["contracts"] == 5 and pos[0]["cost_cents"] == 360 and abs(pos[0]["fee_cents"] - 7) < 1e-6 and pos[0]["order_id"] == "ord1"
    orders = s.conn.execute("SELECT book, order_id, status, tif, post_only, subaccount, client_order_id FROM kalshi_orders ORDER BY id").fetchall()
    assert [(o["book"], o["status"]) for o in orders] == [("live_kalshi_taker", "executed"), ("live_kalshi_maker", "resting")] and orders[1]["subaccount"] == 7
    assert out["taker"]["entered"] == 1 and out["maker"]["posted"] == 1 and out["wealth"] > 90 and out["blocked"] is None
    marks = s.conn.execute("SELECT book, wealth FROM wealth_marks WHERE beat_id = %s ORDER BY book", (s.beat,)).fetchall()
    assert [m["book"] for m in marks] == ["live_kalshi", "live_kalshi_maker", "live_kalshi_taker"]
    # a fill on the resting maker order (from the stream's fill channel) opens the maker position
    s.conn.execute("INSERT INTO kalshi_fills (trade_id, order_id, book, ticker, side, price_cents, count, fee_cents, is_taker, action, ts) VALUES ('f1', 'ord2', 'live_kalshi_maker', 'T2', 'yes', 69, 3, 0, false, 'buy', %s)", (s.m1,))
    beat2 = s.conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,2,0) RETURNING id", (s.run, s.m1)).fetchone()["id"]
    out2 = mirror.minute(_ctx(s.conn, s.m1 + timedelta(minutes=1), quotes), run_id=s.run, beat_id=beat2, taker_entries=[], maker_plan={"posted": [], "canceled": 0, "replaced": 0})
    assert out2["maker_fills"] == 1 and len(P.open_positions(s.conn, L.BOOKS["maker"])) == 1
    # the exchange settles T1 yes: payout and fee from its row; T2 from the market result at face value
    s.rest.settle_rows = [{"ticker": "T1", "market_result": "yes", "revenue_dollars": "5.00", "fee_dollars": "0.00"}]
    beat3 = s.conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,3,0) RETURNING id", (s.run, s.m1)).fetchone()["id"]
    out3 = mirror.minute(_ctx(s.conn, s.m1 + timedelta(minutes=2), quotes, {"T2": "no"}), run_id=s.run, beat_id=beat3, taker_entries=[], maker_plan={"posted": [], "canceled": 0, "replaced": 0})
    assert out3["settled"] == 2
    rows = {r["ticker"]: r for r in s.conn.execute("SELECT ticker, status, payout_cents, realized_cents FROM kalshi_positions WHERE book LIKE 'live_%'").fetchall()}
    assert rows["T1"]["status"] == "settled" and rows["T1"]["payout_cents"] == 500 and abs(rows["T1"]["realized_cents"] - (500 - 360 - 7)) < 1e-6
    assert rows["T2"]["status"] == "settled" and rows["T2"]["payout_cents"] == 0 and rows["T2"]["realized_cents"] == -207


def test_replaced_paper_orders_cancel_live_and_the_kill_switch_liquidates(setup, monkeypatch):
    s = setup; mirror = L.KalshiLiveMirror(rest=s.rest); quotes = {"T2": (68.0, 70.0, 50.0, 50.0), "T3": (40.0, 42.0, 50.0, 50.0)}
    plan = {"posted": [{"order_row": 0, "decision_id": s.did2, "ticker": "T2", "side": "yes", "price_cents": 69, "count": 3, "expiration_ts": s.m1.timestamp() + 3600, "strategy": "ev"}],
            "canceled": 0, "replaced": 0}
    mirror.minute(_ctx(s.conn, s.m1, quotes), run_id=s.run, beat_id=s.beat, taker_entries=[], maker_plan=plan)
    # the paper arm replaced its order this minute (a paper row with the same decision id left 'resting')
    s.conn.execute("INSERT INTO kalshi_orders (book, decision_id, ticker, side, price_cents, count, tif, post_only, status, updated_at) VALUES ('paper_kalshi_maker', %s, 'T2', 'yes', 69, 3, 'good_till_canceled', true, 'replaced', %s)",
                   (s.did2, s.m1 + timedelta(seconds=30)))
    beat2 = s.conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,2,0) RETURNING id", (s.run, s.m1)).fetchone()["id"]
    out = mirror.minute(_ctx(s.conn, s.m1 + timedelta(minutes=1), quotes), run_id=s.run, beat_id=beat2, taker_entries=[], maker_plan={"posted": [], "canceled": 0, "replaced": 1})
    assert out["maker"]["canceled"] == 1 and ("cancel", "ord1") in s.rest.calls
    assert s.conn.execute("SELECT status FROM kalshi_orders WHERE order_id = 'ord1'").fetchone()["status"] == "canceled"
    # kill switch on: a resting order is cancelled and an open position sold IOC at the bid
    P.open_position(s.conn, book=L.BOOKS["taker"], ticker="T3", side="yes", contracts=4, cost_cents=160, fee_cents=2, avg_price=40, decision_id=None, order_id="x", strategy="ev", arm="taker",
                    score=0.1, line=0.0, ts=s.m1)
    s.conn.execute("INSERT INTO kalshi_orders (book, order_id, decision_id, ticker, side, price_cents, count, tif, post_only, status, subaccount) VALUES ('live_kalshi_maker', 'ord9', %s, 'T2', 'yes', 68, 2, 'good_till_canceled', true, 'resting', 7)", (s.did2,))
    s.conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = 'test' WHERE id = %s", (rails.KALSHI_CIRCUIT,))
    beat3 = s.conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active) VALUES (%s,%s,3,0) RETURNING id", (s.run, s.m1)).fetchone()["id"]
    out = mirror.minute(_ctx(s.conn, s.m1 + timedelta(minutes=2), quotes), run_id=s.run, beat_id=beat3, taker_entries=[{"ticker": "T3", "side": "no", "contracts": 1, "limit_price_cents": 60,
                        "vwap_cents": 60, "cost_cents": 60, "decision_id": s.did, "edge": 0.1, "line": 0, "strategy": "ev"}], maker_plan={"posted": [], "canceled": 0, "replaced": 0})
    assert out["blocked"] == "kill switch" and out["taker"]["entered"] == 0 and out["liquidated"] == 1 and ("cancel", "ord9") in s.rest.calls
    sells = [c[1] for c in s.rest.calls if c[0] == "create" and c[1]["action"] == "sell"]
    assert len(sells) == 1 and sells[0]["ticker"] == "T3" and sells[0]["price"] == 40 and sells[0]["tif"] == "immediate_or_cancel"
    p = s.conn.execute("SELECT status, realized_cents FROM kalshi_positions WHERE ticker = 'T3'").fetchone()
    assert p["status"] == "closed" and p["realized_cents"] < 0


def test_order_failures_feed_the_kalshi_circuit(setup, monkeypatch):
    s = setup
    class Broken(FakeRest):
        def create_order(self, *a, **k):
            raise L.KalshiApiError("POST", "/portfolio/orders", 400, {"message": "insufficient balance"})
    mirror = L.KalshiLiveMirror(rest=Broken())
    out = mirror.minute(_ctx(s.conn, s.m1, {"T1": (70.0, 72.0)}), run_id=s.run, beat_id=s.beat,
                        taker_entries=[{"ticker": "T1", "side": "yes", "contracts": 2, "limit_price_cents": 72, "vwap_cents": 73, "cost_cents": 146, "decision_id": s.did, "edge": 0.1, "line": 0, "strategy": "ev"}],
                        maker_plan={"posted": [], "canceled": 0, "replaced": 0})
    assert out["taker"]["entered"] == 0 and rails.load_circuit(s.conn, rails.KALSHI_CIRCUIT).fail_count == 1
    assert s.conn.execute("SELECT status, error FROM kalshi_orders").fetchone()["status"] == "error"
    assert rails.load_circuit(s.conn, rails.SOLANA_CIRCUIT).fail_count == 0                    # the memecoin circuit is untouched
