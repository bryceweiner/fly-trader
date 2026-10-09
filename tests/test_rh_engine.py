"""The live side of the second chain: an RH engine reads rh_minutes/rh_meta in ETH with the chain inputs set and the
K-scaled gates; each chain's engine feeds only its own books and waits only on its own stream; paper RH books pay the
Pons costs in ETH against the RH bankroll; circuit 3 and the RH handover leave Solana's untouched."""
import json
import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config, markets
from fly_trader.agent import handover, minute_engine as me, rails
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker
from fly_trader.market.exit_cost import PONS_LABEL, exit_cost_fraction, fee_fraction, impact_fraction
from fly_trader.train.decisions import X_COLS

M0 = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
MINT = "0x00000000000000000000000000000000000000e1"


@pytest.fixture
def rh_rows(db_conn):
    k = markets.RH.k()
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM rh_minutes WHERE mint = %s", (MINT,)); cur.execute("DELETE FROM rh_meta WHERE mint = %s", (MINT,))
        cur.execute("INSERT INTO rh_minutes (mint, ts, pool_id, quote_asset, quote_class, open, high, low, close, buy_eth, sell_eth, n_buys, n_sells, n_traders, resq_eth, fee_rate) "
                    "VALUES (%s, %s, '0xpool', '0xstock', 'stock', 1e-6, 1.2e-6, 0.9e-6, 1.1e-6, %s, %s, 3, 1, 4, %s, 0.029)",
                    (MINT, M0, 6.0 * k, 2.0 * k, 55.0 * k))
        cur.execute("INSERT INTO rh_meta (mint, graduated_at, quote_class, supply, ttg_min, dev_sol) VALUES (%s, %s, 'stock', 1e9, 40.0, %s)",
                    (MINT, M0 - timedelta(hours=3), 0.1 * k))
    db_conn.commit()
    yield k
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM rh_minutes WHERE mint = %s", (MINT,)); cur.execute("DELETE FROM rh_meta WHERE mint = %s", (MINT,))
    db_conn.commit()


def test_rh_minute_features_in_eth_with_chain_inputs(db_conn, rh_rows):
    k = rh_rows; e = me.MinuteEngine(market=markets.RH)
    agg = e._aggregate(db_conn, M0, M0 + timedelta(minutes=1))
    assert list(agg) == [MINT] and agg[MINT]["resq"] == pytest.approx(55.0 * k) and agg[MINT]["program_label"] == PONS_LABEL
    x, info = e._features(db_conn, MINT, agg[MINT], M0.timestamp() + 60)
    col = {n: x[X_COLS.index(n)] for n in ("chain_rh", "qc_stock", "qc_stable", "qc_btc", "ttg_min", "age_known")}
    assert col == {"chain_rh": 1.0, "qc_stock": 1.0, "qc_stable": 0.0, "qc_btc": 0.0, "ttg_min": 40.0, "age_known": 1.0}
    assert info["chain"] == "rh" and info["decimals"] == 18 and info["age_h"] == pytest.approx(3.0 + 1 / 60) and not info["broken"]
    # the same pool in SOL would pass the Solana gate: in ETH it passes the K-scaled one (and a Solana engine's gate would refuse it)
    info = {**info, "logvol_15m": math.log1p(20.0 * k)}
    assert e.eligible(info) and not me.MinuteEngine().eligible(info)


def test_solana_rows_keep_the_chain_inputs_off(db_conn):
    e = me.MinuteEngine()
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    x, info = e._features(db_conn, "Cpump", a, 1_800_000_000.0)
    assert all(x[X_COLS.index(c)] == 0.0 for c in ("chain_rh", "qc_stable", "qc_btc", "qc_stock")) and info["chain"] == "sol"


def _status(conn, key, through):
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                 (key, json.dumps({"flushed_through": datetime.fromtimestamp(through, timezone.utc).isoformat()})))


def test_a_lagging_rh_stream_never_holds_solana_back(db_conn):
    m1 = math.floor(time.time() / 60) * 60
    sol, rh = me.MinuteEngine(), me.MinuteEngine(market=markets.RH)
    with transaction() as conn:
        _status(conn, "pumpstream_status", m1 - 60); _status(conn, "rh_stream_status", m1 - 3600)
    assert sol.ready_through(m1) == m1 and rh.ready_through(m1) == m1 - 3540         # each waits on its own stream only
    with transaction() as conn:
        conn.execute("DELETE FROM ui_settings WHERE key = 'rh_stream_status'")


class _Book:
    def __init__(self, name, chain):
        self.name, self.chain, self.done, self.seen = name, chain, False, []

    def on_bars(self, t, bars):
        pass

    def on_minute(self, ctx):
        self.seen.append(ctx.market.chain); return {}

    def open_mints(self, conn):
        return set()

    def finish(self):
        pass


def test_each_engine_feeds_only_its_chain_books(db_conn, rh_rows, monkeypatch):
    books = [_Book("selector", "sol"), _Book("selector_rh", "rh")]
    rh = me.MinuteEngine(market=markets.RH); rh.books = books
    monkeypatch.setattr(rh, "stream_fresh", lambda t: True)
    rh.run_minute(M0.timestamp() + 60)
    assert books[1].seen == ["rh"] and books[0].seen == []
    assert [b.name for b in me.MinuteEngine(books).my_books()] == ["selector"]


def test_paper_rh_book_pays_pons_costs_in_eth(db_conn, monkeypatch):
    book = markets.RH.fly_book; k = markets.RH.k(); size, resq, p0 = 0.1 * k, 60.0 * k, 1e-6
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE book = %s", (book,)); conn.execute("DELETE FROM fills WHERE book = %s", (book,))
        b = PaperBroker(book)
        fill = b.buy(conn, decision_id=None, mint=MINT, pool="0xpool", size_sol=size, price=p0, res_quote_sol=resq, mcap_sol=None, decimals=18,
                     program_label=PONS_LABEL, pool_fee=0.029)
        assert fill.ok
        pos = ledger.open_positions(conn, book)
        assert len(pos) == 1 and pos[0]["chain"] == "rh" and pos[0]["decimals"] == 18 and pos[0]["program_label"] == PONS_LABEL
        assert ledger.paper_cash(conn, book) == pytest.approx(markets.RH.capital() - size)
        out = b.sell(conn, position=pos[0], decision_id=None, price=1.1 * p0, res_quote_sol=resq, mcap_sol=None, program_label=PONS_LABEL,
                     forced_kind=None, pool_fee=0.029)
        tokens = size * (1 - fee_fraction(size, None, 0.029, PONS_LABEL)) / (p0 * (1 + impact_fraction(size, resq, PONS_LABEL)))
        gross = int(tokens * 1e18) / 1e18 * 1.1 * p0
        want = gross * (1 - exit_cost_fraction(gross, resq, None, PONS_LABEL, 0.029)) - size
        realized = conn.execute("SELECT realized_sol FROM positions WHERE id = %s", (out.position_id,)).fetchone()["realized_sol"]
        assert realized == pytest.approx(want, rel=1e-9) and ledger.paper_cash(conn, book) == pytest.approx(markets.RH.capital() + want)
        assert ledger.paper_cash(conn, markets.SOL.fly_book) != pytest.approx(markets.RH.capital() + want)     # the Solana bankroll is its own
        conn.execute("DELETE FROM positions WHERE book = %s", (book,)); conn.execute("DELETE FROM fills WHERE book = %s", (book,))


def test_circuit_3_is_isolated(db_conn):
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, peak_wealth=NULL, entries_paused=false WHERE id IN (1, 3)")
        assert rails.check_drawdown(conn, 0.01, 1.0, circuit_id=rails.RH_CIRCUIT, drawdown=config.RH_KILL_SWITCH_DRAWDOWN, unit="ETH")
        for _ in range(config.CIRCUIT_THRESHOLD):
            rails.record_failure(conn, "rh boom", circuit_id=rails.RH_CIRCUIT)
        assert rails.load_circuit(conn, 3).kill_switch and rails.load_circuit(conn, 3).tripped
        assert not rails.load_circuit(conn, 1).kill_switch and not rails.load_circuit(conn, 1).tripped
        rails.record_notional(conn, 0.02, market="rh")
        assert rails.notional_24h(conn, "rh") >= 0.02 and rails.notional_24h(conn) == rails.notional_24h(conn, "sol")
        conn.execute("UPDATE circuit_state SET fail_count=0, tripped=false, kill_switch=false, kill_reason=NULL, peak_wealth=NULL WHERE id = 3")
        conn.execute("DELETE FROM notional_ledger WHERE market = 'rh'")


def test_handover_is_per_chain(db_conn):
    with transaction() as conn:
        before = handover.state(conn)
        conn.execute("DELETE FROM ui_settings WHERE key = 'handover_rh'")
        handover.record(conn, {"why": "test"}, "rh")
        assert handover.state(conn, "rh") is not None and handover.state(conn) == before
        conn.execute("DELETE FROM ui_settings WHERE key = 'handover_rh'")


def test_admission_per_chain_and_the_rh_fly_shares_the_solana_brain(db_conn, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(config, "RH_ENABLED", True)
    calls = []
    def try_start(live, chain="sol", brain=None):
        calls.append((chain, brain.name if brain else None))
        return SimpleNamespace(name="fly" if chain == "sol" else "fly_rh", chain=chain), ""
    from fly_trader.agent import fly_session, selector_session
    monkeypatch.setattr(fly_session, "try_start", try_start)
    monkeypatch.setattr(selector_session, "SelectorBook", lambda chain="sol": SimpleNamespace(name="selector" if chain == "sol" else "selector_rh", chain=chain))
    with transaction() as conn:
        conn.execute("DELETE FROM ui_settings WHERE key = 'handover_rh'")
        handover.record(conn, {"why": "test"}, "rh")
        sol_handed = handover.state(conn) is not None
    new = me.admit_books(me.MinuteEngine(), live=False)
    assert calls == [("sol", None), ("rh", "fly")]
    names = [b.name for b in new]
    assert "fly" in names and "fly_rh" in names and "selector_rh" not in names         # RH handed over: its selector stays off
    assert ("selector" in names) == (not sol_handed)
    with transaction() as conn:
        conn.execute("DELETE FROM ui_settings WHERE key = 'handover_rh'")


def test_rh_minutes_a_few_minutes_old_are_traded_solana_only_the_current_one():
    m1 = 1_800_000_000.0
    sol, rh = me.MinuteEngine(), me.MinuteEngine(market=markets.RH)
    assert sol.tradeable(m1, m1) and not sol.tradeable(m1 - 60, m1)
    assert rh.tradeable(m1 - 180, m1) and rh.tradeable(m1, m1) and not rh.tradeable(m1 - 300, m1)
