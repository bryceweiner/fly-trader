"""Correctness (2026-10-24): a per-strategy P(clean) -- net ≥ +5 % over the hold, no checkpoint below -15 % -- trained on
all of the strategy's candidates. Behind the EV line it sizes the Kelly bands instead of the EV margin on Solana, its
lowest band is Kelly 0 (the gate), Robinhood Chain keeps its EV bands, and every decision carries it as the console's
correctness %. The random-removal test clips returns so one moonshot cannot decide it."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import paper_trading, sizing
from fly_trader.db.connection import transaction
from fly_trader.execution.broker_paper import PaperBroker
from fly_trader.train import strategies as S
from fly_trader.train.decisions import X_COLS, DecisionSet

C = {k: i for i, k in enumerate(X_COLS)}


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CACHE_DIR", tmp_path / "stack_cache")


def _set(fh: dict, n: int, mints=None, day_rows: int = 100) -> DecisionSet:
    d0 = datetime(2026, 7, 1, tzinfo=timezone.utc); k = np.arange(n)
    ts = np.array([(d0 + timedelta(days=int(i // day_rows), minutes=int(i % day_rows) * 3)).timestamp() for i in k])
    day = np.array([(d0 + timedelta(days=int(i // day_rows))).date() for i in k], dtype=object)
    r = fh[240]
    X = np.zeros((n, len(X_COLS)), np.float32)
    X[:, X_COLS.index("meta_known")] = 1.0; X[:, X_COLS.index("curve_known")] = 1.0     # creator facts known (selector.in_universe)
    return DecisionSet(X=X, y=(r > 0).astype(np.int8), fwd=r, fwd_pess=r, day=day, ts=ts,
                       mint=np.array(mints if mints is not None else [f"m{i}" for i in k]), cols=list(X_COLS), horizon_s=14400.0, fwd_h=fh)


def test_a_clean_trade_ends_up_and_never_dips_below_its_floor():
    n = 5; fh = {h: np.zeros(n, np.float32) for h in S.HOLDS_MIN}
    fh[240][:] = [0.06, 0.06, 0.04, np.nan, 0.30]
    fh[60][1] = -0.20                                              # ends up, but dipped past -15 % on the way
    fh[300] = np.full(n, -0.9, np.float32)                         # a checkpoint past the hold is not looked at
    y = S.clean_label(_set(fh, n), 240)
    assert y[0] == 1 and y[1] == 0 and y[2] == 0 and np.isnan(y[3]) and y[4] == 1
    assert S.clean_label(_set(fh, n), 30)[0] == 0                   # over a 30-minute hold the return is the 30-minute one (0)


def test_one_moonshot_no_longer_decides_the_random_removal_test():
    """A filter that drops every loser and, with them, one +200,000 % trade: unclipped the moonshot alone made random
    removal look better (it survives most draws); clipped at +300 % the filter wins, as it should."""
    n = 1000; r = np.full(n, 0.02, np.float32); r[:300] = -0.10; r[300] = 2000.0
    ds = _set({h: r.copy() for h in S.HOLDS_MIN}, n)
    before = np.ones(n, bool); after = before.copy(); after[:301] = False
    ok, info = S.beats_random_removal(ds, before, after, 600.0, r, ds.day[0])
    assert ok and info["filtered_total"] == pytest.approx(699 * 0.02, rel=1e-4) and info["random_total"] < 0
    old_random = float(np.mean([np.delete(r.astype(np.float64), np.random.default_rng(s).choice(n, 301, replace=False)).sum() for s in range(S.RANDOM_SEEDS)]))
    assert old_random > info["filtered_total"]                     # what the unclipped test compared against


class _Id:
    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class _Score:
    """Predicts the EV score from one input column."""
    def __init__(self, j):
        self.j = j

    def predict(self, X):
        return np.asarray(X)[:, self.j]


class _P:
    """P(clean) from one input column."""
    classes_ = np.array([0, 1])

    def __init__(self, j):
        self.j = j

    def predict_proba(self, X):
        p = np.asarray(X)[:, self.j]
        return np.c_[1 - p, p]


EV_TABLE = [{"lo": 0.0, "n": 100, "mean": 0.01, "win": 0.5, "kelly": 0.1}]
C_TABLE = [{"lo": 0.0, "n": 50, "mean": -0.01, "win": 0.4, "kelly": 0.0, "gate": True}, {"lo": 0.1, "n": 50, "mean": 0.03, "win": 0.6, "kelly": 0.3},
           {"lo": 0.25, "n": 50, "mean": 0.06, "win": 0.7, "kelly": 0.6}]


def _models():
    cols = ["imb_5m", "ret_1m"]                                    # imb_5m carries the EV score, ret_1m the P(clean)
    clean = {"model": _P(1), "scaler": _Id(), "p0": 0.3, "cut": 0.4, "table": C_TABLE, "chains": ["sol"], "ret": 0.05, "dd": -0.15}
    return {"strategies": {"ev": {"gbm": _Score(0), "scaler": _Id(), "cols": cols, "line": 0.02, "hold_min": 240, "thr": {}, "high": None, "meta": None,
                                  "hours": [], "regimes": [], "sizing": EV_TABLE, "clean": clean}}, "veto": None, "combine": "kelly"}


def test_on_solana_the_ev_line_screens_and_correctness_sizes_and_gates():
    X = np.zeros((5, len(X_COLS)), np.float32)
    X[:, C["imb_5m"]] = [0.05, 0.05, 0.05, 0.01, 0.05]             # the last but one fails the EV line (0.02)
    X[:, C["ret_1m"]] = [0.35, 0.65, 0.35, 0.90, np.nan]            # P(clean)
    X[2, C["chain_rh"]] = 1.0                                       # a Robinhood Chain row: correctness does not size or gate it
    X[4, C["ret_1m"]] = 0.5
    d = S.decide(_models(), X, list(X_COLS), 0.0)
    assert not d["allow"][0] and d["reason"][0].startswith("correctness 35 %")            # Solana, lowest band: the gate
    assert d["allow"][1] and d["tables"][1] is C_TABLE and d["size_margin"][1] == pytest.approx(0.35) and d["pclean"][1] == pytest.approx(0.65)
    assert d["allow"][2] and d["tables"][2] is EV_TABLE and np.isnan(d["size_margin"][2]) and d["pclean"][2] == pytest.approx(0.35)
    assert d["strategy"][3] is None and np.isnan(d["pclean"][3])                         # under the EV line: no strategy, no P(clean)
    s, why = sizing.size_position(float(d["score"][1]), float(d["threshold"][1]), d["tables"][1], 10.0, 10.0, 1e6, margin=float(d["size_margin"][1]))
    assert s > 0 and why.startswith("correctness band from +0.250: Kelly 0.60")
    s0, why0 = sizing.size_position(0.05, 0.02, C_TABLE, 10.0, 10.0, 1e6, margin=0.05)
    assert s0 == 0.0 and why0 == "no edge in this correctness band"
    s_ev, why_ev = sizing.size_position(0.05, 0.02, EV_TABLE, 10.0, 10.0, 1e6, margin=float("nan"))       # NaN: the EV margin, as before
    assert s_ev > 0 and "over the line" in why_ev


def _ctx(conn, m1, mints):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in mints}, prices={m: 1.0 for m in mints}, resqs={m: 1000.0 for m in mints},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=0.0)


def _info(m):
    return {"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": 1.0, "mcap": 1e6, "decimals": 6, "program_label": None, "fee_rate": None}


def test_the_book_sizes_at_the_correctness_margin_and_records_the_correctness(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0)
    book = "replay_correctness_test"; run = str(uuid.uuid4()); broker = PaperBroker(book); m0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    mints = ["LOWpump", "HIGHpump"]
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE book = %s", (book,)); conn.execute("DELETE FROM wealth_marks WHERE book = %s", (book,))
        conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
        plan = paper_trading.plan_minute(_ctx(conn, m0, mints), book=book, run_id=run, beat_no=1, broker=broker, kind="selector", mints=mints,
                                         infos=[_info(m) for m in mints], scores=np.array([0.05, 0.05]), threshold=np.array([0.02, 0.02]), table=None,
                                         horizon_s=14400.0, holds=np.array([14400.0, 14400.0]), strategies=np.array(["capitulation"] * 2, dtype=object),
                                         allow=np.array([True, True]), reasons=np.array(["trade", "trade"], dtype=object), tables=[C_TABLE] * 2,
                                         size_margins=np.array([0.05, 0.30]), correctness=np.array([0.35, 0.60]))
        assert [e["mint"] for e in plan["entries"]] == ["HIGHpump"]
        e = plan["entries"][0]
        assert e["size_margin"] == pytest.approx(0.30) and e["correctness"] == pytest.approx(0.60)          # what a live mirror sizes the wallet's buy from
        rows = {r["mint"]: r for r in conn.execute("SELECT mint, kind, rail, detail FROM decisions WHERE run_id = %s", (run,)).fetchall()}
        assert rows["LOWpump"]["kind"] == "blocked" and rows["LOWpump"]["rail"] == "sizing" and rows["LOWpump"]["detail"]["correctness"] == pytest.approx(0.35)
        assert rows["HIGHpump"]["kind"] == "selector_enter" and rows["HIGHpump"]["detail"]["correctness"] == pytest.approx(0.60)
        assert rows["HIGHpump"]["detail"]["sizing"].startswith("correctness band from +0.250")
        conn.execute("DELETE FROM decisions WHERE run_id = %s", (run,)); conn.execute("DELETE FROM beats WHERE run_id = %s", (run,))


def test_the_stack_keeps_correctness_where_it_ranks_inside_the_ev_picks(monkeypatch):
    """Planted: inside the EV signal's picks a third of the trades lose, invisible to the EV inputs. A P(clean) that knows
    them (stubbed here: the component's judgement is the subject) grows the evaluation half's bankroll faster than the EV
    margin: kept, its losing band measured at Kelly 0, its cut at the first band with an edge, and the final model carries it."""
    monkeypatch.setattr(S, "STRATEGIES", {"ev": {**S.STRATEGIES["ev"], "holds": (240,)}})
    monkeypatch.setattr(S, "GROUPS", {})
    monkeypatch.setattr(S, "LEGACY_COLS", ["imb_5m", "ret_1m", "age_known", "log_age_h"])
    monkeypatch.setattr(S, "HOLDOUT_DAYS", 0)
    monkeypatch.setattr(S, "_min_trades", lambda: 30)
    rng = np.random.default_rng(3); days, per_day = 100, 400; n = days * per_day
    sig = rng.normal(size=n).astype(np.float32); bad = rng.random(n) < 0.3
    good = sig > 1.2
    fwd = np.where(good, np.where(bad, -0.05, np.where(rng.random(n) < 0.85, 0.06, -0.03)), rng.normal(-0.01, 0.03, n)).astype(np.float32)
    ds = _set({h: fwd.copy() for h in S.HOLDS_MIN}, n, mints=[f"m{i % 150}" for i in range(n)], day_rows=per_day)
    ds.X[:, C["imb_5m"]] = sig; ds.X[:, C["ret_1m"]] = rng.normal(0, 0.02, n)
    p_stub = np.where(bad, 0.1, 0.8).astype(np.float32) + rng.normal(0, 0.01, n).astype(np.float32)
    monkeypatch.setattr(S, "fit_correctness", lambda ds_, f, uni, stop=None: p_stub)
    st = S.fit_stack(ds)
    c = next(x for x in st.components if x["name"] == "correctness: ev")
    assert c["passed"], c["reason"]
    f = next(x for x in st.fits if x.name == "ev")
    tab = f.clean["table"]
    assert tab[0]["kelly"] == 0.0 and f.clean["cut"] == pytest.approx(f.clean["p0"] + tab[1]["lo"])        # measured: the losing band has no edge
    g = c["evaluation"]["growth_per_trade"]; assert g["correctness_bands"] > g["ev_bands"]
    assert f.clean["chains"] == ["sol"] and c["evaluation"]["growth_per_trade"]["n_correctness"] > 0
    m = S.final_models(ds, st)["strategies"]["ev"]["clean"]
    assert m is not None and {"model", "scaler", "p0", "cut", "table", "chains"} <= set(m) and m["table"][0]["kelly"] == 0.0
