"""The Kalshi fly: its features enter the photoreceptors of a graph that keeps the optic lobes, its head predicts a
probability (scale 1) whose edge over the arm's price is what it trades, the memecoin FlyNet is unchanged by the new
parameters, the plastic rule learns from settlement outcomes, snapshots keep their own kind and version, and the replay
lets a tag come due at its market's settlement and judges with the deploy rule."""
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from fly_trader.brain import connectome as C, plastic
from fly_trader.kalshi import fly as KF, fly_replay as KR
from fly_trader.kalshi.features import K_COLS
from fly_trader.kalshi.strategies import KALSHI, effective
from fly_trader.train import fly_selector
from fly_trader.train.decisions import DecisionSet
from fly_trader.train.scaling import RobustScaler

POPS = {"KC": (0, 20), "MBON_APP": (20, 22), "MBON_AV": (22, 24), "MBON_OTHER": (24, 26), "ORN_FOOD": (26, 30), "VISUAL": (30, 70), "DESCENDING": (70, 80)}


def _graph(seed=0):
    """A stand-in brain with an optic lobe: 40 VISUAL neurons of which the first 24 are photoreceptors (R1R6), wired
    into interneurons (the other VISUAL rows) → KCs and DNs; 4 ORNs the Kalshi fly must never use."""
    g = torch.Generator().manual_seed(seed); pre, post = [], []
    for a, b in ((range(30, 54), range(54, 70)), (range(54, 70), range(0, 20)), (range(54, 70), range(70, 80)), (range(20, 26), range(70, 80)),
                 (range(26, 30), range(0, 20))):
        for j in b:
            for i in torch.randint(a.start, a.stop, (6,), generator=g).tolist():
                pre.append(i); post.append(j)
    vals = (torch.rand(len(pre), generator=g) * 5 + 1) * torch.where(torch.rand(len(pre), generator=g) < 0.8, 1.0, -1.0)
    W = torch.randint(1, 12, (20, 6), generator=g).float() * (torch.rand(20, 6, generator=g) < 0.4)
    fg = np.full(80, 9); fg[30:54] = 0; fg[54:70] = 2
    return SimpleNamespace(N=80, indices=torch.tensor([post, pre]), values_raw=vals, pop_ranges=dict(POPS), W_KM0=W, M_KM=W > 0,
                           device=torch.device("cpu"), s=0.05, spectral_radius_raw=None, flybrain_group=fg,
                           flybrain_group_names=["VIS_R1R6", "VIS_R7R8", "VIS_ME", "LPTC", "LO", "X", "Y", "Z", "W", "CENTRAL"])


def _ds(days=14, per_day=400, seed=0, settle_h=6.0):
    """Kalshi rows over K_COLS: side ask around 75c, the outcome following ret_6h (a signal the fly can find), every
    position settling ``settle_h`` hours after its row (hold_s), the taker label at the effective ask, the maker one filled
    on the rows whose ask later dipped."""
    rng = np.random.default_rng(seed); n = days * per_day; c = {k: i for i, k in enumerate(K_COLS)}
    X = np.zeros((n, len(K_COLS)), np.float32)
    ask = np.clip(rng.normal(75, 8, n), 40, 97).round(); X[:, c["side_ask"]] = ask; X[:, c["side_bid"]] = ask - 2; X[:, c["side_mid"]] = ask - 1
    X[:, c["yes_ask"]] = ask; X[:, c["yes_bid"]] = ask - 2; X[:, c["mid"]] = ask - 1; X[:, c["spread"]] = 2; X[:, c["side_yes"]] = 1
    X[:, c["fee_mult"]] = 1.0; X[:, c["maker_fee"]] = 0.0
    fee = 0.07 * (ask / 100) * (1 - ask / 100) * 100; X[:, c["eff_price"]] = ask + fee
    X[:, c["ret_6h"]] = rng.normal(size=n); X[:, c["taker_imb_1h"]] = rng.normal(size=n); X[:, c["log_vol_24h"]] = np.log1p(rng.exponential(500, n))
    X[:, c["log_oi"]] = np.log1p(rng.exponential(2000, n)); X[:, c["log_h_to_close"]] = np.log1p(settle_h)
    p = 1 / (1 + np.exp(-(1.2 * (ask - 75) / 10 + 1.5 * X[:, c["ret_6h"]])))
    y = (rng.random(n) < p).astype(np.float32)
    eff = X[:, c["eff_price"]]; fwd_pess = ((100 * y - eff) / eff).astype(np.float32); fwd = ((100 * y - ask) / ask).astype(np.float32)
    rest = ask - 1; filled = rng.random(n) < 0.6
    maker = np.where(filled, (100 * y - rest) / rest, np.nan).astype(np.float32)
    d0 = datetime(2026, 5, 1, tzinfo=timezone.utc); k = np.arange(n)
    ts = np.array([(d0 + timedelta(days=int(i // per_day), minutes=int(i % per_day))).timestamp() for i in k])
    day = np.array([(d0 + timedelta(days=int(i // per_day))).date() for i in k], dtype=object)
    mint = np.array([f"T{i % 150}:{'yes' if i % 2 else 'no'}" for i in k])
    ds = DecisionSet(X=X, y=(fwd_pess > 0).astype(np.int8), fwd=fwd, fwd_pess=fwd_pess, day=day, ts=ts, mint=mint, cols=list(K_COLS), horizon_s=settle_h * 3600.0,
                     fwd_h={"maker": maker})
    ds.hold_s = np.full(n, settle_h * 3600.0); ds.outcome = y; ds.side = np.where(k % 2 == 1, "yes", "no"); ds.ticker = np.array([m.split(":")[0] for m in mint])
    ds.category = np.full(n, "politics", dtype=object)
    return ds


def _teacher(ds, train):
    """A stack of two strategies (taker ``favorite``, maker ``ev``) fitted directly: classifiers of the outcome."""
    models = {"strategies": {}, "combine": "score"}
    for name, arm, cols in (("favorite", "taker", ["side_ask", "ret_6h"]), ("ev", "maker", ["side_ask", "ret_6h", "taker_imb_1h"])):
        ci = [ds.cols.index(x) for x in cols]; rows = np.flatnonzero(train & KALSHI.base_mask(name, ds.X, ds.cols))
        sc = RobustScaler.fit(ds.X[np.ix_(rows, ci)], seed=1); m = HistGradientBoostingClassifier(max_iter=40, random_state=1).fit(sc.transform(ds.X[np.ix_(rows, ci)]), ds.outcome[rows].astype(int))
        models["strategies"][name] = {"gbm": m, "scaler": sc, "cols": cols, "line": 0.0, "hold_min": arm, "thr": {}, "high": None, "sizing": [], "meta": None}
    return KF.KalshiTeacher(models, RobustScaler.fit(ds.X[train], seed=7), ds.cols)


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(KF.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(KR.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(fly_selector, "EPOCHS", 2)


def test_photoreceptor_rows_are_the_visual_r1r6_neurons():
    g = _graph(); rows = C.photoreceptor_rows(g)
    assert rows.tolist() == list(range(30, 54))
    bad = SimpleNamespace(pop_ranges={"KC": (0, 20)})
    with pytest.raises(ValueError):
        C.photoreceptor_rows(bad)


def test_flynet_takes_custom_afferent_rows_and_the_memecoin_default_is_unchanged():
    g = _graph(); rows = C.photoreceptor_rows(g)
    k = fly_selector.FlyNet(g, obs_dim=5, device="cpu", aff_rows=rows, scale=1.0)
    assert tuple(k.w_in.shape) == (24, 5) and k.aff_rows.tolist() == rows.tolist() and k.scale == 1.0
    m = fly_selector.FlyNet(g, obs_dim=5, device="cpu")
    assert tuple(m.w_in.shape) == (4, 5) and m.aff_rows.tolist() == [26, 27, 28, 29] and m.scale == fly_selector.SCALE
    x = torch.randn(3, 5); _, _, _, H = k.forward_parts_all_h(x)
    assert H.shape == (3, 80)


def _boot(ds, start):
    torch.manual_seed(0)
    train = ds.day < ds.days[start] - timedelta(days=KF.CALIB_DAYS + KF.PURGE_DAYS)
    return KF.bootstrap(ds, ds.days[start], g=_graph(), device="cpu", teacher=_teacher(ds, train), teacher_note="test stack")


def test_bootstrap_predicts_probabilities_trades_edges_and_calibrates_its_own_lines():
    ds = _ds(); fly, info = _boot(ds, 12)
    assert isinstance(fly, KF.KalshiFlyModel) and fly.strategies == ["favorite", "ev"] and [fly.arm(0), fly.arm(1)] == ["taker", "maker"]
    P = fly.score_all(ds.X[:500]); E = fly.edge_all(ds.X[:500], P)
    assert P.shape == (500, 2) and (P >= 0).all() and (P <= 1).all()
    assert np.allclose(E[:, 0], P[:, 0] - effective(ds.X[:500], ds.cols, "taker") / 100.0)
    assert np.allclose(E[:, 1], P[:, 1] - effective(ds.X[:500], ds.cols, "maker") / 100.0)
    assert (effective(ds.X[:500], ds.cols, "maker") < effective(ds.X[:500], ds.cols, "taker")).all()     # resting inside the ask is the cheaper way in
    trig = fly.triggers(ds.X[:500], ds.cols)
    assert trig[:, 1].all() and (trig[:, 0] == (ds.X[:500, ds.cols.index("side_ask")] >= 70)).all()
    assert info["photoreceptors"] == 24 and info["calibration"]["trades"] > 0 and set(info["lines"]) == {"favorite", "ev"}
    assert info["calibration"]["per_strategy"]["favorite"]["arm"] == "taker"
    # the fly found the planted signal: higher p̂ where ret_6h is high
    hi = ds.X[:500, ds.cols.index("ret_6h")] > 1; lo = ds.X[:500, ds.cols.index("ret_6h")] < -1
    assert P[hi, 0].mean() > P[lo, 0].mean()
    d = KF.kalshi_decide(fly, ds.X[:500], ds.cols, E, ds.hold_s[:500])
    picked = np.array([s is not None for s in d["strategy"]])
    assert (d["hold_s"][picked] == ds.hold_s[:500][picked]).all() and (d["hold_s"][~picked] == 0).all()


def test_snapshots_keep_their_own_kind_version_and_afferents(db_conn, tmp_path, monkeypatch):
    from fly_trader import config
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path)
    ds = _ds(days=12, per_day=300); fly, info = _boot(ds, 10)
    path, sid = KF.save(fly, info)
    assert path.parent.name == KF.SUBDIR and path.name.startswith(KF.PREFIX)
    row = db_conn.execute("SELECT kind, note FROM brain_snapshots WHERE id = %s", (sid,)).fetchone()
    assert row["kind"] == KF.KIND
    back = KF.load(path, g=_graph(), device="cpu")
    assert isinstance(back, KF.KalshiFlyModel)
    assert back.net.aff_rows.tolist() == list(range(30, 54)) and back.net.scale == 1.0 and back.lines == fly.lines
    assert np.allclose(back.score_all(ds.X[:50]), fly.score_all(ds.X[:50]), atol=1e-5)
    assert KF.latest_current(db_conn)["id"] == sid
    assert fly_selector.latest_current(db_conn, "fly_selector", fly_selector.FLY_VERSION) is None or fly_selector.latest_current(db_conn)["id"] != sid   # not the memecoin fly's
    assert KF.deployable({**info, "data": KF.KALSHI_FLY_VERSION})[0] == (info["gates_ok"] and info["calibration"]["trades"] >= 100 and (info["calibration"]["mean"] or 0) > 0)
    assert KF.deployable({**info, "data": fly_selector.FLY_VERSION})[0] is False


def test_plasticity_learns_from_the_settlement_outcome():
    ds = _ds(days=12, per_day=300); fly, _ = _boot(ds, 10)
    bank = plastic.PlasticBank(fly.net, [(0.0, math.inf), (1e-3, 3.0)], KF.SCALE, learn=fly.net.learn.cpu().numpy(), read=fly.net.read.cpu().numpy())
    X = ds.X[:64]; Y, u0, k = fly.parts_all(X); bank.estimate_nu(Y[:, 0], u0, k)
    tags = plastic.Tags(keys=[(i, 0) for i in range(64)], ts=ds.ts[:64], y_dn=Y[:, 0], u0=u0, k=k, w=torch.ones(2, 64), s=np.zeros(64, np.int64))
    p0, _ = bank.predict(Y[:, 0], u0, k, s=np.zeros(64, np.int64))
    st = bank.update(tags, torch.tensor(ds.outcome[:64]), float(ds.ts[63]) + 1)
    assert abs(st["mean_delta"][0] - float((torch.tensor(ds.outcome[:64]) - p0[0]).clamp(-0.5, 0.5).mean())) < 1e-4     # δ = y − p̂
    assert st["drift"][0] == 0.0 and st["drift"][1] > 0                                                               # the frozen arm never moves


def test_replay_tags_come_due_at_settlement_and_the_verdict_is_versioned(monkeypatch):
    ds = _ds(days=16, per_day=400); fly, boot = _boot(ds, 9)
    seen = []; real = plastic.PlasticBank.update

    def spy(self, tags, r, t):
        assert (tags.ts + ds.hold_s[0] + KR.LABEL_LAG_S <= t + 1e-6).all()          # the market has settled
        assert set(r.tolist()) <= {0.0, 1.0}                                            # the reward is the outcome
        seen.append(len(tags)); return real(self, tags, r, t)

    monkeypatch.setattr(plastic.PlasticBank, "update", spy)
    out = KR.run(ds=ds, fly=fly, boot=boot, start_day=9, configs=[(0.0, math.inf), (1e-3, 3.0)], save_verdict=False)
    assert sum(seen) > 0 and out["data"] == KF.KALSHI_FLY_VERSION and out["arms"] == ["taker", "maker"]
    assert isinstance(out["passed"], bool) and "reason" in out and len(out["configs"]) == 2
    assert out["learning"] and all(len(d["drift"]) == 2 for d in out["learning"])
    if out.get("chosen"):
        assert out["chosen"]["alpha"] > 0
