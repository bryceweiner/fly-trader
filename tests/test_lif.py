"""Batched LIF kernel: toy connectomes with explicit tensors, then the real artifact if it is built."""
import math

import numpy as np
import pytest
import torch

from fly_trader.brain import lif as L

LEAK, THETA, REFR = 0.95, 1.0, 3
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def toy(pop_ranges, N, indices=None, values=None, device="cpu", s=1.0, **kw):
    if indices is None:
        indices = np.zeros((2, 0), dtype=np.int64)
        values = np.zeros(0, dtype=np.float32)
    return L.Connectome.from_arrays(np.asarray(indices), np.asarray(values, dtype=np.float32), N, pop_ranges,
                                    device=torch.device(device), s=s, **kw)


def make_lif(conn, batch=1, **kw):
    kw.setdefault("leak", LEAK)
    kw.setdefault("theta", THETA)
    kw.setdefault("refractory", REFR)
    kw.setdefault("kwta", False)
    kw.setdefault("noise", 0.0)
    return L.LIF(conn, batch, ticks=50, readout_window=30, **kw)


@pytest.mark.parametrize("device", DEVICES)
def test_single_neuron_steady_state_and_first_spike(device):
    conn = toy({"OTHER": (0, 1)}, 1, device=device)
    # sub-threshold: V -> I / (1 - leak) = 0.8
    lif = make_lif(conn)
    lif.set_input(torch.full((1, 1), 0.04))
    r = lif.run(300)
    expected = 0.04 / (1 - LEAK) * (1 - LEAK ** 300)
    assert r.total_spikes == 0
    assert abs(float(lif.V[0, 0]) - expected) < 1e-4
    assert abs(expected - 0.8) < 1e-6
    # supra-threshold: V_t = 2 (1 - 0.95^t) >= 1 first at t = 14
    lif = make_lif(conn)
    lif.set_input(torch.full((1, 1), 0.10))
    first = None
    for t in range(1, 40):
        r = lif.run(1)
        if r.total_spikes > 0:
            first = t
            break
        assert abs(float(lif.V[0, 0]) - 2.0 * (1 - LEAK ** t)) < 1e-4
    expected_first = math.ceil(math.log(0.5) / math.log(LEAK))
    assert first == expected_first == 14
    assert float(lif.V[0, 0]) == 0.0  # reset after the spike


@pytest.mark.parametrize("device", DEVICES)
def test_refractory_holds_potential(device):
    conn = toy({"OTHER": (0, 1)}, 1, device=device)
    lif = make_lif(conn, refractory=REFR)
    lif.set_input(torch.full((1, 1), 0.10))
    spikes = []
    for t in range(1, 40):
        r = lif.run(1)
        spikes.append(r.total_spikes)
        if r.total_spikes > 0:
            break
    assert spikes[-1] == 1
    for _ in range(REFR):                       # held at 0 for `refractory` ticks
        r = lif.run(1)
        assert r.total_spikes == 0
        assert float(lif.V[0, 0]) == 0.0
    r = lif.run(1)                              # integration resumes: V = 0 * leak + I
    assert abs(float(lif.V[0, 0]) - 0.10) < 1e-6
    # integration restarts from 0, so the second spike comes 14 ticks after the hold ends
    n = 1                                       # the tick above (V = 0.10) is integration tick 1
    while True:
        n += 1
        assert n < 40
        if lif.run(1).total_spikes > 0:
            break
    assert n == 14


@pytest.mark.parametrize("device", DEVICES)
def test_kwta_limits_kc_spikes_per_column(device):
    n = 100
    conn = toy({"KC": (0, n)}, n, device=device)
    B = 3
    lif = make_lif(conn, batch=B, kwta=True, kwta_frac=0.05)
    assert lif.k == 5
    I = 0.30 + 0.001 * torch.arange(n, dtype=torch.float32)[:, None].expand(n, B)
    lif.set_input(I)
    seen = 0
    for _ in range(30):
        r = lif.run(1)
        per_col = r.spike_sum.sum(0).cpu()
        assert (per_col <= lif.k).all()
        if per_col.max() > 0:
            if seen == 0:                                   # first firing tick: the largest potentials win
                winners = torch.nonzero(r.spike_sum[:, 0].cpu()).flatten().tolist()
                assert winners == list(range(n - lif.k, n))
            seen += 1
    assert seen > 0
    # without k-WTA all neurons fire on the same tick
    lif2 = make_lif(conn, batch=B, kwta=False)
    lif2.set_input(I)
    r = lif2.run(4)
    assert r.total_spikes == n * B


@pytest.mark.parametrize("device", DEVICES)
def test_two_neuron_chain_one_tick_delay(device):
    # W[post=1, pre=0] = 1.0 (excitatory), neuron 0 driven to fire on tick 1
    conn = toy({"OTHER": (0, 2)}, 2, indices=[[1], [0]], values=[1.0], device=device)
    lif = make_lif(conn)
    lif.set_input(torch.tensor([[1.0], [0.0]]))
    r1 = lif.run(1)
    assert r1.spike_sum[:, 0].cpu().tolist() == [1.0, 0.0]
    r2 = lif.run(1)
    assert r2.spike_sum[:, 0].cpu().tolist() == [0.0, 1.0]   # propagated with one tick of delay
    # inhibitory edge never drives the target
    conn_i = toy({"OTHER": (0, 2)}, 2, indices=[[1], [0]], values=[-1.0], device=device)
    lif = make_lif(conn_i)
    lif.set_input(torch.tensor([[1.0], [0.0]]))
    r = lif.run(20)
    assert float(r.spike_sum[1, 0]) == 0.0
    assert float(lif.V[1, 0]) < 0.0


def test_scale_and_clip():
    conn = toy({"OTHER": (0, 2)}, 2, indices=[[1], [0]], values=[4.0], s=0.5)
    assert float(conn.W.to_dense()[1, 0]) == 1.0        # clipped at w_clip
    conn.scale(0.1)
    assert abs(float(conn.W.to_dense()[1, 0]) - 0.4) < 1e-6


@pytest.mark.skipif(not L.CURRENT_FILE.exists(), reason="connectome not built")
def test_real_connectome_silence_and_odor():
    conn = L.Connectome.load()
    assert conn.N == 139_255 and conn.n_KC == 5_177 and conn.n_MBON == 96
    assert conn.W_KM0.shape == (5_177, 96) and conn.M_KM.shape == (5_177, 96)
    assert conn.mbon_app_cols.numel() + conn.mbon_av_cols.numel() <= 96
    assert conn.glomerulus_of_orn.shape[0] == conn.n("ORN_FOOD")
    assert isinstance(conn.pop_ranges, dict) and conn.pop_ranges["KC"] == (0, 5_177)
    lif = L.LIF(conn, 8, ticks=50)
    r = lif.run(50)
    assert r.total_spikes == 0 and not r.nan_flag
    A = np.zeros((conn.n_glomeruli, 8), dtype=np.float32)
    rng = np.random.default_rng(0)
    for b in range(8):
        A[rng.choice(conn.n_glomeruli, 12, replace=False), b] = 0.10
    lif.set_glomerulus_input(A)
    r = lif.run(50)
    assert r.total_spikes > 0 and not r.nan_flag
    assert r.m_hat.shape == (8,) and r.mbon_app_rate.shape == (8,) and r.dan_pam_rate.shape == (8,)
    assert r.kc_rates.shape == (5_177, 8) and r.pop_rates.shape == (conn.P, 8) and r.kc_active_frac.shape == (8,)
    assert torch.isfinite(r.m_hat).all() and (r.m_hat.abs() < 1).all()
    lif.reset([0, 1])
    assert float(lif.V[:, :2].abs().sum()) == 0.0
