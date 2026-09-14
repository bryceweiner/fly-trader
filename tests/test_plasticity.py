import numpy as np
import torch

from fly_trader import config
from fly_trader.brain.plasticity import Journal, Plasticity


def _make(tmp_path, eta=0.1, mask=None):
    W0 = torch.full((6, 4), 0.5)
    M = torch.ones(6, 4) if mask is None else mask
    s = torch.tensor([1.0, 1.0, -1.0, -1.0])  # cols 0-1 approach, 2-3 avoid
    p = Plasticity(W0, M, s, batch=2, eta=eta, journal=Journal(tmp_path))
    p.E = torch.zeros(6, 2)
    p.E[:3, 0] = 1.0          # KCs 0-2 eligible in slot 0
    return p


def test_reward_depresses_avoid_and_damped_potentiates_approach(tmp_path):
    p = _make(tmp_path)
    W_before = p.W.clone()
    st = p.step(torch.tensor([1.0, 0.0]), torch.tensor([True, True]))
    dW = p.W - W_before
    assert torch.all(dW[:3, 2:] < 0)                      # avoid columns depressed
    assert torch.all(dW[:3, :2] > 0)                      # approach potentiated ...
    assert torch.allclose(dW[:3, :2], -config.BETA_POT * dW[:3, 2:], atol=1e-6)  # ... damped by β_pot
    assert torch.all(dW[3:] == 0)                         # ineligible KCs untouched
    assert st.n_slots == 2 and st.n_neg == 6 and st.n_pos == 6


def test_punishment_reverses_sign(tmp_path):
    p = _make(tmp_path)
    W_before = p.W.clone()
    p.step(torch.tensor([-1.0, 0.0]), torch.tensor([True, False]))
    dW = p.W - W_before
    assert torch.all(dW[:3, :2] < 0) and torch.all(dW[:3, 2:] > 0)


def test_learn_mask_and_connectome_mask(tmp_path):
    mask = torch.ones(6, 4); mask[0, 2] = 0.0
    p = _make(tmp_path, mask=mask)
    W_before = p.W.clone()
    p.step(torch.tensor([1.0, 1.0]), torch.tensor([False, True]))   # slot 0 not learning, slot 1 has no eligibility
    assert torch.equal(p.W, W_before)
    p.step(torch.tensor([1.0, 0.0]), torch.tensor([True, False]))
    assert p.W[0, 2] == W_before[0, 2]                     # masked synapse never changes


def test_frobenius_cap_and_clip(tmp_path):
    p = _make(tmp_path, eta=config.ETA_MAX)
    p.frob_max = 1e-3
    st = p.step(torch.tensor([3.0, 0.0]), torch.tensor([True, False]))
    assert st.frob_capped and st.frob <= 1e-3 + 1e-9
    p2 = _make(tmp_path, eta=config.ETA_MAX)
    p2.frob_max = 1e9
    for _ in range(200):
        p2.step(torch.tensor([-3.0, 0.0]), torch.tensor([True, False]))
    assert float(p2.W.min()) >= 0.0 and float(p2.W.max()) <= p2.w_max
