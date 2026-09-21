"""Mushroom-body plasticity: the local three-factor rule is exactly gradient descent on the weighted squared prediction
error, the shadow is the bootstrap fly, changes stay on connectome synapses within their bounds and step cap, decay has
the exact half-life, and tags come back as they were pushed, only once due."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fly_trader.brain import plastic
from fly_trader.train.fly_selector import SCALE, FlyNet

POPS = {"KC": (0, 20), "MBON_APP": (20, 22), "MBON_AV": (22, 24), "MBON_OTHER": (24, 26), "ORN_FOOD": (26, 56), "DESCENDING": (70, 80)}


def _graph(seed=0):
    g = torch.Generator().manual_seed(seed); pre, post = [], []
    for a, b in ((range(26, 56), range(0, 20)), (range(26, 56), range(56, 70)), (range(56, 70), range(70, 80)), (range(20, 26), range(70, 80))):
        for j in b:
            for i in torch.randint(a.start, a.stop, (6,), generator=g).tolist():
                pre.append(i); post.append(j)
    vals = (torch.rand(len(pre), generator=g) * 5 + 1) * torch.where(torch.rand(len(pre), generator=g) < 0.8, 1.0, -1.0)
    W = torch.randint(1, 12, (20, 6), generator=g).float() * (torch.rand(20, 6, generator=g) < 0.4)
    return SimpleNamespace(N=80, indices=torch.tensor([post, pre]), values_raw=vals, pop_ranges=dict(POPS), W_KM0=W, M_KM=W > 0,
                           device=torch.device("cpu"), s=0.05, spectral_radius_raw=None)


@pytest.fixture()
def net():
    torch.manual_seed(0)
    n = FlyNet(_graph(), obs_dim=5, device="cpu")
    with torch.no_grad():                                                   # non-trivial frozen normalisation and readout
        n.mb_norm.running_mean.uniform_(-0.2, 0.2); n.mb_norm.running_var.uniform_(0.05, 0.3)
        n.c_free.uniform_(-0.3, 0.3); n.gain.uniform_(0.8, 1.2)
    return n.eval()


def _parts(net, B=32, seed=1):
    torch.manual_seed(seed)
    with torch.no_grad():
        return net.forward_parts(torch.randn(B, 5) * 2)


def test_local_rule_equals_autograd_gradient_descent(net):
    y_dn, u0, k = _parts(net)
    bank = plastic.PlasticBank(net, [(0.005, math.inf)], SCALE, nu=1.0, step_cap=1e9)
    bank.D[0] = (torch.rand_like(bank.w0) - 0.5) * 0.5 * bank.w0             # start away from zero, well inside [−w0, 2·w0]
    D0 = bank.D[0].clone()
    with torch.no_grad():
        yhat = net.readout(y_dn, u0, k, D0) / SCALE
    r = yhat + torch.randn_like(yhat) * 0.01                                 # errors well inside the clips
    w = torch.where(torch.rand(len(r)) < 0.3, plastic.TOP_WEIGHT, 1.0)
    Dp = D0.clone().requires_grad_(True)
    loss = 0.5 * (w * (r - net.readout(y_dn, u0, k, Dp) / SCALE) ** 2).sum()
    loss.backward()
    expected = D0 - 0.005 * Dp.grad * bank.mask
    tags = plastic.Tags(keys=list(range(len(r))), ts=np.zeros(len(r)), y_dn=y_dn, u0=u0, k=k, w=w[None])
    bank.update(tags, r, t=0.0)
    assert torch.allclose(bank.D[0], expected, atol=1e-7, rtol=1e-4)


def test_shadow_is_the_bootstrap_fly_and_predictions_match_the_net(net):
    y_dn, u0, k = _parts(net)
    bank = plastic.PlasticBank(net, [(1e-3, 7.0), (0.0, math.inf)], SCALE)
    with torch.no_grad():
        assert torch.allclose(bank.frozen(y_dn, u0), net.readout(y_dn, u0, k) / SCALE, atol=1e-6)
        bank.D[0] = torch.rand_like(bank.w0) * bank.w0
        s, _ = bank.predict(y_dn, u0, k)
        assert torch.allclose(s[0], net.readout(y_dn, u0, k, bank.D[0]) / SCALE, atol=1e-6)
        assert torch.allclose(s[1], bank.frozen(y_dn, u0), atol=1e-6)


def test_changes_stay_on_synapses_within_bounds_and_step_cap(net):
    y_dn, u0, k = _parts(net)
    bank = plastic.PlasticBank(net, [(1e6, math.inf)], SCALE, nu=1.0)       # absurd rate: the caps must hold
    r = torch.ones(len(y_dn)); w = torch.ones(1, len(r))
    tags = plastic.Tags(list(range(len(r))), np.zeros(len(r)), y_dn, u0, k, w)
    before = bank.D.clone(); st = bank.update(tags, r, 0.0)
    assert st["capped"] == [True]
    assert float(torch.linalg.norm(bank.D - before)) <= plastic.STEP_CAP * bank.w0_norm * (1 + 1e-5)
    for _ in range(3000):
        bank.update(tags, r, 0.0)
    assert (bank.D[:, ~bank.mask.bool()] == 0).all()
    assert (bank.D >= -bank.w0 - 1e-7).all() and (bank.D <= 2 * bank.w0 + 1e-7).all()


def test_decay_has_the_exact_half_life(net):
    bank = plastic.PlasticBank(net, [(1e-3, 3.0), (1e-3, math.inf)], SCALE)
    bank.D[:] = bank.w0 * 0.5; bank.decay_to(100.0); bank.decay_to(100.0 + 3 * plastic.DAY_S)
    assert torch.allclose(bank.D[0], bank.w0 * 0.25) and torch.allclose(bank.D[1], bank.w0 * 0.5)


def test_bias_toward_depression_is_optional_and_state_round_trips(net):
    y_dn, u0, k = _parts(net)
    bank = plastic.PlasticBank(net, [(0.05, 7.0)], SCALE, nu=1.0, beta_pot=0.5, step_cap=1e9)
    bank.update(plastic.Tags(list(range(len(y_dn))), np.zeros(len(y_dn)), y_dn, u0, k, torch.ones(1, len(y_dn))), torch.randn(len(y_dn)) * 0.05, 5.0)
    other = plastic.PlasticBank(net, [(0.05, 7.0)], SCALE)
    other.load_state(bank.state())
    assert torch.allclose(other.D, bank.D) and other.t_last == 5.0 and other.nu == 1.0 and other.beta_pot == 0.5
    with pytest.raises(ValueError):
        plastic.PlasticBank(net, [(0.1, 7.0)], SCALE).load_state(bank.state())
    one = plastic.PlasticBank(net, [(0.0, 1.0), (0.05, 7.0)], SCALE); one.D[1] = bank.D[0]
    assert torch.allclose(one.select(1).D[0], bank.D[0])


def test_pending_tags_return_as_pushed_and_only_when_due(net):
    y_dn, u0, k = _parts(net, B=10)
    q = plastic.PendingTags(net.n_kc, net.k_active)
    q.push([("a", i) for i in range(4)], np.full(4, 0.0), 100.0, y_dn[:4], u0[:4], k[:4], torch.ones(2, 4))
    q.push([("b", i) for i in range(6)], np.full(6, 60.0), 160.0, y_dn[4:], u0[4:], k[4:], torch.full((2, 6), 10.0))
    assert len(q) == 10 and q.pop_due(99.0) is None
    t = q.pop_due(130.0)
    assert t.keys == [("a", i) for i in range(4)] and len(q) == 6
    assert torch.allclose(t.k, k[:4]) and torch.allclose(t.u0, u0[:4]) and tuple(t.w.shape) == (2, 4)
    t2 = q.pop_due(1e9)
    assert len(t2) == 6 and len(q) == 0 and float(t2.w[0, 0]) == 10.0
    assert plastic.row_weights(torch.tensor([[0.1, 0.3]]), torch.tensor([0.2])).tolist() == [[0.0, 1.0]]   # below the line teaches nothing
    two = plastic.row_weights(torch.tensor([[0.1, 0.3], [0.1, 0.3]]), torch.tensor([0.2, 0.5]))
    assert two.tolist() == [[0.0, 1.0], [0.0, 0.0]]              # each configuration judges against its own line
