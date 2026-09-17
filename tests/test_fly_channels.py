"""One dopamine channel per strategy: compartments come from the raw DAN→MBON mass; channels are balanced by KC→MBON
mass and keep both valences; a strategy's outcome changes only its own KC→MBON synapses; tags stay in due order; one
strategy behaves exactly as before."""
import math

import numpy as np
import pytest
import torch

from fly_trader.brain import connectome as cn, plastic
from fly_trader.train.fly_selector import SCALE, FlyNet
from tests.test_fly_plastic import _graph, _parts


def test_compartments_from_raw_dan_mass():
    pops = {"KC": (0, 2), "MBON_APP": (2, 4), "MBON_AV": (4, 5), "MBON_OTHER": (5, 6), "DAN_PAM": (6, 8), "DAN_PPL1": (8, 9)}
    post = torch.tensor([2, 2, 3, 4, 4]); pre = torch.tensor([6, 7, 7, 8, 6]); v = torch.tensor([5.0, 1.0, 3.0, 4.0, 1.0])
    comp, names = cn.mbon_compartments(torch.stack([post, pre]), v, pops, cell_type=np.array(["", "", "", "", "", "", "PAM01", "PAM02", "PPL101"]))
    assert names == ["PAM01", "PAM02", "PPL101"] and list(comp) == [0, 1, 2, -1]      # MBON 5 gets no DAN


def test_assign_channels_balance_valence_and_fallbacks():
    comp = np.array([0, 0, 1, 2, 3, 4, 5, -1]); mass = np.array([10, 10, 8, 6, 9, 5, 2, 7.0]); sign = np.array([1, 1, 1, 1, -1, -1, -1, 0])
    learn, read = plastic.assign_channels(comp, mass, sign, 2)
    assert (learn.sum(0) <= 1).all() and not learn[:, 7].any() and read[:, 7].all()   # unclaimed MBON: read by all, learned by none
    for s in range(2):
        assert learn[s, :4].any() and learn[s, 4:7].any()                           # both valences in every channel
    l1, r1 = plastic.assign_channels(comp, mass, sign, 1)
    assert l1.all() and r1.all()                                                     # one strategy: as before
    lf, _ = plastic.assign_channels(np.full(8, -1), mass, sign, 3)
    assert lf.sum() == 8 and (lf.sum(0) == 1).all()                                  # no DAN data: each MBON its own compartment


@pytest.fixture()
def net():
    torch.manual_seed(0)
    n = FlyNet(_graph(), obs_dim=5, device="cpu")
    with torch.no_grad():
        n.mb_norm.running_mean.uniform_(-0.2, 0.2); n.mb_norm.running_var.uniform_(0.05, 0.3); n.c_free.uniform_(-0.3, 0.3)
    return n.eval()


def test_a_strategys_outcome_changes_only_its_own_channel(net):
    y_dn, u0, k = _parts(net)
    sign = net.c_sign.numpy(); learn, read = plastic.assign_channels(np.arange(6), np.ones(6), np.where(sign == 0, 1, sign), 2)
    bank = plastic.PlasticBank(net, [(0.05, math.inf)], SCALE, nu=1.0, step_cap=1e9, learn=learn, read=read)
    r = torch.ones(len(y_dn)) * 0.05
    tags = plastic.Tags(list(range(len(r))), np.zeros(len(r)), y_dn, u0, k, torch.ones(1, len(r)), s=np.zeros(len(r), int))
    bank.update(tags, r, 0.0)
    moved = (bank.D[0].abs().sum(0) > 0).numpy()
    assert moved[learn[0]].any() and not moved[learn[1]].any()                       # channel 1 untouched by strategy 0's outcome
    per = plastic.PlasticBank(net, [(0.0, math.inf), (0.05, math.inf)], SCALE, nu=1.0, step_cap=1e9, learn=learn, read=read)
    per.D[1] = bank.D[0]
    one = per.select_per_strategy([1, 0])
    assert torch.allclose(one.D[0][:, learn[0]], bank.D[0][:, learn[0]]) and (one.D[0][:, learn[1]] == 0).all()


def test_pending_tags_stay_in_due_order(net):
    y_dn, u0, k = _parts(net, B=3)
    q = plastic.PendingTags(net.n_kc, net.k_active)
    q.push(["long"], np.zeros(1), 14400.0, y_dn[:1], u0[:1], k[:1], torch.ones(1, 1), s=2)
    q.push(["short"], np.zeros(1), 600.0, y_dn[1:2], u0[1:2], k[1:2], torch.ones(1, 1), s=1)
    t = q.pop_due(1000.0)
    assert t.keys == ["short"] and list(t.s) == [1] and len(q) == 1
    assert q.pop_due(20000.0).keys == ["long"]
