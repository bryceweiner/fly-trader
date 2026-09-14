import numpy as np, torch, pytest
from fly_trader.brain.policy import ConnectomePolicy


class TinyConnectome:
    def __init__(self):
        self.N = 40
        # 8 afferent (ORN_FOOD 0-7), 24 intrinsic (8-31), 8 descending (32-39); random sparse edges
        rng = np.random.default_rng(0)
        pre = rng.integers(0, 32, 300); post = rng.integers(8, 40, 300)
        self.indices = torch.tensor(np.stack([post, pre]), dtype=torch.int64)
        self.values_raw = torch.tensor(rng.choice([-3.0, 2.0, 5.0], 300), dtype=torch.float32)
        self.pop_ranges = {"ORN_FOOD": (0, 8), "DESCENDING": (32, 40)}
        self.device = torch.device("cpu")
        self.s = 0.1
        self.spectral_radius_raw = 10.0


def test_shapes_and_gradients_and_dale():
    c = TinyConnectome()
    pol = ConnectomePolicy(c, obs_dim=5, d_enc=4, k_steps=3, device="cpu")
    obs = torch.randn(6, 5); h = pol.init_hidden(6)
    out = pol(obs, h)
    assert out.mu.shape == (6,) and out.value.shape == (6,) and out.h.shape == (40, 6)
    u, logp = pol.sample(out.mu, out.log_std)
    a = pol.to_action(u)
    assert torch.all((a > 0) & (a < 1))
    logp_fixed = pol.log_prob(out.mu, out.log_std, u.detach())   # PPO evaluates log-probs at stored actions
    loss = -(logp_fixed * torch.randn(6)).mean() + out.value.pow(2).mean()
    loss.backward()
    assert pol.theta.grad is not None and pol.theta.grad.abs().sum() > 0
    assert pol.w_in.grad.abs().sum() > 0 and pol.pi_head.weight.grad.abs().sum() > 0
    w = pol.weights()
    assert torch.all(torch.sign(w) == pol.sign)          # Dale's law: signs never flip
    assert pol.describe()["afferent"] == 8 and pol.describe()["efferent"] == 8
