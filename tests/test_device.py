"""One resolver for CPU, CUDA and MPS (a named backend that is missing falls back with one warning), batch sizes from
free memory and the graph's size (reproducing the measured calibration), and the network's arithmetic agreeing with
the CPU on whichever accelerator this machine has."""
import logging

import numpy as np
import pytest
import torch

from fly_trader.brain import device, plastic
from fly_trader.train.fly_selector import SCALE, FlyNet
from tests.test_fly_plastic import _graph

ACCELERATORS = [d for d in ("cuda", "mps") if device._available(d)]


def test_resolve_honours_the_name_and_falls_back_with_one_warning(caplog, monkeypatch):
    device.reset()
    assert device.resolve("cpu").type == "cpu" and device.resolve("CPU").type == "cpu"
    assert device.resolve("auto").type == (ACCELERATORS[0] if ACCELERATORS else "cpu")
    missing = next((d for d in ("cuda", "mps") if d not in ACCELERATORS), None)
    if missing:
        with caplog.at_level(logging.WARNING, logger="fly_trader.brain.device"):
            assert device.resolve(missing).type == "cpu" and device.resolve(missing + ":0").type == "cpu"
            assert device.resolve(missing).type == "cpu"
        assert sum("not available" in r.message for r in caplog.records) == 2               # once per name: cached
    with caplog.at_level(logging.WARNING, logger="fly_trader.brain.device"):
        assert device.resolve("tpu").type == "cpu"
    assert any("not cpu, cuda, mps or auto" in r.message for r in caplog.records)
    monkeypatch.setattr(device.config, "DEVICE", "cpu"); device.reset()
    assert device.resolve().type == "cpu"                                                    # the default reads config.DEVICE
    device.reset()


def test_rows_for_follows_memory_and_the_graph_and_reproduces_the_calibration():
    edges, n = 1_039_659, 41_756                                                             # the FAFB v783 sub-graph
    free = int((19.5 * 2**30 + device.FIXED_BYTES) / device.BUDGET_SHARE)                   # what 2,048 rows measured at
    assert 1800 <= device.rows_for(edges, n, ceiling=4096, free=free) <= 2300
    small, big = (device.rows_for(edges, n, ceiling=4096, free=gb * 2**30) for gb in (8, 24))
    assert device.MIN_ROWS <= small < big                                                    # an 8 GB card gets fewer rows than a 24 GB one
    assert device.rows_for(edges, n, ceiling=4096, free=8 * 2**30, training=True) < small    # training keeps every step: fewer still
    assert device.rows_for(edges, n, ceiling=256, free=10**13) == 256                        # the caller's ceiling
    assert device.rows_for(edges, n, ceiling=4096, free=0) == device.MIN_ROWS                # never zero
    assert device.rows_for(100, 80, ceiling=500, free=2**30) == 500                          # a toy graph fits whole


def test_the_memory_override_caps_the_budget(monkeypatch):
    monkeypatch.setenv(device.MEMORY_GB_ENV, "2")
    assert device.free_bytes(torch.device("cpu")) == 2 * 2**30
    monkeypatch.setenv(device.MEMORY_GB_ENV, "lots")
    assert device.free_bytes(torch.device("cpu")) > 2 * 2**30                                # unparseable: measured instead


def test_describe_batch_rows_and_seeding():
    assert device.describe(torch.device("cpu")).startswith("cpu (")
    net = FlyNet(_graph(), obs_dim=5, device="cpu")
    assert net.batch_rows(ceiling=2048) == 2048 and net.batch_rows(ceiling=64, training=True) == 64   # a toy graph: the ceilings bind
    device.seed_all(3); a = torch.randn(3); device.seed_all(3)
    assert torch.equal(a, torch.randn(3))
    device.empty_cache(torch.device("cpu"))                                                  # a no-op that must not raise


@pytest.mark.parametrize("acc", ACCELERATORS or [pytest.param("none", marks=pytest.mark.skip(reason="no accelerator on this machine"))])
def test_the_network_and_its_plasticity_agree_with_the_cpu_on(acc):
    torch.manual_seed(0); g = _graph()
    cpu = FlyNet(g, obs_dim=5, device="cpu").eval(); other = FlyNet(g, obs_dim=5, device=acc).eval()
    other.load_state_dict({k: v.to(acc) for k, v in cpu.state_dict().items()})
    x = torch.randn(64, 5)
    with torch.no_grad():
        pa = cpu.forward_parts_all_h(x); pb = other.forward_parts_all_h(x.to(acc))
    for a, b in zip(pa, pb):
        assert torch.allclose(a, b.cpu(), atol=2e-4, rtol=1e-3)                              # scatter, k-WTA and the decoder agree
    assert ((pa[2] > 0) == (pb[2].cpu() > 0)).float().mean() > 0.98                          # the same Kenyon cells win
    learn = np.ones((1, cpu.n_mbon), dtype=bool); B = 16; keys = [(0.0, f"m{i}", "ev") for i in range(B)]
    out = []
    for net, (y, u, k, _) in ((cpu, pa), (other, pb)):
        bank = plastic.PlasticBank(net, [(1e-3, 1.0)], SCALE, learn=learn, read=learn)
        bank.estimate_nu(y[:B, 0], u[:B], k[:B])
        tags = plastic.Tags(keys, np.zeros(B), y[:B, 0], u[:B], k[:B], torch.ones(1, B, device=bank.dev))
        bank.update(tags, torch.linspace(-0.05, 0.05, B, device=bank.dev), 60.0)
        pred, _ = bank.predict(y[:B, 0], u[:B], k[:B], s=np.zeros(B, dtype=np.int64))
        out.append((bank.D.detach().cpu(), pred.detach().cpu(), float(bank.drift()[0])))
    assert torch.allclose(out[0][0], out[1][0], atol=1e-7, rtol=1e-3)                        # the learned change
    assert torch.allclose(out[0][1], out[1][1], atol=2e-4, rtol=1e-3)                        # and the prediction it feeds
    assert out[0][2] == pytest.approx(out[1][2], rel=1e-3, abs=1e-7)
