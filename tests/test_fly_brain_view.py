"""The console's 3D brain view: the network's activity comes out of the forward pass without changing any score, is
kept as one small file per minute, the most-changed KC→MBON synapses rank by their share of the base weight, and the
neurons' positions join the connectome's index the way the fly's sub-graph is indexed."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from fly_trader.brain import activity, geometry, pathways
from fly_trader.train.fly_selector import FlyModel, FlyNet
from fly_trader.train.scaling import RobustScaler
from tests.test_fly_plastic import _graph


def test_the_activity_comes_out_of_the_forward_pass_without_changing_it():
    torch.manual_seed(0)
    net = FlyNet(_graph(), obs_dim=5, device="cpu").eval(); x = torch.randn(16, 5)
    with torch.no_grad():
        y, u, k, h = net.forward_parts_all_h(x)
    assert tuple(h.shape) == (16, 80) and float(h.abs().max()) <= 1.0                        # every neuron, tanh rates
    assert (h[:, :20] >= 0).all() and int((h[:, :20] > 0).sum(1).max()) <= net.k_active     # the KC block after k-WTA
    y2, u2, k2 = net.forward_parts_all(x)
    assert torch.equal(y, y2) and torch.equal(u, u2) and torch.equal(k, k2)                  # the scoring path is untouched
    X = np.random.default_rng(0).normal(size=(16, 5)).astype(np.float32)
    fly = FlyModel(net, RobustScaler.fit(X), [f"f{i}" for i in range(5)], 30)
    a, b = fly.parts_all(X), fly.parts_all_h(X)
    assert len(b) == 4 and all(torch.equal(p, q) for p, q in zip(a, b[:3])) and tuple(b[3].shape) == (16, 80)


def test_activity_files_round_trip_and_prune(tmp_path):
    q = activity.quantise(np.array([1.0, -1.0, 0.5, 1.3, 0.0]))
    assert q.dtype == np.int8 and q.tolist() == [127, -127, 64, 127, 0]
    assert activity.quantise(torch.tensor([0.25])).tolist() == [32]
    d = tmp_path / "activity"; t = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc).timestamp()
    for dt in (0, 60, 7200):
        activity.write(d, t + dt, activity.quantise(np.full(8, 0.1 * (dt // 60 + 1))), n_rows=9, n_cand=3, connectome="c.npz")
    (d / ".20260922T1300.tmp.npz").write_bytes(b"partial")                                  # an interrupted write
    assert activity.listing(d) == ["20260922T1200", "20260922T1201", "20260922T1400"] and activity.newest(d) == "20260922T1400"
    r = activity.load(activity.path_of(d, "20260922T1201"))
    assert r["h"].tolist() == [25] * 8 and r["t"] == t + 60 and (r["n_rows"], r["n_cand"], r["connectome"]) == (9, 3, "c.npz")
    assert activity.parse("20260922T1201") == t + 60
    assert activity.prune(d, before=t + 30) == 1 and activity.listing(d) == ["20260922T1201", "20260922T1400"]


def test_top_pathways_rank_by_share_of_the_base_weight_and_diff_against_a_reference():
    kc = np.array([0, 1, 2, 3, 4, 5]); mbon = np.array([0, 1, 2, 0, 1, 2]); w0 = np.array([1.0, 2.0, 1.0, 4.0, 1.0, 2.0])
    D = np.array([0.5, -1.0, 0.0, 0.4, 0.3, -0.2]); learn = np.array([[True, False, False], [False, True, False]])
    items = pathways.top_pathways(D, w0, kc, mbon, learn, ["ev", "cap"], n_app=1, n_av=1, top_k=3, mbon_offset=100)
    assert [i["kc"] for i in items] == [0, 1, 4]                                              # ratios 0.5, 0.5, 0.3 (0.1, 0.1 dropped; 0 skipped)
    assert items[0] == {"kc": 0, "mbon": 100, "delta": 0.5, "ratio": 0.5, "strategy": "ev", "valence": "approach"}
    assert items[1]["delta"] < 0 and items[1]["strategy"] == "cap" and items[1]["valence"] == "avoid"
    assert pathways.top_pathways(D, w0, kc, mbon, learn, ["ev", "cap"], 1, 1, top_k=6)[-1]["strategy"] == "shared"   # column 2: no owner
    ref = D.copy(); ref[0] = 0.5                                                                # unchanged since the reference
    since = pathways.top_pathways(D, w0, kc, mbon, learn, ["ev", "cap"], 1, 1, top_k=6, D_ref=ref)
    assert 0 not in [i["kc"] for i in since]
    assert pathways.top_pathways(np.zeros(6), w0, kc, mbon, learn, ["ev", "cap"], 1, 1) == []


def test_reference_snapshot_prefers_the_newest_at_least_a_day_old(tmp_path):
    now = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc); rows = []
    for hours, boot, exists in ((1, 62, True), (25, 62, True), (26, 62, False), (30, 61, True), (40, 62, True)):
        p = tmp_path / f"snap_{hours}.pt"
        if exists:
            p.write_bytes(b"x")
        rows.append({"ts": now - timedelta(hours=hours), "path": str(p), "note": json.dumps({"bootstrap_id": boot})})
    assert pathways.reference_snapshot(rows, now.timestamp(), 62).endswith("snap_25.pt")       # newest ≥ 24 h, right bootstrap, file present
    assert pathways.reference_snapshot(rows[:1], now.timestamp(), 62).endswith("snap_1.pt")     # nothing old enough: the oldest there is
    assert pathways.reference_snapshot(rows, now.timestamp(), 99) is None


def _npz(path: Path, sha: str):
    pops = {"KC": [0, 2], "MBON_APP": [2, 3], "MBON_AV": [3, 4], "MBON_OTHER": [4, 4], "VISUAL": [4, 6], "OTHER": [6, 8]}
    W = np.zeros((2, 2), np.float32); W[0, 0] = 3; W[1, 1] = 1
    np.savez(path, N=8, indices=np.zeros((2, 0), np.int64), values_raw=np.zeros(0, np.float32), pop_ranges=json.dumps(pops),
             pop_order=np.array(list(pops)), root_ids=np.arange(10, 18, dtype=np.int64), cell_type=np.array(["KCa", "KCb", "M1", "M2", "V", "V", "X", "KCa"]),
             W_KM0=W, M_KM=W != 0, content_sha256=sha)


def _tsv(path: Path):
    rows = ["\t".join(["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "side"])]
    rows.append("10\t100\t100\t10\t200\t200\t20\tleft")            # soma and pos: soma wins
    rows.append("11\t400\t400\t40\t\t\t\tright")                  # pos only
    rows.append("12\t\t\t\t\t\t\tleft")                           # neither: its population's centroid
    rows.append("13\t1000\t1000\t100\t1000\t1000\t100\tleft")
    rows.append("14\t5\t5\t5\t5\t5\t5\tleft"); rows.append("15\t5\t5\t5\t5\t5\t5\tleft")   # VISUAL: dropped
    rows.append("16\t0\t0\t0\t0\t0\t0\tleft"); rows.append("17\t2000\t2000\t200\t\t\t\tleft")
    path.write_text("\n".join(rows) + "\n")


def test_build_geometry_joins_positions_scales_and_is_idempotent(tmp_path):
    npz, tsv, out = tmp_path / "c.npz", tmp_path / "ann.tsv", tmp_path / "static"
    _npz(npz, "sha-a"); _tsv(tsv)
    m = geometry.build_geometry(out, npz, tsv)
    xyz = np.frombuffer((out / "geometry.bin").read_bytes(), dtype=np.float32).reshape(-1, 3)
    assert m["n"] == 6 and xyz.shape == (6, 3) and m["excluded"] == ["VISUAL"] and "VISUAL" not in m["pop_ranges"]
    assert m["pop_ranges"] == {"KC": [0, 2], "MBON_APP": [2, 3], "MBON_AV": [3, 4], "OTHER": [4, 6]}      # re-indexed as the sub-graph
    assert m["pop_order"] == ["KC", "MBON_APP", "MBON_AV", "OTHER"] and m["coverage"] == {"soma": 3, "pos": 2, "centroid": 1}
    raw = xyz + np.array(m["center_um"], np.float32)                                                       # undo the centring
    assert np.allclose(raw[0], [200 * 0.004, 200 * 0.004, 20 * 0.04], atol=1e-4)                           # soma preferred, voxel → µm
    assert np.allclose(raw[1], [400 * 0.004, 400 * 0.004, 40 * 0.04], atol=1e-4)                           # pos when no soma
    assert np.allclose(raw[2], raw[[0, 1, 3, 4, 5]].mean(0), atol=1e-3)                                     # alone in MBON_APP: the global centroid
    assert np.allclose(xyz.mean(0), 0, atol=1e-4)
    names = m["cell_type"]["names"]; assert [names[i] for i in m["cell_type"]["index"]] == ["KCa", "KCb", "M1", "M2", "X", "KCa"]
    mt = (out / "geometry.bin").stat().st_mtime_ns
    assert geometry.build_geometry(out, npz, tsv) == m and (out / "geometry.bin").stat().st_mtime_ns == mt   # no rewrite
    _npz(npz, "sha-b")
    assert geometry.build_geometry(out, npz, tsv)["connectome_sha256"] == "sha-b"                            # a new connectome rebuilds
