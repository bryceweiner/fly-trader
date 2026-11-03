"""The connectome as the fly sees it: scale.json's pathway gains drop the silenced edges, the mushroom-body block
(KC→MBON, kept apart from the sparse graph by the build) survives a sub-connectome, and the real artifact has no
KC→MBON edge in its sparse graph."""
import json

import numpy as np
import pytest

from fly_trader.brain import connectome as cn

POPS = {"KC": [0, 4], "MBON_APP": [4, 5], "MBON_AV": [5, 6], "MBON_OTHER": [6, 7], "DAN_PAM": [7, 8], "VISUAL": [8, 10], "DESCENDING": [10, 12]}


def _artifact(tmp_path, monkeypatch, gains):
    post, pre = [4, 0, 10, 11, 9, 10], [7, 7, 4, 5, 8, 9]       # DAN→MBON, DAN→KC, MBON→DN ×2, VISUAL→VISUAL, VISUAL→DN
    W = np.zeros((4, 3), np.float32); W[0, 0] = 7; W[1, 1] = 2; W[3, 2] = 1
    path = tmp_path / "t.npz"
    np.savez(path, N=12, indices=np.array([post, pre], np.int64), values_raw=np.array([3, 2, 5, -4, 1, 6], np.float32),
             pop_ranges=json.dumps(POPS), W_KM0=W, M_KM=W != 0, spectral_radius_raw=10.0, content_sha256="x", s0=0.1)
    monkeypatch.setattr(cn, "SCALE_FILE", tmp_path / "scale.json")
    (tmp_path / "scale.json").write_text(json.dumps({"connectome": "t.npz", "s": 0.05, "pathway_gains": gains}))
    return path


def test_pathway_gains_drop_silenced_edges_and_the_mb_block_loads(tmp_path, monkeypatch):
    c = cn.Connectome(_artifact(tmp_path, monkeypatch, {"DAN_PAM->MBON_APP": 0.0, "DAN_PAM->KC": 0.0, "MBON_AV->DESCENDING": 0.5}), device="cpu")
    pre = c.indices[1].tolist()
    assert 7 not in pre and c.values_raw.numel() == 4                               # both DAN edges gone
    assert float(c.values_raw[c.indices[1] == 5]) == pytest.approx(-2.0)           # scaled, sign kept
    assert c.s == pytest.approx(0.05) and tuple(c.W_KM0.shape) == (4, 3) and int(c.M_KM.sum()) == 3


def test_sub_connectome_keeps_the_mushroom_body_and_refuses_to_drop_it(tmp_path, monkeypatch):
    c = cn.Connectome(_artifact(tmp_path, monkeypatch, {}), device="cpu")
    sub = cn.SubConnectome(c, exclude=("VISUAL",))
    assert sub.N == 10 and sub.pop_ranges["KC"] == (0, 4) and sub.pop_ranges["DESCENDING"] == (8, 10)
    assert sub.W_KM0 is c.W_KM0 and sub.M_KM is c.M_KM
    assert sub.indices.shape[1] == 4 and int(sub.indices[1].max()) <= 7            # both VISUAL edges dropped
    with pytest.raises(ValueError, match="mushroom body"):
        cn.SubConnectome(c, exclude=("KC",))


def test_real_artifact_has_no_kc_to_mbon_edge_in_the_sparse_graph():
    try:
        c = cn.Connectome.load(device="cpu")
    except FileNotFoundError:
        pytest.skip("no connectome artifact")
    (k0, k1), m0, m1 = c.pop_ranges["KC"], c.pop_ranges["MBON_APP"][0], c.pop_ranges["MBON_OTHER"][1]
    post, pre = c.indices[0], c.indices[1]
    assert not bool(((pre >= k0) & (pre < k1) & (post >= m0) & (post < m1)).any())
    assert tuple(c.W_KM0.shape) == (k1 - k0, m1 - m0) and bool(c.M_KM.any())
    d0, d1 = c.pop_ranges["DAN_PAM"]
    if c.pathway_gains.get("DAN_PAM->MBON_APP") == 0.0:
        assert not bool(((pre >= d0) & (pre < d1) & (post >= m0) & (post < c.pop_ranges["MBON_APP"][1])).any())
