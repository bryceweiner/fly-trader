"""The console split and the two-fly brain view: three navigation sections with the Kalshi pages, every page compiles,
the full-brain geometry carries a sub→full index map per fly, and the version-2 payload places each fly's activity."""
import ast
import base64
import json
from pathlib import Path

import numpy as np

from fly_trader.brain import geometry
from fly_trader.ui import brain3d
from tests.test_fly_brain_view import _npz, _tsv

UI = Path(__file__).resolve().parents[1] / "fly_trader" / "ui"


def test_navigation_has_three_sections_and_every_page_compiles():
    tree = ast.parse((UI / "app.py").read_text())
    sections = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pages" for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            for k, v in zip(node.value.keys, node.value.values):
                sections[k.value] = [c.args[0].value for c in v.elts]
    assert list(sections) == ["Memecoins", "Prediction markets", "System"]
    assert sections["Prediction markets"] == ["app_pages/kalshi_overview.py", "app_pages/kalshi_trades.py", "app_pages/kalshi_model.py", "app_pages/kalshi_data.py"]
    assert sections["System"] == ["app_pages/processes.py", "app_pages/safety.py"]
    for rel in [p for s in sections.values() for p in s]:
        compile((UI / rel).read_text(), rel, "exec")


def test_full_geometry_maps_each_fly_into_the_whole_brain(tmp_path, monkeypatch):
    npz, tsv, out = tmp_path / "c.npz", tmp_path / "ann.tsv", tmp_path / "static"
    _npz(npz, "sha-f"); _tsv(tsv)
    monkeypatch.setattr(geometry, "FLY_EXCLUDES", {"memecoin": ("VISUAL",), "kalshi": ("OTHER",)})
    m = geometry.build_full_geometry(out, npz, tsv)
    xyz = np.frombuffer((out / "geometry_full.bin").read_bytes(), dtype=np.float32).reshape(-1, 3)
    assert m["n"] == 8 and xyz.shape == (8, 3) and m["excluded"] == [] and m["pop_ranges"]["VISUAL"] == [4, 6]
    assert set(m["flies"]) == {"memecoin", "kalshi"}
    mi = np.frombuffer((out / m["flies"]["memecoin"]["index"]).read_bytes(), dtype=np.int32); ki = np.frombuffer((out / m["flies"]["kalshi"]["index"]).read_bytes(), dtype=np.int32)
    assert mi.tolist() == [0, 1, 2, 3, 6, 7] and m["flies"]["memecoin"]["n"] == 6           # VISUAL (4, 5) dropped, the rest in order
    assert ki.tolist() == [0, 1, 2, 3, 4, 5] and m["flies"]["kalshi"]["n"] == 6             # OTHER dropped
    mt = (out / "geometry_full.bin").stat().st_mtime_ns
    assert geometry.build_full_geometry(out, npz, tsv) == m and (out / "geometry_full.bin").stat().st_mtime_ns == mt
    sub = geometry.build_geometry(out, npz, tsv)                                             # the sub-graph geometry lives beside it, untouched
    assert sub["n"] == 6 and (out / "geometry.bin").exists()


def test_payload_flies_places_each_flys_activity_and_flags_a_foreign_connectome(monkeypatch):
    monkeypatch.setattr(brain3d.st, "get_option", lambda k: "")
    meta = {"connectome_sha256": "abc", "connectome_file": "c.npz", "n": 8, "flies": {"memecoin": {"n": 6, "index": "index_memecoin.bin"}, "kalshi": {"n": 6, "index": "index_kalshi.bin"}}}
    act = {"h": np.arange(6, dtype=np.int8), "t": 1_790_000_000.0, "n_rows": 40, "n_cand": 7, "connectome": "c.npz"}
    pth = {"mode": "bootstrap", "key": "k", "reference": None, "items": [{"kc": 0, "mbon": 2, "delta": 0.1, "ratio": 0.2, "strategy": "ev", "valence": "approach"}], "note": None, "changed": 1}
    f1 = brain3d.fly_payload("kalshi", meta, act, pth, ["favorite", "ev"])
    assert f1["index"] == {"url": "/app/static/brain/index_kalshi.bin?v=abc", "n": 6} and f1["n"] == 6 and f1["candidates"] == 7 and f1["minute"].startswith("2026-")
    assert np.frombuffer(base64.b64decode(f1["activity_b64"]), dtype=np.int8).tolist() == list(range(6)) and f1["strategies"] == ["favorite", "ev"]
    f2 = brain3d.fly_payload("memecoin", meta, {**act, "connectome": "other.npz"}, pth, ["ev"])
    assert f2["activity_b64"] is None and "another connectome" in f2["note"]
    p = brain3d.payload_flies(meta, [f1, f2], "kalshi")
    assert p["version"] == 2 and p["central_owner"] == "kalshi" and p["geometry"]["url"].endswith("geometry_full.bin?v=abc") and p["geometry"]["n"] == 8
    assert [f["name"] for f in p["flies"]] == ["kalshi", "memecoin"] and "another connectome" in p["message"]
    json.dumps(p)
