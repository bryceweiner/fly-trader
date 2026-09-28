"""Model files that cannot run code when they load.

A pickle (``joblib.load``, ``torch.load(weights_only=False)``) executes whatever its author put in it, and model
releases install on the hosted fly without a human step (ops/release.py). So every model the engine loads goes through
here:

* fly checkpoints and plastic banks: ``torch.load(weights_only=True)`` -- tensors, numbers, strings, lists and dicts
  only; anything else refuses to load. ``torch_save`` re-reads what it wrote the same way, so a checkpoint that would
  not load safely is caught where it is made, not on the server.
* selectors: skops (``.skops``), whose loader builds only the types listed in ``TRUSTED_SELECTOR_TYPES`` (plus the
  scikit-learn / numpy types skops itself trusts). Older ``.joblib`` selectors are refused; ``fly-trader models
  convert`` rewrites this machine's own ones once.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

TRUSTED_SELECTOR_TYPES = [
    "fly_trader.train.scaling.RobustScaler",
    "fly_trader.train.selector.SelectorModel",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]
# what a release's models/ may hold (connectome .npz loads with allow_pickle=False, wallet skill is parquet, geometry
# is raw floats + JSON); TABLE and current.txt are plain text
SAFE_SUFFIXES = {".pt", ".skops", ".npz", ".json", ".bin", ".parquet", ".txt", ".sha256", ""}


class UnsafeModel(RuntimeError):
    pass


def torch_load(path: str | Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def torch_save(obj, path: str | Path) -> None:
    import torch
    torch.save(obj, path)
    try:
        torch_load(path)
    except Exception as e:
        Path(path).unlink(missing_ok=True)
        raise UnsafeModel(f"{Path(path).name} would not load with weights_only=True (a non-tensor, non-primitive value?): {e}") from e


def save_selector(m, path: str | Path) -> None:
    import skops.io as sio
    sio.dump(m, path)
    untrusted = set(sio.get_untrusted_types(file=path)) - set(TRUSTED_SELECTOR_TYPES)
    if untrusted:
        Path(path).unlink(missing_ok=True)
        raise UnsafeModel(f"selector holds types outside TRUSTED_SELECTOR_TYPES: {sorted(untrusted)}")


def load_selector(path: str | Path):
    import skops.io as sio
    p = Path(path)
    if p.suffix != ".skops":
        raise UnsafeModel(f"{p.name}: only .skops selectors load (a pickle can run code); run `fly-trader models convert`")
    return sio.load(p, trusted=TRUSTED_SELECTOR_TYPES)


def check_file(path: str | Path) -> None:
    """Raise UnsafeModel unless ``path`` is a model file that loads without running code (ops/release.apply_release)."""
    p = Path(path)
    if p.suffix not in SAFE_SUFFIXES:
        raise UnsafeModel(f"{p.name}: {p.suffix} files are not accepted in a release (pickle formats can run code)")
    if p.suffix == ".pt":
        try:
            torch_load(p)
        except Exception as e:
            raise UnsafeModel(f"{p.name} does not load with weights_only=True: {e}") from e
    elif p.suffix == ".skops":
        import skops.io as sio
        untrusted = set(sio.get_untrusted_types(file=p)) - set(TRUSTED_SELECTOR_TYPES)
        if untrusted:
            raise UnsafeModel(f"{p.name} holds untrusted types {sorted(untrusted)}")


def convert_legacy() -> list[dict]:
    """Rewrite this machine's own ``.joblib`` selector snapshots as ``.skops`` (path and sha256 updated in place, ids
    kept so pins still hold). Only for files this machine trained: it unpickles them once to convert."""
    import joblib
    from ..db.connection import transaction
    done = []
    with transaction() as conn:
        rows = conn.execute("SELECT id, path FROM brain_snapshots WHERE kind = 'selector' AND path LIKE '%%.joblib'").fetchall()
        for r in rows:
            src = Path(r["path"])
            if not src.exists():
                continue
            dst = src.with_suffix(".skops")
            save_selector(joblib.load(src), dst)
            sha = hashlib.sha256(dst.read_bytes()).hexdigest()
            conn.execute("UPDATE brain_snapshots SET path = %s, sha256 = %s WHERE id = %s", (str(dst), sha, r["id"]))
            done.append({"id": int(r["id"]), "path": str(dst), "sha256": sha})
    return done
