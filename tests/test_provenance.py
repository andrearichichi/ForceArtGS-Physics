"""The vendored code is the validated code.

Always: every vendored module imports and exposes the recorded names.
With FORCEARTGS_PHYSICS_ORIGINAL_ROOT=<dir containing comparison/ and AKF_PINN/>: the original files still have the
recorded SHA-256, and the copied source text of every name is identical to the original (akf_pinn files: identical
after the documented import rewrite)."""
import importlib
import json
import re
from pathlib import Path

import pytest

from conftest import original_root

PKG = Path(__file__).resolve().parents[1] / "src" / "forceartgs_physics" / "_validated"
PROV = json.loads((PKG / "PROVENANCE.json").read_text())


@pytest.mark.parametrize("mod", sorted(PROV["modules"]))
def test_vendored_names_present(mod):
    m = importlib.import_module(f"forceartgs_physics._validated.{mod[:-3]}")
    for n in PROV["modules"][mod]["names"]:
        assert hasattr(m, n), (mod, n)


def _roots():
    r = original_root()
    if r is None:
        pytest.skip("FORCEARTGS_PHYSICS_ORIGINAL_ROOT not set")
    return {"comparison/freeartgs_all_objects": r / "comparison/freeartgs_all_objects", "AKF_PINN": r / "AKF_PINN"}


def _orig(origin):
    roots = _roots()
    for k, v in roots.items():
        if origin.startswith(k + "/"):
            return v / origin[len(k) + 1:]
    raise KeyError(origin)


@pytest.mark.original_tree
@pytest.mark.parametrize("mod", sorted(PROV["modules"]))
def test_copied_text_identical_to_original(mod):
    import hashlib
    from vendor_validated import segments
    info = PROV["modules"][mod]
    src = _orig(info["origin"])
    assert hashlib.sha256(src.read_bytes()).hexdigest() == info["sha256"], "original changed since vendoring"
    orig = segments(src, info["names"])
    mine = segments(PKG / mod, info["names"])
    for n in info["names"]:
        assert mine[n] == orig[n], (mod, n)


@pytest.mark.original_tree
@pytest.mark.parametrize("rel", sorted(PROV["akf_pinn_whole_files"]))
def test_akf_pinn_files_identical_modulo_imports(rel):
    """Vendored file == original after exactly the documented edits (relative imports, tomllib guard)."""
    import hashlib
    from vendor_validated import relativize
    info = PROV["akf_pinn_whole_files"][rel]
    src = _orig(info["origin"])
    assert hashlib.sha256(src.read_bytes()).hexdigest() == info["sha256"]
    expected = relativize(src.read_text(), rel[len("akf_pinn/"):].count("/"))
    if rel.endswith("config.py"):
        expected = expected.replace("import tomllib\n", "try:  # vendoring patch: Python 3.10 has no tomllib (only used by "
                                    "load_config)\n    import tomllib\nexcept ModuleNotFoundError:  # pragma: no cover\n"
                                    "    tomllib = None\n")
    assert (PKG / rel).read_text() == expected, rel
