"""Re-fit the validated estimators on the bundled A3 inputs and compare with the ORIGINAL fits.

slow (CPU): Backprop, AKF (AKF-improved) and AKF-benchmark on the 10 objects. With the reference environment
            (torch 2.5.1, numpy 1.26.4, scipy 1.17.1, 1 thread) the results are expected to be BITWISE identical;
            the asserted tolerance is rel 1e-9 on I, c and q_hat to allow a different BLAS/libm build.
gpu       : PINN (seed 0) and MLP (seed 0). GPU kernels are not bitwise reproducible across runs/devices; the
            tolerance is |dI/I|, |dc/c| <= max(1e-3, half the across-seed range of the original fits).
Select objects with FORCEARTGS_PHYSICS_OBJECTS=Knife_103706,... (default: all for slow, Knife_103706 for gpu).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from forceartgs_physics.adapters.forcesapien import OBJECTS

SLOW_OBJ = os.environ.get("FORCEARTGS_PHYSICS_OBJECTS", ",".join(OBJECTS)).split(",")
GPU_OBJ = os.environ.get("FORCEARTGS_PHYSICS_OBJECTS", "Knife_103706").split(",")
OUT = os.environ.get("FORCEARTGS_PHYSICS_REFIT_OUT")


def _record(slug, method, r, ref, extra):
    if OUT:
        p = Path(OUT) / f"{slug}.{method}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(slug=slug, method=method, inertia=r.parameters.inertia if r.parameters else None,
                                     damping=r.parameters.damping if r.parameters else None, ref=ref, runtime_s=r.runtime_s,
                                     per_seed=r.per_seed, **extra), default=float, indent=1))


@pytest.mark.slow
@pytest.mark.parametrize("slug", SLOW_OBJ)
@pytest.mark.parametrize("method,key,refsrc", [("Backprop", "Backprop/s0", "a3"), ("AKF-benchmark", "AKF/s0", "a3"),
                                               ("AKF", None, "akf")])
def test_cpu_refit_identical(bench, slug, method, key, refsrc):
    from forceartgs_physics import api
    ref = bench.reference("a3_fits.json.gz")[slug][key] if refsrc == "a3" else bench.reference("akf_improved_fits.json.gz")[slug]
    r = api.identify(bench.interaction_A3(slug), method, diagnose_identifiability=False, encoding="cartesian")
    dq = float(np.max(np.abs(np.asarray(r.q_fit["q"]) - ref["q_hat"])))
    _record(slug, method, r, dict(inertia=ref["inertia"], damping=ref["damping"]),
            dict(max_abs_dq=dq, bitwise=bool(r.parameters.inertia == ref["inertia"] and r.parameters.damping == ref["damping"] and dq == 0)))
    assert r.parameters.inertia == pytest.approx(ref["inertia"], rel=1e-9)
    assert r.parameters.damping == pytest.approx(ref["damping"], rel=1e-9)
    assert dq <= 1e-9 * np.ptp(ref["q_hat"])
    if method == "AKF":
        assert r.per_seed[0]["diagnostics"]["selected"] == ref["selected"]


@pytest.mark.gpu
@pytest.mark.parametrize("slug", GPU_OBJ)
@pytest.mark.parametrize("method,orig,seeds", [("PINN", "PINN-V2", 3), ("MLP", "MLP-Joint", 5)])
def test_gpu_refit_close(bench, slug, method, orig, seeds, tmp_path):
    from forceartgs_physics import api
    fits = bench.reference("a3_fits.json.gz")[slug]
    ref = fits[f"{orig}/s0"]
    allI = [fits[f"{orig}/s{k}"]["inertia"] for k in range(seeds)]
    allc = [fits[f"{orig}/s{k}"]["damping"] for k in range(seeds)]
    tolI = max(1e-3, 0.5 * np.ptp(allI) / np.median(allI))
    tolc = max(1e-3, 0.5 * np.ptp(allc) / np.median(allc))
    r = api.identify(bench.interaction_A3(slug), method, seeds=[0], diagnose_identifiability=False, encoding="cartesian",
                     workdir=str(tmp_path))
    dI, dc = r.parameters.inertia / ref["inertia"] - 1, r.parameters.damping / ref["damping"] - 1
    _record(slug, method, r, dict(inertia=ref["inertia"], damping=ref["damping"]),
            dict(rel_dI=dI, rel_dc=dc, tol_I=tolI, tol_c=tolc, bitwise=bool(dI == 0 and dc == 0)))
    assert abs(dI) <= tolI and abs(dc) <= tolc
