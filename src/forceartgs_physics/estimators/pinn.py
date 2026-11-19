"""PINN (PINN-V2 of the CVPR robust_v2 sprint, the 'PINN' of the A3 benchmark).

Trajectory network (akf_pinn TrajectoryMLP, 4x64 tanh, hard q0) jointly trained with log I, log c on
  data loss  + integral (first-integral) physics residual I (q'(t) - q'(t0)) + c (q(t) - q(t0)) - J(t), J = int tau
  normalised by rms(J), + initial-condition loss;
parameters started at the shared integral-LS estimate; 2000 data-only Adam steps, 20000 joint Adam steps (lr 1e-3),
up to 10000 L-BFGS iterations in chunks of 100 with the V1 stopping rule; 4096 uniform collocation points; float64.
Frozen in results/force_estimators_cvpr_robust_v2/protocol/METHODS_FREEZE.json; seeds 0, 1, 2 (median).
Code: robust_v2/pinn_v2.py::run with run_v2.VARIANTS['V2'], vendored verbatim; driver robust_v2/run_heldout.py.
"""
from __future__ import annotations

import numpy as np

from .._validated import common as H
from .._validated import pinn_v2 as P
from .._validated.run_v2 import VARIANTS
from . import _bridge
from .base import Estimator


class PINN(Estimator):
    name = "PINN"
    default_seeds = (0, 1, 2)
    frozen_hyperparameters = dict(variant="V2", **VARIANTS["V2"], **{k: v for k, v in P.V2Config().__dict__.items()
                                                                     if k not in ("seed", *VARIANTS["V2"])})
    provenance = dict(code="force_estimators/robust_v2/pinn_v2.py::run", config="run_v2.VARIANTS['V2']",
                      freeze="results/force_estimators_cvpr_robust_v2/protocol/METHODS_FREEZE.json",
                      driver="robust_v2/run_heldout.py")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        dev = _bridge.default_device(self.config.device)
        cfg = P.V2Config(seed=seed, **VARIANTS["V2"])
        r = H.restore(P.run(d.to(dev), cfg, init), sc)
        finite = np.isfinite([r["inertia"], r["damping"], *r["q_hat"]]).all() and min(r["inertia"], r["damping"]) > 0
        r["status"] = "ok" if finite else "numerical_failure"
        r["device"] = dev
        sd = r.get("diagnostics", {}).pop("state_dict", None)
        if sd is not None:
            r["diagnostics"]["network_state_dict_omitted"] = True
        r["diagnostics"].pop("history", None)
        return r
