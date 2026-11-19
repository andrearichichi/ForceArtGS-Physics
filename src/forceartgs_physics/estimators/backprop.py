"""Backprop: direct optimisation through the exact ZOH solution (validated 'Backprop primary').

Implementation: controlled_fairness/cpu.py::backprop (vendored verbatim), called as run.py::execute did for
variant 'primary': 35 starts on the generic SUPPORT grid (log10 I in [-3, 3], log10 decay time in [-2, 2],
nondimensional), 3000 Adam steps (lr 0.05), top-8 L-BFGS-B refinement (500 iterations), offset and initial
velocity in closed form (least squares) at every evaluation. CPU, deterministic.
"""
from __future__ import annotations

import time

from .._validated import common as H
from .._validated import cpu
from . import _bridge
from .base import Estimator


class Backprop(Estimator):
    name = "Backprop"
    frozen_hyperparameters = dict(starts="7x5 grid on SUPPORT", support=H.SUPPORT, adam_steps=3000, adam_lr=0.05,
                                  lbfgsb_top_k=8, lbfgsb_maxiter=500, device="cpu")
    provenance = dict(code="force_estimators/controlled_fairness/cpu.py::backprop", variant="primary",
                      driver="controlled_fairness/run.py::execute via robust_v2/run_heldout.py")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        t0 = time.perf_counter()
        r = H.restore(cpu.backprop(d, "native", H.SUPPORT, False, False), sc)
        r["status"] = "ok" if _bridge.finite_ok(r, w.q_obs, sc["Q"]) else "numerical_failure"
        r.setdefault("runtime_s", time.perf_counter() - t0)
        return r
