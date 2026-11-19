"""Integral least squares (closed form, GT-free) - the shared initializer of every validated estimator
(controlled_fairness/common.py::ls). Exposed as a fast baseline; ``fit_many`` stacks several interactions with
per-interaction nuisance terms (an extension, not part of the validated benchmark)."""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..schemas import IdentificationResult, PhysicalParameters
from .._validated import improved_methods as M
from ..forces.projection import generalized_effort
from .base import Estimator


class IntegralLS(Estimator):
    name = "IntegralLS"
    provenance = dict(code="force_estimators/controlled_fairness/common.py::ls")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        I, c = init["inertia"] * sc["inertia_to_physical"], init["damping"] * sc["damping_to_physical"]
        return dict(status="ok" if (np.isfinite([I, c]).all() and min(I, c) > 0) else "numerical_failure", inertia=I,
                    damping=c, q_hat=list(map(float, w.q_obs)), convergence="closed form", diagnostics=dict(init))

    def fit_many(self, interactions):
        """Stacked integral equations  I (q-q0) + c int(q-q0) + a_k (t-t0) + b_k = iint tau  for k interactions,
        sharing I, c; nuisance a_k, b_k per interaction. Physical units (no nondimensionalisation)."""
        t0 = time.perf_counter()
        rows, rhs = [], []
        K = len(interactions)
        for k, x in enumerate(interactions):
            x.require_identifiable()
            t, q = x.trajectory.t_s, x.trajectory.q
            eff = generalized_effort(x.joint, x.forces)
            tf, tau = eff.t_s, eff.levels
            knots, jumps = M.force_knots(tf, tau, float(t[0]))
            J = .5 * np.maximum(t[:, None] - knots[None, :], 0.) ** 2 @ jumps
            X = np.zeros((len(t), 2 + 2 * K))
            X[:, 0] = q - q[0]
            X[:, 1] = cumulative_trapezoid(q - q[0], t, initial=0)
            X[:, 2 + 2 * k] = t - t[0]; X[:, 3 + 2 * k] = 1.0
            rows.append(X); rhs.append(J)
        X, J = np.vstack(rows), np.concatenate(rhs)
        cs = np.linalg.norm(X, axis=0); cs[cs == 0] = 1
        coef = np.linalg.lstsq(X / cs, J, rcond=None)[0] / cs
        jt = interactions[0].joint_type
        ok = bool(np.isfinite(coef[:2]).all() and min(coef[:2]) > 0)
        return IdentificationResult(
            estimator="IntegralLS(stacked)", status="ok" if ok else "numerical_failure",
            parameters=PhysicalParameters(jt, float(coef[0]), float(coef[1])) if ok else None,
            interactions=[x.name for x in interactions],
            diagnostics=dict(condition=float(np.linalg.cond(X / cs)), n_rows=len(J)), runtime_s=time.perf_counter() - t0)
