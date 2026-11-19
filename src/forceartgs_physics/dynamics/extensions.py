"""EXTENSIONS for real data - NOT part of the ForceSAPIEN model and never used by the validated estimators.

    I q'' + c q' + Fc * s(q') + g(q) + b = tau(t)

  Fc   Coulomb friction [N*m | N]; s = sign (``smooth=None``) or tanh(q'/smooth)
  Fs   static (breakaway) level of a Karnopp stick model: the joint stays at rest while |tau - g - b| <= Fs
  g(q) configuration-dependent load, e.g. gravity of a lid ``m g r cos(q + phi)`` (user callable)
  b    constant generalized effort (sensor bias projected on the joint, or any constant load)

Identifiability warning (measured on Hoi! and REAL, see docs/REAL_DATA.md): in a one-direction push Fc, b and any
constant force are carried by the SAME regressor and cannot be separated; ``identify_extended_ls`` reports this
instead of returning numbers for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid


@dataclass
class ExtendedModel:
    inertia: float
    damping: float
    coulomb: float = 0.0
    static: Optional[float] = None
    bias: float = 0.0
    load: Optional[Callable[[np.ndarray], np.ndarray]] = None
    smooth: Optional[float] = None
    meta: dict = field(default_factory=dict)


def simulate(model: ExtendedModel, effort, t_query, t0=0.0, q0=0.0, v0=0.0, max_step_s=1e-3) -> np.ndarray:
    """Forward simulation (RK4 for the smooth model, semi-implicit stick-slip for the Karnopp model)."""
    tq = np.asarray(t_query, float)
    T = np.r_[t0, np.sort(tq)]
    out = {}
    q, v, t = float(q0), float(v0), float(t0)

    def f(q, v, t):
        tau = float(effort.at(t)) - model.bias - (model.load(np.array([q]))[0] if model.load else 0.0)
        if model.static is not None and abs(v) < 1e-9 and abs(tau) <= model.static:
            return v, 0.0, True
        s = np.tanh(v / model.smooth) if model.smooth else np.sign(v if abs(v) > 1e-12 else tau)
        return v, (tau - model.damping * v - model.coulomb * s) / model.inertia, False

    for tn in T[1:]:
        while t < tn - 1e-15:
            h = min(max_step_s, tn - t)
            if model.smooth or model.static is not None:
                _, a, stuck = f(q, v, t)
                v_new = 0.0 if stuck else v + h * a
                if not stuck and model.coulomb and np.sign(v_new) != np.sign(v) and v != 0:
                    v_new = 0.0                                    # friction cannot reverse the motion
                q, v = q + h * v_new, v_new
            else:
                k1 = f(q, v, t); k2 = f(q + h / 2 * k1[0], v + h / 2 * k1[1], t + h / 2)
                k3 = f(q + h / 2 * k2[0], v + h / 2 * k2[1], t + h / 2); k4 = f(q + h * k3[0], v + h * k3[1], t + h)
                q += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += h
        out[tn] = q
    return np.array([out[x] for x in tq])


def identify_extended_ls(t, q, effort, terms=("coulomb", "bias"), cond_limit=1e6, rel_std_limit=0.10):
    """Integral least squares for I, c and the requested extra terms (integrated once, no differentiation of q):

        I (q'(t) - q'(t0)) + c (q(t) - q0) + Fc int sign(q') + b (t - t0) = int tau
    with q'(t) - q'(t0) eliminated by integrating once more (same construction as the validated integral-LS):
        I (q - q0) + c int (q - q0) - I v0 (t - t0) + Fc iint sign(q') + b (t-t0)^2/2 = iint tau

    sign(q') comes from finite differences of q. Returns estimates with standard errors and, for every term, an
    explicit ``identifiable`` flag (relative std <= rel_std_limit and condition number <= cond_limit). A term that is
    not identifiable is reported as such and its value is NOT a result."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    tau = effort.at(t)
    T = t - t[0]
    J = cumulative_trapezoid(cumulative_trapezoid(tau, t, initial=0), t, initial=0)
    cols, names = [q - q[0], cumulative_trapezoid(q - q[0], t, initial=0), -T], ["inertia", "damping", "inertia_x_v0"]
    if "coulomb" in terms:
        s = np.sign(np.gradient(q, t))
        cols.append(cumulative_trapezoid(cumulative_trapezoid(s, t, initial=0), t, initial=0)); names.append("coulomb")
    if "bias" in terms:
        cols.append(0.5 * T ** 2); names.append("bias")
    X = np.column_stack(cols)
    cs = np.linalg.norm(X, axis=0); cs[cs == 0] = 1
    coef, *_ = np.linalg.lstsq(X / cs, J, rcond=None)
    coef = coef / cs
    r = J - X @ coef
    dof = max(len(t) - len(names), 1)
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    cond = float(np.linalg.cond(X / cs))
    out = {"condition_number": cond, "residual_rms": float(np.sqrt(s2)), "terms": {}}
    # Physical consistency: I, c and Fc cannot be negative. A negative estimate means the model does not describe the
    # data (or the terms are confounded); then NO term of this fit is reported as identified.
    nonneg = ("inertia", "damping", "coulomb")
    violations = [n for i, n in enumerate(names) if n in nonneg and coef[i] < 0]
    for i, n in enumerate(names):
        se = float(np.sqrt(max(cov[i, i], 0)))
        rel = se / abs(coef[i]) if coef[i] != 0 else np.inf
        out["terms"][n] = dict(value=float(coef[i]), std=se, rel_std=float(rel),
                               identifiable=bool(rel <= rel_std_limit and cond <= cond_limit and not violations))
    out["physically_consistent"] = not violations
    if violations:
        out["verdict"] = (f"physically inconsistent estimate ({', '.join(violations)} < 0): model misspecified or terms "
                          "confounded; no term is identified")
    C = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)) + 1e-300)
    out["correlation"] = {f"{a}~{b}": float(C[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if j > i}
    return out
