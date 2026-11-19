"""GT-free identifiability diagnostic of (I, c) for one interaction.

Port of robust_v2/identifiability.py::estimator_side (the diagnostic pre-registered in METHODS_FREEZE.json,
'reliability_flag_gt_free'); its exact-ZOH solver ``solve`` is vendored verbatim. Computed on the
nondimensionalised data of the validated pipeline, at the shared integral-LS estimate:

  1. integral regressor X = [q - q0, int (q - q0), t - t0, 1]: condition number, |corr| of the I and c columns;
  2. output sensitivity S = dq(t_obs)/d(log I, log c, v0, q0), Fisher F = S^T S / sigma^2 (sigma = rms LS residual),
     Cramer-Rao relative std of I and c and the log I - log c correlation.

Verdict (pre-registered threshold): I and c are SEPARATELY identifiable iff both Cramer-Rao relative stds
are <= 10 %. Otherwise the result says which parameter is not identifiable; when |corr(log I, log c)| > 0.95 it
also states that only a combination is determined. The CRLB is a LOCAL bound for white q noise: correlated
visual errors and model errors are not included, so 'identifiable' is necessary, not sufficient, for accuracy.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from .._validated import identifiability_rv2 as _rv2
from ..estimators import _bridge

CRLB_LIMIT_PCT = 10.0
DT = torch.float64


def _estimator_side(d, ls):
    """Body of identifiability.estimator_side from ``d, sc = H.normalize(w)`` on (same statements)."""
    t, q = d.t_obs.numpy(), d.q_obs.numpy()
    X = np.column_stack([q - q[0], cumulative_trapezoid(q - q[0], t, initial=0), t - t[0], np.ones(len(t))])
    Xn = X / np.linalg.norm(X, axis=0)
    sv = np.linalg.svd(Xn, compute_uv=False)
    corr_Ic = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    tf = d.force_t.numpy()
    tau = d.generalized_input(d.force_t, torch.ones((), dtype=DT)).numpy()
    knots = np.r_[t[0], tf[tf > t[0]]]
    lev = np.r_[np.interp(t[0], tf, tau), tau[tf > t[0]]]
    p0 = np.array([math.log(ls["inertia"]), math.log(ls["damping"]), ls["v0"], ls["q0"]])
    f = lambda p: _rv2.solve(math.exp(p[0]), math.exp(p[1]), p[3], p[2], knots, lev, t)
    base = f(p0)
    sigma = float(np.sqrt(np.mean((base - q) ** 2)))
    S = np.empty((len(t), 4))
    for j in range(4):
        h = 1e-5 * max(1.0, abs(p0[j]))
        e = np.zeros(4); e[j] = h
        S[:, j] = (f(p0 + e) - f(p0 - e)) / (2 * h)
    Fi = S.T @ S / max(sigma, 1e-12) ** 2
    cov = np.linalg.pinv(Fi)
    Sn = S[:, :2] / np.linalg.norm(S[:, :2], axis=0)
    svs = np.linalg.svd(Sn, compute_uv=False)
    active = np.interp(t, tf, np.abs(tau)) > 0.5 * np.abs(tau).max()
    return dict(n_obs=len(t), fraction_obs_during_force=float(active.mean()),
                integral_cond=float(sv[0] / sv[-1]), integral_sv_min=float(sv[-1]), integral_corr_I_c_columns=float(corr_Ic),
                ls_decay_time_over_window=float(ls["inertia"] / ls["damping"]),
                fit_sigma_nd=sigma, sens_cond_logI_logc=float(svs[0] / svs[-1]),
                sens_corr_logI_logc=float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])),
                crlb_rel_std_I_pct=float(100 * math.sqrt(cov[0, 0])), crlb_rel_std_c_pct=float(100 * math.sqrt(cov[1, 1])))


def diagnose(interaction, encoding="auto") -> dict:
    """Identifiability report for one interaction (no estimator needed, no ground truth)."""
    out = dict(kinematic_status=interaction.joint.status.value)
    if interaction.joint.status.value != "identified":
        out.update(identifiable=False, verdict=f"kinematics {interaction.joint.status.value}: not identifiable")
        return out
    eff = np.abs(_bridge.to_window(interaction, encoding).effort())
    tq = interaction.trajectory.t_s
    if not np.any(eff > 0) or interaction.forces.t_s[np.argmax(eff > 0)] > tq[-1]:
        out.update(identifiable=False, verdict="no applied effort during the observed motion: unobservable")
        return out
    with _bridge.single_thread(1):
        w, d, sc, init = _bridge.prepare(interaction, encoding)
        r = _estimator_side(d, init)
    out.update(r)
    okI, okc = r["crlb_rel_std_I_pct"] <= CRLB_LIMIT_PCT, r["crlb_rel_std_c_pct"] <= CRLB_LIMIT_PCT
    out["identifiable_inertia"], out["identifiable_damping"] = bool(okI), bool(okc)
    out["identifiable"] = bool(okI and okc)
    if out["identifiable"]:
        out["verdict"] = "I and c separately identifiable (CRLB <= 10 %)"
    else:
        bad = [n for n, ok in (("inertia", okI), ("damping", okc)) if not ok]
        v = f"NOT separately identifiable: CRLB > 10 % for {', '.join(bad)}"
        if abs(r["sens_corr_logI_logc"]) > 0.95:
            v += f"; |corr(log I, log c)| = {abs(r['sens_corr_logI_logc']):.3f}: only a combination of I and c is determined"
        out["verdict"] = v
    out["uncertainty"] = dict(method="local Cramer-Rao bound at the integral-LS estimate (white q noise)",
                              inertia_rel_std=r["crlb_rel_std_I_pct"] / 100, damping_rel_std=r["crlb_rel_std_c_pct"] / 100)
    out["threshold_pct"] = CRLB_LIMIT_PCT
    return out
