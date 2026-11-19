"""Generalized effort on the joint.

prismatic : tau = a . F                                  [N]
revolute  : tau = a . ((p - o) x F) + a . M              [N*m]
with a the unit joint axis, o a point on it, p the application point, F the force ON the moving part and M an
optional pure couple. Sign: positive tau drives positive q (same axis convention as the kinematics).

``generalized_effort`` uses the same formula as the validated code (forceartgs_physics._validated.replay.effort,
itself identical to akf_pinn.physics.common) and is the signed projection to use for new data.

``lever_factor`` is the ForceSAPIEN A3 -> B1-B10 benchmark force model (tau_B = |kappa_A| |F_B|). It discards the
force direction and therefore the SIGN (laptop_stress_v2: 30/100 gross failures on reversed pushes); it is kept
only to reproduce that benchmark and must not be used for interactions whose push can reverse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._validated import replay as _rp


@dataclass
class GeneralizedEffort:
    """Piecewise-constant effort: ``levels[k]`` is held on [t_s[k], t_s[k+1]) (forward ZOH), the first level
    also before t_s[0]. Units: N*m (revolute) or N (prismatic)."""

    t_s: np.ndarray
    levels: np.ndarray
    joint_type: str
    model: str

    def at(self, t) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def generalized_effort(joint, forces) -> GeneralizedEffort:
    """Signed projection of a ForceRecord on a JointKinematics (same frame required)."""
    if joint.frame != forces.frame:
        raise ValueError(f"joint frame {joint.frame!r} != force frame {forces.frame!r}")
    P = forces.point_m if forces.point_m is not None else np.zeros_like(forces.force_N)
    tau = np.asarray(_rp.effort(joint.joint_type, joint.axis, joint.origin, P, forces.force_N), float)
    if forces.torque_Nm is not None:
        if joint.joint_type == "revolute":
            tau = tau + forces.torque_Nm @ joint.axis
        # a pure couple does no work on a prismatic joint
    return GeneralizedEffort(forces.t_s.copy(), tau, joint.joint_type, "signed vector projection")


def lever_factor(effort_A: np.ndarray, force_A: np.ndarray, active_A: np.ndarray) -> float:
    """kappa = LS ratio of the A generalized effort to |F_A| over the active samples (post_a3.evaluate)."""
    e, act = np.asarray(effort_A, float), np.asarray(active_A, bool)
    Fm = np.linalg.norm(np.asarray(force_A, float), axis=1)
    return float(e[act] @ Fm[act] / (Fm[act] @ Fm[act]))


def lever_factor_effort(kappa: float, forces, joint_type: str) -> GeneralizedEffort:
    """Benchmark force model tau = |kappa| |F| (sign-blind, see module docstring)."""
    return GeneralizedEffort(forces.t_s.copy(), abs(kappa) * np.linalg.norm(forces.force_N, axis=1), joint_type,
                             "lever factor |kappa_A| |F| (ForceSAPIEN A3 benchmark protocol, sign-blind)")
