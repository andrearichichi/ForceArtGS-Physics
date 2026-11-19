"""Explicit frame transforms. The library never transforms anything implicitly."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    """x_dst = s * R @ x_src + t  (points);  v_dst = R @ v_src  (directions, forces, torques).

    ``s`` is a pure number (e.g. a reconstruction-to-metric scale); forces and torques are not scaled
    (a torque transported between frames with a scale change must be recomputed from force and point).
    """

    R: np.ndarray
    t: np.ndarray
    s: float
    src: str
    dst: str

    def __post_init__(self):
        R = np.asarray(self.R, float).reshape(3, 3)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
            raise ValueError("R must be a proper rotation")
        if not (np.isfinite(self.s) and self.s > 0):
            raise ValueError("scale must be positive")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", np.asarray(self.t, float).reshape(3))

    def point(self, x):
        return self.s * (np.asarray(x, float) @ self.R.T) + self.t

    def direction(self, v):
        return np.asarray(v, float) @ self.R.T

    def joint(self, joint):
        """Re-express a JointKinematics in ``dst``. Prismatic q (metres) are NOT rescaled here: if s != 1 the
        caller must rescale q explicitly (see adapters.forceartgs, which records the scale it applies)."""
        if joint.frame != self.src:
            raise ValueError(f"joint is in {joint.frame!r}, transform expects {self.src!r}")
        return replace(joint, axis=self.direction(joint.axis), origin=self.point(joint.origin), frame=self.dst,
                       origin_std_m=None if joint.origin_std_m is None else joint.origin_std_m * self.s)

    def forces(self, rec):
        if rec.frame != self.src:
            raise ValueError(f"force record is in {rec.frame!r}, transform expects {self.src!r}")
        if self.s != 1.0 and rec.torque_Nm is not None:
            raise ValueError("cannot transport pure couples through a scaled transform")
        return replace(rec, force_N=self.direction(rec.force_N),
                       point_m=None if rec.point_m is None else self.point(rec.point_m),
                       torque_Nm=None if rec.torque_Nm is None else self.direction(rec.torque_Nm), frame=self.dst,
                       calibration={**rec.calibration, f"transform_{self.src}_to_{self.dst}": dict(s=self.s)})
