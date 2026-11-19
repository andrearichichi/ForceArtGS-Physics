"""Synthetic ROBOT record consistent with a ForceArtGS synthetic scene (used by examples and tests).

ForceArtGS' own synthetic generator (estimotion_2part/joint_discovery/synthetic.py::make_scene) moves the joint
with q(t) = q_max (3 s^2 - 2 s^3), s = t / t_end, but its contact force is a constant 6 N (not dynamics-consistent).
This module builds what a robot would have recorded if it had produced that SAME motion on a joint with known
(I, c): joint torque tau = I q'' + c q' (exact, analytic), applied as a force at a TCP point that moves with the
part, recorded at 50 Hz like the REAL LeRobot episodes: EE position + rotation vector in base_link, wrench in the
EE (= sensor) axes, REACTION sign (force on the robot = -force on the object), TCP = EE + R_ee (0, 0, 0.1876).

Nothing here is used by the estimators except through the public adapters; I and c are only used to generate
the record and to check the result.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

TCP_OFFSET = np.array([0.0, 0.0, 0.1876])


def gt_q(t, q_max, t_end):
    s = np.clip(np.asarray(t, float) / t_end, 0, 1)
    q = q_max * (3 * s ** 2 - 2 * s ** 3)
    qd = q_max * (6 * s - 6 * s ** 2) / t_end * ((t >= 0) & (t <= t_end))
    qdd = q_max * (6 - 12 * s) / t_end ** 2 * ((t >= 0) & (t <= t_end))
    return q, qd, qdd


def rotate(axis, ang, v):
    return Rotation.from_rotvec(np.outer(ang, axis)).apply(v)


def robot_record(kind, axis, origin, q_max, t_end, inertia, damping, fps=50.0, contact_lever_m=0.25):
    """Arrays in LeRobot semantics: t, ee_pos [N,3], ee_rotvec [N,3], wrench_sensor [N,6] (reaction, EE axes),
    contact point p0 at q = 0 (on the moving part)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.arange(0.0, t_end + 1e-9, 1.0 / fps)
    # mid-interval samples of tau make the forward-ZOH record accurate to O(h^2) in q
    tm = np.minimum(t + 0.5 / fps, t_end)
    q, qd, qdd = gt_q(tm, q_max, t_end)
    tau = inertia * qdd + damping * qd
    if kind == "revolute":
        o = np.asarray(origin, float)
        perp = np.cross(axis, [0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0, 0])
        perp /= np.linalg.norm(perp)
        p0 = o + contact_lever_m * np.cross(perp, axis)          # point on the part, distance contact_lever_m from the axis
        r = rotate(axis, gt_q(t, q_max, t_end)[0], p0 - o)       # contact point follows the part (at sample times)
        p = o + r
        F = tau[:, None] * np.cross(axis, r) / np.sum(r * r, 1)[:, None]   # a . (r x F) = tau
    else:
        p0 = np.asarray(origin, float)
        p = p0 + gt_q(t, q_max, t_end)[0][:, None] * axis
        F = tau[:, None] * axis[None]
    R_ee = Rotation.from_euler("xyz", [np.pi, 0.0, 0.3])       # tool pointing down, fixed orientation
    ee_rotvec = np.tile(R_ee.as_rotvec(), (len(t), 1))
    ee_pos = p - R_ee.apply(TCP_OFFSET)
    F_sensor_reaction = -R_ee.inv().apply(F)                    # recorded: force ON the robot, EE axes
    wrench = np.c_[F_sensor_reaction, np.zeros_like(F)]          # torques unused (reference undocumented)
    return dict(t=t, ee_pos=ee_pos, ee_rotvec=ee_rotvec, wrench=wrench, frame_index=np.arange(len(t)), fps=fps,
                tau_true=tau, F_object=F, p_contact=p)


def write_parquet(path, rec):
    import pyarrow as pa
    import pyarrow.parquet as pq
    state = np.c_[rec["ee_pos"], rec["ee_rotvec"], np.zeros((len(rec["t"]), 6))]
    tab = pa.table({"observation.state": [list(map(float, r)) for r in state],
                    "observation.wrench": [list(map(float, r)) for r in rec["wrench"]],
                    "frame_index": rec["frame_index"].astype(np.int64), "timestamp": rec["t"].astype(np.float32),
                    "episode_index": np.zeros(len(rec["t"]), np.int64)})
    pq.write_table(tab, path)
    return path
