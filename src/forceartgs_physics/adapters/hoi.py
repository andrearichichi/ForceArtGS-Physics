"""Hoi! adapter.

Hoi! has NO q(t) ground truth (only Leica axis/origin/type and open/close windows, evaluation only) but its gripper
sessions DO record a wrist force/torque sensor. Our freeartgs-hoi force pipeline (freeartgs-hoi/multicam/force,
reports in reports/force/) produced per-interaction CSVs (freeartgs-hoi-data/force/<version>/<target>/<target>.csv +
report.json) with q from the gripper TCP on the packaged joint and the signed generalized force on the object along
the packaged axis, with an uncertainty envelope (axis +-2 deg, hinge +-22 mm, lag +-5 ms).

``load_force_csv`` turns one such CSV into an Interaction in the JOINT-ALIGNED EFFORT FRAME: axis = e_x, origin = 0;
the generalized force is carried exactly as a pure force along e_x (prismatic) or a pure couple about e_x (revolute),
so ``forces.projection`` returns gen_force unchanged. The Cartesian force/contact point of the CSV stay in
``metadata`` (Leica frame) for re-projection with another joint.

Known limits measured by that pipeline (docs/REAL_DATA.md): tool inertia not compensated (5-14 % of the force),
bias drift 0.36-1.2 N per interaction, damping not identifiable even on clean synthetic data, Coulomb friction,
bias and any constant load share one regressor in one-direction interactions.

Sealed test locations (wardrobe_1, bedroom_10, kitchen_14, livingroom_1) must never be read; this loader refuses
files whose report names such a session.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..schemas import ForceRecord, Interaction, JointKinematics, KinematicStatus, Trajectory

SEALED = ("wardrobe_1_", "bedroom_10_", "kitchen_14_", "livingroom_1_")
FRAME = "joint_aligned_effort_frame"


def load_force_csv(csv_path, joint_type: str, *, only_window: bool = True, name: str | None = None) -> Interaction:
    p = Path(csv_path)
    rep = json.loads((p.parent / "report.json").read_text()) if (p.parent / "report.json").exists() else {}
    sess = str(rep.get("session", ""))
    if any(sess.startswith(s) for s in SEALED):
        raise PermissionError(f"{p}: session {sess} is a sealed Hoi! test location")
    rows = list(csv.DictReader(open(p)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    keep = col("in_window") > 0 if only_window else np.ones(len(rows), bool)
    t, q, g = col("t_s")[keep], col("q_tcp")[keep], col("gen_force")[keep]
    lo, hi = col("gen_force_lo")[keep], col("gen_force_hi")[keep]
    valid = col("contact_valid")[keep] > 0
    z = np.zeros((len(t), 3))
    F, M = z.copy(), None
    if joint_type == "prismatic":
        F[:, 0] = g
    else:
        M = z.copy(); M[:, 0] = g
    joint = JointKinematics(joint_type=joint_type, axis=[1, 0, 0], origin=[0, 0, 0], frame=FRAME,
                            status=KinematicStatus.IDENTIFIED,
                            source=dict(note="packaged joint of the Hoi! force pipeline; effort already projected",
                                        report=str(p.parent / "report.json")))
    forces = ForceRecord(t_s=t, force_N=F, torque_Nm=M, point_m=z, frame=FRAME, contact_valid=valid,
                         force_std_N=((hi - lo) / 2).tolist(),
                         calibration=dict(rep.get("frozen", {}), source="freeartgs-hoi force pipeline (signed gen_force)"),
                         source=dict(csv=str(p), session=sess))
    traj = Trajectory(t_s=t, q=q, zero_reference=str(rep.get("dataset", {}).get("q", "from gripper TCP")),
                      source=dict(csv=str(p), kinematics="gripper TCP on the packaged joint (not ForceArtGS)"))
    meta = dict(report_checks=rep.get("checks"),
                cartesian=dict(frame="Hoi! Leica scan frame", F_applied_world_N=np.c_[col("F_applied_world_x_N"),
                               col("F_applied_world_y_N"), col("F_applied_world_z_N")][keep].tolist(),
                               contact_m=np.c_[col("contact_x_m"), col("contact_y_m"), col("contact_z_m")][keep].tolist()))
    return Interaction(name=name or p.stem, joint=joint, trajectory=traj, forces=forces, metadata=meta)
