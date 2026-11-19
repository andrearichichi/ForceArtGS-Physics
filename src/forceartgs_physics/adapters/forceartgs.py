"""Adapter for Samuele Mara's ForceArtGS (https://github.com/SamueleMara/ForceArtGS, interface audited at 886d49e).

ForceArtGS discovers the joint; this library only CONSUMES its kinematics. Two kinematic contracts are supported
(see docs/FORCEARTGS_INTERFACE.md for the full key lists and file:line references):

  Level A  joint authority   <branch>/joint_authority/v<NNN>/{authority.json, joint_states.npz}
           (schema 'forceartgs.joint_authority/1'; written by integration_contract.JointAuthority.write)
  Level B  discovery run      <run>/joint_candidates.json (+ joint_states.npz), written by joint_discovery.cli

Both give, in the world frame W of metadata/cameras.json (robot base_link on the real scenes, metres):
joint type, axis (sign-canonicalised together with q), origin (revolute), q(t) [rad | m] referenced to 0 at
``reference_time_index``, ``timestamps_s`` (= source_frame_index / fps on real scenes), validity mask and optional
per-sample sigma. The decision / decision_status is mapped to schemas.KinematicStatus and anything other than an
accepted revolute/prismatic joint is NOT identifiable (fail closed).

Robot forces (LeRobot parquet, 50 Hz): EE position (m) + rotation vector in base_link, wrench (N, N*m) in the sensor
axes, declared equal to the EE axes; the recorded force is the reaction ON the robot, so the force ON the object is
-1 x recorded (configs/real_robot_tools.json 'object_side_force_multiplier': -1). The application point is the TCP
p = p_ee + R_ee t_ee_tcp. The recorded torques are NOT used: their reference point is undocumented (audit gap G4);
the joint torque is rebuilt from (p - o) x F by the projection.

The old FreeArtGS pipeline outputs (``twopart_blend/estimated_joint.json`` + ``splatfacto-art/articulation.json``)
are NOT metric (nerfstudio-normalised frame) and carry no timestamps; they are refused here. The ForceSAPIEN A3
inputs were produced from that pipeline by our own registration (fork align_prediction + capture->force similarity)
and are read with adapters.forcesapien.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import ForceRecord, Interaction, JointKinematics, KinematicStatus, Trajectory

AUTHORITY_SCHEMA = "forceartgs.joint_authority/1"
DECISION_STATUS = {"revolute": KinematicStatus.IDENTIFIED, "prismatic": KinematicStatus.IDENTIFIED,
                   "static/no_motion": KinematicStatus.STATIC, "ambiguous/unobservable": KinematicStatus.AMBIGUOUS,
                   # physical selection v2 labels
                   "IDENTIFIED_REVOLUTE": KinematicStatus.IDENTIFIED, "IDENTIFIED_PRISMATIC": KinematicStatus.IDENTIFIED,
                   "TYPE_UNOBSERVABLE": KinematicStatus.UNOBSERVABLE, "MODEL_INCONSISTENT": KinematicStatus.REJECTED,
                   "NO_SIGNIFICANT_MOTION": KinematicStatus.STATIC}
AUTHORITY_STATUS = {"identified": KinematicStatus.IDENTIFIED, "ambiguous": KinematicStatus.AMBIGUOUS,
                    "candidate": KinematicStatus.UNKNOWN, "provisional": KinematicStatus.UNKNOWN}


def _trajectory(t, q, valid, sigma, ref_index, source) -> Trajectory:
    t, q = np.asarray(t, float), np.asarray(q, float)
    valid = np.ones(len(q), bool) if valid is None else np.asarray(valid, bool)
    valid &= np.isfinite(q)
    s = None if sigma is None else np.asarray(sigma, float)
    if s is not None and not np.all(np.isfinite(s[valid])):
        s = None
    return Trajectory(t_s=t[valid], q=q[valid], q_std=None if s is None else s[valid],
                      zero_reference=f"q = 0 at ForceArtGS reference_time_index {ref_index}",
                      source=dict(source, n_invalid_dropped=int((~valid).sum())))


def load_discovery_run(run_dir, kind: str | None = None, frame: str = "forceartgs_world",
                       allow_override: bool = False) -> tuple[JointKinematics, Trajectory]:
    """Level B. ``kind`` defaults to the run's decision. Asking for a kind different from the decision requires
    allow_override=True and is recorded (the kinematics are then NOT marked identified)."""
    run_dir = Path(run_dir)
    jc = json.loads((run_dir / "joint_candidates.json").read_text())
    decision = jc.get("decision")
    status = DECISION_STATUS.get(decision, KinematicStatus.UNKNOWN)
    chosen = kind or (decision if decision in ("revolute", "prismatic") else None)
    if chosen is None:
        # nothing usable: return the best-documented candidate with the non-identified status, so callers can report it
        chosen = next(iter(jc["candidates"]))
    if kind is not None and kind != decision:
        if not allow_override:
            raise ValueError(f"requested kind {kind!r} but ForceArtGS decided {decision!r}; pass allow_override=True")
        status = KinematicStatus.UNKNOWN
    c = jc["candidates"][chosen]
    units = c["q_units"]
    if (chosen == "revolute") != (units == "rad"):
        raise ValueError(f"inconsistent q_units {units!r} for {chosen}")
    origin = c["joint_origin_world"] if c.get("joint_origin_world") is not None else [0.0, 0.0, 0.0]
    obs = c.get("observability") or {}
    joint = JointKinematics(joint_type=chosen, axis=c["joint_axis_world"], origin=origin, frame=frame, status=status,
                            axis_std_rad=None if obs.get("axis_sigma_deg") is None else float(np.deg2rad(obs["axis_sigma_deg"])),
                            origin_std_m=obs.get("hinge_line_sigma_m"),
                            source=dict(level="B joint_candidates.json", run=str(run_dir), decision=decision,
                                        world_frame=c.get("world_frame"), sign_convention=c.get("sign_convention"),
                                        origin_convention=c.get("origin_convention"), observability=obs,
                                        origin_is_placeholder=c.get("joint_origin_world") is None))
    traj = _trajectory(c["timestamps_s"], c["q"], c.get("q_valid"), c.get("q_uncertainty"), c.get("reference_time_index"),
                       dict(level="B", run=str(run_dir), q_units=units))
    return joint, traj


def load_authority(authority_dir, frame: str = "forceartgs_world", allow_provisional: bool = False):
    """Level A. Verifies the schema and the joint_states.npz hash recorded in authority.json (as
    JointAuthority.load does) without importing ForceArtGS."""
    import hashlib
    d = Path(authority_dir)
    meta = json.loads((d / "authority.json").read_text())
    if meta.get("schema_version") != AUTHORITY_SCHEMA:
        raise ValueError(f"unsupported authority schema {meta.get('schema_version')!r}")
    h = hashlib.sha256((d / "joint_states.npz").read_bytes()).hexdigest()
    if h != meta.get("joint_states_sha256"):
        raise ValueError("joint_states.npz does not match the hash recorded in authority.json")
    z = np.load(d / "joint_states.npz")
    status = AUTHORITY_STATUS.get(meta.get("decision_status"), KinematicStatus.UNKNOWN)
    if allow_provisional and meta.get("decision_status") == "provisional":
        status = KinematicStatus.IDENTIFIED
    jt = meta["joint_type"]
    if jt not in ("revolute", "prismatic"):
        status = KinematicStatus.STATIC if jt == "static" else KinematicStatus.UNKNOWN
        jt = "prismatic"                       # placeholder type; status prevents any identification
    joint = JointKinematics(joint_type=jt, axis=meta["axis_world"] or [1, 0, 0], origin=meta["origin_world"] or [0, 0, 0],
                            frame=frame, status=status,
                            source=dict(level="A joint authority", dir=str(d), authority_id=meta.get("authority_id"),
                                        decision_status=meta.get("decision_status"), world_frame=meta.get("world_frame"),
                                        fit_covariance_status=meta.get("fit_covariance_status")))
    traj = _trajectory(z["timestamp_s"], z["q"], z["q_valid"], z["q_sigma"] if "q_sigma" in z.files else None,
                       meta.get("reference_time_index"), dict(level="A", authority_id=meta.get("authority_id")))
    return joint, traj


def refuse_legacy(run_dir) -> None:
    raise ValueError(f"{run_dir}: legacy FreeArtGS output (estimated_joint.json / articulation.json) is in a normalised, "
                     "non-metric frame without timestamps; register it to metric world first (see adapters.forcesapien)")


# ---------------------------------------------------------------------------------------------- robot data
def _rotvec_to_R(rv):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(rv, float)).as_matrix()


def robot_forces_from_arrays(t_s, ee_pos_m, ee_rotvec, wrench_sensor, t_ee_tcp_m, *, wrench_frame="ee",
                             object_side_multiplier=-1.0, frame="base_link", R_ee_sensor=None, contact_valid=None,
                             source=None) -> ForceRecord:
    """Build the force ON the object at the TCP, in ``frame`` (= robot base). wrench_frame: 'ee' (sensor axes = EE
    axes, the declared convention of the REAL recordings) or 'base'. Torques are dropped (reference undocumented)."""
    t = np.asarray(t_s, float)
    p_ee = np.asarray(ee_pos_m, float)
    R = _rotvec_to_R(ee_rotvec)                                   # [N,3,3] base <- ee
    W = np.asarray(wrench_sensor, float)
    F = W[:, :3]
    if wrench_frame == "ee":
        Rs = np.eye(3) if R_ee_sensor is None else np.asarray(R_ee_sensor, float)
        F = np.einsum("nij,jk,nk->ni", R, Rs, F)
    elif wrench_frame != "base":
        raise ValueError("wrench_frame must be 'ee' or 'base'")
    F = object_side_multiplier * F
    p_tcp = p_ee + np.einsum("nij,j->ni", R, np.asarray(t_ee_tcp_m, float))
    return ForceRecord(t_s=t, force_N=F, point_m=p_tcp, frame=frame, contact_valid=contact_valid, wrench_frame=wrench_frame,
                       tcp=dict(t_ee_tcp_m=list(map(float, t_ee_tcp_m)), convention="T_base_tcp = T_base_ee @ T_ee_tcp"),
                       calibration=dict(object_side_multiplier=object_side_multiplier, sensor_axes=wrench_frame,
                                        torques="dropped: wrench torque reference point undocumented",
                                        bias="NOT removed (use forces.preprocessing.remove_constant_bias with a declared "
                                             "contact-free window)"),
                       source=source or {})


def load_lerobot_parquet(parquet, t_ee_tcp_m, *, fps: float | None = 50.0, time_source="frame_index", **kw) -> ForceRecord:
    """LeRobot episode parquet (columns observation.state [EE pos(3), rotvec(3), ...], observation.wrench [6],
    frame_index, timestamp). ``time_source='frame_index'`` gives t = frame_index / fps, the SAME clock ForceArtGS
    uses for ``timestamps_s`` (audit gap G2: neither is a hardware stamp); 'timestamp' uses the parquet column.
    Needs the optional dependency pyarrow (pip install -e .[real])."""
    import pyarrow.parquet as pq
    tab = pq.read_table(parquet).to_pydict()
    state = np.asarray(tab["observation.state"], float)
    wrench = np.asarray(tab["observation.wrench"], float)
    if time_source == "frame_index":
        t = np.asarray(tab["frame_index"], float) / float(fps)
    elif time_source == "timestamp":
        t = np.asarray(tab["timestamp"], float)
    else:
        raise ValueError(time_source)
    return robot_forces_from_arrays(t, state[:, :3], state[:, 3:6], wrench, t_ee_tcp_m,
                                    source=dict(parquet=str(parquet), time_source=time_source, fps=fps), **kw)


def interaction(name, joint, trajectory, forces, metadata=None) -> Interaction:
    """Assemble; the joint frame name must equal the force frame name (declare both as the robot base on real scenes
    only when the ForceArtGS world is the robot base, i.e. every camera of metadata/cameras.json declares base_link)."""
    return Interaction(name=name, joint=joint, trajectory=trajectory, forces=forces, metadata=metadata or {})
