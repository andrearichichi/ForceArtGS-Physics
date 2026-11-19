"""REAL robot dataset (Samuele's Doosan pushes; LeRobot parquet per episode, 50 Hz).

Contents (REAL_AUDIT 2026-11-11 + ForceArtGS audit): observation.state[:3] EE position in base_link [m], [3:6] rotation
vector, [6:12] hand channels (all zero); observation.wrench 6 values [N, N*m]; no per-sensor timestamps;
frame_timestamps.csv maps video frames to parquet rows. The robot PUSHES (no grasp) along straight lines at
0.05-0.10 m/s. Tool transform: configs/real_robot_tools.json of ForceArtGS (pointed tip +0.1876 m along EE +Z for
most scenes; three white-drawer recordings have an uncalibrated mount and must not be used).

Kinematics come from ForceArtGS (adapters.forceartgs); as of 2026-11-19 no ForceArtGS joint-authority run exists on
Leonardo for REAL scenes, so this module only provides the force side.

Identifiability warning: constant-speed pushes give almost no acceleration information: mass is not identifiable
and viscous vs Coulomb friction are collinear (REAL_AUDIT). The library's identifiability diagnostic will say so.
"""
from __future__ import annotations

from .forceartgs import load_lerobot_parquet, robot_forces_from_arrays

# Historical reference only. Never applied automatically by the measured-data reader.
HISTORICAL_POINTED_TCP_EE_M = (0.0, 0.0, 0.1876)

__all__ = ["load_lerobot_parquet", "robot_forces_from_arrays", "HISTORICAL_POINTED_TCP_EE_M", "load_observed_forces_json"]


def load_observed_forces_json(path):
    """Read preprocessed measured forces; never infer calibration or synthesize force.

    SI arrays must already share the joint clock and frame. Calibration entries
    describe the measurement/procedure, not merely a boolean approval.
    """
    import json
    from pathlib import Path
    from ..schemas import ForceRecord

    d = json.loads(Path(path).read_text())
    if d.get("source", {}).get("kind") != "measured":
        raise ValueError("real input requires source.kind='measured'")
    if not d["source"].get("recording"):
        raise ValueError("declare source.recording")
    calibration = d.get("calibration", {})
    for key in ("frame_registration", "force_sign", "sensor_axes", "application_point", "bias", "time_alignment"):
        if not isinstance(calibration.get(key), str) or not calibration[key].strip():
            raise ValueError(f"declare measured calibration/procedure: {key}")
    if d.get("torque_Nm") is not None and not calibration.get("torque_reference"):
        raise ValueError("measured couples require calibration.torque_reference")
    return ForceRecord(t_s=d["t_s"], force_N=d["force_N"], point_m=d.get("point_m"),
                       torque_Nm=d.get("torque_Nm"), frame=d["frame"],
                       calibration=calibration, source=d["source"])
