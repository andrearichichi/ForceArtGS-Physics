"""ForceSAPIEN adapters.

* ``load_estimator_input``: the Protocol-C / A3 ``<slug>.*.estimator_input.json`` (FreeArtGS/ForceArtGS-fork
  kinematics + simulator force record), read exactly as controlled_fairness/common.py::load did: ALL observations
  (the file's ``estimation_window_end_s`` is NOT applied by the A3 fits), the whole force record, joint from
  ``joint_force_world``. Frame: the ForceSAPIEN simulator world (metres), in which the joint was registered
  (fork align_prediction + capture->force similarity, see the file's ``registration``).
* ``load_force_record``: a reduced B force record of the bundle or a raw ``*.force_record.json`` / ``simulation.json``.
* ``A3Benchmark``: the bundled 10-object benchmark (benchmark/forcesapien_a3). Truth is reachable only through
  ``A3Benchmark.truth`` (evaluation code); no estimator path opens it.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

from ..schemas import ForceRecord, Interaction, JointKinematics, KinematicStatus, Trajectory

FRAME = "forcesapien_world"
OBJECTS = ["FoldingChair_102255", "Knife_103706", "Laptop_10211", "Oven_101917", "Refrigerator_10905", "Scissors_11100",
           "Stapler_103111", "StorageFurniture_45135", "USB_100109", "WashingMachine_103776"]
B_INTERACTIONS = [f"B{j}" for j in range(1, 11)]


def _read(p: Path):
    p = Path(p)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return json.loads(p.read_text())


def load_estimator_input(path, name: str | None = None) -> Interaction:
    x = _read(path)
    j, f, o = x["joint_force_world"], x["force"], x["observations"]
    if x.get("leak_guard", {}).get("violations"):
        raise ValueError(f"{path}: input carries leak-guard violations")
    joint = JointKinematics(joint_type=j["type"], axis=j["axis"], origin=j["origin"], frame=FRAME,
                            status=KinematicStatus.IDENTIFIED,
                            source=dict(file=str(path), geometry=x.get("geometry"), registration_method=(x.get("registration") or {}).get("method")))
    traj = Trajectory(t_s=o["t"], q=o["q"], zero_reference="q(first observation) = 0, sign from q_sign_from_force",
                      source=dict(q_sign=x.get("q_sign"), recorded_sample=o.get("recorded_sample")))
    forces = ForceRecord(t_s=f["t"], force_N=f["force_world"], point_m=f["application_point_world"], frame=FRAME,
                         contact_valid=f.get("active"), source=dict(interpolation=f.get("interpolation"), t_stop_s=f.get("t_stop_s")),
                         calibration=dict(note="simulator applied force (exact, no sensor model)"))
    if joint.joint_type != x["joint_type"]:
        raise ValueError("joint_type mismatch inside estimator input")
    return Interaction(name=name or x.get("slug", Path(path).stem), joint=joint, trajectory=traj, forces=forces,
                       metadata=dict(units=x.get("units"), estimation_window_end_s=x.get("estimation_window_end_s"),
                                     window_applied=False, initial_state=x.get("initial_state"),
                                     effort_on_force_grid=x.get("effort_on_force_grid")))


def load_force_record(path) -> ForceRecord:
    x = _read(path)
    if "samples" in x and isinstance(x["samples"], dict):          # simulation.json
        smp = x["samples"]["force"]
    elif "samples" in x:                                             # *.force_record.json
        smp = x["samples"]
    else:                                                            # bundle (reduced) record
        return ForceRecord(t_s=x["time_s"], force_N=x["force_world"], point_m=x["application_point_world"], frame=FRAME,
                           contact_valid=x["force_active"], source=dict(file=str(path), interaction=x.get("interaction")))
    return ForceRecord(t_s=[s["time_s"] for s in smp],
                       force_N=[s.get("force_vector_world", s.get("applied_force_world")) for s in smp],
                       point_m=[s["application_point_world"] for s in smp], frame=FRAME,
                       contact_valid=[bool(s.get("force_active", True)) for s in smp], source=dict(file=str(path)))


class A3Benchmark:
    """Accessor of benchmark/forcesapien_a3 (built by tools/build_forcesapien_a3_bundle.py)."""

    def __init__(self, root=None):
        if root is None:
            root = os.environ.get("FORCEARTGS_PHYSICS_BENCHMARK")
            if not root:
                raise FileNotFoundError("ForceSAPIEN data are distributed separately. Set "
                                        "FORCEARTGS_PHYSICS_BENCHMARK to an authorized external bundle directory.")
        self.root = Path(root)
        if not (self.root / "MANIFEST.json").exists():
            raise FileNotFoundError(f"benchmark bundle not found at {self.root}")

    objects = OBJECTS
    interactions_B = B_INTERACTIONS

    def interaction_A3(self, slug) -> Interaction:
        return load_estimator_input(self.root / "inputs" / f"{slug}.A3.estimator_input.json", name=f"{slug}.A3")

    def forces_B(self, slug, b) -> ForceRecord:
        return load_force_record(self.root / "forces_B" / f"{slug}.{b}.json.gz")

    def reference(self, name):
        p = self.root / "reference" / name
        return _read(p)

    def truth(self, slug, interaction):
        """EVALUATION ONLY (simulator ground truth)."""
        return _read(self.root / "truth" / f"{slug}.{interaction}.truth.json.gz")
