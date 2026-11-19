"""End-to-end on real ForceArtGS outputs (Level B, synthetic scenes of ForceArtGS' own generator) + a robot record
written as a LeRobot parquet: load -> joint/q(t) -> robot forces -> estimators -> save -> predict -> export."""
import json

import numpy as np
import pytest

import forceartgs_synthetic_robot as S
from forceartgs_physics import api
from forceartgs_physics.adapters import forceartgs as FA
from forceartgs_physics.export import write_dynamics_yaml
from forceartgs_physics.schemas import KinematicStatus, NotIdentifiableError

I_TRUE, C_TRUE = 0.05, 0.15


@pytest.fixture(scope="module")
def revolute(fixtures_dir, tmp_path_factory):
    joint, traj = FA.load_discovery_run(fixtures_dir / "forceartgs_synthetic_revolute_s0", frame="base_link")
    rec = S.robot_record("revolute", [1, 0, 0], [0, 0.1, 0.13], -np.radians(40), 1.1, I_TRUE, C_TRUE)
    try:
        path = S.write_parquet(tmp_path_factory.mktemp("robot") / "episode.parquet", rec)
        forces = FA.load_lerobot_parquet(path, S.TCP_OFFSET, fps=50.0, frame="base_link")
    except ImportError:
        forces = FA.robot_forces_from_arrays(rec["frame_index"] / 50.0, rec["ee_pos"], rec["ee_rotvec"], rec["wrench"],
                                             S.TCP_OFFSET, frame="base_link")
    return FA.interaction("forceartgs_synthetic_revolute", joint, traj, forces), rec


def test_kinematics_contract(revolute):
    x, rec = revolute
    assert x.joint.status is KinematicStatus.IDENTIFIED and x.joint_type == "revolute"
    assert np.degrees(np.arccos(abs(x.joint.axis @ [1, 0, 0]))) < 0.5
    assert len(x.trajectory.t_s) == 12 and x.trajectory.q[0] == 0.0
    assert np.max(np.abs(x.forces.force_N - rec["F_object"])) < 1e-6           # reaction sign + EE axes undone
    assert np.max(np.abs(x.forces.point_m - rec["p_contact"])) < 1e-6           # TCP offset applied


def test_ambiguous_forceartgs_decision_blocks_physics(fixtures_dir):
    joint, traj = FA.load_discovery_run(fixtures_dir / "forceartgs_synthetic_prismatic_s0", frame="base_link")
    assert joint.status is KinematicStatus.AMBIGUOUS
    rec = S.robot_record("prismatic", [0, -1, 0], [0, -0.21, 0.05], 0.12, 1.1, 2.0, 3.0)
    forces = FA.robot_forces_from_arrays(rec["frame_index"] / 50.0, rec["ee_pos"], rec["ee_rotvec"], rec["wrench"],
                                         S.TCP_OFFSET, frame="base_link")
    x = FA.interaction("prismatic_ambiguous", joint, traj, forces)
    with pytest.raises(NotIdentifiableError):
        api.identify(x, "IntegralLS")
    with pytest.raises(ValueError):
        FA.load_discovery_run(fixtures_dir / "forceartgs_synthetic_prismatic_s0", kind="prismatic", frame="base_link")


@pytest.mark.parametrize("method,tol", [("IntegralLS", 0.02), ("Backprop", 0.02), ("AKF", 0.02)])
def test_cpu_estimators_recover_parameters(revolute, method, tol, tmp_path):
    x, _ = revolute
    r = api.identify(x, method)
    assert r.status == "ok" and r.identifiability["identifiable"]
    assert abs(r.parameters.inertia / I_TRUE - 1) < tol and abs(r.parameters.damping / C_TRUE - 1) < tol
    p = api.save_parameters(r, tmp_path / "res.json")
    params = api.load_parameters(p)
    pred = api.predict(params, x.forces, x.joint, x.trajectory.t_s)
    ev = api.evaluate_prediction(pred, x.trajectory.t_s, x.trajectory.q)
    assert ev["nrmse_pct"] < 2.0
    y = write_dynamics_yaml(r, tmp_path / "physics" / "dynamics.yaml")
    txt = y.read_text()
    assert "parttwin.dynamics/v1" in txt and "mode: \"identified\"" in txt and "inertia_com_link_kg_m2: null" in txt



def test_installed_optional_parquet_dependency_imports():
    import importlib.util
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("optional pyarrow not installed")
    # An installed but incompatible Arrow wheel must fail, rather than hide behind
    # the synthetic array fallback in the integration fixture.
    import pyarrow.parquet
    assert pyarrow.parquet is not None
