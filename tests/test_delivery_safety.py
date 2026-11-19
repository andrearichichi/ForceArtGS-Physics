"""Safety of the handoff boundaries; no estimator algorithms are changed."""
import json
from dataclasses import replace

import pytest

from forceartgs_physics import api
from forceartgs_physics.adapters.real import load_observed_forces_json
from forceartgs_physics.export.parttwin import dynamics_document, write_dynamics_yaml
from forceartgs_physics.schemas import IdentificationResult, NotIdentifiableError, PhysicalParameters


def result(joint_type="prismatic"):
    return IdentificationResult("test", "ok", PhysicalParameters(joint_type, 2., 3., identifiable=True), ["x"],
                                identifiability={"identifiable": True, "kinematic_status": "identified"})


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_effective_mass_is_not_link_mass_or_tensor(kind):
    doc = dynamics_document(result(kind))
    assert all(v is None for v in doc["links"]["moving"].values())
    key = "effective_mass_kg" if kind == "prismatic" else "effective_inertia_about_axis_kg_m2"
    assert doc["joint_space_identification"][key] == 2.
    assert doc["joint"]["damping"] == 3.


@pytest.mark.parametrize("case", ["ambiguous", "unobservable", "static", "rejected", "unknown", "failed",
                                  "missing_diagnostic", "missing_kinematic_status", "negative_diagnostic", "missing_parameters",
                                  "parameter_conflict", "unassessed_parameters", "partial", "nan", "inf", "negative", "zero", "negative_damping", "zero_damping"])
def test_invalid_physics_never_exported(case, tmp_path):
    r = result()
    if case in ["ambiguous", "unobservable", "static", "rejected", "unknown"]:
        r.identifiability["kinematic_status"] = case
    elif case == "failed":
        r.status = "failed"
    elif case == "missing_diagnostic":
        r.identifiability = None
    elif case == "missing_kinematic_status":
        del r.identifiability["kinematic_status"]
    elif case == "negative_diagnostic":
        r.identifiability["identifiable"] = False
    elif case == "missing_parameters":
        r.parameters = None
    elif case in ("parameter_conflict", "unassessed_parameters"):
        r.parameters.identifiable = False if case == "parameter_conflict" else None
    elif case == "partial":
        r.identifiability["identifiable_damping"] = False
    elif case in ("negative_damping", "zero_damping"):
        r.parameters.damping = -1. if case == "negative_damping" else 0.
    else:
        r.parameters.inertia = {"nan": float("nan"), "inf": float("inf"), "negative": -1., "zero": 0.}[case]
    target = tmp_path / "physics.yaml"
    with pytest.raises(NotIdentifiableError):
        write_dynamics_yaml(r, target)
    assert not target.exists()
    with pytest.raises(NotIdentifiableError):
        api.save_parameters(r, tmp_path / "parameters.json")
    # Research reports retain failed candidates, but cannot be loaded as deployable physics.
    report = r.save(tmp_path / "report.json")
    with pytest.raises(NotIdentifiableError):
        api.load_parameters(report)


@pytest.mark.parametrize("identifiable", [False, None])
def test_bare_parameters_require_verdict(identifiable, tmp_path):
    p = PhysicalParameters("revolute", 1., 2., identifiable=identifiable)
    with pytest.raises(NotIdentifiableError):
        api.save_parameters(p, tmp_path / "p.json")


def test_measured_reader_requires_calibration_and_never_synthesizes(tmp_path):
    d = dict(t_s=[0., 1.], force_N=[[1., 0., 0.], [0., 0., 0.]], frame="verified",
             source=dict(kind="measured", recording="episode"))
    path = tmp_path / "forces.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="calibration"):
        load_observed_forces_json(path)
    keys = ["frame_registration", "force_sign", "sensor_axes", "application_point", "bias", "time_alignment"]
    d["calibration"] = {k: "measured procedure reference" for k in keys}
    path.write_text(json.dumps(d))
    f = load_observed_forces_json(path)
    assert f.force_N.tolist() == d["force_N"] and f.calibration == d["calibration"]
    d["source"]["kind"] = "synthetic"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="measured"):
        load_observed_forces_json(path)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_extension_coefficient_refuses_export(value):
    ext = {"physically_consistent": True, "terms": {"coulomb": {"identifiable": True, "value": value}}}
    with pytest.raises(NotIdentifiableError):
        dynamics_document(result(), extension_fit=ext)


def test_non_identifiable_extension_coefficient_is_unknown():
    ext = {"physically_consistent": True, "terms": {"coulomb": {"identifiable": False, "value": 5.}}}
    assert dynamics_document(result(), extension_fit=ext)["joint"]["coulomb_friction"] is None


@pytest.mark.parametrize("extension", [{}, {"physically_consistent": False}])
def test_inconsistent_or_unassessed_extension_blocks_asset(extension):
    with pytest.raises(NotIdentifiableError):
        dynamics_document(result(), extension_fit=extension)


def test_measured_cli_requires_observed_force_file():
    from forceartgs_physics.cli import main
    with pytest.raises(SystemExit, match="forces-json"):
        main(["identify", "--forceartgs-run", "/unused", "--level", "A", "--out", "/unused.json"])


def test_synthetic_demo_refuses_other_runs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "examples" / "01_forceartgs_end_to_end.py"
    out = tmp_path / "must_not_be_created"
    run = subprocess.run([sys.executable, str(example), "--run", str(tmp_path), "--out", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 2 and "synthetic demo only" in run.stderr
    assert not out.exists()



def test_cli_cannot_mix_benchmark_and_measured_sources():
    from forceartgs_physics.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["diagnose", "--forceartgs-run", "/unused", "--estimator-input", "/unused.json"])
    assert exc.value.code == 2
