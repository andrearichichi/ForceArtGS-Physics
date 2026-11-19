import numpy as np
import pytest

from forceartgs_physics.schemas import (ForceRecord, Interaction, JointKinematics, KinematicStatus, NotIdentifiableError,
                                        PhysicalParameters, Trajectory)
from forceartgs_physics.units import units_for


def _inter(status="identified", force_frame="W"):
    j = JointKinematics("revolute", [0, 0, 2.0], [0, 0, 0], "W", status=status)
    tr = Trajectory([0, 1, 2], [0, .1, .2])
    f = ForceRecord([0, 1], [[1, 0, 0], [0, 0, 0]], force_frame, point_m=[[0, 1, 0], [0, 1, 0]])
    return Interaction("x", j, tr, f)


def test_axis_normalised_and_units():
    i = _inter()
    assert np.allclose(i.joint.axis, [0, 0, 1])
    assert units_for("revolute")["inertia"] == "kg*m^2" and units_for("prismatic")["damping"] == "N*s/m"


def test_frame_mismatch_is_refused():
    with pytest.raises(ValueError, match="frame"):
        _inter(force_frame="other")


@pytest.mark.parametrize("st", ["ambiguous", "unobservable", "static", "rejected", "unknown"])
def test_non_identified_kinematics_refused(st):
    i = _inter(status=st)
    with pytest.raises(NotIdentifiableError):
        i.require_identifiable()


def test_bad_inputs():
    with pytest.raises(ValueError):
        Trajectory([0, 0, 1], [0, 1, 2])
    with pytest.raises(ValueError):
        JointKinematics("spherical", [1, 0, 0], [0, 0, 0], "W")
    with pytest.raises(ValueError):
        Interaction("x", JointKinematics("revolute", [0, 0, 1], [0, 0, 0], "W"), Trajectory([0, 1], [0, 1]),
                    ForceRecord([0, 1], [[1, 0, 0]] * 2, "W"))          # revolute without application points


def test_parameters_roundtrip(tmp_path):
    from forceartgs_physics import api
    p = PhysicalParameters("prismatic", 2.0, 4.0, identifiable=True)
    api.save_parameters(p, tmp_path / "p.json")
    q = api.load_parameters(tmp_path / "p.json")
    assert q.inertia == 2.0 and q.damping == 4.0 and q.decay_time_s == 0.5 and q.units["inertia"] == "kg"
