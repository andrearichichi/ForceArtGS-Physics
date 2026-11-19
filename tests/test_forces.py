import numpy as np
import pytest

from forceartgs_physics.forces import SimilarityTransform, generalized_effort, lever_factor
from forceartgs_physics.forces.preprocessing import apply_contact_mask, remove_constant_bias
from forceartgs_physics.schemas import ForceRecord, JointKinematics


def test_projection_signs():
    j = JointKinematics("revolute", [0, 0, 1], [0, 0, 0], "W")
    f = ForceRecord([0, 1], [[0, 1, 0], [0, -1, 0]], "W", point_m=[[1, 0, 0], [1, 0, 0]], torque_Nm=[[0, 0, .5], [0, 0, 0]])
    assert np.allclose(generalized_effort(j, f).levels, [1.5, -1.0])
    jp = JointKinematics("prismatic", [2, 0, 0], [0, 0, 0], "W")
    fp = ForceRecord([0, 1], [[3, 1, 0], [-1, 0, 0]], "W")
    assert np.allclose(generalized_effort(jp, fp).levels, [3, -1])


def test_similarity_transform_preserves_effort():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation
    T = SimilarityTransform(Rotation.random(random_state=1).as_matrix(), [1, 2, 3], 1.0, "A", "B")
    j = JointKinematics("revolute", rng.normal(size=3), rng.normal(size=3), "A")
    f = ForceRecord([0, 1, 2], rng.normal(size=(3, 3)), "A", point_m=rng.normal(size=(3, 3)))
    assert np.allclose(generalized_effort(j, f).levels, generalized_effort(T.joint(j), T.forces(f)).levels)


def test_preprocessing_logged():
    f = ForceRecord([0, 1, 2, 3, 4], np.ones((5, 3)), "W", contact_valid=[0, 1, 1, 0, 0])
    g = apply_contact_mask(f)
    assert g.force_N[0].sum() == 0 and g.force_N[1].sum() == 3 and g.calibration["contact_mask_applied"] == 3
    h = remove_constant_bias(f, (0, 2))
    assert np.allclose(h.force_N, 0) and "bias_removed" in h.calibration


def test_bundle_effort_matches_stored_effort(bench):
    """The signed projection reproduces the effort stored in every A3 input (computed by build_inputs_C.effort)."""
    for slug in bench.objects:
        x = bench.interaction_A3(slug)
        e = generalized_effort(x.joint, x.forces).levels
        ref = np.asarray(x.metadata["effort_on_force_grid"])
        assert np.max(np.abs(e - ref)) <= 1e-12 * np.max(np.abs(ref)), slug
