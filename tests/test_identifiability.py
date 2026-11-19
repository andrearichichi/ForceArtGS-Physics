import numpy as np
import pytest

from forceartgs_physics.dynamics.extensions import ExtendedModel, identify_extended_ls, simulate
from forceartgs_physics.dynamics.viscous import propagate_from_state
from forceartgs_physics.forces.projection import GeneralizedEffort
from forceartgs_physics.schemas import ForceRecord, Interaction, JointKinematics, Trajectory
from forceartgs_physics.validation.identifiability import diagnose


def _prismatic(t_obs, q, tf, f, status="identified"):
    j = JointKinematics("prismatic", [1, 0, 0], [0, 0, 0], "W", status=status)
    F = np.zeros((len(tf), 3)); F[:, 0] = f
    return Interaction("syn", j, Trajectory(t_obs, q), ForceRecord(tf, F, "W"))


def test_push_and_decay_is_identifiable():
    tf = np.arange(0, 6, 1 / 50); f = np.where(tf < 2, 1.0, 0.0)
    t = np.arange(0.02, 6, 1 / 30)
    q = propagate_from_state(0.5, 0.25, GeneralizedEffort(tf, f, "prismatic", ""), t) + 1e-5 * np.random.default_rng(0).normal(size=len(t))
    r = diagnose(_prismatic(t, q, tf, f))
    assert r["identifiable"] and r["crlb_rel_std_I_pct"] < 1


def test_steady_push_mass_not_identifiable():
    """REAL-like constant-velocity push: the force balances damping, acceleration carries no information."""
    t = np.arange(0.5, 3, 1 / 25); tf = np.arange(0, 3, 1 / 50)
    f = np.full(len(tf), 2.0)
    q = 0.05 * t + 2e-4 * np.random.default_rng(1).normal(size=len(t))     # steady motion observed after transient
    r = diagnose(_prismatic(t, q, tf, f))
    assert not r["identifiable"] and "NOT separately identifiable" in r["verdict"]


def test_ambiguous_status_short_circuits():
    t = np.linspace(0, 1, 10)
    r = diagnose(_prismatic(t, t, t, np.ones(10), status="ambiguous"))
    assert r["identifiable"] is False and "ambiguous" in r["verdict"]


def test_coulomb_and_bias_collinear_in_one_direction_push():
    tf = np.arange(0, 4, 1 / 100); f = np.where(tf < 3, 6.0, 0.0)
    eff = GeneralizedEffort(tf, f, "prismatic", "")
    t = np.arange(0, 2.5, 1 / 30)
    q = simulate(ExtendedModel(2.0, 1.0, coulomb=1.5), eff, t)
    out = identify_extended_ls(t, q, eff, terms=("coulomb", "bias"))
    assert not out["terms"]["coulomb"]["identifiable"] or not out["terms"]["bias"]["identifiable"]
    assert abs(out["correlation"]["coulomb~bias"]) > 0.99
