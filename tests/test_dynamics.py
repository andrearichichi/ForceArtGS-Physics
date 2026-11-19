import numpy as np

from forceartgs_physics._validated import identifiability_rv2 as rv2
from forceartgs_physics._validated.akf_pinn.physics.integrator import rk4_integrate
from forceartgs_physics.dynamics.viscous import propagate_from_rest, propagate_from_state
from forceartgs_physics.forces.projection import GeneralizedEffort

EFF = GeneralizedEffort(np.arange(0.1, 3.0, 1 / 30), np.r_[np.full(60, 0.7), np.zeros(27)], "revolute", "test")


def test_state_solver_reduces_to_validated_solver():
    tq = np.linspace(0, 3, 301)
    a = propagate_from_rest(0.02, 0.01, EFF, tq)
    b = propagate_from_state(0.02, 0.01, EFF, tq, 0.0, 0.0, 0.0)
    assert np.array_equal(a, b)


def test_against_independent_solver_and_rk4():
    import torch
    tq = np.linspace(0.5, 3, 200)
    I, c, q0, v0 = 0.02, 0.01, 0.3, -0.2
    a = propagate_from_state(I, c, EFF, tq, 0.5, q0, v0)
    knots = np.r_[0.5, EFF.t_s[EFF.t_s > 0.5]]
    lev = np.r_[EFF.at(0.5), EFF.levels[EFF.t_s > 0.5]]
    b = rv2.solve(I, c, q0, v0, knots, lev, tq)               # robust_v2 identifiability solver (independent code)
    assert np.max(np.abs(a - b)) < 1e-10
    f = lambda z, t: torch.stack([z[1], (torch.as_tensor(float(EFF.at(float(t))), dtype=z.dtype) - c * z[1]) / I])
    z = torch.tensor([q0, v0], dtype=torch.float64)
    t_prev, out = 0.5, []
    for t in tq:
        z = rk4_integrate(f, z, t_prev, t, 1e-3) if t > t_prev else z
        t_prev = t; out.append(float(z[0]))
    assert np.max(np.abs(a - np.array(out))) < 5e-3 * np.ptp(a)   # RK4 across ZOH jumps is only first order


def test_small_damping_limit_is_stable():
    q = propagate_from_rest(1.0, 1e-14, EFF, [2.0])
    assert np.isfinite(q).all()
