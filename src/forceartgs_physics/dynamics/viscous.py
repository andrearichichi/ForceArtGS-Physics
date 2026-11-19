"""Exact solution of ``I q'' + c q' = tau(t)`` for a piecewise-constant (forward ZOH) effort.

``propagate_from_rest`` is the validated benchmark solver (oneplus10/replay.py ``propagate``, vendored verbatim):
rest at t = 0, first effort level held back to t = 0. ``propagate_from_state`` generalises it to an arbitrary start
(t0, q0, v0) with the same cancellation-free closed form; with t0 = 0, q0 = v0 = 0 it reproduces the validated solver
(tests/test_dynamics.py).
"""
from __future__ import annotations

import math

import numpy as np

from .._validated import replay as _rp


def propagate_from_rest(inertia: float, damping: float, effort, t_query) -> np.ndarray:
    """Benchmark solver. ``effort`` is a GeneralizedEffort (levels held on [t_k, t_{k+1}), first level from 0)."""
    knots, levels = _rp.zoh_levels(np.asarray(effort.t_s, float), np.asarray(effort.levels, float))
    return _rp.propagate(float(inertia), float(damping), knots, levels, np.asarray(t_query, float))


def _phi(x):
    if x < 1e-2:
        return (1 - x / 2 + x * x / 6 - x ** 3 / 24 + x ** 4 / 120,
                .5 - x / 6 + x * x / 24 - x ** 3 / 120 + x ** 4 / 720)
    em1 = math.expm1(-x)
    return -em1 / x, (x + em1) / (x * x)


def propagate_from_state(inertia: float, damping: float, effort, t_query, t0: float = 0.0, q0: float = 0.0,
                         v0: float = 0.0, return_velocity: bool = False):
    """Exact propagation from (q0, v0) at t0 (all t_query >= t0). Effort level at time t is the last sample with
    t_k <= t (or the first sample if t < t_s[0])."""
    I, c = float(inertia), float(damping)
    if not (I > 0 and c >= 0):
        raise ValueError("inertia must be > 0 and damping >= 0")
    k = c / I
    tf, lev = np.asarray(effort.t_s, float), np.asarray(effort.levels, float)
    tq = np.asarray(t_query, float)
    if np.any(tq < t0 - 1e-12):
        raise ValueError("query times must not precede t0")
    if t0 <= tf[0]:
        # the validated layout (replay.zoh_levels): first level held from t0 to t_1, no knot at t_0
        knots, levels = np.r_[t0, tf[1:]], lev
    else:
        after = tf > t0
        j0 = int(np.searchsorted(tf, t0, side="right")) - 1
        knots = np.r_[t0, tf[after]]
        levels = np.r_[lev[j0], lev[after]]
    edges = np.r_[knots, np.inf]

    def step(q, v, tau, h):
        x = k * h
        p1, p2 = _phi(x)
        a = tau / I
        return q + v * h * p1 + a * h * h * p2, v * math.exp(-x) + a * h * p1

    order = np.argsort(tq)
    qo, vo = np.empty(len(tq)), np.empty(len(tq))
    q, v, now, i = float(q0), float(v0), float(t0), 0
    for idx in order:
        t = tq[idx]
        while edges[i + 1] <= t:
            q, v = step(q, v, levels[i], edges[i + 1] - now)
            now = edges[i + 1]
            i += 1
        qo[idx], vo[idx] = step(q, v, levels[i], t - now)
    return (qo, vo) if return_velocity else qo
