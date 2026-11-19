from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor


def rk4_integrate(
    derivative: Callable[[Tensor, Tensor], Tensor],
    state: Tensor,
    t0: Tensor | float,
    t1: Tensor | float,
    max_step_s: float,
) -> Tensor:
    """Differentiable RK4 propagation over one arbitrary time interval."""
    start = torch.as_tensor(t0, dtype=state.dtype, device=state.device)
    end = torch.as_tensor(t1, dtype=state.dtype, device=state.device)
    duration = float((end - start).detach().cpu())
    if duration < 0.0:
        raise ValueError("t1 must not precede t0")
    if duration == 0.0:
        return state
    steps = max(1, math.ceil(duration / max_step_s))
    h = (end - start) / steps
    current, time = state, start
    for _ in range(steps):
        k1 = derivative(current, time)
        k2 = derivative(current + 0.5 * h * k1, time + 0.5 * h)
        k3 = derivative(current + 0.5 * h * k2, time + 0.5 * h)
        k4 = derivative(current + h * k3, time + h)
        current = current + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        time = time + h
    return current


def rk4_integrate_with_jacobian(
    derivative_and_jacobian: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    state: Tensor,
    t0: Tensor | float,
    t1: Tensor | float,
    max_step_s: float,
) -> tuple[Tensor, Tensor]:
    """RK4 propagation of state and analytical variational transition.

    ``state`` may be ``[D]`` or ``[..., D]``. The resulting transition matrix
    is ``[..., D, D]`` and remains on the state's CPU/CUDA device.
    """
    start = torch.as_tensor(t0, dtype=state.dtype, device=state.device)
    end = torch.as_tensor(t1, dtype=state.dtype, device=state.device)
    duration = float((end - start).detach().cpu())
    if duration < 0.0:
        raise ValueError("t1 must not precede t0")
    dimension = state.shape[-1]
    transition = torch.eye(dimension, dtype=state.dtype, device=state.device).expand(
        state.shape[:-1] + (dimension, dimension)
    ).clone()
    if duration == 0.0:
        return state, transition
    steps = max(1, math.ceil(duration / max_step_s))
    h = (end - start) / steps
    current, time = state, start
    for _ in range(steps):
        k1_state, a1 = derivative_and_jacobian(current, time)
        k1_transition = a1 @ transition

        state2 = current + 0.5 * h * k1_state
        transition2 = transition + 0.5 * h * k1_transition
        k2_state, a2 = derivative_and_jacobian(state2, time + 0.5 * h)
        k2_transition = a2 @ transition2

        state3 = current + 0.5 * h * k2_state
        transition3 = transition + 0.5 * h * k2_transition
        k3_state, a3 = derivative_and_jacobian(state3, time + 0.5 * h)
        k3_transition = a3 @ transition3

        state4 = current + h * k3_state
        transition4 = transition + h * k3_transition
        k4_state, a4 = derivative_and_jacobian(state4, time + h)
        k4_transition = a4 @ transition4

        current = current + h * (k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state) / 6.0
        transition = transition + h * (
            k1_transition + 2.0 * k2_transition + 2.0 * k3_transition + k4_transition
        ) / 6.0
        time = time + h
    return current, transition
