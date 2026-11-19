from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..config import DampingMode


def positive(raw: Tensor) -> Tensor:
    return torch.exp(torch.clamp(raw, -30.0, 30.0))


def physical_parameters(raw_inertia: Tensor, raw_second: Tensor, mode: DampingMode) -> tuple[Tensor, Tensor, Tensor | None]:
    inertia = positive(raw_inertia)
    second = positive(raw_second)
    if mode == "free_c":
        return inertia, second, None
    if mode == "decay_time":
        return inertia, inertia / second, second
    raise ValueError(f"unsupported damping parameterization: {mode}")


def acceleration(velocity: Tensor, generalized_effort: Tensor, inertia: Tensor, damping: Tensor) -> Tensor:
    return (generalized_effort - damping * velocity) / inertia


def state_derivative(
    state: Tensor,
    time: Tensor,
    generalized_input: Callable[[Tensor, Tensor], Tensor],
    damping_parameterization: DampingMode = "free_c",
) -> Tensor:
    inertia, damping, _ = physical_parameters(state[..., 2], state[..., 3], damping_parameterization)
    effort = generalized_input(time, inertia)
    qddot = acceleration(state[..., 1], effort, inertia, damping)
    zeros = torch.zeros_like(qddot)
    return torch.stack((state[..., 1], qddot, zeros, zeros), dim=-1)


def state_derivative_and_jacobian(
    state: Tensor,
    time: Tensor,
    generalized_input_with_derivative: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    damping_parameterization: DampingMode = "free_c",
) -> tuple[Tensor, Tensor]:
    """Known dynamics and the closed-form continuous Jacobian ``df/dz``."""
    inertia, damping, decay_time = physical_parameters(
        state[..., 2], state[..., 3], damping_parameterization
    )
    effort, d_effort = generalized_input_with_derivative(time, inertia)
    velocity = state[..., 1]
    qddot = acceleration(velocity, effort, inertia, damping)
    zeros = torch.zeros_like(qddot)
    derivative = torch.stack((velocity, qddot, zeros, zeros), dim=-1)
    if damping_parameterization == "free_c":
        d_acceleration_inertia = d_effort / inertia - qddot
        d_acceleration_second = -damping * velocity / inertia
    else:
        assert decay_time is not None
        d_acceleration_inertia = (d_effort - effort) / inertia
        d_acceleration_second = velocity / decay_time
    first_row = torch.stack((zeros, torch.ones_like(zeros), zeros, zeros), dim=-1)
    second_row = torch.stack(
        (zeros, -damping / inertia, d_acceleration_inertia, d_acceleration_second), dim=-1
    )
    zero_row = torch.stack((zeros, zeros, zeros, zeros), dim=-1)
    jacobian = torch.stack((first_row, second_row, zero_row, zero_row), dim=-2)
    return derivative, jacobian


def residual(
    qdot: Tensor,
    qddot: Tensor,
    generalized_effort: Tensor,
    inertia: Tensor,
    damping: Tensor,
) -> Tensor:
    """The one shared equation used by both PINN and filter validation."""
    return inertia * qddot + damping * qdot - generalized_effort
