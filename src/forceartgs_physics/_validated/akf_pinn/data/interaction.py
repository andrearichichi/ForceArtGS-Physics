from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch
from torch import Tensor

from ..physics.common import (
    compute_prismatic_generalized_force,
    compute_revolute_generalized_torque,
    generalized_gravity,
)
from ..physics.interpolation import InterpolationMode, interpolate


@dataclass
class InteractionData:
    joint_type: Literal["prismatic", "revolute"]
    t_obs: Tensor
    q_obs: Tensor
    force_t: Tensor
    force_world: Tensor
    application_point_world: Tensor
    axis_world: Tensor
    origin_world: Tensor
    com_world: Tensor | None = None
    gravity_world: Tensor | None = None
    revolute_mass: Tensor | None = None
    interpolation: InterpolationMode = "zoh"
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.joint_type not in {"prismatic", "revolute"}:
            raise ValueError("joint_type must be prismatic or revolute")
        if self.t_obs.ndim != 1 or self.q_obs.shape != self.t_obs.shape:
            raise ValueError("t_obs and q_obs must be matching 1D tensors")
        if self.force_t.ndim != 1 or self.force_world.shape != (len(self.force_t), 3):
            raise ValueError("force_world must have shape [len(force_t), 3]")
        if len(self.t_obs) < 2 or not bool(torch.all(self.t_obs[1:] > self.t_obs[:-1])):
            raise ValueError("at least two strictly increasing observation times are required")
        if not bool(torch.all(self.force_t[1:] > self.force_t[:-1])):
            raise ValueError("force times must be strictly increasing")
        for name in ("application_point_world", "axis_world", "origin_world"):
            value = getattr(self, name)
            if value.shape not in {(3,), (len(self.force_t), 3)}:
                raise ValueError(f"{name} must have shape [3] or [len(force_t), 3]")

    @property
    def device(self) -> torch.device:
        return self.t_obs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.t_obs.dtype

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "InteractionData":
        def move(value: Any) -> Any:
            return value.to(device=device, dtype=dtype or value.dtype) if isinstance(value, Tensor) else value

        ground_truth = None if self.ground_truth is None else {
            key: move(value) for key, value in self.ground_truth.items()
        }
        return replace(
            self,
            t_obs=move(self.t_obs),
            q_obs=move(self.q_obs),
            force_t=move(self.force_t),
            force_world=move(self.force_world),
            application_point_world=move(self.application_point_world),
            axis_world=move(self.axis_world),
            origin_world=move(self.origin_world),
            com_world=move(self.com_world),
            gravity_world=move(self.gravity_world),
            revolute_mass=move(self.revolute_mass),
            ground_truth=ground_truth,
        )

    def _at(self, value: Tensor | None, time: Tensor) -> Tensor | None:
        if value is None:
            return None
        if value.ndim == 1:
            return value.expand(time.shape + (3,))
        return interpolate(self.force_t, value, time, self.interpolation)

    def generalized_input_with_derivative(self, time: Tensor, inertia_or_mass: Tensor) -> tuple[Tensor, Tensor]:
        """Return effort and its derivative with respect to log inertia/mass."""
        force = interpolate(self.force_t, self.force_world, time, self.interpolation)
        axis = self._at(self.axis_world, time)
        origin = self._at(self.origin_world, time)
        point = self._at(self.application_point_world, time)
        assert axis is not None and origin is not None and point is not None
        if self.joint_type == "prismatic":
            effort = compute_prismatic_generalized_force(axis, force)
        else:
            effort = compute_revolute_generalized_torque(axis, origin, point, force)
        derivative = torch.zeros_like(effort + inertia_or_mass * 0.0)
        if self.gravity_world is not None and bool(torch.any(self.gravity_world != 0)):
            gravity = self.gravity_world.expand(time.shape + (3,))
            gravity_effort = generalized_gravity(
                self.joint_type,
                axis,
                origin,
                gravity,
                inertia_or_mass=inertia_or_mass,
                com_world=self._at(self.com_world, time),
                mass=self.revolute_mass,
            )
            effort = effort + gravity_effort
            if self.joint_type == "prismatic":
                derivative = gravity_effort
        return effort, derivative

    def generalized_input(self, time: Tensor, inertia_or_mass: Tensor) -> Tensor:
        return self.generalized_input_with_derivative(time, inertia_or_mass)[0]
