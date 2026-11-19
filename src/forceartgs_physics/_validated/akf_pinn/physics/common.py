from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


JointType = Literal["prismatic", "revolute"]


def unit(vector: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if bool(torch.any(norm <= torch.finfo(vector.dtype).eps)):
        raise ValueError("joint axis must be non-zero")
    return vector / norm


def compute_prismatic_generalized_force(axis_world: Tensor, force_world: Tensor) -> Tensor:
    return torch.sum(unit(axis_world) * force_world, dim=-1)


def compute_revolute_generalized_torque(
    axis_world: Tensor,
    origin_world: Tensor,
    application_point_world: Tensor,
    force_world: Tensor,
) -> Tensor:
    moment = torch.linalg.cross(application_point_world - origin_world, force_world, dim=-1)
    return torch.sum(unit(axis_world) * moment, dim=-1)


def generalized_gravity(
    joint_type: JointType,
    axis_world: Tensor,
    origin_world: Tensor,
    gravity_world: Tensor,
    *,
    inertia_or_mass: Tensor,
    com_world: Tensor | None = None,
    mass: Tensor | None = None,
) -> Tensor:
    if joint_type == "prismatic":
        return inertia_or_mass * torch.sum(unit(axis_world) * gravity_world, dim=-1)
    if com_world is None or mass is None:
        raise ValueError("revolute gravity requires com_world and a physical mass")
    gravity_force = mass * gravity_world
    moment = torch.linalg.cross(com_world - origin_world, gravity_force, dim=-1)
    return torch.sum(unit(axis_world) * moment, dim=-1)

