from __future__ import annotations

import torch
from torch import Tensor, nn


def trajectory_derivatives(
    model: nn.Module, time: Tensor, *, create_graph: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    time_leaf = time.detach().clone().requires_grad_(True)
    q = model(time_leaf)
    qdot = torch.autograd.grad(q, time_leaf, torch.ones_like(q), create_graph=True)[0]
    qddot = torch.autograd.grad(qdot, time_leaf, torch.ones_like(qdot), create_graph=create_graph)[0]
    return q, qdot, qddot
