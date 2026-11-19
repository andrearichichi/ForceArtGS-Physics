from __future__ import annotations

import torch
from torch import Tensor, nn


class TrajectoryMLP(nn.Module):
    """Tanh trajectory with hard q0 and optional hard initial velocity."""

    def __init__(
        self,
        t_min: Tensor,
        t_max: Tensor,
        initial_position: Tensor,
        initial_velocity: Tensor | float | None,
        hidden_layers: int = 4,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        if hidden_layers < 1 or hidden_width < 1:
            raise ValueError("the trajectory network needs non-empty hidden layers")
        self.register_buffer("t_min", t_min.detach().clone())
        self.register_buffer("t_span", (t_max - t_min).detach().clone())
        self.register_buffer("initial_position", initial_position.detach().clone())
        self.hard_velocity = initial_velocity is not None
        self.register_buffer(
            "initial_velocity",
            torch.as_tensor(0.0 if initial_velocity is None else initial_velocity, device=t_min.device, dtype=t_min.dtype).detach().clone(),
        )
        layers: list[nn.Module] = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, time: Tensor) -> Tensor:
        normalized = 2.0 * (time - self.t_min) / self.t_span - 1.0
        unconstrained = self.layers(normalized.reshape(-1, 1)).reshape(time.shape)
        elapsed = time - self.t_min
        if not self.hard_velocity:
            return self.initial_position + elapsed * unconstrained
        return self.initial_position + self.initial_velocity * elapsed + elapsed.square() / self.t_span * unconstrained
