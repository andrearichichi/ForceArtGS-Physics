from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
try:  # vendoring patch: Python 3.10 has no tomllib (only used by load_config)
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None
from typing import Any, Literal, TypeVar

import torch


DampingMode = Literal["free_c", "decay_time"]


@dataclass
class AKFConfig:
    max_step_s: float = 1.0 / 240.0
    position_process_std: float = 1e-5
    velocity_process_std: float = 1e-3
    parameter_process_std: float = 1e-4
    position_measurement_std: float = 1e-4
    initial_velocity: float = 0.0
    initial_position_std: float = 1e-2
    initial_velocity_std: float = 1.0
    initial_parameter_std: float = 2.0
    inertia_guess: float = 1.0
    damping_guess: float = 0.1
    damping_parameterization: DampingMode = "free_c"
    jacobian_mode: Literal["analytic", "autodiff"] = "analytic"


@dataclass
class AKFSearchConfig:
    inertia_guesses: list[float] = field(
        default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )
    damping_guesses: list[float] = field(
        default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )


@dataclass
class PlainPINNConfig:
    seed: int = 0
    hidden_layers: int = 4
    hidden_width: int = 64
    adam_iterations: int = 50_000
    adam_lr: float = 1e-3
    lbfgs_iterations: int = 1_000
    lbfgs_chunk_iterations: int = 100
    lbfgs_max_eval_per_chunk: int = 125
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-15
    convergence_relative_loss: float = 1e-10
    convergence_relative_parameter: float = 1e-5
    convergence_gradient: float = 1e-8
    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_ic: float = 1.0
    inertia_guess: float = 1.0
    damping_guess: float = 0.1
    physics_points: int = 4096


ConfigType = TypeVar("ConfigType", AKFConfig, AKFSearchConfig, PlainPINNConfig)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str = "float64") -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as exc:
        raise ValueError("dtype must be float32 or float64") from exc


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def config_section(document: dict[str, Any], name: str, config_type: type[ConfigType]) -> ConfigType:
    values = document.get(name, {})
    allowed = {item.name for item in fields(config_type)}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown [{name}] fields: {', '.join(sorted(unknown))}")
    return config_type(**values)
