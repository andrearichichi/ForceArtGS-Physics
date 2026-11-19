from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


InterpolationMode = Literal["zoh", "linear"]


def interpolate(times: Tensor, values: Tensor, query: Tensor, mode: InterpolationMode = "zoh") -> Tensor:
    """Interpolate a time series on CPU or CUDA without smoothing jumps."""
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty 1D tensor")
    if values.shape[0] != len(times):
        raise ValueError("values and times must have the same leading dimension")
    if len(times) > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    original_shape = query.shape
    flat = query.reshape(-1)
    right = torch.searchsorted(times, flat, right=True)
    left = (right - 1).clamp(0, len(times) - 1)
    if mode == "zoh" or len(times) == 1:
        result = values[left]
    elif mode == "linear":
        upper = right.clamp(0, len(times) - 1)
        t0, t1 = times[left], times[upper]
        weight = torch.where(t1 > t0, (flat - t0) / (t1 - t0), torch.zeros_like(flat))
        weight = weight.clamp(0.0, 1.0)
        weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
        result = values[left] + weight * (values[upper] - values[left])
    else:
        raise ValueError(f"unsupported interpolation mode: {mode}")
    return result.reshape(original_shape + values.shape[1:])

