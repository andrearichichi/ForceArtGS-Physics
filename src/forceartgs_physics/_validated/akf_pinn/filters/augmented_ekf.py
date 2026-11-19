from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from collections.abc import Sequence

import torch
from torch import Tensor

from ..config import AKFConfig, AKFSearchConfig
from ..data.interaction import InteractionData
from ..physics.dynamics import (
    physical_parameters,
    state_derivative,
    state_derivative_and_jacobian,
)
from ..physics.integrator import rk4_integrate, rk4_integrate_with_jacobian


@dataclass
class AKFResult:
    times: Tensor
    filtered_state: Tensor
    filtered_covariance: Tensor
    predicted_state: Tensor
    predicted_covariance: Tensor
    transition_jacobians: Tensor
    innovations: Tensor
    innovation_variances: Tensor
    dense_t: Tensor
    dense_q: Tensor
    dense_qdot: Tensor
    dense_qddot: Tensor
    inertia: Tensor
    damping: Tensor
    decay_time: Tensor | None
    runtime_s: float


@dataclass
class BatchedAKFResult:
    """GPU-friendly multi-start result; batch is dimension 1 after time."""

    times: Tensor
    filtered_state: Tensor
    filtered_covariance: Tensor
    predicted_state: Tensor
    predicted_covariance: Tensor
    transition_jacobians: Tensor
    innovations: Tensor
    innovation_variances: Tensor
    inertia: Tensor
    damping: Tensor
    decay_time: Tensor | None
    innovation_scores: Tensor
    runtime_s: float

    @property
    def best_index(self) -> int:
        return int(torch.argmin(self.innovation_scores).item())


@dataclass
class AKFSearchResult:
    result: AKFResult
    selected_initial_inertia: float
    selected_initial_damping: float
    innovation_nll: float
    candidate_count: int
    failed_candidate_count: int
    search_runtime_s: float
    runtime_s: float


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise FloatingPointError(f"non-finite {name}")


def _symmetrize(value: Tensor) -> Tensor:
    return 0.5 * (value + value.mT)


def _transition(data: InteractionData, config: AKFConfig, t0: Tensor, t1: Tensor, state: Tensor) -> Tensor:
    return rk4_integrate(
        lambda z, t: state_derivative(z, t, data.generalized_input, config.damping_parameterization),
        state,
        t0,
        t1,
        config.max_step_s,
    )


def _analytic_transition(
    data: InteractionData, config: AKFConfig, t0: Tensor, t1: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    return rk4_integrate_with_jacobian(
        lambda z, t: state_derivative_and_jacobian(
            z, t, data.generalized_input_with_derivative, config.damping_parameterization
        ),
        state,
        t0,
        t1,
        config.max_step_s,
    )


def _dense_reconstruction(
    data: InteractionData,
    config: AKFConfig,
    filtered: Tensor,
    dense_t: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if bool(torch.any(dense_t[1:] < dense_t[:-1])):
        raise ValueError("dense output times must be sorted")
    if float(dense_t[0]) < float(data.t_obs[0]):
        raise ValueError("dense output may not begin before the first observation")
    raw_parameters = filtered[-1, 2:].detach()
    states: list[Tensor] = []
    accelerations: list[Tensor] = []
    for requested in dense_t:
        left = int(torch.searchsorted(data.t_obs, requested, right=True).clamp(max=len(data.t_obs)).item()) - 1
        left = max(0, min(left, len(data.t_obs) - 1))
        initial = torch.cat((filtered[left, :2], raw_parameters))
        propagated = _transition(data, config, data.t_obs[left], requested, initial)
        derivative = state_derivative(propagated, requested, data.generalized_input, config.damping_parameterization)
        states.append(propagated)
        accelerations.append(derivative[1])
    dense_state = torch.stack(states)
    return dense_state[:, 0], dense_state[:, 1], torch.stack(accelerations)


def run_akf(
    data: InteractionData,
    config: AKFConfig | None = None,
    *,
    dense_t: Tensor | None = None,
) -> AKFResult:
    """Run the continuous-discrete augmented Kalman filter."""
    config = config or AKFConfig()
    if min(config.inertia_guess, config.damping_guess, config.position_measurement_std) <= 0.0:
        raise ValueError("initial physical guesses and measurement std must be positive")
    started = time.perf_counter()
    dtype, device = data.dtype, data.device
    second_guess = config.damping_guess
    if config.damping_parameterization == "decay_time":
        second_guess = config.inertia_guess / config.damping_guess
    state = torch.tensor(
        [float(data.q_obs[0]), config.initial_velocity, math.log(config.inertia_guess), math.log(second_guess)],
        dtype=dtype,
        device=device,
    )
    covariance = torch.diag(torch.tensor(
        [
            config.initial_position_std**2,
            config.initial_velocity_std**2,
            config.initial_parameter_std**2,
            config.initial_parameter_std**2,
        ],
        dtype=dtype,
        device=device,
    ))
    filtered_states, filtered_covariances = [state], [covariance]
    predicted_states, predicted_covariances = [state], [covariance]
    transitions: list[Tensor] = []
    innovations = [torch.zeros((), dtype=dtype, device=device)]
    innovation_variances = [torch.as_tensor(config.position_measurement_std**2, dtype=dtype, device=device)]
    measurement = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
    identity = torch.eye(4, dtype=dtype, device=device)
    measurement_covariance = torch.tensor([[config.position_measurement_std**2]], dtype=dtype, device=device)

    for index in range(1, len(data.t_obs)):
        previous = filtered_states[-1].detach()
        transition = lambda z: _transition(data, config, data.t_obs[index - 1], data.t_obs[index], z)
        if config.jacobian_mode == "analytic":
            predicted_state, jacobian = _analytic_transition(
                data, config, data.t_obs[index - 1], data.t_obs[index], previous
            )
        elif config.jacobian_mode == "autodiff":
            predicted_state = transition(previous)
            jacobian = torch.func.jacrev(transition)(previous)
        else:
            raise ValueError(f"unsupported Jacobian mode: {config.jacobian_mode}")
        dt = data.t_obs[index] - data.t_obs[index - 1]
        process_variances = torch.tensor(
            [
                config.position_process_std**2,
                config.velocity_process_std**2,
                config.parameter_process_std**2,
                config.parameter_process_std**2,
            ],
            dtype=dtype,
            device=device,
        ) * dt
        predicted_covariance = _symmetrize(jacobian @ filtered_covariances[-1] @ jacobian.mT + torch.diag(process_variances))
        innovation = data.q_obs[index] - predicted_state[0]
        innovation_covariance = measurement @ predicted_covariance @ measurement.mT + measurement_covariance
        gain = torch.linalg.solve(innovation_covariance, (predicted_covariance @ measurement.mT).mT).mT
        state = predicted_state + gain[:, 0] * innovation
        correction = identity - gain @ measurement
        covariance = _symmetrize(
            correction @ predicted_covariance @ correction.mT + gain @ measurement_covariance @ gain.mT
        )
        for name, value in (("state", state), ("covariance", covariance), ("Jacobian", jacobian)):
            _finite(name, value)
        predicted_states.append(predicted_state.detach())
        predicted_covariances.append(predicted_covariance.detach())
        filtered_states.append(state.detach())
        filtered_covariances.append(covariance.detach())
        transitions.append(jacobian.detach())
        innovations.append(innovation.detach())
        innovation_variances.append(innovation_covariance.squeeze().detach())

    filtered_state = torch.stack(filtered_states)
    filtered_covariance = torch.stack(filtered_covariances)
    predicted_state = torch.stack(predicted_states)
    predicted_covariance = torch.stack(predicted_covariances)
    transition_jacobians = torch.stack(transitions)
    inertia, damping, decay_time = physical_parameters(
        filtered_state[-1, 2], filtered_state[-1, 3], config.damping_parameterization
    )
    if dense_t is None:
        dense_t = torch.linspace(data.t_obs[0], data.t_obs[-1], 500, device=device, dtype=dtype)
    dense_q, dense_qdot, dense_qddot = _dense_reconstruction(data, config, filtered_state, dense_t)
    return AKFResult(
        times=data.t_obs,
        filtered_state=filtered_state,
        filtered_covariance=filtered_covariance,
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        transition_jacobians=transition_jacobians,
        innovations=torch.stack(innovations),
        innovation_variances=torch.stack(innovation_variances),
        dense_t=dense_t,
        dense_q=dense_q,
        dense_qdot=dense_qdot,
        dense_qddot=dense_qddot,
        inertia=inertia,
        damping=damping,
        decay_time=decay_time,
        runtime_s=time.perf_counter() - started,
    )


def run_akf_batched(
    data: InteractionData,
    inertia_guesses: Sequence[float] | Tensor,
    damping_guesses: Sequence[float] | Tensor,
    config: AKFConfig | None = None,
) -> BatchedAKFResult:
    """Run many AKF hypotheses in one batched CPU/CUDA tensor pipeline.

    Analytical Jacobians avoid one autodiff graph per hypothesis. Use
    ``result.best_index`` to select the lowest innovation-likelihood start.
    """
    config = config or AKFConfig()
    if config.jacobian_mode != "analytic":
        raise ValueError("batched AKF currently requires jacobian_mode='analytic'")
    started = time.perf_counter()
    dtype, device = data.dtype, data.device
    inertia_guess = torch.as_tensor(inertia_guesses, dtype=dtype, device=device).flatten()
    damping_guess = torch.as_tensor(damping_guesses, dtype=dtype, device=device).flatten()
    if inertia_guess.shape != damping_guess.shape or len(inertia_guess) == 0:
        raise ValueError("inertia_guesses and damping_guesses must be equally sized non-empty vectors")
    if not bool(torch.all((inertia_guess > 0) & (damping_guess > 0))):
        raise ValueError("all physical guesses must be positive")
    batch = len(inertia_guess)
    second_guess = damping_guess
    if config.damping_parameterization == "decay_time":
        second_guess = inertia_guess / damping_guess
    state = torch.stack(
        (
            data.q_obs[0].expand(batch),
            torch.full_like(inertia_guess, config.initial_velocity),
            torch.log(inertia_guess),
            torch.log(second_guess),
        ),
        dim=-1,
    )
    base_covariance = torch.diag(torch.tensor(
        [
            config.initial_position_std**2,
            config.initial_velocity_std**2,
            config.initial_parameter_std**2,
            config.initial_parameter_std**2,
        ],
        dtype=dtype,
        device=device,
    ))
    covariance = base_covariance.expand(batch, 4, 4).clone()
    filtered_states, filtered_covariances = [state], [covariance]
    predicted_states, predicted_covariances = [state], [covariance]
    transitions: list[Tensor] = []
    innovations = [torch.zeros(batch, dtype=dtype, device=device)]
    measurement_variance = torch.as_tensor(config.position_measurement_std**2, dtype=dtype, device=device)
    innovation_variances = [measurement_variance.expand(batch)]
    measurement = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).expand(batch, 1, 4)
    identity = torch.eye(4, dtype=dtype, device=device).expand(batch, 4, 4)

    for index in range(1, len(data.t_obs)):
        predicted_state, jacobian = _analytic_transition(
            data, config, data.t_obs[index - 1], data.t_obs[index], filtered_states[-1]
        )
        dt = data.t_obs[index] - data.t_obs[index - 1]
        process_variances = torch.tensor(
            [
                config.position_process_std**2,
                config.velocity_process_std**2,
                config.parameter_process_std**2,
                config.parameter_process_std**2,
            ],
            dtype=dtype,
            device=device,
        ) * dt
        predicted_covariance = _symmetrize(
            jacobian @ filtered_covariances[-1] @ jacobian.mT + torch.diag(process_variances)
        )
        innovation = data.q_obs[index] - predicted_state[:, 0]
        innovation_covariance = (
            measurement @ predicted_covariance @ measurement.mT
            + measurement_variance.reshape(1, 1, 1)
        )
        gain = torch.linalg.solve(
            innovation_covariance, (predicted_covariance @ measurement.mT).mT
        ).mT
        state = predicted_state + gain[..., 0] * innovation[:, None]
        correction = identity - gain @ measurement
        covariance = _symmetrize(
            correction @ predicted_covariance @ correction.mT
            + measurement_variance * (gain @ gain.mT)
        )
        for name, value in (("batched state", state), ("batched covariance", covariance)):
            _finite(name, value)
        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        filtered_states.append(state)
        filtered_covariances.append(covariance)
        transitions.append(jacobian)
        innovations.append(innovation)
        innovation_variances.append(innovation_covariance[..., 0, 0])

    filtered_state = torch.stack(filtered_states)
    filtered_covariance = torch.stack(filtered_covariances)
    predicted_state = torch.stack(predicted_states)
    predicted_covariance = torch.stack(predicted_covariances)
    transition_jacobians = torch.stack(transitions)
    inertia, damping, decay_time = physical_parameters(
        filtered_state[-1, :, 2], filtered_state[-1, :, 3], config.damping_parameterization
    )
    innovation_tensor = torch.stack(innovations)
    variance_tensor = torch.stack(innovation_variances)
    scores = 0.5 * torch.sum(
        torch.log(variance_tensor[1:]) + innovation_tensor[1:] ** 2 / variance_tensor[1:], dim=0
    )
    return BatchedAKFResult(
        times=data.t_obs,
        filtered_state=filtered_state,
        filtered_covariance=filtered_covariance,
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        transition_jacobians=transition_jacobians,
        innovations=innovation_tensor,
        innovation_variances=variance_tensor,
        inertia=inertia,
        damping=damping,
        decay_time=decay_time,
        innovation_scores=scores,
        runtime_s=time.perf_counter() - started,
    )


def run_akf_search(
    data: InteractionData,
    config: AKFConfig | None = None,
    search: AKFSearchConfig | None = None,
    *,
    dense_t: Tensor | None = None,
) -> AKFSearchResult:
    """Select an AKF initialization by innovation NLL, then rerun that filter."""
    config = config or AKFConfig()
    search = search or AKFSearchConfig()
    candidates: list[tuple[float, float, float]] = []
    failed = 0
    search_started = time.perf_counter()

    def evaluate(pairs: list[tuple[float, float]]) -> None:
        nonlocal failed
        try:
            batch = run_akf_batched(
                data,
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                config,
            )
            for index, (inertia, damping) in enumerate(pairs):
                score = float(batch.innovation_scores[index])
                if not math.isfinite(score):
                    raise FloatingPointError("non-finite innovation score")
                candidates.append((score, inertia, damping))
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            if len(pairs) == 1:
                failed += 1
            else:
                midpoint = len(pairs) // 2
                evaluate(pairs[:midpoint])
                evaluate(pairs[midpoint:])

    evaluate([
        (inertia, damping)
        for inertia in search.inertia_guesses
        for damping in search.damping_guesses
    ])
    search_runtime = time.perf_counter() - search_started
    if not candidates:
        raise RuntimeError("all AKF initialization candidates failed")
    score, inertia_guess, damping_guess = min(candidates)
    selected = replace(config, inertia_guess=inertia_guess, damping_guess=damping_guess)
    result = run_akf(data, selected, dense_t=dense_t)
    return AKFSearchResult(
        result=result,
        selected_initial_inertia=inertia_guess,
        selected_initial_damping=damping_guess,
        innovation_nll=score,
        candidate_count=len(candidates) + failed,
        failed_candidate_count=failed,
        search_runtime_s=search_runtime,
        runtime_s=search_runtime + result.runtime_s,
    )
