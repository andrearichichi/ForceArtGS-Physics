"""High-level API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import estimators as _est
from .dynamics.viscous import propagate_from_rest, propagate_from_state
from .forces.projection import GeneralizedEffort, generalized_effort, lever_factor_effort
from .schemas import IdentificationResult, PhysicalParameters, Prediction
from .validation import identifiability as _idf
from .validation.metrics import nrmse_pct


def identify(interaction, method: str = "Backprop", *, diagnose_identifiability: bool = True, **config) -> IdentificationResult:
    """Identify (I, c) of ``interaction`` with one estimator ('PINN', 'Backprop', 'MLP', 'AKF', 'AKF-benchmark',
    'IntegralLS'). config: seeds, device, encoding, threads, workdir. The identifiability diagnostic is attached
    to the result and to ``parameters.identifiable`` / ``parameters.uncertainty``; it never alters the estimate."""
    res = _est.get(method, **config).fit(interaction)
    if diagnose_identifiability:
        rep = _idf.diagnose(interaction, config.get("encoding", "auto"))
        res.identifiability = rep
        if res.parameters is not None:
            res.parameters.identifiable = rep.get("identifiable")
            res.parameters.uncertainty = rep.get("uncertainty")
    return res


def identify_many(interactions: Sequence, method: str = "Backprop", **config) -> IdentificationResult:
    """Several interactions of the same joint (see Estimator.fit_many: per-interaction fits, median aggregation)."""
    return _est.get(method, **config).fit_many(list(interactions))


def compare(interaction, methods: Iterable[str] = _est.VALIDATED, **config) -> dict:
    """Run several estimators on the same interaction. Returns {method: IdentificationResult}."""
    out = {}
    rep = _idf.diagnose(interaction, config.get("encoding", "auto"))
    for m in methods:
        out[m] = identify(interaction, m, diagnose_identifiability=False, **config)
        out[m].identifiability = rep
        if out[m].parameters is not None:
            out[m].parameters.identifiable = rep.get("identifiable")
            out[m].parameters.uncertainty = rep.get("uncertainty")
    return out


def predict(parameters: PhysicalParameters, forces=None, joint=None, t_query=None, *, effort: GeneralizedEffort | None = None,
            initial_state: dict | None = None) -> Prediction:
    """Open-loop prediction with frozen parameters.

    The effort is the signed projection of ``forces`` on ``joint`` (both in the same frame), or a given
    GeneralizedEffort. Without ``initial_state`` the joint starts at rest at t = 0 and q is the displacement (the
    benchmark convention, validated solver). With initial_state = {t_s, q, qdot} the exact solution starts there."""
    if effort is None:
        if forces is None or joint is None:
            raise ValueError("give either effort or forces + joint")
        if joint.joint_type != parameters.joint_type:
            raise ValueError("joint type differs from the identified parameters")
        effort = generalized_effort(joint, forces)
    tq = np.asarray(forces.t_s if t_query is None and forces is not None else t_query, float)
    if initial_state is None:
        q = propagate_from_rest(parameters.inertia, parameters.damping, effort, tq)
        ini = dict(t_s=0.0, q=0.0, qdot=0.0, convention="rest at t = 0, displacement")
    else:
        q = propagate_from_state(parameters.inertia, parameters.damping, effort, tq, initial_state["t_s"],
                                 initial_state["q"], initial_state["qdot"])
        ini = dict(initial_state)
    return Prediction(t_s=tq, q=q, parameters=parameters, effort_model=effort.model, initial_state=ini,
                      effort_t_s=effort.t_s, effort=effort.levels)


def predict_lever_factor(parameters: PhysicalParameters, kappa: float, forces, t_query=None) -> Prediction:
    """ForceSAPIEN benchmark force model tau = |kappa| |F| (sign-blind; only for reproducing the A3 benchmark)."""
    eff = lever_factor_effort(kappa, forces, parameters.joint_type)
    return predict(parameters, effort=eff, t_query=forces.t_s if t_query is None else t_query)


def evaluate_prediction(pred: Prediction, observed_t, observed_q, align_start: bool = True) -> dict:
    """nRMSE (range-normalised) of a prediction against an observed q(t) of the new interaction. With
    align_start the observation is expressed as displacement from its first sample (the prediction starts at rest)."""
    qo = np.asarray(observed_q, float)
    qp = np.interp(np.asarray(observed_t, float), pred.t_s, pred.q)
    if align_start:
        qo = qo - qo[0]
        qp = qp - qp[0]
    return dict(nrmse_pct=nrmse_pct(qp, qo), rmse=float(np.sqrt(np.mean((qp - qo) ** 2))), n=len(qo))


def save_parameters(result_or_params, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_or_params.require_exportable()
    if isinstance(result_or_params, IdentificationResult):
        return result_or_params.save(path)
    path.write_text(json.dumps(result_or_params.to_dict(), indent=1))
    return path


def load_parameters(path) -> PhysicalParameters:
    d = json.loads(Path(path).read_text())
    if "parameters" in d and "estimator" in d:
        result = IdentificationResult.load(path)
        result.require_exportable()
        return result.parameters
    params = PhysicalParameters.from_dict(d)
    params.require_exportable()
    return params
