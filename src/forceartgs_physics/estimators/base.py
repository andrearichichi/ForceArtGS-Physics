"""Common estimator interface."""
from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import __version__
from ..schemas import IdentificationResult, NotIdentifiableError, PhysicalParameters
from . import _bridge


@dataclass
class EstimatorConfig:
    """Options shared by all estimators. Algorithmic hyperparameters of the validated methods are FROZEN inside
    the vendored code and are reported (not settable) in ``frozen_hyperparameters``."""

    seeds: Sequence[int] = (0,)
    device: Optional[str] = None             # None/'auto' = cuda if available (PINN, MLP); Backprop/AKF always CPU
    encoding: str = "auto"                   # see estimators._bridge.to_window
    threads: Optional[int] = 1               # 1 = as in the validated runs (bitwise reproducibility on CPU)
    workdir: Optional[str] = None            # MLP checkpoints / progress files (temporary dir if None)
    extra: dict = field(default_factory=dict)


class Estimator(abc.ABC):
    name: str = ""
    frozen_hyperparameters: dict = {}
    provenance: dict = {}
    supports_multi_interaction = False
    default_seeds: Sequence[int] = (0,)

    def __init__(self, config: EstimatorConfig | None = None, **kw):
        self.config = config or EstimatorConfig(seeds=tuple(self.default_seeds))
        for k, v in kw.items():
            setattr(self.config, k, v)

    # ------------------------------------------------------------------ to implement
    @abc.abstractmethod
    def _fit_seed(self, inter, w, d, sc, init, seed: int) -> dict:
        """Return dict(inertia, damping [physical units], q_hat [list, on the observation times], convergence,
        diagnostics, runtime_s, status)."""

    # ------------------------------------------------------------------ public
    def fit(self, interaction) -> IdentificationResult:
        """Identify (I, c) from one interaction. Multi-seed methods report every seed and use the MEDIAN over seeds
        of I and of c (separately), the aggregation of the A3 benchmark (post_a3.score)."""
        interaction.require_identifiable()
        t0 = time.perf_counter()
        with _bridge.single_thread(self.config.threads):
            w, d, sc, init = _bridge.prepare(interaction, self.config.encoding)
            per_seed = []
            for s in self.config.seeds:
                try:
                    r = self._fit_seed(interaction, w, d, sc, init, int(s))
                except Exception as exc:  # noqa: BLE001  a failure is a result
                    r = dict(status="failed", error=f"{type(exc).__name__}: {exc}", inertia=None, damping=None, q_hat=None)
                r["seed"] = int(s)
                per_seed.append(r)
        ok = [r for r in per_seed if r.get("status") == "ok"]
        jt = interaction.joint_type
        if ok:
            I = float(np.median([r["inertia"] for r in ok]))
            c = float(np.median([r["damping"] for r in ok]))
            params = PhysicalParameters(jt, I, c)
            best = min(ok, key=lambda r: abs(np.log(r["inertia"] / I)) + abs(np.log(r["damping"] / c)))
            qh = np.asarray(best["q_hat"], float)
            qo = interaction.trajectory.q
            rng = float(np.ptp(qo)) or 1.0
            metrics = dict(q_fit_rmse=float(np.sqrt(np.mean((qh - qo) ** 2))),
                           q_fit_nrmse_pct=100 * float(np.sqrt(np.mean((qh - qo) ** 2))) / rng,
                           normalisation="range of the observed q", units=interaction.units["q"])
            q_fit = dict(t_s=interaction.trajectory.t_s.tolist(), q=qh.tolist(), seed=best["seed"])
            status = "ok"
        else:
            params, metrics, q_fit = None, {}, None
            status = "numerical_failure" if any(r.get("status") == "numerical_failure" for r in per_seed) else "failed"
        return IdentificationResult(
            estimator=self.name, status=status, parameters=params, interactions=[interaction.name],
            config=dict(asdict(self.config), frozen_hyperparameters=self.frozen_hyperparameters), q_fit=q_fit,
            fit_metrics=metrics, per_seed=[{k: v for k, v in r.items() if k != "q_hat"} for r in per_seed],
            diagnostics=dict(scales=sc, shared_ls=init, encoding=w.meta.get("encoding"),
                             aggregation="median over seeds of I and c" if len(per_seed) > 1 else "single seed"),
            runtime_s=time.perf_counter() - t0,
            provenance=dict(library_version=__version__, implementation=self.provenance, environment=_bridge.env_note(),
                            input_fingerprint=interaction.fingerprint()))

    def fit_many(self, interactions) -> IdentificationResult:
        """Several interactions of the same joint. The validated estimators are single-interaction methods; they
        are NOT changed into a joint multi-interaction fit. This fits each interaction separately and aggregates
        log I and log c by the median, reporting the spread. Use estimators.integral_ls.IntegralLS.fit_many for a
        joint (stacked) closed-form fit."""
        res = [self.fit(x) for x in interactions]
        ok = [r for r in res if r.status == "ok"]
        if not ok:
            raise NotIdentifiableError("no interaction could be identified")
        jt = ok[0].parameters.joint_type
        if any(r.parameters.joint_type != jt for r in ok):
            raise ValueError("interactions of different joint types")
        lI = np.log([r.parameters.inertia for r in ok]); lc = np.log([r.parameters.damping for r in ok])
        params = PhysicalParameters(jt, float(np.exp(np.median(lI))), float(np.exp(np.median(lc))),
                                    extra=dict(aggregation="median of per-interaction estimates",
                                               log_spread=dict(inertia=float(np.ptp(lI)), damping=float(np.ptp(lc)))))
        return IdentificationResult(estimator=self.name, status="ok", parameters=params,
                                    interactions=[x.name for x in interactions], config=res[0].config,
                                    diagnostics=dict(per_interaction=[r.to_dict() for r in res]),
                                    runtime_s=sum(r.runtime_s or 0 for r in res), provenance=res[0].provenance)
