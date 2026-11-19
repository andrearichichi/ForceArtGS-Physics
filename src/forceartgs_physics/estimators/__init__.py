"""The four validated estimators behind one interface, plus two reference variants.

=============  ===========================================================  ======
name           implementation                                               device
=============  ===========================================================  ======
PINN           PINN-V2 (integral residual, LS start), seeds 0-2, median     GPU*
Backprop       exact-ZOH multi-start Adam + L-BFGS-B                        CPU
MLP            MLP-Joint (label-free amortised), seeds 0-4, median          GPU*
AKF            AKF-improved (LS start, replay-selected R/P0)                CPU
AKF-benchmark  original A3 'AKF' row (native NLL start)                     CPU
IntegralLS     shared closed-form initializer                               CPU
=============  ===========================================================  ======
* runs on CPU too (slower); the reference results were produced on A100 GPUs.
"""
from .akf import AKF, AKFBenchmark
from .backprop import Backprop
from .base import Estimator, EstimatorConfig
from .integral_ls import IntegralLS
from .mlp import MLP
from .pinn import PINN

REGISTRY = {"PINN": PINN, "Backprop": Backprop, "MLP": MLP, "AKF": AKF, "AKF-benchmark": AKFBenchmark,
            "IntegralLS": IntegralLS}
VALIDATED = ("PINN", "Backprop", "AKF", "MLP")


def get(name: str, **kw) -> Estimator:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown estimator {name!r}; available: {sorted(REGISTRY)}") from None
    cfg = EstimatorConfig(seeds=tuple(kw.pop("seeds", cls.default_seeds)), **kw)
    return cls(cfg)


__all__ = ["Estimator", "EstimatorConfig", "PINN", "Backprop", "MLP", "AKF", "AKFBenchmark", "IntegralLS",
           "REGISTRY", "VALIDATED", "get"]
