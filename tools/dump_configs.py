#!/usr/bin/env python3
"""Write configs/*.json from the code (the frozen hyperparameters live in the vendored code; this documents them)."""
import json
from pathlib import Path

from forceartgs_physics import estimators as E
from forceartgs_physics.validation import forcesapien_a3 as P
from forceartgs_physics.validation.identifiability import CRLB_LIMIT_PCT

out = Path(__file__).resolve().parents[1] / "configs"
out.mkdir(exist_ok=True)
doc = {name: dict(default_seeds=list(cls.default_seeds), frozen_hyperparameters=cls.frozen_hyperparameters,
                  provenance=cls.provenance) for name, cls in E.REGISTRY.items()}
(out / "estimators_frozen.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
(out / "forcesapien_a3_protocol.json").write_text(json.dumps(dict(
    objects=P.OBJECTS, train="A3", test=P.B_INTERACTIONS, freeze="I, c from A3",
    force_model_B="tau_B = |kappa_A3| |F_B|, forward ZOH, first level held back to t=0, rest at t=0",
    kappa="LS ratio of A3 generalized effort (FreeArtGS geometry) to |F_A3| over active samples",
    score="nRMSE(q_pred, push_sign * q_rel_B) / range, evaluation only",
    aggregation="median over seeds per B; median over B1-B10 per object; median over objects",
    pass_threshold_pct=10.0, identifiability_crlb_limit_pct=CRLB_LIMIT_PCT,
    estimator_names={"PINN": "PINN-V2", "Backprop": "Backprop", "MLP": "MLP-Joint", "AKF": "AKF-improved",
                     "AKF-benchmark": "AKF (original A3 row)"}), indent=1) + "\n")
print("wrote", out)
