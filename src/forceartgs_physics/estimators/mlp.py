"""MLP (MLP-Joint): amortised, label-free joint estimation of q(t), I and c.

Per interaction, an MLP is trained on 20000 synthetic trajectories of the SAME model and the SAME measured effort,
with parameters drawn from the generic SUPPORT prior (no ground truth, no external pre-trained weights): trunk
N-256-256-128 SiLU, parameter head (log decay time, nondimensional log I), J-invariant motion head; loss
L_par + L_obs + 0.3 L_phys; AdamW 1e-3, cosine schedule, 200 epochs, early stopping on synthetic validation
(patience 40). Seeds 0-4 (median). Code: controlled_fairness/mlp.py::train + mlp_joint/mlp_joint.py, vendored
verbatim. The 'pre-training' IS this per-interaction synthetic training; checkpoints are written to ``workdir``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .._validated import common as H
from .._validated import mlp as _mlp
from .._validated import mlp_joint as MJ
from . import _bridge
from .base import Estimator


class MLP(Estimator):
    name = "MLP"
    default_seeds = (0, 1, 2, 3, 4)
    frozen_hyperparameters = dict(variant="primary", n_synth=20000, epochs=200, noise="mixture", lambda_phys=0.3,
                                  prior=H.prior(H.SUPPORT), **{k: v for k, v in MJ.Config().__dict__.items()
                                                               if k not in ("n_synth", "epochs", "noise", "lambda_phys")})
    provenance = dict(code="force_estimators/controlled_fairness/mlp.py::train + mlp_joint/mlp_joint.py",
                      variant="primary", driver="controlled_fairness/run.py::execute via robust_v2/run_heldout.py")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        dev = _bridge.default_device(self.config.device)
        wd = Path(self.config.workdir) if self.config.workdir else Path(tempfile.mkdtemp(prefix="fagp_mlp_"))
        dst = wd / inter.name / f"MLP-Joint_primary_s{seed}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        r, _snap = _mlp.train(d, seed, "primary", dev, dst, False)
        r = H.restore(r, sc)
        r["status"] = "ok" if _bridge.finite_ok(r, w.q_obs, sc["Q"]) else "numerical_failure"
        r["device"] = dev
        r["diagnostics"].pop("history", None)
        r["checkpoint"] = str(dst.with_name(dst.stem + "_final.pt"))
        return r
