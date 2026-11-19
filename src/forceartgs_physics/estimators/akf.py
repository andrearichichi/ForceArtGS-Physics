"""AKF: augmented-state extended Kalman filter, state [q, q', log I, log c].

Two variants, both on the vendored akf_pinn filter (RK4 transition with analytic Jacobian, Joseph update):

* ``AKF`` (DEFINITIVE, 'AKF-improved' of results/forcesapien_a3_akf): filter started at the shared integral-LS
  estimate; measurement std m_R * sigma and initial log-parameter std P0 selected over
  m_R in {1,2,5,10,20,50,100} x P0 in {ln 100, ln 10} by the open-loop replay error of the final (I, c) on the
  TRAINING interaction only. Code: akf_a3/akf_variants.py (fit_grid) + akf_a3/dev_probe.py (base_cfg, run_variant,
  replay), vendored verbatim; this wrapper reproduces akf_variants.main for 'AKF-improved'.
  Caveat (AKF_A3_PROTOCOL.md): the selection rule was written after B1-B10 had been used in earlier benchmarks, so
  its B result is exploratory; the planned confirmation set C1-C10 was never scored.
* ``AKFBenchmark`` ('AKF' row of A3_RESULTS): native 35-start innovation-NLL search, R = sigma, P0 = ln 100
  (controlled_fairness/cpu.py::akf). Kept for reproducing the original table; it is not the recommended variant.
"""
from __future__ import annotations

import time
from dataclasses import replace

from .._validated import akf_variants as AV
from .._validated import common as H
from .._validated import cpu
from .._validated import dev_probe as D
from . import _bridge
from .base import Estimator


class AKF(Estimator):
    name = "AKF"
    frozen_hyperparameters = dict(start="shared integral-LS (I, c), q0 = first observation", m_R=AV.M_R,
                                  P0={k: float(v) for k, v in AV.P0.items()}, selection="min open-loop replay nRMSE on training data",
                                  process_noise="position 0.1 sigma, velocity sigma, parameters 0",
                                  sigma="robust local-interpolation noise of q (mlp_joint.robust_noise_var), floor 1e-4",
                                  max_step="1/240 s", device="cpu")
    provenance = dict(code="force_estimators/akf_a3/akf_variants.py (AKF-improved) + akf_a3/dev_probe.py + akf_pinn",
                      results="results/forcesapien_a3_akf", protocol="AKF_A3_PROTOCOL.md")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        t0 = time.perf_counter()
        cfg, sigma = D.base_cfg(d, sc)
        ls = init
        QV = (ls["q0"], ls["v0"])
        LS0 = (ls["inertia"], ls["damping"])
        full = [(p, m) for p in AV.P0 for m in AV.M_R]
        best, runs = AV.fit_grid(d, sc, cfg, sigma, LS0, full, QV)
        I, c = best["I"], best["c"]
        c2 = replace(cfg, position_measurement_std=best["m_R"] * sigma, initial_parameter_std=AV.P0[best["P0"]],
                     inertia_guess=LS0[0], damping_guess=LS0[1])
        r = D.run_akf(d, c2, dense_t=d.t_obs[:1])
        q_hat = (sc["q0"] + sc["Q"] * r.filtered_state[:, 0].numpy()).tolist()
        q_rep = (sc["q0"] + sc["Q"] * D.replay(d, I / sc["inertia_to_physical"], c / sc["damping_to_physical"], *QV)).tolist()
        return dict(status="ok", inertia=I, damping=c, q_hat=q_hat, runtime_s=time.perf_counter() - t0,
                    convergence=f"finite filter pass; selected P0={best['P0']} m_R={best['m_R']} by replayA "
                                f"{best['replay_A_nrmse_pct']:.4f}% (NIS {best['nis_mean']:.2f})",
                    diagnostics=dict(selected=dict(P0=best["P0"], m_R=best["m_R"]), sigma=sigma, q_replay_train=q_rep,
                                     grid=[{k: g.get(k) for k in ("P0", "m_R", "ok", "I", "c", "replay_A_nrmse_pct", "nis_mean")}
                                           for g in runs]))


class AKFBenchmark(Estimator):
    name = "AKF-benchmark"
    frozen_hyperparameters = dict(starts="7x5 grid on SUPPORT, innovation NLL", R="sigma", P0="ln 100", device="cpu")
    provenance = dict(code="force_estimators/controlled_fairness/cpu.py::akf", variant="primary",
                      note="the AKF row of results/forcesapien_a3/A3_RESULTS.md (superseded by AKF-improved)")

    def _fit_seed(self, inter, w, d, sc, init, seed):
        r = H.restore(cpu.akf(d, sc, "native", 1.0, False), sc)
        r["status"] = "ok" if _bridge.finite_ok(r, w.q_obs, sc["Q"]) else "numerical_failure"
        return r
