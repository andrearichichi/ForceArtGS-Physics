"""The ForceSAPIEN A3 -> B1-B10 protocol, statement for statement as force_estimators/a3/post_a3.py.

1. Identify (I, c) on A3 (estimator input only).
2. Freeze. For each B interaction: kappa = LS ratio of the A3 generalized effort (FreeArtGS geometry) to |F_A3| over
   the active samples; tau_B = |kappa| |F_B| (forward ZOH on the recorded times, first level held back to t = 0);
   exact solution from rest at t = 0 (oneplus10/replay.py ``propagate``). No B kinematics, no GT.
3. Score (EVALUATION ONLY): nRMSE of the prediction against push-sign * q_rel of the B truth, normalised by its range;
   A3: I/c relative errors, q_hat nRMSE against sign(axis_est . axis_gt) * (q_gt(t) - q_gt(t0)), normalised by the GT
   range over the observation window.
4. Aggregate: per B interaction the median over seeds; per object the median over B1-B10; across objects the median
   of per-object values; |I err|, |c err|: median over objects of |median over seeds|. PASS: B nRMSE <= 10 %.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .._validated import replay as _rp
from ..adapters.forcesapien import B_INTERACTIONS, OBJECTS, A3Benchmark
from .metrics import GROSS_FAILURE_PCT

med = lambda v: float(np.median(v)) if len(v) else float("nan")


def kappa_from_input(meta_effort, forces) -> float:
    e = np.asarray(meta_effort, float)
    act = np.asarray(forces.contact_valid, bool)
    Fm = np.linalg.norm(forces.force_N, axis=1)
    return float(e[act] @ Fm[act] / (Fm[act] @ Fm[act]))


def predict_B(I, c, kappa, forces_B) -> np.ndarray:
    kn, lev = _rp.zoh_levels(forces_B.t_s, abs(kappa) * np.linalg.norm(forces_B.force_N, axis=1))
    return _rp.propagate(float(I), float(c), kn, lev, forces_B.t_s)


def score_B(q_pred, truth_B) -> float:
    eff = np.asarray(truth_B["effort_gt_joint"])
    push = math.copysign(1.0, eff[np.flatnonzero(np.abs(eff) > 0)[0]])
    refb = push * np.asarray(truth_B["q_rel"])
    return 100 * float(np.sqrt(np.mean((np.asarray(q_pred) - refb) ** 2))) / float(np.ptp(refb))


def score_A3(inter, I, c, q_hat, truth_A):
    t = inter.trajectory.t_s
    tt, gq = np.asarray(truth_A["t"]), np.asarray(truth_A["q_rel"])
    ax = np.asarray(truth_A["axis_world"]) / np.linalg.norm(truth_A["axis_world"])
    ref = np.sign(inter.joint.axis @ ax) * (np.interp(t, tt, gq) - np.interp(t[0], tt, gq))
    mask = (tt >= t[0] - 1e-12) & (tt <= t[-1] + 1e-12)
    norm = float(np.ptp(gq[mask]))
    out = dict(I_err_pct=100 * (I / truth_A["inertia"] - 1), c_err_pct=100 * (c / truth_A["damping"] - 1))
    if q_hat is not None:
        out["q_nrmse_pct"] = 100 * float(np.sqrt(np.mean((np.asarray(q_hat) - ref) ** 2))) / norm
    return out


def evaluate(estimates: dict, bench: A3Benchmark | None = None, objects=OBJECTS, with_predictions=False) -> dict:
    """estimates[slug][method] = list of dict(seed, inertia, damping, q_hat|None, status). Returns per-object
    summaries and the across-object table (the A3_RESULTS.json structure)."""
    bench = bench or A3Benchmark()
    per_obj, preds = {}, {}
    for slug in objects:
        inter = bench.interaction_A3(slug)
        kappa = kappa_from_input(inter.metadata["effort_on_force_grid"], inter.forces)
        trA = bench.truth(slug, "A3")
        per_obj[slug], preds[slug] = {}, {}
        for m, fits in estimates[slug].items():
            ok = [f for f in fits if f.get("status", "ok") == "ok" and f["inertia"] and f["damping"]
                  and np.isfinite([f["inertia"], f["damping"]]).all() and min(f["inertia"], f["damping"]) > 0]
            A = [score_A3(inter, f["inertia"], f["damping"], f.get("q_hat"), trA) for f in ok]
            perB = {}
            for b in B_INTERACTIONS:
                FB, trB = bench.forces_B(slug, b), bench.truth(slug, b)
                vals = []
                for f in fits:
                    if f in ok:
                        qp = predict_B(f["inertia"], f["damping"], kappa, FB)
                        vals.append(score_B(qp, trB))
                        if with_predictions:
                            preds[slug].setdefault(b, {})[f"{m}/s{f.get('seed', 0)}"] = qp
                    else:
                        vals.append(float("inf"))
                perB[b] = med(vals)
            v = np.array(list(perB.values()), float)
            per_obj[slug][m] = dict(I_err_pct=med([a["I_err_pct"] for a in A]), c_err_pct=med([a["c_err_pct"] for a in A]),
                                    qA3_nrmse_pct=med([a["q_nrmse_pct"] for a in A if "q_nrmse_pct" in a]),
                                    B=perB, B_median=float(np.median(v)), B_pass=int((v <= GROSS_FAILURE_PCT).sum()),
                                    kappa=kappa, n_fits_ok=len(ok))
    methods = sorted({m for s in per_obj.values() for m in s})
    table = {}
    for m in methods:
        rows = [per_obj[s][m] for s in objects if m in per_obj[s]]
        table[m] = dict(I_err_median=med([abs(r["I_err_pct"]) for r in rows]), c_err_median=med([abs(r["c_err_pct"]) for r in rows]),
                        B_object_median=med([r["B_median"] for r in rows]), B_pass=sum(r["B_pass"] for r in rows),
                        B_total=10 * len(rows))
    out = dict(per_object=per_obj, per_method=table)
    if with_predictions:
        out["predictions"] = preds
    return out


def reference_estimates(bench: A3Benchmark | None = None) -> dict:
    """The ORIGINAL fits (benchmark/forcesapien_a3/reference) in the format of ``evaluate``: PINN (PINN-V2 s0-2),
    Backprop (s0), MLP (MLP-Joint s0-4), AKF (AKF-improved s0), AKF-benchmark (A3 'AKF' s0)."""
    bench = bench or A3Benchmark()
    fits, akf = bench.reference("a3_fits.json.gz"), bench.reference("akf_improved_fits.json.gz")
    names = {"PINN-V2": "PINN", "Backprop": "Backprop", "MLP-Joint": "MLP", "AKF": "AKF-benchmark"}
    est = defaultdict(lambda: defaultdict(list))
    for slug, rows in fits.items():
        for key, r in rows.items():
            m, s = key.split("/s")
            est[slug][names[m]].append(dict(seed=int(s), inertia=r["inertia"], damping=r["damping"], q_hat=r["q_hat"],
                                            status=r["status"]))
        a = akf[slug]
        est[slug]["AKF"].append(dict(seed=0, inertia=a["inertia"], damping=a["damping"], q_hat=a["q_hat"], status=a["status"]))
    return {k: dict(v) for k, v in est.items()}
