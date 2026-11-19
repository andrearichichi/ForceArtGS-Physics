#!/usr/bin/env python3
"""Reproduce the ForceSAPIEN A3 -> B1-B10 table.

    python examples/02_forcesapien_benchmark.py                      # score the ORIGINAL fits (CPU, ~1 min)
    python examples/02_forcesapien_benchmark.py --refit Backprop AKF # also re-fit (CPU, ~1-2 min per object each)
    python examples/02_forcesapien_benchmark.py --refit PINN MLP --device cuda   # GPU, ~25 min per object and seed

Requires an authorized external bundle via FORCEARTGS_PHYSICS_BENCHMARK.
No dataset is distributed here.

Protocol: identify on A3 (external inputs), freeze I and c, predict B1-B10 open loop with tau_B = |kappa_A3| |F_B|
from rest, score nRMSE against the simulator truth (evaluation only). See docs/BENCHMARK.md.
"""
import argparse
import json

from forceartgs_physics import api
from forceartgs_physics.adapters.forcesapien import A3Benchmark
from forceartgs_physics.validation import forcesapien_a3 as P

ap = argparse.ArgumentParser()
ap.add_argument("--refit", nargs="*", default=[])
ap.add_argument("--objects", nargs="*", default=A3Benchmark.objects)
ap.add_argument("--device")
ap.add_argument("--out", default="outputs/a3_table.json")
a = ap.parse_args()
try:
    bench = A3Benchmark()
except FileNotFoundError as exc:
    ap.error(str(exc))
est = {s: v for s, v in P.reference_estimates(bench).items() if s in a.objects}
for m in a.refit:
    for s in a.objects:
        r = api.identify(bench.interaction_A3(s), m, diagnose_identifiability=False, encoding="cartesian",
                         **({"device": a.device} if a.device else {}))
        est[s][f"{m} (refit)"] = [dict(seed=x["seed"], inertia=x.get("inertia"), damping=x.get("damping"),
                                       status=x.get("status")) for x in r.per_seed]
        print(s, m, r.status, r.parameters and (r.parameters.inertia, r.parameters.damping), flush=True)
res = P.evaluate(est, bench, a.objects)
import pathlib
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(f"\n{'method':22s} {'|I err|':>8s} {'|c err|':>8s} {'B nRMSE':>8s}  PASS   (object medians, %)")
for m, v in sorted(res["per_method"].items(), key=lambda kv: kv[1]["B_object_median"]):
    print(f"{m:22s} {v['I_err_median']:8.3f} {v['c_err_median']:8.3f} {v['B_object_median']:8.3f}  {v['B_pass']}/{v['B_total']}")
