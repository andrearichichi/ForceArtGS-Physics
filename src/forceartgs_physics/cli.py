"""Command line.

  forceartgs-physics identify  --estimator-input X.json --method Backprop --out result.json
  forceartgs-physics identify  --forceartgs-run RUN --forces-json forces.json --frame base_link --method AKF --out result.json
  forceartgs-physics diagnose  --estimator-input X.json
  forceartgs-physics benchmark-a3 [--refit Backprop AKF ...] [--objects ...] --out table.json
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api
from .adapters import forceartgs as FA
from .adapters import forcesapien as FS


def _interaction(a):
    if a.estimator_input:
        return FS.load_estimator_input(a.estimator_input)
    if a.forceartgs_run:
        if not a.forces_json or not a.frame:
            raise SystemExit("measured runs require --forces-json and --frame; see docs/QUICKSTART_SAMUELE.md")
        from .adapters.real import load_observed_forces_json
        joint, traj = (FA.load_authority(a.forceartgs_run, frame=a.frame) if a.level == "A"
                       else FA.load_discovery_run(a.forceartgs_run, frame=a.frame))
        forces = load_observed_forces_json(a.forces_json)
        return FA.interaction(a.name or "measured_interaction", joint, traj, forces)
    raise SystemExit("give --estimator-input or --forceartgs-run + --forces-json + --frame")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forceartgs-physics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("identify", "diagnose"):
        s = sub.add_parser(name)
        inputs = s.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--estimator-input")
        inputs.add_argument("--forceartgs-run")
        s.add_argument("--level", choices=["A", "B"], default="B")
        s.add_argument("--forces-json", help="calibrated observed forces (docs/QUICKSTART_SAMUELE.md)")
        s.add_argument("--frame", help="verified common metric world frame")
        s.add_argument("--name")
        if name == "identify":
            s.add_argument("--method", default="Backprop")
            s.add_argument("--device")
            s.add_argument("--seeds", type=int, nargs="+")
            s.add_argument("--out", required=True)
    b = sub.add_parser("benchmark-a3")
    b.add_argument("--refit", nargs="*", default=[], help="estimators to re-run (default: score the reference fits)")
    b.add_argument("--objects", nargs="*", default=FS.OBJECTS)
    b.add_argument("--device")
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "identify":
        kw = {k: v for k, v in dict(device=a.device, seeds=a.seeds).items() if v is not None}
        r = api.identify(_interaction(a), a.method, **kw)
        r.save(a.out)
        p = r.parameters
        print(r.status, None if p is None else f"I={p.inertia:.6g} {p.units['inertia']}  c={p.damping:.6g} {p.units['damping']}",
              (r.identifiability or {}).get("verdict"))
    elif a.cmd == "diagnose":
        from .validation.identifiability import diagnose
        print(json.dumps(diagnose(_interaction(a)), indent=1, default=float))
    else:
        from .validation import forcesapien_a3 as P
        try:
            bench = FS.A3Benchmark()
        except FileNotFoundError as exc:
            ap.error(str(exc))
        est = P.reference_estimates(bench)
        for m in a.refit:
            for slug in a.objects:
                kw = {"device": a.device} if a.device else {}
                r = api.identify(bench.interaction_A3(slug), m, diagnose_identifiability=False, **kw)
                est[slug][m + " (refit)"] = [dict(seed=s["seed"], inertia=s.get("inertia"), damping=s.get("damping"),
                                                  q_hat=None, status=s.get("status")) for s in r.per_seed]
        res = P.evaluate({s: est[s] for s in a.objects}, bench, a.objects)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        for m, v in res["per_method"].items():
            print(f"{m:22s} |I| {v['I_err_median']:.3f}%  |c| {v['c_err_median']:.3f}%  B median {v['B_object_median']:.3f}%  "
                  f"PASS {v['B_pass']}/{v['B_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
