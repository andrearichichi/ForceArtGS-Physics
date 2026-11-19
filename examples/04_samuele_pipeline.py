#!/usr/bin/env python3
"""Measured data only: ForceArtGS output + preprocessed observed forces.

Writes a diagnostic research report, never a simulation-ready asset.
Calibration, synchronization and model adequacy must be checked on the episode.
See docs/QUICKSTART_SAMUELE.md. No synthetic force or calibration fallback.
"""
import argparse

from forceartgs_physics import api
from forceartgs_physics.adapters import forceartgs as FA
from forceartgs_physics.adapters.real import load_observed_forces_json
from forceartgs_physics.validation.identifiability import diagnose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--level", choices=["A", "B"], required=True)
    p.add_argument("--forces-json", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["IntegralLS", "AKF", "Backprop"], default="AKF")
    a = p.parse_args()
    loader = FA.load_authority if a.level == "A" else FA.load_discovery_run
    joint, trajectory = loader(a.run, frame=a.frame)
    forces = load_observed_forces_json(a.forces_json)
    x = FA.interaction("measured_episode", joint, trajectory, forces)
    report = diagnose(x)
    print(report["verdict"])
    if not report["identifiable"]:
        p.exit(2, "Not identifiable: no physical parameters exported.\n")
    result = api.identify(x, a.method, device="cpu")
    result.save(a.out)
    print(f"Diagnostic report: {a.out}; status={result.status}")
    print("Model-conditional estimates only: measured-data model adequacy still requires review.")
    if result.status != "ok":
        p.exit(2)


if __name__ == "__main__":
    main()
