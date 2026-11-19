#!/usr/bin/env python3
"""Smallest possible use: arrays in, parameters out (no dataset adapter).

A drawer (prismatic, axis +x in frame 'W') is pushed with 4 N for 1.5 s and released; q is observed at 30 Hz with
0.1 mm noise. Effective mass 1.2 kg, damping 0.8 N s/m.
"""
import numpy as np

from forceartgs_physics import api
from forceartgs_physics.dynamics.viscous import propagate_from_rest
from forceartgs_physics.forces.projection import GeneralizedEffort
from forceartgs_physics.schemas import ForceRecord, Interaction, JointKinematics, Trajectory

tf = np.arange(0, 6, 1 / 100)
F = np.zeros((len(tf), 3)); F[tf < 1.5, 0] = 4.0                       # N, on the moving part, frame W
t = np.arange(1 / 30, 6, 1 / 30)
q = propagate_from_rest(1.2, 0.8, GeneralizedEffort(tf, F[:, 0], "prismatic", "truth"), t)
q += np.random.default_rng(0).normal(0, 1e-4, len(t))

x = Interaction("drawer_demo",
                JointKinematics("prismatic", axis=[1, 0, 0], origin=[0, 0, 0], frame="W"),
                Trajectory(t_s=t, q=q),
                ForceRecord(t_s=tf, force_N=F, frame="W"))
r = api.identify(x, "Backprop")
p = r.parameters
print(f"I = {p.inertia:.4f} {p.units['inertia']}  c = {p.damping:.4f} {p.units['damping']}  "
      f"identifiable: {p.identifiable}  ({r.identifiability['verdict']})")
