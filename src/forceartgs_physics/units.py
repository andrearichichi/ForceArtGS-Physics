"""Explicit SI conventions. Nothing in the library converts units implicitly.

The 1-DoF model is ``I q'' + c q' = tau``. Units depend only on the joint type:

=============  ==================  ==================
quantity       revolute            prismatic
=============  ==================  ==================
q              rad                 m
q'             rad/s               m/s
tau (effort)   N*m                 N
I (inertia)    kg*m^2              kg (effective mass)
c (damping)    N*m*s/rad           N*s/m
=============  ==================  ==================

Positions are metres, forces Newtons, torques Newton-metres, times seconds.
"""
from __future__ import annotations

JOINT_TYPES = ("revolute", "prismatic")

UNITS = {
    "revolute": {"q": "rad", "qdot": "rad/s", "effort": "N*m", "inertia": "kg*m^2", "damping": "N*m*s/rad",
                 "time": "s", "position": "m", "force": "N", "torque": "N*m"},
    "prismatic": {"q": "m", "qdot": "m/s", "effort": "N", "inertia": "kg", "damping": "N*s/m",
                  "time": "s", "position": "m", "force": "N", "torque": "N*m"},
}


def units_for(joint_type: str) -> dict:
    """Unit table for a joint type; raises on anything other than revolute/prismatic."""
    if joint_type not in UNITS:
        raise ValueError(f"joint_type must be one of {JOINT_TYPES}, got {joint_type!r}")
    return dict(UNITS[joint_type])
