"""Interaction -> inputs of the validated code, exactly as the A3 drivers built them.

Original call chain (robust_v2/run_heldout.py, controlled_fairness/run.py::execute):
    w, info = common.load(slug)          # estimator_input.json -> estimators.Window
    d, sc   = common.normalize(w)        # nondimensionalisation + force truncated at the last observation
    init    = common.ls(d)               # shared integral-LS initializer
    r       = <method>(d, ...)           # nondimensional estimate
    r       = common.restore(r, sc)      # back to physical units
``to_window`` replaces only the file read of ``common.load``; everything else is the vendored code.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch

from .._validated import common as H
from .._validated import estimators as E
from ..forces.projection import generalized_effort


def to_window(inter, encoding: str = "auto") -> E.Window:
    """encoding:
      'cartesian' : force vectors, application points, axis and origin are passed as in the benchmark inputs
                    (bitwise-identical path for ForceSAPIEN);
      'effort'    : the signed generalized effort is computed by forces.projection (adds measured couples) and
                    passed as an exact scalar: a unit axis e_x with force [tau, 0, 0], so that a . F == tau exactly;
      'auto'      : 'effort' if the record has couples (torque_Nm), else 'cartesian'.
    """
    f, j, tr = inter.forces, inter.joint, inter.trajectory
    if encoding == "auto":
        encoding = "effort" if f.torque_Nm is not None else "cartesian"
    if encoding == "cartesian":
        if f.torque_Nm is not None:
            raise ValueError("cartesian encoding cannot carry measured couples; use encoding='effort'")
        P = f.point_m if f.point_m is not None else np.zeros_like(f.force_N)
        return E.Window(joint_type=j.joint_type, t_obs=tr.t_s, q_obs=tr.q, t_force=f.t_s, force_world=f.force_N,
                        point_world=P, axis_world=j.axis, origin_world=j.origin,
                        meta={"slug": inter.name, "encoding": "cartesian"})
    if encoding == "effort":
        tau = generalized_effort(j, f).levels
        F = np.zeros((len(tau), 3)); F[:, 0] = tau
        return E.Window(joint_type="prismatic", t_obs=tr.t_s, q_obs=tr.q, t_force=f.t_s, force_world=F,
                        point_world=np.zeros_like(F), axis_world=np.array([1.0, 0.0, 0.0]), origin_world=np.zeros(3),
                        meta={"slug": inter.name, "encoding": "effort", "physical_joint_type": j.joint_type})
    raise ValueError(encoding)


def prepare(inter, encoding="auto"):
    w = to_window(inter, encoding)
    d, sc = H.normalize(w)
    init = H.ls(d)
    return w, d, sc, init


@contextlib.contextmanager
def single_thread(n: int | None = 1):
    """The validated runs used torch.set_num_threads(1) (set at import of common.py / cells_C.py) and
    OMP_NUM_THREADS=1. CPU float results depend on the reduction order, so bitwise reproduction needs the same."""
    if n is None:
        yield
        return
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def default_device(device: str | None) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def finite_ok(r: dict, q_obs, Q) -> bool:
    """The validity rule of controlled_fairness/run.py::execute."""
    finite = np.isfinite([r["inertia"], r["damping"], *r["q_hat"]]).all() and min(r["inertia"], r["damping"]) > 0
    return bool(finite and np.max(np.abs(np.asarray(r["q_hat"]) - q_obs)) / Q <= 1e6)


def env_note() -> dict:
    return dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                cuda=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
