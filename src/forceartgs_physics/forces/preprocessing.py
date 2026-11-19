"""Explicit, logged force preprocessing. Each function returns a NEW ForceRecord whose ``calibration`` dict records
what was done; nothing is applied automatically by the estimators."""
from __future__ import annotations

from dataclasses import replace

import numpy as np


def apply_contact_mask(rec):
    """Zero force (and couple) where ``contact_valid`` is False: off-contact sensor readings are not applied forces."""
    if rec.contact_valid is None:
        raise ValueError("record has no contact_valid mask")
    m = rec.contact_valid
    F = np.where(m[:, None], rec.force_N, 0.0)
    M = None if rec.torque_Nm is None else np.where(m[:, None], rec.torque_Nm, 0.0)
    return replace(rec, force_N=F, torque_Nm=M, calibration={**rec.calibration, "contact_mask_applied": int((~m).sum())})


def remove_constant_bias(rec, window):
    """Subtract the mean force (and couple) over samples with t in [window[0], window[1]] (a declared no-contact window).

    A bias estimated this way is NOT separable from a constant physical force (e.g. Coulomb friction) acting in the
    same window; only use a window where the object is known to be free of contact."""
    m = (rec.t_s >= window[0]) & (rec.t_s <= window[1])
    if m.sum() < 3:
        raise ValueError("bias window contains fewer than 3 samples")
    bF = rec.force_N[m].mean(0)
    bM = None if rec.torque_Nm is None else rec.torque_Nm[m].mean(0)
    return replace(rec, force_N=rec.force_N - bF, torque_Nm=None if bM is None else rec.torque_Nm - bM,
                   calibration={**rec.calibration, "bias_removed": dict(window_s=list(window), force_N=bF.tolist(),
                                                                         torque_Nm=None if bM is None else bM.tolist())})


def shift_time(rec, lag_s: float):
    """t_force <- t_force + lag_s (synchronise the force clock with the camera clock; positive = force is late)."""
    return replace(rec, t_s=rec.t_s + float(lag_s), calibration={**rec.calibration, "time_shift_s": float(lag_s)})
