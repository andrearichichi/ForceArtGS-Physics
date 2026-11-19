"""Metrics of the A3 benchmark (post_a3.score)."""
from __future__ import annotations

import numpy as np


def nrmse_pct(pred, ref, norm=None) -> float:
    """100 * RMSE(pred - ref) / norm; norm defaults to the range (max - min) of ref."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    n = float(np.ptp(ref)) if norm is None else float(norm)
    return 100 * float(np.sqrt(np.mean((pred - ref) ** 2))) / n


def rel_err_pct(est, true) -> float:
    return 100 * (float(est) / float(true) - 1)


GROSS_FAILURE_PCT = 10.0     # post_a3.GROSS: a B interaction 'passes' iff its nRMSE <= 10 %
