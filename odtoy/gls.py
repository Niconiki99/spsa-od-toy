"""Baseline: generalised least squares on a linearised simulator.

Around the prior the simulator is approximated by its assignment matrix,
flows ~= A @ od. The corrected OD then solves a bounded least-squares
problem in the cell factors u = od / prior:

    min  || (A' u - y) / s ||^2  +  reg * || u - 1 ||^2 ,   u >= 0

with A' = A[sensors] scaled by the prior and s the same relative scale
used in ``count_loss``. Re-linearising a few times (one simulation each)
accounts for congestion feedback.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .calibrate import active_cells
from .scenario import Day, Scenario


def gls_step(A_fit: np.ndarray, counts_fit: np.ndarray, od_prior_active: np.ndarray, reg: float, floor: float = 1.0) -> np.ndarray:
    """Solve one bounded GLS problem; returns the factors u on the active cells."""
    scale = np.maximum(counts_fit, floor)
    A_rel = (A_fit * od_prior_active[None, :]) / scale[:, None]
    y_rel = counts_fit / scale
    n = od_prior_active.shape[0]
    M = np.vstack([A_rel, np.sqrt(reg) * np.eye(n)])
    b = np.concatenate([y_rel, np.sqrt(reg) * np.ones(n)])
    res = lsq_linear(M, b, bounds=(0.0, np.inf), lsmr_tol="auto")
    return res.x


def calibrate_od_gls(
    sc: Scenario,
    day: Day,
    sensors_fit: np.ndarray,
    n_outer: int = 3,
    reg: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Iterated linearised GLS starting from the prior. Returns the corrected OD."""
    active = active_cells(sc.od_prior)
    counts_fit = day.counts[np.isin(sc.sensors, sensors_fit)]
    od = sc.od_prior.copy()
    for _ in range(n_outer):
        _, A = sc.sim.assignment_matrix(od, seed=seed)
        A_fit = A[sensors_fit][:, active.ravel()]
        u = gls_step(A_fit, counts_fit, od[active], reg)
        od = od.copy()
        od[active] = od[active] * u
    return od
