"""Baseline calibration: SPSA directly on the OD cells.

Parameters are log-factors on the non-zero cells of the prior, so the
corrected matrix stays positive and theta = 0 is the prior itself.
"""

from __future__ import annotations

import numpy as np

from .metrics import count_loss
from .scenario import Day, Scenario
from .spsa import spsa


def active_cells(od_prior: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells that carry demand in the prior."""
    return od_prior > 0


def od_from_log_factors(od_prior: np.ndarray, theta: np.ndarray, active: np.ndarray) -> np.ndarray:
    """OD = prior * exp(theta) on the active cells, prior elsewhere."""
    od = od_prior.copy()
    od[active] = od_prior[active] * np.exp(theta)
    return od


def make_day_loss(sc: Scenario, day: Day, sensors_fit: np.ndarray, reg: float = 0.0):
    """Build ``loss(theta, seed)`` for one day.

    The seed goes to the simulator, so an SPSA perturbation pair shares
    the same daily conditions. ``reg`` penalises the mean squared
    log-factor, i.e. the distance from the prior.
    """
    active = active_cells(sc.od_prior)

    def loss(theta: np.ndarray, seed: int) -> float:
        flows = sc.sim(od_from_log_factors(sc.od_prior, theta, active), seed=seed)
        return count_loss(flows, day.counts[np.isin(sc.sensors, sensors_fit)], sensors_fit) + reg * float(np.mean(theta**2))

    return loss


def calibrate_od_spsa(
    sc: Scenario,
    day: Day,
    sensors_fit: np.ndarray,
    n_iter: int = 200,
    a: float = 1.0,
    c: float = 0.05,
    reg: float = 0.0,
    clip: float | None = None,
    seed: int = 0,
    **spsa_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SPSA on the log-factors of every active cell. Returns (od_est, history)."""
    active = active_cells(sc.od_prior)
    loss = make_day_loss(sc, day, sensors_fit, reg=reg)
    theta, history = spsa(loss, np.zeros(active.sum()), n_iter, a=a, c=c, clip=clip, seed=seed, **spsa_kwargs)
    return od_from_log_factors(sc.od_prior, theta, active), history
