"""Losses and diagnostics for comparing simulated flows with sensor counts."""

from __future__ import annotations

import numpy as np


def count_loss(
    flows: np.ndarray,
    counts: np.ndarray,
    sensors: np.ndarray,
    floor: float = 1.0,
) -> float:
    """Mean squared *relative* error at the sensor links.

    Residuals are divided by the observed count (floored at ``floor``
    vehicles), so that a 10% miss on a quiet link weighs as much as a
    10% miss on a busy one. This matches the multiplicative noise model
    of the sensors and is the objective every optimiser will minimise.
    """
    scale = np.maximum(counts, floor)
    resid = (flows[sensors] - counts) / scale
    return float(np.mean(resid**2))


def geh(flows: np.ndarray, counts: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """GEH statistic per sensor, the usual traffic-model acceptance measure.

    GEH = sqrt(2 (m - c)^2 / (m + c)); practitioners consider a link
    acceptable below 5 and a model acceptable when most links pass.
    """
    m = flows[sensors]
    c = counts
    denom = np.maximum(m + c, 1e-9)
    return np.sqrt(2.0 * (m - c) ** 2 / denom)


def geh_pass_rate(flows: np.ndarray, counts: np.ndarray, sensors: np.ndarray, threshold: float = 5.0) -> float:
    """Fraction of sensors with GEH below ``threshold``."""
    return float(np.mean(geh(flows, counts, sensors) < threshold))


def od_rel_error(od_est: np.ndarray, od_true: np.ndarray) -> float:
    """Mean absolute relative error over the cells with non-zero true demand.

    Only computable on synthetic data, where the true OD is known: this
    is the number that tells whether a method is estimating demand or
    merely fitting the sensors.
    """
    mask = od_true > 0
    return float(np.mean(np.abs(od_est[mask] / od_true[mask] - 1.0)))


def od_total_error(od_est: np.ndarray, od_true: np.ndarray) -> float:
    """Signed relative error on total trips (positive = over-estimate)."""
    return float(od_est.sum() / od_true.sum() - 1.0)
