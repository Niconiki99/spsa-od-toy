"""Synthetic demand: a structured mean OD matrix, daily variations, and a biased prior."""

from __future__ import annotations

import numpy as np

from .network import Network


def zone_coords(net: Network, cols: int) -> np.ndarray:
    """(row, col) grid coordinates of every zone node."""
    z = net.zones
    return np.stack([z // cols, z % cols], axis=1).astype(float)


def gravity_od(
    net: Network,
    cols: int,
    total_trips: float,
    decay: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Gravity-model OD: productions x attractions x exp(-distance/decay).

    Productions and attractions are gamma-distributed per zone, so some
    zones are much busier than others. Manhattan distance on the grid is
    the deterrence variable. The matrix is scaled to ``total_trips`` and
    has a zero diagonal.
    """
    rng = np.random.default_rng(seed)
    n = net.n_zones
    prod = rng.gamma(shape=2.0, scale=1.0, size=n)
    attr = rng.gamma(shape=2.0, scale=1.0, size=n)
    xy = zone_coords(net, cols)
    dist = np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2)
    od = prod[:, None] * attr[None, :] * np.exp(-dist / decay)
    np.fill_diagonal(od, 0.0)
    return od * (total_trips / od.sum())


def sample_day_od(
    od_mean: np.ndarray,
    rng: np.random.Generator,
    scale_sd: float = 0.10,
    zone_sd: float = 0.15,
    cell_sd: float = 0.10,
) -> np.ndarray:
    """Draw one day's true OD as multiplicative log-normal deviations.

    Three layers of variability, all centred on the mean:
    - a global scale (busy vs quiet day),
    - one factor per origin zone and one per destination zone
      (structured deviations, the kind a low-dimensional correction
      should be able to capture),
    - independent noise per cell.
    """
    n = od_mean.shape[0]
    scale = np.exp(scale_sd * rng.standard_normal())
    orig = np.exp(zone_sd * rng.standard_normal(n))
    dest = np.exp(zone_sd * rng.standard_normal(n))
    cell = np.exp(cell_sd * rng.standard_normal((n, n)))
    od = od_mean * scale * orig[:, None] * dest[None, :] * cell
    np.fill_diagonal(od, 0.0)
    return od


def make_prior(
    od_mean: np.ndarray,
    rng: np.random.Generator,
    scale_bias: float = 0.9,
    zone_sd: float = 0.25,
    cell_sd: float = 0.20,
) -> np.ndarray:
    """A biased estimate of the mean OD, standing in for a planning model.

    Same multiplicative structure as ``sample_day_od`` but with a fixed
    global bias (models typically under- or over-estimate total demand)
    and larger zone/cell errors. This is the matrix every calibration
    method starts from; how far it is from the daily truth is the
    problem to solve.
    """
    n = od_mean.shape[0]
    orig = np.exp(zone_sd * rng.standard_normal(n))
    dest = np.exp(zone_sd * rng.standard_normal(n))
    cell = np.exp(cell_sd * rng.standard_normal((n, n)))
    od = od_mean * scale_bias * orig[:, None] * dest[None, :] * cell
    np.fill_diagonal(od, 0.0)
    return od
