"""SPSA calibration of the OD matrix under two parametrisations.

- ``cells``: one log-factor per non-zero cell of the prior (baseline)
- ``zones``: a global log-scale plus one log-factor per origin zone and
  one per destination zone, i.e. 2n+1 parameters instead of ~n^2

In both cases the corrected matrix stays positive and theta = 0 is the
prior itself.
"""

from __future__ import annotations

import numpy as np

from .metrics import count_loss
from .scenario import Day, Scenario
from .spsa import spsa

#: Largest log-factor allowed on a cell. exp(5) is already an extreme
#: correction, and unbounded factors make the BPR costs overflow.
MAX_LOG_FACTOR = 5.0


def active_cells(od_prior: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells that carry demand in the prior."""
    return od_prior > 0


def od_from_log_factors(od_prior: np.ndarray, theta: np.ndarray, active: np.ndarray) -> np.ndarray:
    """OD = prior * exp(theta) on the active cells, prior elsewhere."""
    od = od_prior.copy()
    od[active] = od_prior[active] * np.exp(np.clip(theta, -MAX_LOG_FACTOR, MAX_LOG_FACTOR))
    return od


def od_from_zone_factors(od_prior: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """OD = prior * exp(s) * exp(o)[:, None] * exp(d)[None, :] with theta = [s, o, d]."""
    n = od_prior.shape[0]
    t = np.clip(theta, -MAX_LOG_FACTOR, MAX_LOG_FACTOR)
    s, o, d = t[0], t[1 : 1 + n], t[1 + n :]
    return od_prior * np.exp(s) * np.exp(o)[:, None] * np.exp(d)[None, :]


def parametrisation(od_prior: np.ndarray, kind: str):
    """Return ``(n_params, theta -> od)`` for ``kind`` in {"cells", "zones"}."""
    if kind == "cells":
        active = active_cells(od_prior)
        return int(active.sum()), lambda theta: od_from_log_factors(od_prior, theta, active)
    if kind == "zones":
        return 2 * od_prior.shape[0] + 1, lambda theta: od_from_zone_factors(od_prior, theta)
    raise ValueError(f"unknown parametrisation {kind!r}")


def make_day_loss(sc: Scenario, day: Day, sensors_fit: np.ndarray, reg: float = 0.0, kind: str = "cells"):
    """Build ``loss(theta, seed)`` for one day.

    The seed goes to the simulator, so an SPSA perturbation pair shares
    the same daily conditions. ``reg`` penalises the mean squared
    log-factor, i.e. the distance from the prior.
    """
    _, to_od = parametrisation(sc.od_prior, kind)
    counts_fit = day.counts[np.isin(sc.sensors, sensors_fit)]

    def loss(theta: np.ndarray, seed: int) -> float:
        flows = sc.sim(to_od(theta), seed=seed)
        return count_loss(flows, counts_fit, sensors_fit) + reg * float(np.mean(theta**2))

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
    kind: str = "cells",
    **spsa_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SPSA under the chosen parametrisation. Returns (od_est, history)."""
    n_params, to_od = parametrisation(sc.od_prior, kind)
    loss = make_day_loss(sc, day, sensors_fit, reg=reg, kind=kind)
    theta, history = spsa(loss, np.zeros(n_params), n_iter, a=a, c=c, clip=clip, seed=seed, **spsa_kwargs)
    return to_od(theta), history
