"""Sensor placement and noisy count measurements."""

from __future__ import annotations

import numpy as np

from .network import Network


def choose_sensors(net: Network, n_sensors: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n_sensors`` distinct links to instrument, sorted by link id."""
    if not 0 < n_sensors <= net.n_links:
        raise ValueError(f"n_sensors must be in 1..{net.n_links}")
    return np.sort(rng.choice(net.n_links, size=n_sensors, replace=False))


def measure(flows: np.ndarray, sensors: np.ndarray, rng: np.random.Generator, noise_sd: float = 0.05) -> np.ndarray:
    """Counts at the sensor links with multiplicative Gaussian noise.

    Relative noise keeps large and small links equally reliable in
    percentage terms, which is roughly how loop detectors behave.
    """
    obs = flows[sensors] * (1.0 + noise_sd * rng.standard_normal(sensors.shape[0]))
    return np.maximum(obs, 0.0)


def split_sensors(sensors: np.ndarray, n_holdout: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split sensors into (fit, holdout) sets.

    Methods only see the fit set; the holdout set measures whether a
    correction improves flows where no count constrained it.
    """
    if not 0 <= n_holdout < sensors.shape[0]:
        raise ValueError("n_holdout must leave at least one sensor to fit")
    perm = rng.permutation(sensors.shape[0])
    return np.sort(sensors[perm[n_holdout:]]), np.sort(sensors[perm[:n_holdout]])
