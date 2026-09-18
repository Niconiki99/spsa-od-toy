"""Simultaneous Perturbation Stochastic Approximation (Spall, 1992).

Works on any loss ``loss(theta, seed) -> float``. The seed lets the
caller use common random numbers: both sides of a perturbation pair are
evaluated with the same seed, which removes most of the simulator noise
from the difference.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Loss = Callable[[np.ndarray, int], float]


def spsa_gradient(
    loss: Loss,
    theta: np.ndarray,
    c: float,
    rng: np.random.Generator,
    n_dir: int = 1,
) -> tuple[np.ndarray, float]:
    """Two-sided SPSA gradient estimate, averaged over ``n_dir`` directions.

    Returns the estimate and the mean of the evaluated losses, a free
    estimate of ``loss(theta)`` that costs no extra simulation.
    """
    g = np.zeros_like(theta, dtype=float)
    l_mean = 0.0
    for _ in range(n_dir):
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        seed = int(rng.integers(1 << 31))
        lp = loss(theta + c * delta, seed)
        lm = loss(theta - c * delta, seed)
        g += (lp - lm) / (2.0 * c * delta)
        l_mean += 0.5 * (lp + lm)
    return g / n_dir, l_mean / n_dir


def spsa(
    loss: Loss,
    theta0: np.ndarray,
    n_iter: int,
    a: float,
    c: float,
    A: float | None = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    n_dir: int = 1,
    clip: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise ``loss`` with Spall's standard gain sequences.

    a_k = a / (A + k + 1)^alpha,  c_k = c / (k + 1)^gamma.
    ``A`` defaults to 10% of ``n_iter``. ``clip`` bounds the L2 norm of
    each gradient estimate. Returns the final theta and the per-iteration
    loss estimates (length ``n_iter``).
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    A = 0.1 * n_iter if A is None else A
    history = np.empty(n_iter)
    for k in range(n_iter):
        ak = a / (A + k + 1.0) ** alpha
        ck = c / (k + 1.0) ** gamma
        g, history[k] = spsa_gradient(loss, theta, ck, rng, n_dir)
        if clip is not None:
            norm = np.linalg.norm(g)
            if norm > clip:
                g *= clip / norm
        theta -= ak * g
    return theta, history
