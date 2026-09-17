"""Black-box simulator wrapper: OD matrix + seed -> link flows.

This is the only interface the optimisers are allowed to see. It counts
how many times it has been called, because the number of simulations is
the cost unit for every method we compare.
"""

from __future__ import annotations

import numpy as np

from .assignment import assign
from .network import Network


class Simulator:
    """Callable ``sim(od, seed) -> flows`` with daily stochastic conditions.

    The seed drives a log-normal jitter on link capacities, standing in
    for whatever makes a real simulator day-dependent (incidents,
    weather, driver behaviour). Two calls with the same OD and seed give
    identical flows, which is what common random numbers rely on.
    """

    def __init__(
        self,
        net: Network,
        n_iter: int = 30,
        cap_jitter: float = 0.1,
        alpha: float = 0.15,
        beta: float = 4.0,
    ):
        self.net = net
        self.n_iter = n_iter
        self.cap_jitter = cap_jitter
        self.alpha = alpha
        self.beta = beta
        self.n_calls = 0

    def daily_network(self, seed: int) -> Network:
        rng = np.random.default_rng(seed)
        jitter = np.exp(self.cap_jitter * rng.standard_normal(self.net.n_links))
        return self.net.with_capacity(self.net.cap * jitter)

    def __call__(self, od: np.ndarray, seed: int = 0) -> np.ndarray:
        self.n_calls += 1
        net = self.daily_network(seed) if self.cap_jitter > 0 else self.net
        return assign(net, od, n_iter=self.n_iter, alpha=self.alpha, beta=self.beta)

    def reset_counter(self) -> None:
        self.n_calls = 0
