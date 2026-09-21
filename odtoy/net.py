"""A small MLP whose weights live in one flat vector.

SPSA optimises a plain parameter array, so the network keeps no state of
its own: ``forward`` takes the weights as an argument and slices them.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected network with tanh hidden layers and a linear output."""

    def __init__(self, sizes: tuple[int, ...]):
        if len(sizes) < 2:
            raise ValueError("sizes needs at least an input and an output layer")
        self.sizes = tuple(int(s) for s in sizes)

    @property
    def n_params(self) -> int:
        return sum(a * b + b for a, b in zip(self.sizes[:-1], self.sizes[1:]))

    def init_params(self, rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
        """Small random weights, zero biases: the output starts near zero."""
        theta = np.zeros(self.n_params)
        i = 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            theta[i : i + a * b] = scale * rng.standard_normal(a * b) / np.sqrt(a)
            i += a * b + b
        return theta

    def forward(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate the network on a single input vector."""
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have {self.n_params} entries, got {theta.shape}")
        h = np.asarray(x, dtype=float)
        i = 0
        for layer, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            W = theta[i : i + a * b].reshape(a, b)
            i += a * b
            bias = theta[i : i + b]
            i += b
            h = h @ W + bias
            if layer < len(self.sizes) - 2:
                h = np.tanh(h)
        return h
