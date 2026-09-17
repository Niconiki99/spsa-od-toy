"""Road network container and a synthetic grid generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Network:
    """Directed road network stored as parallel link arrays.

    Attributes
    ----------
    n_nodes : number of nodes (0..n_nodes-1)
    tail, head : link endpoints, shape (n_links,)
    fft : free-flow travel time per link
    cap : capacity per link (vehicles / period)
    zones : node ids that act as origins/destinations
    """

    n_nodes: int
    tail: np.ndarray
    head: np.ndarray
    fft: np.ndarray
    cap: np.ndarray
    zones: np.ndarray

    @property
    def n_links(self) -> int:
        return int(self.tail.shape[0])

    @property
    def n_zones(self) -> int:
        return int(self.zones.shape[0])

    def with_capacity(self, cap: np.ndarray) -> "Network":
        """Return a copy with a different capacity vector."""
        cap = np.asarray(cap, dtype=float)
        if cap.shape != self.cap.shape:
            raise ValueError("capacity vector has the wrong shape")
        return Network(self.n_nodes, self.tail, self.head, self.fft, cap, self.zones)


def grid_network(
    rows: int,
    cols: int,
    fft: float = 1.0,
    cap_mean: float = 100.0,
    cap_spread: float = 0.3,
    zone_stride: int = 1,
    seed: int = 0,
) -> Network:
    """Build a bidirectional grid with heterogeneous capacities.

    Every pair of 4-neighbours is connected by two opposite links.
    Capacities are drawn log-normally around ``cap_mean`` so that
    some corridors are naturally more attractive than others.
    Zones are the nodes on a sub-grid with step ``zone_stride``.
    """
    if rows < 2 or cols < 2:
        raise ValueError("grid needs at least 2 rows and 2 columns")
    rng = np.random.default_rng(seed)

    def nid(r: int, c: int) -> int:
        return r * cols + c

    tails, heads = [], []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                tails += [nid(r, c), nid(r, c + 1)]
                heads += [nid(r, c + 1), nid(r, c)]
            if r + 1 < rows:
                tails += [nid(r, c), nid(r + 1, c)]
                heads += [nid(r + 1, c), nid(r, c)]

    n_links = len(tails)
    cap = cap_mean * np.exp(cap_spread * rng.standard_normal(n_links))
    zones = np.array(
        [nid(r, c) for r in range(0, rows, zone_stride) for c in range(0, cols, zone_stride)]
    )
    return Network(
        n_nodes=rows * cols,
        tail=np.asarray(tails, dtype=int),
        head=np.asarray(heads, dtype=int),
        fft=np.full(n_links, float(fft)),
        cap=cap,
        zones=zones,
    )
