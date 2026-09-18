"""Static traffic assignment: BPR link costs + method of successive averages.

The all-or-nothing step picks a single shortest path per OD pair, which
makes the mapping OD -> flows piecewise smooth but not differentiable:
exactly the kind of black box we want to calibrate through.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .network import Network


def bpr_time(fft: np.ndarray, cap: np.ndarray, flow: np.ndarray, alpha: float = 0.15, beta: float = 4.0) -> np.ndarray:
    """Bureau of Public Roads volume-delay function."""
    return fft * (1.0 + alpha * (flow / cap) ** beta)


def _link_index(net: Network) -> dict[tuple[int, int], int]:
    return {(int(t), int(h)): i for i, (t, h) in enumerate(zip(net.tail, net.head))}


def all_or_nothing(net: Network, od: np.ndarray, cost: np.ndarray, link_index: dict | None = None) -> np.ndarray:
    """Load every OD pair on its current shortest path.

    Parameters
    ----------
    od : (n_zones, n_zones) demand matrix, zone order follows ``net.zones``
    cost : (n_links,) current link travel times
    """
    if link_index is None:
        link_index = _link_index(net)
    graph = csr_matrix((cost, (net.tail, net.head)), shape=(net.n_nodes, net.n_nodes))
    _, pred = dijkstra(graph, directed=True, indices=net.zones, return_predecessors=True)

    flow = np.zeros(net.n_links)
    for oi, origin in enumerate(net.zones):
        for di, dest in enumerate(net.zones):
            q = od[oi, di]
            if q <= 0.0 or origin == dest:
                continue
            node = int(dest)
            while node != origin:
                prev = int(pred[oi, node])
                if prev < 0:
                    raise RuntimeError(f"no path from zone {origin} to zone {dest}")
                flow[link_index[(prev, node)]] += q
                node = prev
    return flow


def aon_incidence(net: Network, cost: np.ndarray, link_index: dict | None = None) -> np.ndarray:
    """Link-path incidence for the current shortest paths.

    Returns ``P`` of shape (n_links, n_zones * n_zones) with ``P[l, o*n+d] = 1``
    when link ``l`` lies on the shortest path from zone ``o`` to zone ``d``.
    All-or-nothing flows are then ``P @ od.ravel()``.
    """
    if link_index is None:
        link_index = _link_index(net)
    graph = csr_matrix((cost, (net.tail, net.head)), shape=(net.n_nodes, net.n_nodes))
    _, pred = dijkstra(graph, directed=True, indices=net.zones, return_predecessors=True)

    n = net.n_zones
    P = np.zeros((net.n_links, n * n))
    for oi, origin in enumerate(net.zones):
        for di, dest in enumerate(net.zones):
            if origin == dest:
                continue
            node = int(dest)
            while node != origin:
                prev = int(pred[oi, node])
                if prev < 0:
                    raise RuntimeError(f"no path from zone {origin} to zone {dest}")
                P[link_index[(prev, node)], oi * n + di] = 1.0
                node = prev
    return P


def assign(
    net: Network,
    od: np.ndarray,
    n_iter: int = 30,
    alpha: float = 0.15,
    beta: float = 4.0,
    return_matrix: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Equilibrium-ish link flows via the method of successive averages.

    A fixed number of iterations keeps the run time predictable, which is
    what matters for a simulator that will be called thousands of times.
    With ``return_matrix`` the MSA-averaged incidence ``A`` is returned as
    well, so that ``flows == A @ od.ravel()`` holds exactly.
    """
    od = np.asarray(od, dtype=float)
    if od.shape != (net.n_zones, net.n_zones):
        raise ValueError(f"od must have shape {(net.n_zones, net.n_zones)}, got {od.shape}")
    link_index = _link_index(net)

    if not return_matrix:
        flow = all_or_nothing(net, od, net.fft, link_index)
        for k in range(1, n_iter):
            cost = bpr_time(net.fft, net.cap, flow, alpha, beta)
            aon = all_or_nothing(net, od, cost, link_index)
            step = 1.0 / (k + 1)
            flow = (1.0 - step) * flow + step * aon
        return flow

    od_vec = od.ravel()
    A = aon_incidence(net, net.fft, link_index)
    flow = A @ od_vec
    for k in range(1, n_iter):
        cost = bpr_time(net.fft, net.cap, flow, alpha, beta)
        P = aon_incidence(net, cost, link_index)
        step = 1.0 / (k + 1)
        A = (1.0 - step) * A + step * P
        flow = A @ od_vec
    return flow, A
