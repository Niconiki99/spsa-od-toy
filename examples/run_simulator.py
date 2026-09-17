"""Smoke test: build a grid, assign a random OD, print flow statistics and timing."""

import time

import numpy as np

from odtoy import Simulator, grid_network


def main() -> None:
    net = grid_network(rows=6, cols=6, cap_mean=80.0, seed=0)
    sim = Simulator(net, n_iter=30, cap_jitter=0.1)

    rng = np.random.default_rng(42)
    od = rng.gamma(shape=2.0, scale=0.8, size=(net.n_zones, net.n_zones))
    np.fill_diagonal(od, 0.0)

    t0 = time.perf_counter()
    flows = sim(od, seed=0)
    dt = time.perf_counter() - t0

    print(f"nodes={net.n_nodes} links={net.n_links} zones={net.n_zones} trips={od.sum():.0f}")
    print(f"one simulation: {dt*1000:.1f} ms  (calls so far: {sim.n_calls})")
    print(f"flow  min/median/max: {flows.min():.1f} / {np.median(flows):.1f} / {flows.max():.1f}")
    print(f"v/c   median/max:     {np.median(flows/net.cap):.2f} / {(flows/net.cap).max():.2f}")


if __name__ == "__main__":
    main()
