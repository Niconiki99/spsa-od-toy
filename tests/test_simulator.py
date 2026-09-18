import numpy as np

from odtoy import Simulator, grid_network


def _od(n, seed=0):
    od = np.random.default_rng(seed).uniform(0, 20, (n, n))
    np.fill_diagonal(od, 0.0)
    return od


def test_same_seed_same_flows():
    sim = Simulator(grid_network(4, 4))
    od = _od(16)
    assert np.array_equal(sim(od, seed=7), sim(od, seed=7))


def test_different_seed_different_flows():
    sim = Simulator(grid_network(4, 4), cap_jitter=0.3)
    od = _od(16)
    assert not np.array_equal(sim(od, seed=1), sim(od, seed=2))


def test_call_counter():
    sim = Simulator(grid_network(3, 3))
    od = _od(9)
    for _ in range(3):
        sim(od)
    assert sim.n_calls == 3
    sim.reset_counter()
    assert sim.n_calls == 0


def test_no_jitter_is_deterministic_across_seeds():
    sim = Simulator(grid_network(3, 3), cap_jitter=0.0)
    od = _od(9)
    assert np.array_equal(sim(od, seed=1), sim(od, seed=99))


def test_assignment_matrix_counts_a_call():
    sim = Simulator(grid_network(3, 3))
    od = _od(9)
    flows, A = sim.assignment_matrix(od, seed=4)
    assert sim.n_calls == 1
    assert np.allclose(flows, sim(od, seed=4))
    assert np.allclose(A @ od.ravel(), flows)
