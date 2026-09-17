import numpy as np

from odtoy import assign, bpr_time, grid_network
from odtoy.assignment import all_or_nothing


def test_bpr_increasing():
    fft, cap = np.array([1.0]), np.array([10.0])
    t0 = bpr_time(fft, cap, np.array([0.0]))
    t1 = bpr_time(fft, cap, np.array([10.0]))
    assert t0 == fft
    assert t1 > t0


def test_aon_single_pair_uses_manhattan_path():
    net = grid_network(3, 3)
    od = np.zeros((9, 9))
    od[0, 8] = 5.0  # corner to opposite corner, 4 links away
    flow = all_or_nothing(net, od, net.fft)
    assert flow.sum() == 5.0 * 4
    assert np.all((flow == 0) | (flow == 5.0))


def test_assign_conserves_demand_lower_bound():
    net = grid_network(4, 4, seed=3)
    rng = np.random.default_rng(0)
    od = rng.uniform(0, 10, (16, 16))
    np.fill_diagonal(od, 0.0)
    flow = assign(net, od, n_iter=10)
    assert flow.shape == (net.n_links,)
    assert np.all(flow >= 0)
    # every trip uses at least one link
    assert flow.sum() >= od.sum() - 1e-9


def test_assign_zero_demand():
    net = grid_network(3, 3)
    flow = assign(net, np.zeros((9, 9)))
    assert np.all(flow == 0)


def test_assign_spreads_flow_under_congestion():
    """With heavy demand MSA should use more than the single AON path."""
    net = grid_network(3, 3, cap_mean=5.0, cap_spread=0.0)
    od = np.zeros((9, 9))
    od[0, 8] = 50.0
    aon = all_or_nothing(net, od, net.fft)
    eq = assign(net, od, n_iter=30)
    assert np.count_nonzero(eq) > np.count_nonzero(aon)
