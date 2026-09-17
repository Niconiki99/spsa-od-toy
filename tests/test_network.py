import numpy as np

from odtoy import grid_network


def test_grid_link_count():
    net = grid_network(3, 4)
    # horizontal: 3 rows * 3 pairs, vertical: 2 rows * 4 pairs, each bidirectional
    assert net.n_links == 2 * (3 * 3 + 2 * 4)
    assert net.n_nodes == 12
    assert net.n_zones == 12


def test_zone_stride():
    net = grid_network(4, 4, zone_stride=2)
    assert net.n_zones == 4


def test_capacity_is_reproducible():
    a = grid_network(3, 3, seed=1)
    b = grid_network(3, 3, seed=1)
    c = grid_network(3, 3, seed=2)
    assert np.array_equal(a.cap, b.cap)
    assert not np.array_equal(a.cap, c.cap)
    assert np.all(a.cap > 0)
