import numpy as np
import pytest

from odtoy import grid_network
from odtoy.sensors import choose_sensors, measure


def test_choose_sensors_distinct_and_sorted():
    net = grid_network(4, 4)
    s = choose_sensors(net, 10, np.random.default_rng(0))
    assert s.shape == (10,)
    assert len(set(s.tolist())) == 10
    assert np.all(np.diff(s) > 0)
    assert s.max() < net.n_links


def test_choose_sensors_rejects_bad_count():
    net = grid_network(3, 3)
    with pytest.raises(ValueError):
        choose_sensors(net, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        choose_sensors(net, net.n_links + 1, np.random.default_rng(0))


def test_measure_noise_level():
    rng = np.random.default_rng(0)
    flows = np.full(5000, 100.0)
    sensors = np.arange(5000)
    counts = measure(flows, sensors, rng, noise_sd=0.05)
    rel = counts / flows - 1.0
    assert abs(rel.mean()) < 0.01
    assert 0.04 < rel.std() < 0.06
    assert np.all(counts >= 0)


def test_measure_no_noise():
    flows = np.arange(10, dtype=float)
    sensors = np.array([2, 5])
    counts = measure(flows, sensors, np.random.default_rng(0), noise_sd=0.0)
    assert np.array_equal(counts, np.array([2.0, 5.0]))
