import numpy as np

from odtoy.metrics import count_loss, geh, geh_pass_rate


def test_count_loss_zero_on_perfect_match():
    flows = np.array([10.0, 20.0, 30.0, 40.0])
    sensors = np.array([1, 3])
    counts = flows[sensors]
    assert count_loss(flows, counts, sensors) == 0.0


def test_count_loss_is_relative():
    flows = np.array([110.0, 11.0])
    sensors = np.array([0, 1])
    counts = np.array([100.0, 10.0])
    # both links are 10% off -> both residuals 0.1 -> mean squared 0.01
    assert np.isclose(count_loss(flows, counts, sensors), 0.01)


def test_count_loss_floor_protects_zero_counts():
    flows = np.array([5.0])
    sensors = np.array([0])
    counts = np.array([0.0])
    assert np.isfinite(count_loss(flows, counts, sensors, floor=1.0))
    assert np.isclose(count_loss(flows, counts, sensors, floor=1.0), 25.0)


def test_geh_known_value():
    flows = np.array([120.0])
    counts = np.array([100.0])
    sensors = np.array([0])
    # sqrt(2 * 400 / 220)
    assert np.isclose(geh(flows, counts, sensors)[0], np.sqrt(800.0 / 220.0))


def test_geh_pass_rate():
    flows = np.array([100.0, 200.0, 1000.0])
    counts = np.array([100.0, 205.0, 500.0])
    sensors = np.array([0, 1, 2])
    assert np.isclose(geh_pass_rate(flows, counts, sensors), 2.0 / 3.0)
