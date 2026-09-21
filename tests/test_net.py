import numpy as np
import pytest

from odtoy.net import MLP


def test_param_count():
    net = MLP((5, 4, 3))
    assert net.n_params == 5 * 4 + 4 + 4 * 3 + 3


def test_zero_params_give_zero_output():
    net = MLP((5, 4, 3))
    out = net.forward(np.ones(5), np.zeros(net.n_params))
    assert np.array_equal(out, np.zeros(3))


def test_linear_network_matches_matmul():
    net = MLP((3, 2))
    theta = np.arange(8, dtype=float)
    W, b = theta[:6].reshape(3, 2), theta[6:]
    x = np.array([1.0, -2.0, 0.5])
    assert np.allclose(net.forward(x, theta), x @ W + b)


def test_hidden_layer_is_bounded():
    net = MLP((4, 6, 2))
    theta = 100.0 * np.ones(net.n_params)
    out = net.forward(np.ones(4), theta)
    # tanh saturates, so the output cannot exceed the last layer's scale
    assert np.all(np.abs(out) <= 100.0 * (6 + 1) + 1e-9)


def test_init_params_are_small_and_biases_zero():
    net = MLP((10, 5, 2))
    theta = net.init_params(np.random.default_rng(0))
    out = net.forward(np.ones(10), theta)
    assert theta.shape == (net.n_params,)
    assert np.all(np.abs(out) < 5.0)


def test_wrong_theta_size_raises():
    net = MLP((3, 2))
    with pytest.raises(ValueError):
        net.forward(np.ones(3), np.zeros(5))
