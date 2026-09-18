import numpy as np

from odtoy.spsa import spsa, spsa_gradient

H = np.diag([1.0, 4.0, 9.0])
B = np.array([1.0, -2.0, 0.5])


def quad(theta, seed, noise_sd=0.1):
    """Quadratic with seed-driven additive noise; true gradient is H theta + B."""
    noise = noise_sd * np.random.default_rng(seed).standard_normal()
    return 0.5 * theta @ H @ theta + B @ theta + noise


def test_gradient_estimate_is_unbiased():
    # the estimate's spread scales with the gradient norm (cross terms of
    # the perturbation), so test at a point where the gradient is O(1)
    theta = np.array([0.5, 0.25, -0.1])
    rng = np.random.default_rng(0)
    est = np.mean([spsa_gradient(quad, theta, 0.1, rng)[0] for _ in range(4000)], axis=0)
    assert np.allclose(est, H @ theta + B, atol=0.1)


def test_common_random_numbers_cancel_additive_noise():
    theta = np.zeros(3)
    rng = np.random.default_rng(0)
    g, _ = spsa_gradient(lambda t, s: quad(t, s, noise_sd=100.0), theta, 0.1, rng)
    # same seed on both sides -> the huge noise cancels exactly
    assert np.all(np.abs(g) < 10.0)


def test_spsa_converges_on_quadratic():
    theta_star = -np.linalg.solve(H, B)
    theta, hist = spsa(quad, np.zeros(3), n_iter=500, a=0.5, c=0.1, seed=1)
    assert hist.shape == (500,)
    assert np.linalg.norm(theta - theta_star) < 0.1
    assert hist[-50:].mean() < hist[:50].mean()


def test_clip_bounds_step():
    theta, _ = spsa(quad, 100 * np.ones(3), n_iter=1, a=1.0, c=0.1, A=0.0, clip=1.0)
    assert np.linalg.norm(theta - 100 * np.ones(3)) <= 1.0 + 1e-9
