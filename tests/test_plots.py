import numpy as np

from odtoy.plots import cumulative_cost, plot_break_even, plot_convergence, smooth


def test_smooth_removes_spikes():
    y = np.ones(50)
    y[25] = 100.0
    assert np.allclose(smooth(y, 5), 1.0)


def test_smooth_keeps_length_and_short_input():
    y = np.arange(7, dtype=float)
    assert smooth(y, 3).shape == y.shape
    assert np.array_equal(smooth(y, 99), y)


def test_cumulative_cost_crossover():
    days = np.arange(20)
    cheap_setup = cumulative_cost(0, 100, days)
    big_setup = cumulative_cost(1000, 1, days)
    # the trained method starts more expensive and wins after ~10 days
    assert cheap_setup[1] < big_setup[1]
    assert cheap_setup[15] > big_setup[15]


def test_figures_are_written(tmp_path):
    sims = np.arange(100) * 2
    curves = {"a": (sims, np.exp(-sims / 50)), "b": (sims, np.exp(-sims / 100))}
    p1 = plot_convergence(curves, tmp_path / "conv.png")
    p2 = plot_break_even({"per-day": (0, 400), "amortised": (4800, 1)}, 40, tmp_path / "cost.png")
    assert p1.exists() and p1.stat().st_size > 1000
    assert p2.exists() and p2.stat().st_size > 1000
