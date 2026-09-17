import numpy as np

from odtoy.scenario import make_scenario


def test_make_scenario_shapes():
    sc = make_scenario(rows=4, cols=4, n_days=3, n_sensors=8, seed=0)
    n_links = sc.net.n_links
    assert sc.n_sensors == 8
    assert sc.od_prior.shape == (16, 16)
    assert len(sc.days) == 3
    for d in sc.days:
        assert d.od_true.shape == (16, 16)
        assert d.flows_true.shape == (n_links,)
        assert d.counts.shape == (8,)
    assert sc.sim.n_calls == 0


def test_counts_match_true_flows_up_to_noise():
    sc = make_scenario(rows=4, cols=4, n_days=2, n_sensors=8, count_noise_sd=0.0, seed=1)
    for d in sc.days:
        assert np.allclose(d.counts, d.flows_true[sc.sensors])


def test_true_flows_are_reproducible_from_simulator():
    sc = make_scenario(rows=4, cols=4, n_days=2, n_sensors=8, seed=2)
    d = sc.days[0]
    assert np.array_equal(sc.sim(d.od_true, seed=d.seed), d.flows_true)


def test_same_seed_same_scenario():
    a = make_scenario(rows=4, cols=4, n_days=2, n_sensors=5, seed=3)
    b = make_scenario(rows=4, cols=4, n_days=2, n_sensors=5, seed=3)
    assert np.array_equal(a.od_prior, b.od_prior)
    assert np.array_equal(a.days[1].counts, b.days[1].counts)
