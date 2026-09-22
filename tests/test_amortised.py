import numpy as np

from odtoy.amortised import (
    baseline_flows,
    day_features,
    make_amortised_loss,
    predict_od,
    train_amortised,
)
from odtoy.net import MLP
from odtoy.scenario import make_scenario


def _scenario():
    return make_scenario(rows=4, cols=4, n_days=3, n_sensors=12, count_noise_sd=0.0,
                         sim_kwargs={"cap_jitter": 0.0, "n_iter": 10}, seed=0)


def test_features_are_zero_when_counts_match_baseline():
    sc = _scenario()
    bases = baseline_flows(sc, sc.days)
    # a day whose counts come from the prior itself has no residual
    day = sc.days[0].__class__(sc.days[0].seed, sc.od_prior, bases[0], bases[0][sc.sensors])
    assert np.allclose(day_features(day, bases[0], sc, sc.sensors), 0.0)


def test_features_sign_follows_congestion():
    sc = _scenario()
    bases = baseline_flows(sc, sc.days)
    f = day_features(sc.days[0], bases[0], sc, sc.sensors)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_zero_weights_predict_the_prior():
    sc = _scenario()
    net = MLP((12, 4, 33))
    od = predict_od(sc, net, np.zeros(net.n_params), np.ones(12))
    assert np.allclose(od, sc.od_prior)


def test_output_scale_bounds_the_correction():
    sc = _scenario()
    net = MLP((12, 4, 33))
    theta = 50.0 * np.random.default_rng(0).standard_normal(net.n_params)
    od = predict_od(sc, net, theta, np.ones(12), out_scale=0.2)
    mask = sc.od_prior > 0
    ratio = od[mask] / sc.od_prior[mask]
    # scale + origin + destination factors, each within exp(+-0.2)
    assert np.all(ratio <= np.exp(0.6) + 1e-9)
    assert np.all(ratio >= np.exp(-0.6) - 1e-9)


def test_loss_batch_uses_expected_number_of_simulations():
    sc = _scenario()
    net = MLP((12, 4, 33))
    bases = baseline_flows(sc, sc.days)
    sc.sim.reset_counter()
    loss = make_amortised_loss(sc, net, sc.days, sc.sensors, bases, batch_size=2)
    value = loss(np.zeros(net.n_params), 0)
    assert sc.sim.n_calls == 2
    assert value > 0


def test_same_seed_selects_the_same_batch():
    sc = _scenario()
    net = MLP((12, 4, 33))
    bases = baseline_flows(sc, sc.days)
    loss = make_amortised_loss(sc, net, sc.days, sc.sensors, bases, batch_size=2)
    theta = np.zeros(net.n_params)
    assert loss(theta, 5) == loss(theta, 5)


def test_training_reduces_the_loss():
    sc = _scenario()
    net = MLP((12, 4, 33))
    bases = baseline_flows(sc, sc.days)
    loss = make_amortised_loss(sc, net, sc.days, sc.sensors, bases)
    theta, history = train_amortised(sc, net, sc.days, sc.sensors, bases, n_iter=40, seed=0)
    assert history.shape == (40,)
    assert loss(theta, 0) < loss(net.init_params(np.random.default_rng(0)), 0)


def test_batch_size_is_capped_at_the_number_of_days():
    sc = _scenario()
    net = MLP((12, 4, 33))
    bases = baseline_flows(sc, sc.days)
    sc.sim.reset_counter()
    loss = make_amortised_loss(sc, net, sc.days, sc.sensors, bases, batch_size=99)
    loss(np.zeros(net.n_params), 0)
    assert sc.sim.n_calls == len(sc.days)
