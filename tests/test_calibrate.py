import numpy as np

from odtoy.calibrate import active_cells, calibrate_od_spsa, make_day_loss, od_from_log_factors
from odtoy.metrics import count_loss
from odtoy.scenario import make_scenario


def _small_scenario():
    return make_scenario(rows=4, cols=4, n_days=1, n_sensors=12, count_noise_sd=0.0,
                         sim_kwargs={"cap_jitter": 0.0, "n_iter": 10}, seed=0)


def test_zero_theta_returns_prior():
    sc = _small_scenario()
    active = active_cells(sc.od_prior)
    assert active.sum() == 16 * 15
    od = od_from_log_factors(sc.od_prior, np.zeros(active.sum()), active)
    assert np.array_equal(od, sc.od_prior)


def test_loss_at_zero_matches_prior_loss():
    sc = _small_scenario()
    day = sc.days[0]
    loss = make_day_loss(sc, day, sc.sensors)
    flows = sc.sim(sc.od_prior, seed=3)
    assert np.isclose(loss(np.zeros(16 * 15), 3), count_loss(flows, day.counts, sc.sensors))


def test_regularisation_adds_to_loss():
    sc = _small_scenario()
    theta = 0.1 * np.ones(16 * 15)
    l0 = make_day_loss(sc, sc.days[0], sc.sensors, reg=0.0)(theta, 0)
    l1 = make_day_loss(sc, sc.days[0], sc.sensors, reg=1.0)(theta, 0)
    assert np.isclose(l1 - l0, 0.01)


def test_calibration_reduces_count_loss():
    sc = _small_scenario()
    day = sc.days[0]
    before = count_loss(sc.sim(sc.od_prior, seed=0), day.counts, sc.sensors)
    od_est, history = calibrate_od_spsa(sc, day, sc.sensors, n_iter=40, seed=0)
    after = count_loss(sc.sim(od_est, seed=0), day.counts, sc.sensors)
    assert history.shape == (40,)
    assert after < before
    assert np.all(od_est >= 0)
    assert sc.sim.n_calls == 2 * 40 + 2
