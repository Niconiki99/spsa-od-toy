import numpy as np

from odtoy.calibrate import (
    active_cells,
    calibrate_od_spsa,
    make_day_loss,
    od_from_log_factors,
    od_from_zone_factors,
    parametrisation,
)
from odtoy.metrics import count_loss, od_rel_error
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


def test_zone_factors_zero_returns_prior():
    sc = _small_scenario()
    n_params, to_od = parametrisation(sc.od_prior, "zones")
    assert n_params == 2 * 16 + 1
    assert np.allclose(to_od(np.zeros(n_params)), sc.od_prior)


def test_zone_factors_structure():
    sc = _small_scenario()
    theta = np.zeros(33)
    theta[0] = np.log(2.0)      # global scale
    theta[1 + 3] = np.log(3.0)  # origin zone 3
    theta[1 + 16 + 5] = np.log(0.5)  # destination zone 5
    od = od_from_zone_factors(sc.od_prior, theta)
    assert np.isclose(od[3, 5], sc.od_prior[3, 5] * 2.0 * 3.0 * 0.5)
    assert np.isclose(od[0, 1], sc.od_prior[0, 1] * 2.0)


def test_zone_calibration_reduces_loss_and_od_error():
    sc = _small_scenario()
    day = sc.days[0]
    before = count_loss(sc.sim(sc.od_prior, seed=0), day.counts, sc.sensors)
    od_est, _ = calibrate_od_spsa(sc, day, sc.sensors, n_iter=60, a=0.3, kind="zones", seed=0)
    after = count_loss(sc.sim(od_est, seed=0), day.counts, sc.sensors)
    assert after < before
    assert od_rel_error(od_est, day.od_true) < od_rel_error(sc.od_prior, day.od_true)
