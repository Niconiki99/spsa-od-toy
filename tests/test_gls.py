import numpy as np

from odtoy.gls import calibrate_od_gls, gls_step
from odtoy.metrics import count_loss
from odtoy.scenario import make_scenario


def _small_scenario():
    return make_scenario(rows=4, cols=4, n_days=1, n_sensors=12, count_noise_sd=0.0,
                         sim_kwargs={"cap_jitter": 0.0, "n_iter": 10}, seed=0)


def test_gls_step_recovers_exact_factors():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, (30, 10))
    prior = rng.uniform(5, 10, 10)
    u_true = rng.uniform(0.5, 1.5, 10)
    counts = A @ (prior * u_true)
    u = gls_step(A, counts, prior, reg=1e-8)
    assert np.allclose(u, u_true, atol=1e-3)


def test_strong_regularisation_returns_prior():
    sc = _small_scenario()
    od = calibrate_od_gls(sc, sc.days[0], sc.sensors, n_outer=1, reg=1e6)
    assert np.allclose(od, sc.od_prior, rtol=1e-3)


def test_gls_reduces_count_loss():
    sc = _small_scenario()
    day = sc.days[0]
    before = count_loss(sc.sim(sc.od_prior, seed=0), day.counts, sc.sensors)
    od = calibrate_od_gls(sc, day, sc.sensors, n_outer=3)
    after = count_loss(sc.sim(od, seed=0), day.counts, sc.sensors)
    assert after < before
    assert np.all(od >= 0)
    assert sc.sim.n_calls == 3 + 2
