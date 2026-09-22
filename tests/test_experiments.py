import numpy as np

from odtoy.experiments import METRICS, Result, evaluate, run_scenario, summarise
from odtoy.scenario import make_scenario


def test_evaluate_on_the_truth_is_near_zero():
    sc = make_scenario(rows=4, cols=4, n_days=2, n_sensors=10, count_noise_sd=0.0,
                       sim_kwargs={"cap_jitter": 0.0, "n_iter": 10}, seed=0)
    stats = evaluate(sc, sc.days, lambda d: d.od_true, sc.sensors[:6], sc.sensors[6:])
    assert stats.shape == (len(METRICS),)
    assert np.allclose(stats, 0.0, atol=1e-8)


def test_run_scenario_returns_every_method():
    runs = run_scenario(seed=0, n_train=2, n_test=1, n_sensors=10, n_holdout=3,
                        n_iter_day=5, n_iter_net=5)
    assert [r.method for r in runs] == ["prior", "gls", "cells", "zones", "amortised"]
    assert all(r.metrics.shape == (len(METRICS),) for r in runs)
    prior = [r for r in runs if r.method == "prior"][0]
    assert prior.sims_setup == 0 and prior.sims_per_day == 0
    amortised = [r for r in runs if r.method == "amortised"][0]
    # 2 perturbation sides x 2 training days x 5 iterations
    assert amortised.sims_setup == 2 * 2 * 5 and amortised.sims_per_day == 1.0


def test_summarise_quantiles():
    runs = [
        [Result("a", np.array([1.0, 2.0, 3.0, 4.0]), 0, 1.0)],
        [Result("a", np.array([3.0, 4.0, 5.0, 6.0]), 0, 1.0)],
        [Result("a", np.array([5.0, 6.0, 7.0, 8.0]), 0, 1.0)],
    ]
    s = summarise(runs)["a"]
    assert np.allclose(s["median"], [3.0, 4.0, 5.0, 6.0])
    assert np.allclose(s["q25"], [2.0, 3.0, 4.0, 5.0])
    assert s["sims_per_day"] == 1.0
