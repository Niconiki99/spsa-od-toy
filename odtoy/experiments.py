"""Run every calibration method on the same scenarios and aggregate results.

Single runs are noisy: the scenario seed decides the network, the prior
and the days, and SPSA adds its own randomness on top. Every number
worth reporting is a median over several seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amortised import baseline_flows, day_features, predict_od, train_amortised
from .calibrate import calibrate_od_spsa
from .gls import calibrate_od_gls
from .metrics import count_loss, od_rel_error, od_total_error
from .net import MLP
from .scenario import Scenario
from .sensors import split_sensors

METRICS = ("loss_fit", "loss_hold", "od_err", "od_total")


@dataclass(frozen=True)
class Result:
    """Average metrics of one method on one scenario, plus its simulation cost."""

    method: str
    metrics: np.ndarray      # one entry per name in METRICS
    sims_setup: int          # simulations paid once (training)
    sims_per_day: float      # simulations paid for every new day


def evaluate(sc: Scenario, days, od_of_day, sensors_fit, sensors_hold) -> np.ndarray:
    """Mean of METRICS over ``days``, where ``od_of_day(day)`` is the estimate."""
    in_fit = np.isin(sc.sensors, sensors_fit)
    rows = []
    for day in days:
        od = od_of_day(day)
        flows = sc.sim(od, seed=day.seed)
        rows.append((
            count_loss(flows, day.counts[in_fit], sensors_fit),
            count_loss(flows, day.counts[~in_fit], sensors_hold),
            od_rel_error(od, day.od_true),
            od_total_error(od, day.od_true),
        ))
    return np.mean(rows, axis=0)


def run_scenario(
    seed: int,
    rows: int = 4,
    cols: int = 4,
    n_train: int = 12,
    n_test: int = 4,
    n_sensors: int = 24,
    n_holdout: int = 6,
    n_iter_day: int = 200,
    n_iter_net: int = 600,
    hidden: int = 6,
) -> list[Result]:
    """Fit every method on one scenario and score it on the held-out days."""
    from .scenario import make_scenario

    sc = make_scenario(rows=rows, cols=cols, n_days=n_train + n_test, n_sensors=n_sensors, seed=seed)
    fit, hold = split_sensors(sc.sensors, n_holdout, np.random.default_rng(seed))
    train, test = sc.days[:n_train], sc.days[n_train:]
    results = []

    results.append(Result("prior", evaluate(sc, test, lambda d: sc.od_prior, fit, hold), 0, 0.0))

    for name, fn in (
        ("gls", lambda d: calibrate_od_gls(sc, d, fit, n_outer=3, seed=seed)),
        ("cells", lambda d: calibrate_od_spsa(sc, d, fit, n_iter=n_iter_day, kind="cells", seed=seed)[0]),
        ("zones", lambda d: calibrate_od_spsa(sc, d, fit, n_iter=n_iter_day, a=0.3, kind="zones", seed=seed)[0]),
    ):
        sc.sim.reset_counter()
        od_by_day = {d.seed: fn(d) for d in test}
        cost = sc.sim.n_calls / len(test)
        results.append(Result(name, evaluate(sc, test, lambda d: od_by_day[d.seed], fit, hold), 0, cost))

    net = MLP((len(fit), hidden, 2 * sc.net.n_zones + 1))
    base_train = baseline_flows(sc, train)
    base_test = {d.seed: b for d, b in zip(test, baseline_flows(sc, test))}
    sc.sim.reset_counter()
    theta, _ = train_amortised(sc, net, train, fit, base_train, n_iter=n_iter_net, batch_size=4, seed=seed)
    setup = sc.sim.n_calls
    results.append(Result(
        "amortised",
        evaluate(sc, test, lambda d: predict_od(sc, net, theta, day_features(d, base_test[d.seed], sc, fit)), fit, hold),
        setup,
        1.0,
    ))
    return results


def summarise(runs: list[list[Result]]) -> dict[str, dict[str, np.ndarray]]:
    """Median and interquartile range of each metric, per method.

    ``runs`` is one list of Results per scenario seed.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for method in [r.method for r in runs[0]]:
        rows = np.array([[r.metrics for r in run if r.method == method][0] for run in runs])
        ref = [r for r in runs[0] if r.method == method][0]
        out[method] = {
            "median": np.median(rows, axis=0),
            "q25": np.quantile(rows, 0.25, axis=0),
            "q75": np.quantile(rows, 0.75, axis=0),
            "sims_setup": ref.sims_setup,
            "sims_per_day": ref.sims_per_day,
        }
    return out
