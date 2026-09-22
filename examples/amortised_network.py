"""Amortised correction vs per-day calibration, on days the network never saw.

The network is trained once on the training days and then applied to the
test days with a single forward pass. The per-day SPSA calibration is
re-run from scratch on every test day, so it should be the stronger of
the two: the question is how much accuracy the amortised version gives
up, and how much simulation it saves.
"""

import time

import numpy as np

from odtoy.amortised import baseline_flows, day_features, predict_od, train_amortised
from odtoy.calibrate import calibrate_od_spsa
from odtoy.experiments import evaluate
from odtoy.net import MLP
from odtoy.scenario import make_scenario
from odtoy.sensors import split_sensors

N_TRAIN, N_TEST = 12, 4


def show(label, stats, calls):
    print(f"{label:<12} loss_fit={stats[0]:.4f}  loss_hold={stats[1]:.4f}  "
          f"od_err={stats[2]:.3f}  od_total={stats[3]:+.3f}  sims/day={calls}")


def main() -> None:
    sc = make_scenario(rows=4, cols=4, n_days=N_TRAIN + N_TEST, n_sensors=24, seed=0)
    fit, hold = split_sensors(sc.sensors, 6, np.random.default_rng(0))
    train, test = sc.days[:N_TRAIN], sc.days[N_TRAIN:]

    base_train = baseline_flows(sc, train)
    base_test = {d.seed: b for d, b in zip(test, baseline_flows(sc, test))}

    print(f"zones={sc.net.n_zones} sensors={len(fit)}+{len(hold)} "
          f"train_days={N_TRAIN} test_days={N_TEST}\n")
    show("prior", evaluate(sc, test, lambda d: sc.od_prior, fit, hold), 0)

    # per-day calibration: a fresh SPSA run for every test day
    sc.sim.reset_counter()
    per_day = {d.seed: calibrate_od_spsa(sc, d, fit, n_iter=200, a=0.3, kind="zones", seed=0)[0] for d in test}
    show("per-day", evaluate(sc, test, lambda d: per_day[d.seed], fit, hold),
         sc.sim.n_calls // N_TEST)

    # amortised: train once, then one forward pass per test day
    net = MLP((len(fit), 6, 2 * sc.net.n_zones + 1))
    sc.sim.reset_counter()
    t0 = time.perf_counter()
    theta, history = train_amortised(sc, net, train, fit, base_train, batch_size=4, seed=0)
    train_calls = sc.sim.n_calls
    show("amortised", evaluate(sc, test, lambda d: predict_od(sc, net, theta, day_features(d, base_test[d.seed], sc, fit)),
                               fit, hold), 1)
    print(f"\nnetwork: {net.n_params} weights, trained in {time.perf_counter()-t0:.0f}s "
          f"with {train_calls} simulations ({history[:20].mean():.4f} -> {history[-20:].mean():.4f})")


if __name__ == "__main__":
    main()
