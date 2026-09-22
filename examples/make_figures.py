"""Produce the two figures in figures/: convergence and amortisation break-even."""

import numpy as np

from odtoy.amortised import baseline_flows, train_amortised
from odtoy.calibrate import calibrate_od_spsa
from odtoy.net import MLP
from odtoy.plots import plot_break_even, plot_convergence
from odtoy.scenario import make_scenario
from odtoy.sensors import split_sensors

N_TRAIN, N_TEST = 12, 4
N_ITER_DAY, N_ITER_NET, BATCH = 400, 600, 4


def main() -> None:
    sc = make_scenario(rows=4, cols=4, n_days=N_TRAIN + N_TEST, n_sensors=24, seed=0)
    fit, _ = split_sensors(sc.sensors, 6, np.random.default_rng(0))
    train, test = sc.days[:N_TRAIN], sc.days[N_TRAIN:]
    day = test[0]

    curves = {}
    for label, kwargs in (("per cell", {"kind": "cells"}), ("per zone", {"kind": "zones", "a": 0.3})):
        _, history = calibrate_od_spsa(sc, day, fit, n_iter=N_ITER_DAY, seed=0, **kwargs)
        curves[f"one day, {label}"] = (2 * np.arange(1, N_ITER_DAY + 1), history)

    net = MLP((len(fit), 6, 2 * sc.net.n_zones + 1))
    _, history = train_amortised(sc, net, train, fit, baseline_flows(sc, train),
                                 n_iter=N_ITER_NET, batch_size=BATCH, seed=0)
    curves[f"network, {N_TRAIN} days"] = (2 * BATCH * np.arange(1, N_ITER_NET + 1), history)

    p1 = plot_convergence(
        curves,
        "figures/convergence.png",
        title="levels are not comparable: the per-day runs fit one day,\nthe network fits all training days at once",
    )

    train_cost = 2 * BATCH * N_ITER_NET
    p2 = plot_break_even(
        {"calibrate every day": (0, 2 * N_ITER_DAY), "train once, then predict": (train_cost, 1)},
        max_days=40,
        path="figures/break_even.png",
    )
    print(f"wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
