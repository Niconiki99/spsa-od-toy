"""Baselines on one day: SPSA (per cell, per zone) vs linearised GLS.

Reports fit loss, held-out loss and OD recovery, so that fitting the
sensors can be told apart from estimating the demand.
"""

import numpy as np

from odtoy.calibrate import calibrate_od_spsa
from odtoy.gls import calibrate_od_gls
from odtoy.metrics import count_loss, geh_pass_rate, od_rel_error, od_total_error
from odtoy.scenario import make_scenario
from odtoy.sensors import split_sensors


def report(label, sc, day, od, seed=0):
    flows = sc.sim(od, seed=seed)
    fit, hold = sc.sensors_fit, sc.sensors_hold
    in_fit = np.isin(sc.sensors, fit)
    print(
        f"{label:<8} loss_fit={count_loss(flows, day.counts[in_fit], fit):.4f}  "
        f"loss_hold={count_loss(flows, day.counts[~in_fit], hold):.4f}  "
        f"geh_pass={geh_pass_rate(flows, day.counts, sc.sensors):.2f}  "
        f"od_err={od_rel_error(od, day.od_true):.3f}  "
        f"od_total={od_total_error(od, day.od_true):+.3f}"
    )


def main() -> None:
    sc = make_scenario(rows=6, cols=6, n_days=1, n_sensors=40, seed=0)
    sc.sensors_fit, sc.sensors_hold = split_sensors(sc.sensors, 10, np.random.default_rng(0))
    day = sc.days[0]

    report("prior", sc, day, sc.od_prior)
    sc.sim.reset_counter()
    od_est, history = calibrate_od_spsa(sc, day, sc.sensors_fit, n_iter=300, a=1.0, c=0.05, reg=0.01, seed=0)
    n_calls = sc.sim.n_calls
    report("cells", sc, day, od_est)
    print(f"         simulator calls: {n_calls}   loss estimate first/last 20 iters: "
          f"{history[:20].mean():.4f} -> {history[-20:].mean():.4f}")

    sc.sim.reset_counter()
    od_zone, history = calibrate_od_spsa(sc, day, sc.sensors_fit, n_iter=300, a=0.3, c=0.05, reg=0.01,
                                         kind="zones", seed=0)
    n_calls = sc.sim.n_calls
    report("zones", sc, day, od_zone)
    print(f"         simulator calls: {n_calls}   loss estimate first/last 20 iters: "
          f"{history[:20].mean():.4f} -> {history[-20:].mean():.4f}")

    sc.sim.reset_counter()
    od_gls = calibrate_od_gls(sc, day, sc.sensors_fit, n_outer=3, reg=1e-3, seed=0)
    n_calls = sc.sim.n_calls
    report("gls", sc, day, od_gls)
    print(f"         simulator calls: {n_calls}")


if __name__ == "__main__":
    main()
