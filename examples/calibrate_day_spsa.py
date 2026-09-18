"""Baseline: SPSA on the OD cells of one day. Reports fit, holdout and OD recovery."""

import numpy as np

from odtoy.calibrate import calibrate_od_spsa
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
    report("spsa", sc, day, od_est)
    print(f"\nsimulator calls: {n_calls}   loss estimate first/last 20 iters: "
          f"{history[:20].mean():.4f} -> {history[-20:].mean():.4f}")


if __name__ == "__main__":
    main()
