"""Build a scenario and show how far the prior OD is from each day's truth."""

import numpy as np

from odtoy.scenario import make_scenario


def main() -> None:
    sc = make_scenario(rows=6, cols=6, n_days=5, n_sensors=40, seed=0)
    print(f"zones={sc.net.n_zones} links={sc.net.n_links} sensors={sc.n_sensors} days={len(sc.days)}")
    print(f"prior total trips: {sc.od_prior.sum():.0f}  (true mean: {sc.od_mean.sum():.0f})")
    print()
    print("day  trips_true  od_rel_err  count_rel_err(prior)")
    for i, d in enumerate(sc.days):
        mask = d.od_true > 0
        od_err = np.abs(sc.od_prior[mask] / d.od_true[mask] - 1.0).mean()
        flows_prior = sc.sim(sc.od_prior, seed=d.seed)
        cnt_err = np.abs(flows_prior[sc.sensors] / d.counts - 1.0).mean()
        print(f"{i:>3}  {d.od_true.sum():>10.0f}  {od_err:>10.3f}  {cnt_err:>20.3f}")
    print(f"\nsimulator calls used: {sc.sim.n_calls}")


if __name__ == "__main__":
    main()
