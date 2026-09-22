"""Compare every method across several scenario seeds.

Each seed draws a different network, prior and set of days, so the
medians below say more than any single run. Costs are split into what a
method pays once (training) and what it pays for every new day.
"""

import time

import numpy as np

from odtoy.experiments import METRICS, run_scenario, summarise

SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    t0 = time.perf_counter()
    runs = []
    for seed in SEEDS:
        runs.append(run_scenario(seed))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)")

    stats = summarise(runs)
    header = "method     " + "".join(f"{m:>22}" for m in METRICS) + "   sims_setup  sims/day"
    print("\n" + header)
    print("-" * len(header))
    for method, s in stats.items():
        cells = "".join(
            f"{s['median'][i]:>12.3f} [{s['q25'][i]:.3f},{s['q75'][i]:.3f}]".rjust(22)
            for i in range(len(METRICS))
        )
        print(f"{method:<11}{cells}   {s['sims_setup']:>10} {s['sims_per_day']:>9.0f}")
    print(f"\n{len(SEEDS)} seeds, median [q25, q75], total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
