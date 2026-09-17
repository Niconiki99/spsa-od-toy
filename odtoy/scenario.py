"""Bundle everything a calibration experiment needs into one object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .days import gravity_od, make_prior, sample_day_od
from .network import Network, grid_network
from .sensors import choose_sensors, measure
from .simulator import Simulator


@dataclass(frozen=True)
class Day:
    """One day of ground truth and what a calibration method gets to see."""

    seed: int                 # simulator seed for this day
    od_true: np.ndarray       # (n_zones, n_zones), hidden from the methods
    flows_true: np.ndarray    # (n_links,), hidden, for evaluation
    counts: np.ndarray        # (n_sensors,), observed


@dataclass
class Scenario:
    net: Network
    sim: Simulator
    sensors: np.ndarray
    od_mean: np.ndarray       # hidden: the true long-run mean
    od_prior: np.ndarray      # observed: the biased starting matrix
    days: list[Day] = field(default_factory=list)

    @property
    def n_sensors(self) -> int:
        return int(self.sensors.shape[0])


def make_scenario(
    rows: int = 6,
    cols: int = 6,
    n_days: int = 20,
    n_sensors: int = 40,
    total_trips: float = 2000.0,
    cap_mean: float = 80.0,
    count_noise_sd: float = 0.05,
    seed: int = 0,
    sim_kwargs: dict | None = None,
    day_kwargs: dict | None = None,
    prior_kwargs: dict | None = None,
) -> Scenario:
    """Generate a full synthetic scenario from a single seed.

    Ground-truth flows come from the same simulator the methods will use,
    so the only sources of mismatch are demand and measurement noise.
    """
    rng = np.random.default_rng(seed)
    net = grid_network(rows, cols, cap_mean=cap_mean, seed=int(rng.integers(1 << 31)))
    sim = Simulator(net, **(sim_kwargs or {}))
    sensors = choose_sensors(net, n_sensors, rng)

    od_mean = gravity_od(net, cols, total_trips, seed=int(rng.integers(1 << 31)))
    od_prior = make_prior(od_mean, rng, **(prior_kwargs or {}))

    days = []
    for _ in range(n_days):
        day_seed = int(rng.integers(1 << 31))
        od_true = sample_day_od(od_mean, rng, **(day_kwargs or {}))
        flows_true = sim(od_true, seed=day_seed)
        counts = measure(flows_true, sensors, rng, noise_sd=count_noise_sd)
        days.append(Day(day_seed, od_true, flows_true, counts))

    sim.reset_counter()  # ground-truth generation is free
    return Scenario(net, sim, sensors, od_mean, od_prior, days)
