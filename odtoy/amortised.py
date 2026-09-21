"""Amortised correction: one network maps a day's counts to OD factors.

Instead of re-running an optimisation for every day, a single network is
trained across days. It reads how far the measured counts are from a
pre-simulated baseline and outputs zone factors that correct the prior.
The simulator sits between the network's output and the loss, so the
weights are tuned with SPSA rather than backpropagation.

At prediction time a new day costs one network evaluation, not a
calibration run.
"""

from __future__ import annotations

import numpy as np

from .calibrate import od_from_zone_factors
from .metrics import count_loss
from .net import MLP
from .scenario import Day, Scenario


def baseline_flows(sc: Scenario, days: list[Day]) -> list[np.ndarray]:
    """Flows obtained by running the prior OD under each day's conditions.

    This is the "pre-simulated baseline": one simulation per day, paid
    once, that turns raw counts into an interpretable residual.
    """
    return [sc.sim(sc.od_prior, seed=d.seed) for d in days]


def day_features(day: Day, base: np.ndarray, sc: Scenario, sensors_fit: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Relative gap between measured counts and the baseline, at the fit sensors."""
    in_fit = np.isin(sc.sensors, sensors_fit)
    base_fit = np.maximum(base[sensors_fit], floor)
    return day.counts[in_fit] / base_fit - 1.0


def predict_od(sc: Scenario, net: MLP, theta: np.ndarray, features: np.ndarray, out_scale: float = 0.5) -> np.ndarray:
    """Network output -> bounded zone log-factors -> corrected OD.

    ``tanh`` keeps every factor within ``exp(±out_scale)`` of the prior,
    which stops the network from inventing demand where no sensor
    constrains it.
    """
    raw = net.forward(features, theta)
    return od_from_zone_factors(sc.od_prior, out_scale * np.tanh(raw))


def make_amortised_loss(
    sc: Scenario,
    net: MLP,
    days: list[Day],
    sensors_fit: np.ndarray,
    baselines: list[np.ndarray],
    batch_size: int | None = None,
    out_scale: float = 0.5,
    reg: float = 0.0,
):
    """Build ``loss(theta, seed)`` averaged over a batch of days.

    The seed picks the batch as well as driving the simulator, so both
    sides of an SPSA perturbation see exactly the same days: the common
    random numbers cover the sampling too.
    """
    in_fit = np.isin(sc.sensors, sensors_fit)
    feats = [day_features(d, b, sc, sensors_fit) for d, b in zip(days, baselines)]

    def loss(theta: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        idx = range(len(days)) if batch_size is None else rng.choice(len(days), batch_size, replace=False)
        total = 0.0
        for i in idx:
            flows = sc.sim(predict_od(sc, net, theta, feats[i], out_scale), seed=days[i].seed)
            total += count_loss(flows, days[i].counts[in_fit], sensors_fit)
        n = len(days) if batch_size is None else batch_size
        return total / n + reg * float(np.mean(theta**2))

    return loss


def train_amortised(
    sc: Scenario,
    net: MLP,
    days: list[Day],
    sensors_fit: np.ndarray,
    baselines: list[np.ndarray],
    n_iter: int = 600,
    a: float = 1.0,
    c: float = 0.05,
    batch_size: int | None = None,
    out_scale: float = 0.5,
    reg: float = 0.0,
    seed: int = 0,
    **spsa_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Tune the network weights with SPSA. Returns (theta, loss history)."""
    from .spsa import spsa

    loss = make_amortised_loss(sc, net, days, sensors_fit, baselines,
                               batch_size=batch_size, out_scale=out_scale, reg=reg)
    theta0 = net.init_params(np.random.default_rng(seed))
    return spsa(loss, theta0, n_iter, a=a, c=c, seed=seed, **spsa_kwargs)
