"""Figures for the method comparison.

Everything is plotted against the number of simulator calls, not against
iterations: simulations are what a real traffic model actually costs, and
methods that spend them differently are only comparable on that axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt
import numpy as np


def smooth(y: np.ndarray, window: int = 21) -> np.ndarray:
    """Running median, to read a trend through the SPSA noise."""
    if window <= 1 or window > y.size:
        return np.asarray(y, dtype=float)
    pad = window // 2
    padded = np.pad(np.asarray(y, dtype=float), pad, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(y.size)])


def plot_convergence(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    window: int = 21,
    title: str | None = None,
) -> Path:
    """Loss against cumulative simulations, one line per method.

    ``curves`` maps a label to ``(simulations, loss)`` of equal length.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if title:
        ax.set_title(title, fontsize=10)
    for label, (sims, loss) in curves.items():
        ax.plot(sims, smooth(np.asarray(loss), window), label=label, linewidth=1.6)
    ax.set_xlabel("simulator calls")
    ax.set_ylabel("count loss (running median)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def cumulative_cost(setup: float, per_day: float, n_days: np.ndarray) -> np.ndarray:
    """Simulations spent after ``n_days`` days: a fixed setup plus a per-day cost."""
    return setup + per_day * np.asarray(n_days, dtype=float)


def plot_break_even(costs: dict[str, tuple[float, float]], max_days: int, path: str | Path) -> Path:
    """Cumulative simulation cost against the number of days processed.

    ``costs`` maps a label to ``(setup, per_day)``. Where two lines cross
    is the point past which paying a training cost once is cheaper.
    """
    days = np.arange(0, max_days + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for label, (setup, per_day) in costs.items():
        ax.plot(days, cumulative_cost(setup, per_day, days), label=label, linewidth=1.6)
    ax.set_xlabel("days processed")
    ax.set_ylabel("cumulative simulator calls")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
