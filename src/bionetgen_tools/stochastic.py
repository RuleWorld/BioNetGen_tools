"""Ensembles of stochastic trajectories.

``BNGLSimulator.run_replicates`` returns an ``Ensemble``: every replicate of
every observable on a common time grid, with summary statistics and plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Ensemble:
    """Replicate trajectories on a common time grid.

    Attributes:
        time: (n_times,)
        data: observable name -> array of shape (n_replicates, n_times)
        seeds: the seed of each replicate (``simulate(seed=s)`` repeats it)
        method: the simulation method
    """

    time: np.ndarray
    data: Dict[str, np.ndarray]
    seeds: List[Optional[int]] = field(default_factory=list)
    method: str = "ssa"

    @classmethod
    def from_results(cls, results: Sequence, method: str = "ssa") -> "Ensemble":
        if not isinstance(results, (list, tuple)):
            results = [results]
        names = list(results[0].observable_names)
        time = np.asarray(results[0].time, dtype=float)
        data = {
            n: np.vstack([np.asarray(r.observables[n], dtype=float) for r in results])
            for n in names
        }
        return cls(time=time, data=data, seeds=[r.seed for r in results], method=method)

    @property
    def n_replicates(self) -> int:
        return next(iter(self.data.values())).shape[0]

    @property
    def observables(self) -> List[str]:
        return list(self.data)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    def mean(self, name: str) -> np.ndarray:
        return self.data[name].mean(axis=0)

    def std(self, name: str) -> np.ndarray:
        return self.data[name].std(axis=0, ddof=1) if self.n_replicates > 1 else np.zeros_like(self.time)

    def quantiles(self, name: str, q: Sequence[float] = (0.05, 0.5, 0.95)) -> np.ndarray:
        return np.quantile(self.data[name], q, axis=0)

    def fano(self, name: str) -> np.ndarray:
        """Variance / mean over time (1 for a Poisson distribution)."""
        m = self.mean(name)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.std(name) ** 2 / m

    def at_time(self, name: str, t: float) -> np.ndarray:
        """Values of one observable across replicates at the output time nearest t."""
        return self.data[name][:, int(np.argmin(np.abs(self.time - t)))]

    def to_dataframe(self):
        """Long-format table: replicate, time, one column per observable."""
        import pandas as pd

        frames = []
        for i in range(self.n_replicates):
            df = pd.DataFrame({n: v[i] for n, v in self.data.items()})
            df.insert(0, "time", self.time)
            df.insert(0, "replicate", i)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)


def plot_ensemble(
    ens: Ensemble,
    observables: Optional[Sequence[str]] = None,
    n_show: int = 5,
    band: Tuple[float, float] = (0.05, 0.95),
    ode=None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """One panel per observable: a few trajectories, the mean, a quantile band,
    and optionally the ODE solution (a ``SimResult``) for comparison."""
    observables = list(observables or ens.observables)
    figsize = figsize or (10, 3 * len(observables))
    fig, axes = plt.subplots(len(observables), 1, figsize=figsize, squeeze=False, sharex=True)
    for ax, name in zip(axes[:, 0], observables):
        for i in range(min(n_show, ens.n_replicates)):
            ax.plot(ens.time, ens[name][i], lw=0.8, alpha=0.5, color="tab:gray")
        lo, hi = ens.quantiles(name, band)
        ax.fill_between(ens.time, lo, hi, alpha=0.2, color="tab:blue",
                        label=f"{int(band[0]*100)}–{int(band[1]*100)}% of {ens.n_replicates} runs")
        ax.plot(ens.time, ens.mean(name), color="tab:blue", lw=2, label="mean")
        if ode is not None and name in ode:
            ax.plot(ode["time"], ode[name], "k--", lw=1.5, label="ODE")
        ax.set_ylabel(name)
        ax.legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("time")
    plt.tight_layout()
    return fig


def plot_distribution(
    ens: Ensemble,
    observable: str,
    t: Optional[float] = None,
    pool_after: Optional[float] = None,
    ax=None,
    integer_bins: bool = True,
    **hist_kwargs,
):
    """Histogram of an observable across replicates.

    Args:
        t: the time to take values at (default: the last output time)
        pool_after: instead of one time, pool every output time >= this value
            across all replicates (a stationary distribution, if the system
            has settled)
        integer_bins: center bins on integers (counts)
    """
    if pool_after is not None:
        values = ens[observable][:, ens.time >= pool_after].ravel()
        label = f"t ≥ {pool_after:g}"
    else:
        t = ens.time[-1] if t is None else t
        values = ens.at_time(observable, t)
        label = f"t = {t:g}"
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if integer_bins:
        lo, hi = int(np.floor(np.nanmin(values))), int(np.ceil(np.nanmax(values)))
        step = max(1, (hi - lo + 1) // 100)
        edges = np.arange(lo - 0.5, hi + 1.5, step)
        hist_kwargs.setdefault("bins", edges)
    ax.hist(values, alpha=0.7, edgecolor="black", **hist_kwargs)
    ax.set_xlabel(f"{observable} (counts)")
    ax.set_ylabel("frequency")
    ax.set_title(f"{observable} distribution, {label} ({ens.method}, {ens.n_replicates} runs)")
    return ax
