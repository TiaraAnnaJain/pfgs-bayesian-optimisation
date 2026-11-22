"""Figures for a single run (histories) and for comparing runs across seeds."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def mean_and_std(
    runs: pd.DataFrame, column: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Averages a metric over runs at each iteration.

    Runs that stopped early (PFGS and constrained BO can) drop out of later
    iterations, so late means are over fewer runs.

    Parameters:
        runs (pd.DataFrame): output of ``load_runs``.
        column (str): metric column, e.g. "design_space_rrmse".

    Returns:
        iterations (NDArray[np.float64]): iteration numbers.
        mean (NDArray[np.float64]): mean over runs.
        std (NDArray[np.float64]): standard deviation over runs (0 for one run).
    """
    grouped = runs.groupby("iteration")[column]
    mean = grouped.mean()
    std = grouped.std(ddof=0)
    return (
        mean.index.to_numpy(dtype=np.float64),
        mean.to_numpy(dtype=np.float64),
        std.to_numpy(dtype=np.float64),
    )


def plot_comparison(
    curves: dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    title: str,
    nonnegative: bool = False,
) -> plt.Figure:
    """
    Plots the mean of a metric over runs, with a +/- 1 std band, for each curve.

    Parameters:
        curves (dict[str, pd.DataFrame]): legend label -> runs from ``load_runs``.
        column (str): metric column to plot.
        ylabel (str): y-axis label.
        title (str): figure title.
        nonnegative (bool): clip the band at 0, for metrics that can't be negative.

    Returns:
        fig (plt.Figure): comparison figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, runs in curves.items():
        iterations, mean, std = mean_and_std(runs, column)
        n_runs = runs["train_seed"].nunique()
        (line,) = ax.plot(iterations, mean, label=f"{label} (n={n_runs})", lw=2)
        lower = np.maximum(mean - std, 0) if nonnegative else mean - std
        ax.fill_between(
            iterations, lower, mean + std, alpha=0.18, color=line.get_color()
        )

    ax.set(title=title, xlabel="Iteration", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_rrmse_history(
    entire_space_rrmse: Sequence[float], design_space_rrmse: Sequence[float]
) -> plt.Figure:
    """
    Plots the entire-space and design-space RRMSE of a single run over iterations.

    Parameters:
        entire_space_rrmse (Sequence[float]): RRMSE on the entire-space test set.
        design_space_rrmse (Sequence[float]): RRMSE on the design-space test set.

    Returns:
        fig (plt.Figure): RRMSE figure.
    """
    x = range(len(entire_space_rrmse))

    fig, ax = plt.subplots()
    ax.plot(x, entire_space_rrmse, label="RRMSE (Entire space)")
    ax.plot(x, design_space_rrmse, label="RRMSE (Design space)")
    ax.legend()
    ax.set(title="RRMSE over iterations", xlabel="iterations", ylabel="RRMSE")

    return fig


def plot_train_max_history(train_data_max: Sequence[float]) -> plt.Figure:
    """
    Plots the maximum observed objective value of a single run over iterations.

    Parameters:
        train_data_max (Sequence[float]): maximum training output at each iteration.

    Returns:
        fig (plt.Figure): training data maximum figure.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(train_data_max)), train_data_max)
    ax.set(
        title="Maximum value of train data",
        xlabel="iterations",
        ylabel="training data max",
    )

    return fig


def plot_beta_ranges(
    beta_lower: Sequence[float], beta_upper: Sequence[float]
) -> plt.Figure:
    """
    Plots the feasible beta interval found by PFGS at each iteration.

    Parameters:
        beta_lower (Sequence[float]): lower bound of the beta interval per iteration.
        beta_upper (Sequence[float]): upper bound of the beta interval per iteration.

    Returns:
        fig (plt.Figure): beta interval figure.
    """
    fig, ax = plt.subplots()
    for x_i, (lower, upper) in enumerate(zip(beta_lower, beta_upper, strict=True)):
        ax.plot([x_i, x_i], [lower, upper], color="blue")
    ax.set(title="Feasible beta ranges", xlabel="iterations", ylabel="beta")

    return fig


def plot_beta_schedule(
    beta_values: Sequence[float] | NDArray[np.float64],
) -> plt.Figure:
    """
    Plots the beta value used at each iteration of PFGS-guided BO.

    Parameters:
        beta_values (Sequence[float] | NDArray[np.float64]): beta value per iteration.

    Returns:
        fig (plt.Figure): beta schedule figure.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(beta_values)), beta_values)
    ax.set(title="Value of beta", xlabel="iterations", ylabel="beta")

    return fig
