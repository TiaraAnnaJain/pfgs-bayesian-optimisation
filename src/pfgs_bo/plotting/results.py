"""Figures summarising a run (per-run histories) and comparing runs across seeds."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def save_fig(name: str, folder: str, root: Path) -> None:
    """
    Saves figures to specified folder.

    Parameters:
        name (str): name of figure.
        folder (str): path to folder where to save figures.
        root (Path): path to root.
    """
    out_dir = Path(root) / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")


def plot_single_metric(
    x_data: NDArray,
    y_data: NDArray,
    y_label: str,
    fig_name: str,
    root: Path,
) -> None:
    """
    Plots single curve with mean and standard deviation band.

    Parameters:
        x_data (NDArray): x data to plot.
        y_data (NDArray): y data to plot.
        y_label (str): y label to plot.
        fig_name (str): figure name.
        root (Path): path to root.
    """
    mean = np.nanmean(y_data, axis=0)
    std = np.nanstd(y_data, axis=0)

    plt.plot(x_data, mean, label=f"{fig_name}")
    plt.fill_between(x_data, mean - std, mean + std, alpha=0.3, label="±1 std")

    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.legend()

    save_fig(name=fig_name, folder="figures", root=root)
    plt.show()


def plot_multiple_metrics(
    x_data: NDArray, curves: dict, y_label: str, fig_name: str, root: Path
) -> None:
    """
    Plots multiple curves with mean and standard deviation band.

    Parameters:
        x_data (NDArray): x data to plot.
        curves (Dict): dictionary of curves to plot.
        y_label (str): y label to plot.
        fig_name (str): figure name.
        root (Path): path to root.
    """
    for label, y_data in curves.items():
        max_length = max(len(run) for run in y_data)
        padded = np.full((len(y_data), max_length), np.nan)

        for i, run in enumerate(y_data):
            padded[i, : len(run)] = run

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)

        x = np.asarray(x_data)
        if len(x) < max_length:
            x = np.arange(max_length)

        plt.plot(x[:max_length], mean, label=label)
        plt.fill_between(
            x[:max_length],
            mean - std,
            mean + std,
            alpha=0.3,
        )

    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.legend()

    save_fig(name=fig_name, folder="figures", root=root)
    plt.show()


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
