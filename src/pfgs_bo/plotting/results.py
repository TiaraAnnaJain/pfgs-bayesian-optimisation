"""Contains functions used to plot final results and metrics"""

from pathlib import Path
from typing import Dict

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
    folder = Path(root / folder)
    folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(folder / f"{name}.pdf", dpi=300, bbox_inches="tight")


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
    x_data: NDArray, curves: Dict, y_label: str, fig_name: str, root: Path
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
