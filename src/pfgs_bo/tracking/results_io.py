"""Saving and loading experiment results (CSV) and beta ranges (JSON)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def to_csv(results: dict, folder: str | Path, seed: int) -> None:
    """
    Takes results from optimisation scripts and writes them into csv file.

    Parameters:
         results (Dict): results dictionary
         folder (str | Path): folder path
         seed (int): LHS seed
    """
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"seed_{seed}.csv", index=False)


def from_csv(folder: str, x_column: str, y_column: str) -> tuple[NDArray, NDArray]:
    """
    Loads results from csv file to plot. Pads empty runs with NaN's.

    Parameters:
        folder (str): folder path.
        x_column (str): x-axis column name.
        y_column (str): y-axis column name.

    Returns:
        x (NDArray[np.float64]): x-axis array.
        y_all (NDArray[np.float64]): y-axis array.
    """
    files = Path(folder).glob("seed_*.csv")

    runs = []
    x_runs = []

    for file in files:
        df = pd.read_csv(file)

        x = df[x_column].to_numpy()
        y = df[y_column].to_numpy()

        x_runs.append(x)
        runs.append(y)

    max_len = max(len(run) for run in runs)
    y_all = np.full((len(runs), max_len), np.nan)

    for i, run in enumerate(runs):
        y_all[i, : len(run)] = run

    x = np.arange(max_len)

    return x, y_all


def to_json(
    beta_min: list[float], beta_max: list[float], json_file: str | Path
) -> None:
    """
    Writes the beta interval found at each PFGS iteration to a JSON file.

    Parameters:
        beta_min (list[float]): lower bound of the beta interval per iteration.
        beta_max (list[float]): upper bound of the beta interval per iteration.
        json_file (str | Path): file to write; parent folders are created.
    """
    beta_range = {
        "beta_min": beta_min,
        "beta_max": beta_max,
    }

    json_file = Path(json_file)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w") as file:
        json.dump(beta_range, file)


def from_json(json_file: str | Path) -> dict[str, list[float]]:
    """
    Reads beta intervals written by ``to_json``.

    Parameters:
        json_file (str | Path): file to read.

    Returns:
        beta_range (dict[str, list[float]]): "beta_min" and "beta_max" lists.
    """
    with open(json_file) as file:
        beta_range = json.load(file)

    return beta_range
