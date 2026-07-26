"""Adds and loads results from csv file."""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def to_csv(results: Dict, folder: str, seed: int) -> None:
    """
    Takes results from optimisation scripts and writes them into csv file.

    Parameters:
         results (Dict): results dictionary
         folder (str): folder path
         seed (int): LHS seed
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    filepath = os.path.join(folder, f"seed_{seed}.csv")
    df.to_csv(filepath, index=False)


def from_csv(folder: str, x_column: str, y_column: str) -> Tuple[NDArray, NDArray]:
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
    print(Path(folder).absolute())

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
