"""Adds results to csv file."""

import os
from pathlib import Path
from typing import Dict

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
