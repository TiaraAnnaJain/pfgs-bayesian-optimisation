"""Saving and loading run results (CSV) and PFGS beta intervals (JSON)."""

import json
from pathlib import Path

import pandas as pd


def to_csv(results: dict, csv_file: str | Path) -> None:
    """
    Writes the per-iteration results of a run to a CSV file.

    Parameters:
        results (dict): columns to write, e.g. ``RunHistory.to_dict()``.
        csv_file (str | Path): file to write; parent folders are created.
    """
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(csv_file, index=False)


def load_runs(folders: str | Path | list[str | Path]) -> pd.DataFrame:
    """
    Loads every ``train_seed_<n>.csv`` in one or more run folders into one table.

    Parameters:
        folders (str | Path | list[str | Path]): run folder(s), e.g. a
            ``run_dir``. Several folders are pooled, e.g. all "max_overlap"
            beta schedules.

    Returns:
        runs (pd.DataFrame): the CSV columns plus "train_seed" and "folder";
            empty if no CSVs were found.
    """
    if isinstance(folders, (str, Path)):
        folders = [folders]

    frames = []
    for folder in map(Path, folders):
        for csv_file in sorted(folder.glob("train_seed_*.csv")):
            frame = pd.read_csv(csv_file)
            frame["train_seed"] = int(csv_file.stem.removeprefix("train_seed_"))
            frame["folder"] = str(folder)
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
