"""
Makes the comparison figures and summary table from the run CSVs in results/.
"""

import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pfgs_bo.config import PROJECT_ROOT  # noqa: E402
from pfgs_bo.plotting.results import plot_comparison  # noqa: E402
from pfgs_bo.tracking.results_io import load_runs  # noqa: E402

METRICS = {
    "design_space_rrmse": "Design-space RRMSE",
    "train_data_max": "Best observed objective value",
}
SUMMARY_METRICS = ["design_space_rrmse", "entire_space_rrmse", "train_data_max"]

BO_UCB = "bo/UCB_beta1.96"
BO_EI = "bo/EI"
PFGS_UNCONSTRAINED = "pfgs/mean0_std0"

PFGS_BO_SCHEDULES = {
    "Minimum (dynamic)": "min",
    "Maximum (dynamic)": "max",
    "Midpoint (dynamic)": "mid_point",
    "Mean (fixed)": "mean",
    "Median (fixed)": "median",
    "Maximum overlap (fixed)": "max_overlap",
}


def thresholds_label(tag: str) -> str:
    """Turns a folder name like "mean1000_std50" into "µ=1000, σ=50"."""
    match = re.fullmatch(r"mean(.+)_std(.+)", tag)
    return f"µ={match.group(1)}, σ={match.group(2)}" if match else tag


def runs_for(root: Path, pattern: str) -> pd.DataFrame:
    """Loads the runs in the folder(s) under ``root`` matching ``pattern``."""
    return load_runs(sorted(root.glob(pattern)))


def figure_specs(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns the figures to make for one problem, as {name: {label: runs}}.

    Parameters:
        root (Path): the problem's results folder.

    Returns:
        specs (dict[str, dict[str, pd.DataFrame]]): curves for each figure.
    """
    specs: dict[str, dict[str, pd.DataFrame]] = {
        "bo": {
            "BO (UCB, β=1.96)": runs_for(root, BO_UCB),
            "BO (EI)": runs_for(root, BO_EI),
        },
    }

    beta_sources = sorted((root / "pfgs-bo").glob("beta_from_*"))
    for source in beta_sources:
        tag = source.name.removeprefix("beta_from_")
        suffix = f"_{tag}" if len(beta_sources) > 1 else ""
        specs[f"pfgs_bo_schedules{suffix}"] = {
            label: load_runs(source / folder)
            for label, folder in PFGS_BO_SCHEDULES.items()
        }
        baselines = {
            "PFGS-guided BO, maximum overlap": load_runs(source / "max_overlap"),
            "PFGS-guided BO, maximum (dynamic)": load_runs(source / "max"),
            "BO (UCB, β=1.96)": runs_for(root, BO_UCB),
            "PFGS (unconstrained)": runs_for(root, PFGS_UNCONSTRAINED),
        }
        if f"pfgs/{tag}" != PFGS_UNCONSTRAINED:
            baselines[f"PFGS ({thresholds_label(tag)})"] = runs_for(root, f"pfgs/{tag}")
        specs[f"pfgs_bo_vs_baselines{suffix}"] = baselines

    for acquisition, baseline in [("UCB_beta1.96", BO_UCB), ("EI", BO_EI)]:
        name = acquisition.split("_")[0]
        curves = {f"BO ({name}, unconstrained)": runs_for(root, baseline)}
        for folder in sorted((root / "constrained-bo" / acquisition).glob("mean*")):
            curves[thresholds_label(folder.name)] = load_runs(folder)
        specs[f"constrained_bo_{name.lower()}"] = curves

    pfgs_curves = {"PFGS (unconstrained)": runs_for(root, PFGS_UNCONSTRAINED)}
    constrained_pfgs = [
        folder
        for folder in sorted((root / "pfgs").glob("mean*"))
        if f"pfgs/{folder.name}" != PFGS_UNCONSTRAINED
    ]
    for folder in constrained_pfgs:
        pfgs_curves[thresholds_label(folder.name)] = load_runs(folder)
    specs["pfgs_constrained"] = pfgs_curves

    for folder in constrained_pfgs:
        tag = folder.name
        specs[f"constrained_methods_{tag}"] = {
            "Constrained PFGS": load_runs(folder),
            "Constrained BO (EI)": runs_for(root, f"constrained-bo/EI/{tag}"),
            "Constrained BO (UCB, β=1.96)": runs_for(
                root, f"constrained-bo/UCB_beta1.96/{tag}"
            ),
        }

    return specs


def problem_title(label: str) -> str:
    """Turns a results folder name like "ackley_6d" into "Ackley (6-D)"."""
    name, _, dimensions = label.partition("_")
    if dimensions:
        return f"{name.capitalize()} ({dimensions.removesuffix('d')}-D)"
    return name.capitalize()


def make_figures(root: Path) -> list[Path]:
    """
    Writes every figure for one problem that has runs.

    Parameters:
        root (Path): the problem's results folder.

    Returns:
        paths (list[Path]): figures written.
    """
    figures_dir = root / "figures"
    written = []

    for name, curves in figure_specs(root).items():
        curves = {label: runs for label, runs in curves.items() if not runs.empty}
        if len(curves) < 2:
            continue

        for column, ylabel in METRICS.items():
            fig = plot_comparison(
                curves,
                column,
                ylabel,
                title=f"{problem_title(root.name)}: {ylabel}",
                nonnegative=column.endswith("rrmse"),
            )
            figures_dir.mkdir(parents=True, exist_ok=True)
            path = figures_dir / f"{name}_{column}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    return written


def summarise(root: Path) -> pd.DataFrame:
    """
    Summarises every run setting under one problem's results folder.

    Parameters:
        root (Path): the problem's results folder.

    Returns:
        summary (pd.DataFrame): one row per run folder with the number of seeds,
            the mean number of iterations completed, and the mean and standard
            deviation over seeds of each metric at the last iteration.
    """
    rows = []
    for folder in sorted({csv.parent for csv in root.rglob("train_seed_*.csv")}):
        runs = load_runs(folder)
        final = runs.sort_values("iteration").groupby("train_seed").tail(1)
        row = {
            "run": folder.relative_to(root).as_posix(),
            "n_seeds": len(final),
            "iterations_mean": final["iteration"].add(1).mean(),
        }
        for column in SUMMARY_METRICS:
            row[f"final_{column}_mean"] = final[column].mean()
            row[f"final_{column}_std"] = final[column].std(ddof=0)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Makes the figures and summary table for each problem with results."""
    parser = ArgumentParser(description="Make comparison figures from results/.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Results root; defaults to results/ in the project.",
    )
    parser.add_argument(
        "--problem",
        action="append",
        help="Problem(s) to process; defaults to every problem with results.",
    )
    args = parser.parse_args()

    problems = args.problem or sorted(
        path.name for path in args.results_dir.iterdir() if path.is_dir()
    )
    for problem in problems:
        root = args.results_dir / problem
        summary = summarise(root)
        if summary.empty:
            print(f"{problem}: no runs found")
            continue

        summary.to_csv(root / "summary.csv", index=False, float_format="%.6g")
        figures = make_figures(root)
        print(f"{problem}: {len(summary)} run settings, {len(figures)} figures")


if __name__ == "__main__":
    main()
