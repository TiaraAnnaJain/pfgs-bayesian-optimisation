"""Tests for the comparison figures and summary table."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfgs_bo.plotting.results import mean_and_std
from pfgs_bo.tracking.results_io import load_runs, to_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_runs(folder: Path, seeds: list[int], n_iter: int = 4, offset: float = 0):
    """Writes fake run CSVs whose metrics depend on the seed."""
    for seed in seeds:
        to_csv(
            {
                "iteration": range(n_iter),
                "entire_space_rrmse": np.linspace(1, 0.5, n_iter) + offset,
                "design_space_rrmse": np.linspace(2, 1, n_iter) + seed + offset,
                "train_data_max": np.arange(n_iter) + seed,
            },
            folder / f"train_seed_{seed}.csv",
        )


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results" / "himmelblau"
    write_runs(root / "bo/UCB_beta1.96", seeds=[1, 2])
    write_runs(root / "bo/EI", seeds=[1, 2])
    write_runs(root / "constrained-bo/EI/mean1000_std50", seeds=[1, 2], n_iter=3)
    write_runs(root / "pfgs/mean1000_std50", seeds=[1, 2], n_iter=2)
    return tmp_path / "results"


@pytest.fixture
def make_figures():
    return runpy.run_path(str(PROJECT_ROOT / "scripts" / "make_figures.py"))


def test_load_runs_adds_seed_and_pools_folders(results):
    root = results / "himmelblau"

    runs = load_runs([root / "bo/EI", root / "bo/UCB_beta1.96"])

    assert len(runs) == 16
    assert sorted(runs["train_seed"].unique()) == [1, 2]


def test_mean_and_std_over_seeds(results):
    runs = load_runs(results / "himmelblau/bo/EI")

    iterations, mean, std = mean_and_std(runs, "design_space_rrmse")

    np.testing.assert_allclose(iterations, [0, 1, 2, 3])
    # Seeds 1 and 2 differ by exactly 1, so the std is 0.5 at every iteration.
    np.testing.assert_allclose(std, 0.5)
    np.testing.assert_allclose(mean[0], 2 + 1.5)


def test_summary_has_one_row_per_run_setting(results, make_figures):
    summary = make_figures["summarise"](results / "himmelblau")

    assert set(summary["run"]) == {
        "bo/UCB_beta1.96",
        "bo/EI",
        "constrained-bo/EI/mean1000_std50",
        "pfgs/mean1000_std50",
    }
    row = summary.set_index("run").loc["constrained-bo/EI/mean1000_std50"]
    assert row["n_seeds"] == 2
    assert row["iterations_mean"] == 3
    assert row["final_design_space_rrmse_mean"] == pytest.approx(1 + 1.5)


def test_figures_are_written_for_available_comparisons(results, make_figures):
    written = make_figures["make_figures"](results / "himmelblau")
    names = {path.name for path in written}

    assert "bo_design_space_rrmse.png" in names
    assert "constrained_bo_ei_train_data_max.png" in names
    assert "constrained_bo_vs_pfgs_mean1000_std50_design_space_rrmse.png" in names
    # No PFGS-guided BO runs, so those figures are skipped.
    assert not any(name.startswith("pfgs_bo") for name in names)


def test_main_writes_summary_csv(results, make_figures, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--results-dir", str(results)])

    make_figures["main"]()

    summary = pd.read_csv(results / "himmelblau" / "summary.csv")
    assert len(summary) == 4
    assert (results / "himmelblau" / "figures").is_dir()
