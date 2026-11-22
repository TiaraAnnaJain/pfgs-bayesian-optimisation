"""
End-to-end tests: run every example config through its script with tiny settings.

Results and MLflow runs go to a temporary folder, so the project's results/ and
mlflow.db are never touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Small enough to run in seconds, large enough to exercise every step.
TINY = [
    "num_iter=2",
    "test_samples=20",
    "nsga_parameters.population_size=20",
    "nsga_parameters.number_of_generations=2",
    "optimizer_parameters.num_restarts=2",
    "optimizer_parameters.raw_samples=32",
]


def run_script(script: str, config: str, tmp_path: Path, *extra: str) -> None:
    """Runs a script on an example config and fails the test if it errors."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--file",
        config,
        "--train-seed",
        "1",
        "--set",
        f"results_dir={tmp_path / 'results'}",
    ]
    for override in [*TINY, *extra]:
        command += ["--set", override]

    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize(
    ("script", "config", "csv"),
    [
        ("run_pfgs.py", "pfgs.toml", "pfgs/mean0_std0/train_seed_1.csv"),
        ("run_bo.py", "bo_ucb.toml", "bo/UCB_beta1.96/train_seed_1.csv"),
        ("run_bo.py", "bo_ei.toml", "bo/EI/train_seed_1.csv"),
        (
            "run_constrained_bo.py",
            "constrained_bo_ucb.toml",
            "constrained-bo/UCB_beta1.96/mean1000_std100/train_seed_1.csv",
        ),
        (
            "run_constrained_bo.py",
            "constrained_bo_ei.toml",
            "constrained-bo/EI/mean1000_std100/train_seed_1.csv",
        ),
    ],
)
def test_example_config_runs(script, config, csv, tmp_path):
    run_script(script, config, tmp_path)

    assert (tmp_path / "results" / "himmelblau" / csv).is_file()


def test_pfgs_guided_bo_uses_beta_from_pfgs(tmp_path):
    run_script("run_pfgs.py", "pfgs.toml", tmp_path)
    beta_file = tmp_path / "results/himmelblau/pfgs/mean0_std0/beta_train_seed_1.json"
    assert beta_file.is_file()

    run_script("run_pfgs_bo.py", "pfgs_bo.toml", tmp_path)

    schedules = tmp_path / "results/himmelblau/pfgs-bo/beta_from_mean0_std0"
    assert {"min", "max", "mid_point", "mean", "median"} <= {
        path.name for path in schedules.iterdir()
    }


def test_problem_can_be_switched_from_the_command_line(tmp_path):
    run_script("run_bo.py", "bo_ei.toml", tmp_path, "problem=branin")

    assert (tmp_path / "results/branin/bo/EI/train_seed_1.csv").is_file()
