"""Tests for resuming the experiment runner with --skip-existing."""

import json
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    return runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_experiments.py"))


def command(runner, script, config, tmp_path, **overrides):
    experiment = runner["Experiment"](script, config, overrides)
    return [*experiment.command("himmelblau", 1), "--set", f"results_dir={tmp_path}"]


def test_run_is_done_once_its_csv_exists(runner, tmp_path):
    bo = command(runner, "run_bo.py", "bo_ei.toml", tmp_path)
    assert not runner["is_done"](bo)

    csv = tmp_path / "himmelblau/bo/EI/train_seed_1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("iteration\n0\n")

    assert runner["is_done"](bo)


def test_pfgs_guided_bo_is_done_only_when_every_schedule_exists(runner, tmp_path):
    pfgs_bo = command(
        runner,
        "run_pfgs_bo.py",
        "pfgs_bo.toml",
        tmp_path,
        mean_discard_threshold=0,
        std_dev_discard_threshold=0,
    )
    assert not runner["is_done"](pfgs_bo)  # no PFGS beta file yet

    beta = tmp_path / "himmelblau/pfgs/mean0_std0/beta_train_seed_1.json"
    beta.parent.mkdir(parents=True)
    beta.write_text(json.dumps({"beta_min": [0.5, 0.8], "beta_max": [2.0, 1.5]}))
    schedules = ["min", "max", "mid_point", "mean", "median", "max_overlap"]
    root = tmp_path / "himmelblau/pfgs-bo/beta_from_mean0_std0"
    for name in schedules[:-1]:
        (root / name).mkdir(parents=True)
        (root / name / "train_seed_1.csv").write_text("iteration\n0\n")
    assert not runner["is_done"](pfgs_bo)  # max_overlap still missing

    (root / "max_overlap").mkdir()
    (root / "max_overlap/train_seed_1.csv").write_text("iteration\n0\n")
    assert runner["is_done"](pfgs_bo)
