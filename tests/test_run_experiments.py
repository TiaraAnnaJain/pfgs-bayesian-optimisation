"""Tests for the experiment runner: the experiment plan and --skip-existing."""

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
    problem = runner["PROBLEMS"]["himmelblau"]
    return [*experiment.command(problem, 1), "--set", f"results_dir={tmp_path}"]


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
    assert not runner["is_done"](pfgs_bo)

    beta = tmp_path / "himmelblau/pfgs/mean0_std0/beta_train_seed_1.json"
    beta.parent.mkdir(parents=True)
    beta.write_text(json.dumps({"beta_min": [0.5, 0.8], "beta_max": [2.0, 1.5]}))
    schedules = ["min", "max", "mid_point", "mean", "median", "max_overlap"]
    root = tmp_path / "himmelblau/pfgs-bo/beta_from_mean0_std0"
    for name in schedules[:-1]:
        (root / name).mkdir(parents=True)
        (root / name / "train_seed_1.csv").write_text("iteration\n0\n")
    assert not runner["is_done"](pfgs_bo)

    (root / "max_overlap").mkdir()
    (root / "max_overlap/train_seed_1.csv").write_text("iteration\n0\n")
    assert runner["is_done"](pfgs_bo)


def settings(runs):
    """(problem, script, config, thresholds) for each run, for easy comparison."""
    return [
        (
            problem.label,
            experiment.script,
            experiment.config,
            (
                experiment.overrides.get("mean_discard_threshold"),
                experiment.overrides.get("std_dev_discard_threshold"),
            ),
        )
        for problem, experiment in runs
    ]


def test_experiment_a_is_pfgs_then_pfgs_guided_bo_on_himmelblau(runner):
    assert settings(runner["experiment_a"]()) == [
        ("himmelblau", "run_pfgs.py", "pfgs.toml", (1000, 100)),
        ("himmelblau", "run_pfgs_bo.py", "pfgs_bo.toml", (1000, 100)),
    ]


def test_experiment_b_runs_three_methods_unconstrained_and_per_threshold(runner):
    runs = settings(runner["experiment_b"]())

    for label, pairs in runner["CONSTRAINTS"].items():
        own = [run for run in runs if run[0] == label]
        assert len(pairs) == 4
        assert len(own) == 15
        assert {r[3] for r in own if r[1] == "run_pfgs.py"} == {(0, 0), *pairs}
        for config in ["constrained_bo_ei.toml", "constrained_bo_ucb.toml"]:
            assert {r[3] for r in own if r[2] == config} == set(pairs)
        assert {r[2] for r in own if r[1] == "run_bo.py"} == {
            "bo_ei.toml",
            "bo_ucb.toml",
        }


def test_ackley_6d_runs_with_six_dimensions(runner):
    problem = runner["PROBLEMS"]["ackley_6d"]
    command = runner["Experiment"]("run_bo.py", "bo_ei.toml").command(problem, 1)

    assert command[command.index("--dimensions") + 1] == "6"
    assert command[command.index("--problem") + 1] == "ackley"


def test_pfgs_thresholds_without_constrained_bo_are_rejected(runner):
    problem = runner["PROBLEMS"]["branin"]
    pfgs = runner["Experiment"](
        "run_pfgs.py", "pfgs.toml", runner["thresholds"]((1, 2))
    )

    with pytest.raises(ValueError, match="not run with constrained BO"):
        runner["check_constraints"]([(problem, pfgs)])


def test_the_full_plan_passes_the_check_and_shared_runs_happen_once(
    runner, monkeypatch, capsys
):
    runner["check_constraints"](runner["experiment_a"]() + runner["experiment_b"]())

    monkeypatch.setattr("sys.argv", ["run_experiments.py", "--dry-run", "--seeds", "1"])
    runner["main"]()
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 61
    assert len(set(line.split("] ", 1)[1] for line in lines)) == 61
