"""
Runs every experiment from the report, once per training seed.

Each experiment is one of the method scripts with an example config and the
report's settings for that problem. PFGS runs before PFGS-guided BO, which reads
the beta intervals PFGS saves. Results go to ``results/<problem>/...`` and MLflow.

Usage:
    uv run python scripts/run_experiments.py --dry-run     # list the runs
    uv run python scripts/run_experiments.py               # run everything
    uv run python scripts/run_experiments.py --problem branin --seeds 1 2
"""

import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent

TRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEED = 11


@dataclass(frozen=True)
class Experiment:
    """
    One experiment: a method script, its example config and the settings to apply.

    Parameters:
        script (str): script in ``scripts/``.
        config (str): example config in ``configs/``.
        overrides (dict[str, Any]): config values to set with ``--set``.
    """

    script: str
    config: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def command(self, problem: str, train_seed: int) -> list[str]:
        """Returns the command that runs this experiment for one problem and seed."""
        command = [
            sys.executable,
            str(SCRIPTS_DIR / self.script),
            "--file",
            self.config,
            "--problem",
            problem,
            "--train-seed",
            str(train_seed),
            "--test-seed",
            str(TEST_SEED),
        ]
        for key, value in self.overrides.items():
            command += ["--set", f"{key}={value}"]
        return command


def thresholds(mean: float, std_dev: float) -> dict[str, float]:
    """Discard-threshold overrides for PFGS, PFGS-guided BO or constrained BO."""
    return {"mean_discard_threshold": mean, "std_dev_discard_threshold": std_dev}


def constrained_bo_grid(means: list[float], std_devs: list[float]) -> list[Experiment]:
    """Constrained BO with UCB and EI for every pair of discard thresholds."""
    return [
        Experiment("run_constrained_bo.py", config, thresholds(mean, std_dev))
        for config, mean, std_dev in product(
            ["constrained_bo_ucb.toml", "constrained_bo_ei.toml"], means, std_devs
        )
    ]


# The report's experiments, per problem, in the order they must run.
REPORT_EXPERIMENTS: dict[str, list[Experiment]] = {
    "himmelblau": [
        # PFGS without thresholds: the baseline, and the source of the beta intervals.
        Experiment("run_pfgs.py", "pfgs.toml", thresholds(0, 0)),
        # Constrained PFGS, compared with constrained BO (report Fig. 11-12).
        Experiment("run_pfgs.py", "pfgs.toml", thresholds(1000, 100)),
        # PFGS-guided BO, all six beta schedules (report Fig. 3-6).
        Experiment("run_pfgs_bo.py", "pfgs_bo.toml", thresholds(0, 0)),
        # Standard BO baselines.
        Experiment("run_bo.py", "bo_ucb.toml", {"beta": 1.96}),
        Experiment("run_bo.py", "bo_ei.toml"),
        # Constrained BO threshold grid (report Fig. 9-10 and Appendix B.1).
        *constrained_bo_grid(means=[1000, 1200], std_devs=[50, 100]),
    ],
    "branin": [
        Experiment("run_pfgs.py", "pfgs.toml", thresholds(0, 0)),
        Experiment("run_pfgs_bo.py", "pfgs_bo.toml", thresholds(0, 0)),
        Experiment("run_bo.py", "bo_ucb.toml", {"beta": 1.96}),
        Experiment("run_bo.py", "bo_ei.toml"),
        # Constrained BO threshold grid (report Appendix B.2-B.3).
        *constrained_bo_grid(means=[150, 250], std_devs=[0.5, 1]),
    ],
}


def main() -> None:
    """Runs (or lists) the report's experiments."""
    parser = ArgumentParser(description="Run the report's experiments.")
    parser.add_argument(
        "--problem",
        choices=sorted(REPORT_EXPERIMENTS),
        action="append",
        help="Problem(s) to run; defaults to all. Can be repeated.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=TRAIN_SEEDS,
        help=f"Training seeds; defaults to {TRAIN_SEEDS}.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running."
    )
    args = parser.parse_args()

    commands = [
        experiment.command(problem, seed)
        for problem in args.problem or REPORT_EXPERIMENTS
        for experiment in REPORT_EXPERIMENTS[problem]
        for seed in args.seeds
    ]

    for i, command in enumerate(commands, start=1):
        shown = " ".join(["python", f"scripts/{Path(command[1]).name}", *command[2:]])
        print(f"[{i}/{len(commands)}] {shown}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
