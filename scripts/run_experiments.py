"""
Runs every experiment from the report, once per training seed.
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


Thresholds = tuple[float, float]


@dataclass(frozen=True)
class ReportThresholds:
    """
    The discard thresholds the report used for one problem.

    Parameters:
        constrained_bo (list[Thresholds]): pairs run with constrained BO (UCB and EI).
        pfgs (list[Thresholds]): pairs run with PFGS; (0, 0) is unconstrained PFGS.
        beta_source (Thresholds): PFGS pair whose beta intervals PFGS-guided BO uses.
    """

    constrained_bo: list[Thresholds]
    pfgs: list[Thresholds]
    beta_source: Thresholds


REPORT_THRESHOLDS: dict[str, ReportThresholds] = {
    "himmelblau": ReportThresholds(
        constrained_bo=[(1000, 50), (1000, 100), (1200, 50), (1200, 100)],
        pfgs=[(0, 0), (1000, 100)],
        beta_source=(1000, 100),
    ),
    "branin": ReportThresholds(
        constrained_bo=[(150, 0.5), (150, 1), (250, 0.5), (250, 1)],
        pfgs=[(0, 0)],
        beta_source=(0, 0),
    ),
}


def thresholds(pair: Thresholds) -> dict[str, float]:
    """Discard-threshold overrides for PFGS, PFGS-guided BO or constrained BO."""
    mean, std_dev = pair
    return {"mean_discard_threshold": mean, "std_dev_discard_threshold": std_dev}


def report_experiments(problem: str) -> list[Experiment]:
    """
    Returns the report's experiments for one problem, in the order they must run:
    PFGS before PFGS-guided BO, which reads the beta intervals PFGS saves.

    Parameters:
        problem (str): a problem in ``REPORT_THRESHOLDS``.

    Returns:
        experiments (list[Experiment]): experiments to run for each seed.
    """
    report = REPORT_THRESHOLDS[problem]
    pfgs_pairs = list(dict.fromkeys([*report.pfgs, report.beta_source]))

    return [
        *(Experiment("run_pfgs.py", "pfgs.toml", thresholds(t)) for t in pfgs_pairs),
        Experiment("run_pfgs_bo.py", "pfgs_bo.toml", thresholds(report.beta_source)),
        Experiment("run_bo.py", "bo_ucb.toml", {"beta": 1.96}),
        Experiment("run_bo.py", "bo_ei.toml"),
        *(
            Experiment("run_constrained_bo.py", config, thresholds(t))
            for config, t in product(
                ["constrained_bo_ucb.toml", "constrained_bo_ei.toml"],
                report.constrained_bo,
            )
        ),
    ]


def main() -> None:
    """Runs (or lists) the report's experiments."""
    parser = ArgumentParser(description="Run the report's experiments.")
    parser.add_argument(
        "--problem",
        choices=sorted(REPORT_THRESHOLDS),
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
        for problem in args.problem or REPORT_THRESHOLDS
        for experiment in report_experiments(problem)
        for seed in args.seeds
    ]

    for i, command in enumerate(commands, start=1):
        shown = " ".join(["python", f"scripts/{Path(command[1]).name}", *command[2:]])
        print(f"[{i}/{len(commands)}] {shown}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
