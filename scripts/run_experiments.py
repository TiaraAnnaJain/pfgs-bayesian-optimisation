"""
Runs every experiment from the report, once per training seed.
"""

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pfgs_bo.config import load_config, resolve_config
from pfgs_bo.experiment import beta_file, results_csv
from pfgs_bo.pareto_front.beta import SCHEDULE_NAMES, beta_ranges, select_schedules
from pfgs_bo.tracking.mlflow_tracker import prepare_experiments
from pfgs_bo.tracking.results_io import from_json

SCRIPTS_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPTS_DIR.parent / "logs" / "experiments"

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

    def command(self, problem: "Problem", train_seed: int) -> list[str]:
        """Returns the command that runs this experiment for one problem and seed."""
        command = [
            sys.executable,
            str(SCRIPTS_DIR / self.script),
            "--file",
            self.config,
            *problem.args,
            "--train-seed",
            str(train_seed),
            "--test-seed",
            str(TEST_SEED),
        ]
        for key, value in self.overrides.items():
            command += ["--set", f"{key}={value}"]
        return command


@dataclass(frozen=True)
class Problem:
    """
    A benchmark problem as run in the experiments.

    Parameters:
        name (str): benchmark name.
        dimensions (int): number of input dimensions.
    """

    name: str
    dimensions: int = 2

    @property
    def label(self) -> str:
        """Name used on the command line and for results, e.g. "ackley_6d"."""
        return self.name if self.dimensions == 2 else f"{self.name}_{self.dimensions}d"

    @property
    def args(self) -> list[str]:
        """Command-line arguments selecting this problem."""
        return ["--problem", self.name, "--dimensions", str(self.dimensions)]


PROBLEMS = {
    problem.label: problem
    for problem in [
        Problem("himmelblau"),
        Problem("branin"),
        Problem("ackley"),
        Problem("ackley", dimensions=6),
    ]
}

Thresholds = tuple[float, float]
UNCONSTRAINED: Thresholds = (0, 0)

CONSTRAINTS: dict[str, list[Thresholds]] = {
    "himmelblau": [(1000, 50), (1000, 100), (1200, 50), (1200, 100)],
    "branin": [(150, 0.5), (150, 1), (250, 0.5), (250, 1)],
    "ackley": [(4.1, 0.15), (4.1, 0.3), (4.4, 0.15), (4.4, 0.3)],
    "ackley_6d": [(3.85, 0.04), (3.85, 0.08), (3.95, 0.04), (3.95, 0.08)],
}

BASELINE_CONSTRAINTS: dict[str, Thresholds] = {"himmelblau": (1000, 100)}


def thresholds(pair: Thresholds) -> dict[str, float]:
    """Discard-threshold overrides for PFGS, PFGS-guided BO or constrained BO."""
    mean, std_dev = pair
    return {"mean_discard_threshold": mean, "std_dev_discard_threshold": std_dev}


def experiment_a() -> list[tuple[Problem, Experiment]]:
    """
    Experiment A, PFGS-guided hyperparameter selection for BO: PFGS with the
    baseline constraints, then PFGS-guided BO with all six beta schedules.

    Returns:
        runs (list[tuple[Problem, Experiment]]): runs in the order they must run.
    """
    runs = []
    for label, baseline in BASELINE_CONSTRAINTS.items():
        problem = PROBLEMS[label]
        runs.append(
            (problem, Experiment("run_pfgs.py", "pfgs.toml", thresholds(baseline)))
        )
        # One run per schedule, so the schedules can run in parallel.
        runs += [
            (
                problem,
                Experiment(
                    "run_pfgs_bo.py",
                    "pfgs_bo.toml",
                    {**thresholds(baseline), "schedule": schedule},
                ),
            )
            for schedule in SCHEDULE_NAMES
        ]
    return runs


def experiment_b() -> list[tuple[Problem, Experiment]]:
    """
    Experiment B, PFGS-inspired constraints for BO: PFGS, BO-EI and BO-UCB,
    unconstrained and with each pair of discard thresholds, on every problem.

    Returns:
        runs (list[tuple[Problem, Experiment]]): runs for each problem.
    """
    runs = []
    for label, pairs in CONSTRAINTS.items():
        problem = PROBLEMS[label]
        runs += [
            (problem, Experiment("run_pfgs.py", "pfgs.toml", thresholds(pair)))
            for pair in [UNCONSTRAINED, *pairs]
        ]
        for acquisition in ["ei", "ucb"]:
            runs.append((problem, Experiment("run_bo.py", f"bo_{acquisition}.toml")))
            runs += [
                (
                    problem,
                    Experiment(
                        "run_constrained_bo.py",
                        f"constrained_bo_{acquisition}.toml",
                        thresholds(pair),
                    ),
                )
                for pair in pairs
            ]
    return runs


EXPERIMENTS = {"A": experiment_a, "B": experiment_b}


def check_constraints(runs: list[tuple[Problem, Experiment]]) -> None:
    """
    Checks that every threshold pair run with PFGS is also run with constrained BO
    on the same problem; baseline constraints for beta are exempt.

    Parameters:
        runs (list[tuple[Problem, Experiment]]): runs to check.

    Raises:
        ValueError: if a PFGS pair has no matching constrained BO run.
    """

    def pair(experiment: Experiment) -> Thresholds:
        return (
            experiment.overrides["mean_discard_threshold"],
            experiment.overrides["std_dev_discard_threshold"],
        )

    for problem in {problem for problem, _ in runs}:
        own = [experiment for p, experiment in runs if p == problem]
        bo_pairs = {pair(e) for e in own if e.script == "run_constrained_bo.py"} | {
            UNCONSTRAINED
        }
        allowed = bo_pairs | {BASELINE_CONSTRAINTS.get(problem.label, UNCONSTRAINED)}
        for experiment in own:
            if experiment.script == "run_pfgs.py" and pair(experiment) not in allowed:
                raise ValueError(
                    f"{problem.label}: PFGS thresholds {pair(experiment)} are not "
                    "run with constrained BO."
                )


def is_done(command: list[str]) -> bool:
    """
    Returns whether a run has already finished, i.e. its results CSV exists.

    Parameters:
        command (list[str]): command from ``Experiment.command``.

    Returns:
        done (bool): whether to skip this run.
    """
    config = resolve_config(load_config(command[2:]))
    if config["method"] != "BO-PFGS":
        return results_csv(config).is_file()

    pfgs_beta_file = beta_file(config)
    if not pfgs_beta_file.is_file():
        return False
    schedules = select_schedules(
        beta_ranges(from_json(pfgs_beta_file)), config.get("schedule")
    )
    return all(results_csv(config, schedule=name).is_file() for name in schedules)


def shown(command: list[str]) -> str:
    """A command as you would type it, e.g. "python scripts/run_bo.py --file ..."."""
    return " ".join(["python", f"scripts/{Path(command[1]).name}", *command[2:]])


def log_name(command: list[str]) -> str:
    """A readable, unique log file name for a command."""
    args = command[2:]
    parts = [Path(command[1]).stem.removeprefix("run_")]
    for flag in ("--file", "--problem", "--dimensions", "--train-seed"):
        if flag in args:
            parts.append(Path(args[args.index(flag) + 1]).stem)
    parts += [a.split("=", 1)[1] for a in args if "_discard_threshold=" in a]
    parts += [a.split("=", 1)[1] for a in args if a.startswith("schedule=")]
    return "_".join(parts) + ".log"


def stages(commands: list[list[str]]) -> list[list[list[str]]]:
    """
    Splits commands into stages that can each run in parallel: PFGS-guided BO
    reads the beta intervals PFGS saves, so it runs after everything else.
    """
    first = [c for c in commands if Path(c[1]).name != "run_pfgs_bo.py"]
    second = [c for c in commands if Path(c[1]).name == "run_pfgs_bo.py"]
    return [stage for stage in (first, second) if stage]


def run_all(commands: list[list[str]], jobs: int, log_dir: Path) -> list[list[str]]:
    """
    Runs commands, ``jobs`` at a time.

    Parameters:
        commands (list[list[str]]): commands, in the order they must run.
        jobs (int): number of runs at once.
        log_dir (Path): folder for the log files when ``jobs`` > 1.

    Returns:
        failed (list[list[str]]): commands that failed.
    """
    total = len(commands)
    if jobs <= 1:
        for i, command in enumerate(commands, start=1):
            print(f"[{i}/{total}] {shown(command)}", flush=True)
            subprocess.run(command, check=True)
        return []

    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    failed: list[list[str]] = []
    finished = 0

    def run_one(command: list[str]) -> tuple[list[str], int, float, Path]:
        log_file = log_dir / log_name(command)
        start = time.perf_counter()
        with open(log_file, "w") as log:
            log.write(shown(command) + "\n\n")
            log.flush()
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        return command, result.returncode, time.perf_counter() - start, log_file

    for stage in stages(commands):
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in as_completed([pool.submit(run_one, c) for c in stage]):
                command, returncode, seconds, log_file = future.result()
                finished += 1
                status = "done" if returncode == 0 else f"FAILED, see {log_file}"
                print(
                    f"[{finished}/{total}] {status} ({seconds:.0f}s): {shown(command)}"
                )
                if returncode != 0:
                    failed.append(command)

    return failed


def main() -> None:
    """Runs (or lists) the report's experiments."""
    parser = ArgumentParser(description="Run the report's experiments.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        action="append",
        help="A: PFGS-guided BO; B: constraints. Defaults to both. Can be repeated.",
    )
    parser.add_argument(
        "--problem",
        choices=sorted(PROBLEMS),
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Runs to do at once (each uses one CPU thread); 4 suits an 8 GB laptop.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose results already exist, e.g. to resume after a crash.",
    )
    args = parser.parse_args()

    runs = [
        run
        for name in args.experiment or sorted(EXPERIMENTS)
        for run in EXPERIMENTS[name]()
        if args.problem is None or run[0].label in args.problem
    ]
    check_constraints(runs)

    commands = list(
        dict.fromkeys(
            tuple(experiment.command(problem, seed))
            for problem, experiment in runs
            for seed in args.seeds
        )
    )

    if args.skip_existing:
        remaining = [command for command in commands if not is_done(list(command))]
        print(f"Skipping {len(commands) - len(remaining)} runs that already finished.")
        commands = remaining

    if args.dry_run:
        for i, command in enumerate(commands, start=1):
            print(f"[{i}/{len(commands)}] {shown(list(command))}")
        return

    if args.jobs > 1:
        prepare_experiments(sorted({problem.label for problem, _ in runs}))
    failed = run_all([list(command) for command in commands], args.jobs, LOG_DIR)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed; see their logs in {LOG_DIR}.")


if __name__ == "__main__":
    main()
