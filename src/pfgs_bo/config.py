"""
Loads experiment configs from TOML files, with command-line overrides.

A config is resolved in this order, later sources winning:

1. the TOML file given with ``--file`` (a path, or a name inside ``configs/``),
2. ``--problem``, ``--train-seed`` and ``--test-seed``,
3. any number of ``--set key=value`` overrides (values are parsed as TOML, so
   ``--set beta=2.5`` gives a float and ``--set acquisition_function=EI`` a string).
"""

import tomllib
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pfgs_bo.benchmarks import BENCHMARKS, get_benchmark

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _parse_value(text: str) -> Any:
    """Parses a TOML value such as ``2.5``, ``true`` or ``[1, 2]``; else a string."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def _config_path(file: str) -> Path:
    """Resolves ``--file``: an existing path, or a file name inside ``configs/``."""
    path = Path(file)
    if path.is_file():
        return path
    return CONFIG_DIR / file


def apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> None:
    """
    Applies ``key=value`` overrides to a config in place.

    Dotted keys set nested values, e.g. ``nsga_parameters.population_size=50``.

    Parameters:
        config (dict[str, Any]): config to update.
        overrides (Sequence[str]): overrides in ``key=value`` form.

    Raises:
        ValueError: if an override is not in ``key=value`` form.
    """
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key:
            raise ValueError(f"Override {override!r} must be in key=value form.")

        *parents, leaf = key.strip().split(".")
        target = config
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = _parse_value(value.strip())


def load_config(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Loads the config file given on the command line and applies any overrides.

    Parameters:
        argv (Sequence[str] | None): command-line arguments; defaults to
            ``sys.argv[1:]``.

    Returns:
        config (dict[str, Any]): raw config with overrides applied.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--file", required=True, help="Config file: a path or a name in configs/."
    )
    parser.add_argument("--problem", choices=sorted(BENCHMARKS), help="Problem.")
    parser.add_argument("--train-seed", type=int, help="LHS seed for training data.")
    parser.add_argument("--test-seed", type=int, help="LHS seed for the test sets.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config value; can be repeated.",
    )
    args = parser.parse_args(argv)

    with open(_config_path(args.file), "rb") as f:
        config = tomllib.load(f)

    for key, value in (
        ("problem", args.problem),
        ("train_seed", args.train_seed),
        ("test_seed", args.test_seed),
    ):
        if value is not None:
            config[key] = value

    apply_overrides(config, args.overrides)

    return config


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a copy of a validated config with the problem looked up.

    Parameters:
         config (dict[str, Any]): validated raw config.

    Returns:
        resolved (dict[str, Any]): config with "benchmark" (the ``Benchmark``) and
            "simulation" (its objective function) added.
    """
    resolved = dict(config)
    benchmark = get_benchmark(config["problem"])
    resolved["benchmark"] = benchmark
    resolved["simulation"] = benchmark.function

    return resolved


METHODS = ("PFGS", "BO", "BO-PFGS", "constrained-BO")
ACQUISITION_FUNCTIONS = ("UCB", "EI")


def _require(config: dict[str, Any], key: str, message: str) -> Any:
    """Returns ``config[key]``, or raises a ValueError with ``message`` if missing."""
    if config.get(key) is None:
        raise ValueError(message)
    return config[key]


def _require_positive_int(config: dict[str, Any], key: str, description: str) -> None:
    """Checks that ``config[key]`` is an integer greater than 0."""
    value = _require(config, key, f"Please provide {description}.")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(
            f"{description.capitalize()} must be an integer greater than 0."
        )


def validate_config(config: dict[str, Any]) -> None:
    """
    Checks that a raw config has every setting its method needs.

    Parameters:
        config (dict[str, Any]): raw config from ``load_config``.

    Raises:
        ValueError: if a setting is missing or has an unsupported value.
        TypeError: if a setting has the wrong type.
    """
    if not config.get("test_name"):
        raise ValueError("Please provide a test run name.")

    method = _require(
        config,
        "method",
        f"Please provide method from the following: {', '.join(METHODS)}.",
    )
    if method not in METHODS:
        raise ValueError(f"Method {method} is not supported.")

    if method != "BO-PFGS":
        _require_positive_int(config, "num_iter", "number of iterations")

    if method in ("BO", "constrained-BO"):
        acquisition_function = _require(
            config,
            "acquisition_function",
            "Please provide acquisition function from the following: UCB, EI.",
        )
        if acquisition_function not in ACQUISITION_FUNCTIONS:
            raise ValueError("Acquisition function must be either UCB or EI.")
        if acquisition_function == "UCB":
            _require(config, "beta", "Please provide beta parameter.")

    problem = _require(
        config,
        "problem",
        f"Please provide problem from the following: {', '.join(BENCHMARKS)}.",
    )
    get_benchmark(problem)

    _require_positive_int(config, "train_samples", "number of training samples")
    _require_positive_int(config, "test_samples", "number of test samples")

    if method in ("PFGS", "BO-PFGS", "constrained-BO") and (
        config.get("mean_discard_threshold") is None
        or config.get("std_dev_discard_threshold") is None
    ):
        raise ValueError(
            "Please provide mean and standard deviation discard thresholds."
        )

    for seed_name in ("train_seed", "test_seed"):
        if not isinstance(config.get(seed_name), int):
            raise TypeError(f"{seed_name} must be an integer.")
