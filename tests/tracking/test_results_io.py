"""Tests for saving and loading results."""

from pfgs_bo.tracking.results_io import from_json, to_json


def test_beta_json_round_trip_creates_folders(tmp_path):
    path = tmp_path / "results" / "pfgs" / "beta_seed_3.json"

    to_json(beta_min=[0.1, 0.2], beta_max=[1.0, 2.0], json_file=path)

    assert from_json(path) == {"beta_min": [0.1, 0.2], "beta_max": [1.0, 2.0]}
