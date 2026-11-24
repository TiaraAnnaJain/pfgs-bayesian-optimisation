"""Tests for the MLflow tracker's figure logging."""

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import mlflow

from pfgs_bo.tracking.mlflow_tracker import MLflowTracker


def test_logged_figures_leave_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    with MLflowTracker("test-experiment") as tracker:
        fig, _ = plt.subplots()
        tracker._log_fig(fig, "gp_contour/iter_001.png")
        artifacts = Path(urlparse(mlflow.get_artifact_uri()).path)

    assert (artifacts / "gp_contour/iter_001.png").is_file()
    assert list(temp_root.iterdir()) == []
