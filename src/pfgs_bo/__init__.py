"""PFGS principles in Bayesian optimisation: PFGS-guided BO and constrained BO."""

import os
from pathlib import Path

# PySwarms configures logging when it is imported and, by default, writes a
# report.log to the current working directory. Point it at a config without a log
# file instead; this must happen before anything imports pyswarms.
os.environ.setdefault("LOG_CFG", str(Path(__file__).with_name("pyswarms_logging.yaml")))
