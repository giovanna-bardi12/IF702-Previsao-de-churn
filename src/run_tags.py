"""Shared W&B provenance tags for repository experiments."""
from datetime import datetime
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wandb_provenance_tags():
    """Return the source commit and local run date as W&B tags.

    Cluster jobs can set ``GIT_COMMIT`` before syncing a checkout without its
    ``.git`` directory. Local runs resolve the commit directly from Git.
    """
    commit = os.environ.get("GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"

    run_date = datetime.now().astimezone().date().isoformat()
    return [f"commit:{commit}", f"date:{run_date}"]
