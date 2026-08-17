#!/usr/bin/env python3
"""
Runs offline pipeline notebooks in sequence using nbconvert.
Stops immediately if any notebook execution fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import NOTEBOOKS_DIR, get_logger  # noqa: E402

logger = get_logger("run_pipeline")

STAGES = [
    ("ingestion", "01_data_ingestion.ipynb"),
    ("exploration", "02_data_exploration.ipynb"),
    ("embeddings", "03_embeddings_and_index.ipynb"),
    ("label_bootstrapping", "04_label_bootstrapping.ipynb"),
    ("train_classifiers", "05_train_classifiers.ipynb"),
]
STAGE_NAMES = [name for name, _ in STAGES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BookLens offline pipeline end-to-end.")
    parser.add_argument(
        "--from-stage", choices=STAGE_NAMES, default=STAGE_NAMES[0],
        help="First stage to run (default: %(default)s)",
    )
    parser.add_argument(
        "--to-stage", choices=STAGE_NAMES, default=STAGE_NAMES[-1],
        help="Last stage to run (default: %(default)s)",
    )
    parser.add_argument(
        "--then-run-dashboard", action="store_true",
        help="Launch dashboard immediately after pipeline completion",
    )
    return parser.parse_args()


def run_notebook(notebook_filename: str) -> None:
    """Execute one notebook in-place."""
    notebook_path = NOTEBOOKS_DIR / notebook_filename
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(notebook_path),
        ],
        check=True,
        cwd=str(NOTEBOOKS_DIR),
    )


def main() -> None:
    args = parse_args()

    start_index = STAGE_NAMES.index(args.from_stage)
    end_index = STAGE_NAMES.index(args.to_stage)
    if start_index > end_index:
        logger.error("--from-stage %r cannot come after --to-stage %r", args.from_stage, args.to_stage)
        sys.exit(1)

    stages_to_run = STAGES[start_index : end_index + 1]

    for stage_name, notebook_filename in stages_to_run:
        logger.info("Running stage %r (%s)", stage_name, notebook_filename)
        start_time = time.time()

        try:
            run_notebook(notebook_filename)
        except subprocess.CalledProcessError:
            logger.error("Stage %r failed (%s) - stopping pipeline.", stage_name, notebook_filename)
            sys.exit(1)

        duration_minutes = (time.time() - start_time) / 60
        logger.info("Completed %r in %.1f minutes", stage_name, duration_minutes)

    logger.info("Pipeline run complete: %s", ", ".join(name for name, _ in stages_to_run))

    if args.then_run_dashboard:
        logger.info("Starting dashboard...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_dashboard.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
