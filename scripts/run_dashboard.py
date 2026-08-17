#!/usr/bin/env python3
"""
Launcher for the Streamlit dashboard.
Checks that all required data and model artifacts exist before starting.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CATALOG_CSV,
    CATEGORY_MODEL_PATH,
    EMOTION_MODEL_PATH,
    METRICS_PATH,
    VECTOR_STORE_DIR,
    get_logger,
)

logger = get_logger("run_dashboard", log_file="app.log")

REQUIRED_ARTIFACTS = [
    (CATALOG_CSV, "notebook 01 (ingestion) / notebook 04 (labels)"),
    (VECTOR_STORE_DIR, "notebook 03 (embeddings and index)"),
    (CATEGORY_MODEL_PATH, "notebook 05 (train classifiers)"),
    (EMOTION_MODEL_PATH, "notebook 05 (train classifiers)"),
    (METRICS_PATH, "notebook 05 (train classifiers)"),
]


def _artifact_exists(path: Path) -> bool:
    """Check if file exists or directory is non-empty."""
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def check_prerequisites() -> bool:
    """Verify that all pipeline artifacts exist."""
    all_present = True
    for path, source_stage in REQUIRED_ARTIFACTS:
        if not _artifact_exists(path):
            logger.warning("Missing %s (built by %s)", path, source_stage)
            all_present = False
    return all_present


def main() -> None:
    if check_prerequisites():
        logger.info("All pipeline artifacts present - launching Streamlit dashboard")
    else:
        logger.warning("Some artifacts are missing. Live app will use zero-shot fallback.")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "Dashboard.py")],
        check=True,
    )


if __name__ == "__main__":
    main()
