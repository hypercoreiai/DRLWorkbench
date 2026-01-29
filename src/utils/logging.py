# Centralized logging (V3 — PROJECT_OUTLINE Section 9.2)
"""
Create logger that logs to:
1. Console (INFO level).
2. File (DEBUG level) at log_dir/run_id.log.
3. Error file at log_dir/run_id_errors.log.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a logger that writes to console and optionally to files.

    Args:
        log_dir: Directory for log files. If None, only console logging.
        run_id: Identifier for this run; used in log filenames.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir and run_id:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{run_id}.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        err_handler = logging.FileHandler(
            log_path / f"{run_id}_errors.log", encoding="utf-8"
        )
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(fmt)
        logger.addHandler(err_handler)

    return logger
