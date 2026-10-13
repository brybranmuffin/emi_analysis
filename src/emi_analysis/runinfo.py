"""Logging and run-metadata helpers shared by the prep and training entry points."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "emi_analysis"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str | Path | None = None, mode: str = "w") -> logging.Logger:
    """Log to stdout (the SLURM .out file) and, if given, to ``log_file`` with identical lines.

    Warnings and errors are also copied to stderr (the SLURM .err file) so flags and stop reasons are
    visible there. Tracebacks are left to Python's own stderr output to avoid printing them twice.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    errors = logging.StreamHandler(sys.stderr)
    errors.setLevel(logging.WARNING)
    errors.setFormatter(formatter)
    errors.addFilter(lambda record: record.exc_info is None)
    logger.addHandler(errors)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def banner(logger: logging.Logger, title: str, items: dict[str, object] | None = None) -> None:
    """Log a highly visible block, so key numbers stand out in long logs."""
    rule = "=" * 78
    logger.info(rule)
    logger.info(title)
    if items:
        width = max(len(key) for key in items)
        for key, value in items.items():
            logger.info(f"  {key:<{width}} : {value}")
    logger.info(rule)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return f"{out}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "unknown"
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock
