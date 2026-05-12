"""Logging helpers used across modules."""

import logging
from pathlib import Path

from config import LOG_PATH


def _resolve_log_dir() -> Path:
    """Resolve the log directory relative to the project root.

    Args:
        None

    Returns:
        pathlib.Path: Absolute path to the log directory.
    """
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / LOG_PATH).resolve()


def get_logger(name: str, log_file: str) -> logging.Logger:
    """Create or reuse a logger with file and console handlers.

    Args:
        name (str): Logger name.
        log_file (str): Log file name within the log directory.

    Returns:
        logging.Logger: Configured logger instance.

    Side Effects:
        - Creates log directory and log file if missing.
    """
    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
