# src/logger.py
import os
import sys
import logging
from datetime import datetime

def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    Automatically writes both to console and .log file in artifacts/logs/.
    """
    os.makedirs("artifacts/logs", exist_ok=True)
    log_filename = log_file or datetime.now().strftime("log_%Y%m%d_%H%M%S.log")
    log_path = os.path.join("artifacts/logs", log_filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate handlers

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    logger.info(f"Logger initialized. Writing logs to {log_path}")
    return logger
