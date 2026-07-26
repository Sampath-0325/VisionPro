"""
==============================================================
VisionPro Studio
Logger Module
==============================================================

Configurable logging system with file and console output.

Author : VisionPro Studio Team
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ==============================================================
# Color Codes for Console
# ==============================================================

class LogColors:
    """ANSI color codes for styled console output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


# ==============================================================
# Custom Formatter
# ==============================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.CYAN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BG_RED + LogColors.WHITE,
    }

    LEVEL_ICONS = {
        logging.DEBUG: "*",
        logging.INFO: ">",
        logging.WARNING: "!",
        logging.ERROR: "x",
        logging.CRITICAL: "X",
    }

    def format(self, record):

        color = self.LEVEL_COLORS.get(
            record.levelno, LogColors.WHITE
        )

        icon = self.LEVEL_ICONS.get(
            record.levelno, "-"
        )

        timestamp = datetime.fromtimestamp(
            record.created
        ).strftime("%H:%M:%S")

        formatted = (
            f"{LogColors.DIM}{timestamp}{LogColors.RESET} "
            f"{color}{icon} {record.levelname:<8}{LogColors.RESET} "
            f"{record.getMessage()}"
        )

        return formatted


# ==============================================================
# File Formatter
# ==============================================================

class FileFormatter(logging.Formatter):
    """Plain-text formatter for log files."""

    def __init__(self):

        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# ==============================================================
# Logger Setup
# ==============================================================

def setup_logger(
    name="VisionPro",
    level="INFO",
    log_to_file=True,
    log_dir="logs"
):
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_to_file : bool
        Whether to also write logs to a file.
    log_dir : str
        Directory for log files.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ----------------------------------------------------------
    # Console Handler
    # ----------------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    # ----------------------------------------------------------
    # File Handler
    # ----------------------------------------------------------

    if log_to_file:

        log_path = Path(log_dir)

        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_handler = logging.FileHandler(
            log_path / f"visionpro_{timestamp}.log",
            encoding="utf-8"
        )

        file_handler.setFormatter(FileFormatter())

        logger.addHandler(file_handler)

    return logger


# ==============================================================
# Convenience Functions
# ==============================================================

def get_logger(name="VisionPro"):
    """Get existing logger or create a default one."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger


class StageTimer:
    """
    Context manager for timing processing stages.

    Usage
    -----
    with StageTimer("Grayscale Conversion", logger):
        process_grayscale(image)
    """

    def __init__(self, stage_name, logger=None):

        self.stage_name = stage_name
        self.logger = logger or get_logger()
        self.start_time = None
        self.elapsed = None

    def __enter__(self):

        self.start_time = datetime.now()

        self.logger.info(f"Starting: {self.stage_name}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.elapsed = (
            datetime.now() - self.start_time
        ).total_seconds()

        if exc_type:

            self.logger.error(
                f"Failed: {self.stage_name} "
                f"({self.elapsed:.3f}s) — {exc_val}"
            )

        else:

            self.logger.info(
                f"Completed: {self.stage_name} "
                f"({self.elapsed:.3f}s)"
            )

        return False
