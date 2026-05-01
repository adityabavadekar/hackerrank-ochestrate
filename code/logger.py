"""Centralized logger with colorama-colored output for structured, level-tagged console logging."""

import logging
import sys

from colorama import Fore, Style, init

init(autoreset=True)

_LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
}

_LEVEL_TAGS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO ",
    logging.WARNING: "WARN ",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT ",
}


class _ColorFormatter(logging.Formatter):
    """Formats log records with colorama-colored level tags and dim timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        tag = _LEVEL_TAGS.get(record.levelno, record.levelname)
        timestamp = self.formatTime(record, "%H:%M:%S")
        dim = Style.DIM
        reset = Style.RESET_ALL
        prefix = f"{dim}{timestamp}{reset} {color}[{tag}]{reset}"
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return f"{prefix} {message}"


def get_logger(name: str = "orchestrate") -> logging.Logger:
    """Returns a named logger configured with colored console output.

    Idempotent: calling twice with the same name reuses the existing logger
    instead of stacking handlers.

    Args:
        name: Logger hierarchy name. Use __name__ in each module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Module-level shortcuts so callers can do:
#   from code.logger import logger
#   logger.info("...")
logger = get_logger("orchestrate")
