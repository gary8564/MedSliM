import logging
import sys

# ANSI escape sequences for colors
_RESET = "\x1b[0m"
_COLORS = {
    "DEBUG": "\x1b[34m",    # blue
    "INFO": "\x1b[32m",     # green
    "WARNING": "\x1b[33m",  # yellow
    "ERROR": "\x1b[31m",    # red
    "CRITICAL": "\x1b[41m", # white on red bg
}

class ColoredFormatter(logging.Formatter):
    """
    Injects ANSI color codes into the LEVELNAME (and optionally the message).
    """
    def format(self, record):
        level = record.levelname
        if level in _COLORS:
            color = _COLORS[level]
            # colorize levelname
            record.levelname = f"{color}{level}{_RESET}"
            # (optional) colorize the message text itself:
            record.msg = f"{color}{record.msg}{_RESET}"
        return super().format(record)
