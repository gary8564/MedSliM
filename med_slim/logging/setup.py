import logging
import sys
from med_slim.logging.formatters import ColoredFormatter

__all__ = ["init_logging"]

def init_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(filename)s - %(levelname)s : %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure the root logger with a single StreamHandler, colored output,
    and a filename-aware format. Safe to call multiple times (it will no-op
    if handler is already installed).
    """
    root = logging.getLogger()
    # If you re-run init(), don’t attach another handler
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(fmt, datefmt))
    
    root.setLevel(level)
    root.addHandler(handler)