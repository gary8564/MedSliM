"""Shared pytest configuration."""
import os
import tempfile
from pathlib import Path


def pytest_configure() -> None:
    """
    Ensure Python's temp root exists before pytest creates tmp_path fixtures.
    """
    tmp_root = Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(env_name)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)