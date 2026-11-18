import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo changes to the process environment, which the command line controls."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def write(path: Path, text: str) -> Path:
    """Write `text` to `path`, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An empty definitions store directory."""
    root = tmp_path / "definitions"
    root.mkdir()
    return root
