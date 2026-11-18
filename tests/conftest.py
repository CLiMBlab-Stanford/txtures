from pathlib import Path

import pytest


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
