import os
from pathlib import Path

import pytest

HEAVY_MODULES = frozenset({"test_lm.py", "test_lm_integration.py"})
"""Test files that need PyTorch and transformers as soon as they are imported.
With `--fast`, they are not even collected, which also saves importing those
libraries; elsewhere, heavy tests carry the `heavy` marker."""


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        help="skip tests that load neural models (marked heavy), for quick local runs",
    )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--fast") and collection_path.name in HEAVY_MODULES:
        return True
    return None


def tools_ready() -> bool:
    """Whether the tools image of this definition is built in the configured resource store."""
    from txtures.engine.planning import plan_resource
    from txtures.environment import run_context
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore
    from txtures.tools import ToolsImage

    try:
        settings = resolve_settings({})
        image = plan_resource("tools", ToolsImage(), run_context(settings))
        return EntryStore(settings.resource_store).get(image.kind, image.identity) is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    needs_tools = [item for item in items if item.get_closest_marker("tools")]
    if needs_tools and not config.getoption("--fast") and not tools_ready():
        skip = pytest.mark.skip(reason="the tools image is not built; run `txtures tools build`")
        for item in needs_tools:
            item.add_marker(skip)
    if not config.getoption("--fast"):
        return
    kept, skipped = [], []
    for item in items:
        (skipped if item.get_closest_marker("heavy") else kept).append(item)
    if skipped:
        config.hook.pytest_deselected(items=skipped)
        items[:] = kept


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
