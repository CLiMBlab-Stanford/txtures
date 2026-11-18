import os

import pytest
from conftest import write

from txtures.definitions import Definitions, init_store
from txtures.errors import TxturesError


def messages(store) -> list[str]:
    return [str(problem) for problem in Definitions.load(store).validate()]


def test_init_creates_a_valid_store(tmp_path):
    root = tmp_path / "defs"
    init_store(root)
    assert (root / "site" / "site.yml").is_file()
    assert (root / "configs" / "lm_scores" / "pythia410m.yml").is_file()
    assert (root / "workflows" / "pythia.yml").is_file()
    assert messages(root) == []
    assert [p.name for p in tmp_path.iterdir()] == ["defs"]
    assert os.stat(root).st_mode & 0o777 != 0o700


def test_init_example_workflow_resolves_to_pythia(tmp_path):
    root = tmp_path / "defs"
    init_store(root)
    (lm,) = Definitions.load(root).expand("pythia").variants["lm_scores"]
    assert lm.config.model == "EleutherAI/pythia-410m"


def test_init_refuses_an_existing_path(tmp_path):
    root = tmp_path / "defs"
    root.mkdir()
    with pytest.raises(TxturesError, match="already exists"):
        init_store(root)


def test_missing_store_is_an_error(tmp_path):
    with pytest.raises(TxturesError, match="txtures def init"):
        Definitions.load(tmp_path / "absent")


def test_empty_store_is_valid(store):
    assert messages(store) == []


def test_layout_problems_are_all_reported(store):
    write(store / "notes.txt", "")
    write(store / "site" / "other.yml", "")
    write(store / "configs" / "parser" / "a.yml", "")
    write(store / "configs" / "lm_scores" / "bad-name.yml", "")
    write(store / "configs" / "lm_scores" / "wrong.yaml", "")
    write(store / "workflows" / "broken.yml", "lm_scores: [\n")
    found = "\n".join(messages(store))
    assert "notes.txt: unexpected entry in definitions store" in found
    assert "other.yml: unexpected entry in site/" in found
    assert "parser: not a known module directory" in found
    assert "ID 'bad-name' must match" in found
    assert "wrong.yaml: expected a .yml file" in found
    assert "broken.yml:" in found


def test_symlinked_definitions_are_rejected(store, tmp_path):
    target = write(tmp_path / "elsewhere.yml", "model: gpt2\n")
    link = store / "configs" / "lm_scores" / "linked.yml"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert any("symbolic links are not allowed" in m for m in messages(store))


def test_validate_checks_unused_configs_and_every_workflow(store):
    write(store / "configs" / "lm_scores" / "unused.yml", "dtype: float8\n")
    write(store / "configs" / "surprisal" / "main.yml", "extends: other\n")
    write(store / "workflows" / "w.yml", "lm_scores: nothere\n")
    found = "\n".join(messages(store))
    assert "unused.yml: dtype:" in found
    assert "main cannot use extends" in found
    assert "w.yml: no lm_scores config named 'nothere'" in found


def test_heartbeat_keeps_held_locks_fresh(tmp_path):
    from txtures.store import Heartbeat

    path = tmp_path / "entry.lock"
    path.write_text("{}")
    os.utime(path, (0, 0))
    heartbeat = Heartbeat(interval=3600)
    heartbeat.add(path)
    heartbeat.beat()
    assert path.stat().st_mtime > 1e9
    heartbeat.discard(path)
    os.utime(path, (0, 0))
    heartbeat.beat()
    assert path.stat().st_mtime == 0
