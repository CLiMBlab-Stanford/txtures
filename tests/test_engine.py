import json
import os
import threading
from pathlib import Path

import fake_modules
import numpy as np
import pytest
from conftest import write
from fake_modules import CALLS, STATE

from txtures.definitions import Definitions
from txtures.engine import Executor, item_status, plan_request, run_request, table_statuses
from txtures.engine.tables import format_value, replanner
from txtures.errors import DefinitionError, TxturesError
from txtures.modules import RunContext
from txtures.settings import resolve_settings
from txtures.store import EntryStore, try_lock


@pytest.fixture(autouse=True)
def reset_fakes():
    fake_modules.reset()
    yield
    fake_modules.reset()


@pytest.fixture
def env(tmp_path: Path):
    """Settings pointing every location into a temporary directory."""
    locations = {
        key: str(tmp_path / key)
        for key in ("definitions", "inputs", "outputs", "resources", "cache")
    }
    (tmp_path / "definitions").mkdir()
    (tmp_path / "inputs").mkdir()
    settings = resolve_settings(locations, env={}, user_home=tmp_path / "home")
    return settings


def definitions(settings) -> Definitions:
    return Definitions.load(settings.definitions, fake_modules.registry())


def run(settings, name="doc", workflow="main", modules=None):
    plan = plan_request(settings, definitions(settings), name, workflow, modules)
    executor = Executor(
        plan, EntryStore(settings.cache), EntryStore(settings.resource_store), RunContext()
    )
    directory, written = run_request(plan, executor, modules)
    return plan, executor, directory, written


def read_table(directory: Path, name="doc") -> list[list[str]]:
    lines = (directory / f"{name}.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines]


def add_input(settings, text: str, name="doc.tsv") -> None:
    write(settings.inputs / name, text)


DOCS = "document_id\ttext\tcond\na\tThe horse raced.\tx\nb\tIt fell.\ty\nc\tThe end.\tz\n"


# Running and reuse


def test_run_writes_table_columns_and_manifest(env):
    add_input(env, DOCS)
    plan, executor, directory, written = run(env)
    assert written
    assert directory.parent == env.outputs / "doc"
    assert directory.name == f"main-{plan.table_identity[:12]}"
    rows = read_table(directory)
    assert rows[0] == [
        "document_id",
        "region_index",
        "text",
        "char_start",
        "char_end",
        "cond",
        "length",
        "double",
    ]
    assert rows[1] == ["a", "1", "The", "0", "3", "x", "3.0", "6.0"]
    assert rows[3] == ["a", "3", "raced.", "10", "16", "x", "6.0", "12.0"]
    assert len(rows) == 1 + 7
    columns = read_table(directory, "doc.columns")
    assert columns[0] == ["column", "module", "feature", "labels", "variant"]
    assert [row[0] for row in columns[1:]] == ["length", "double"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["digest"] == plan.table_identity
    assert manifest["input"]["name"] == "doc"
    assert {v["module"] for v in manifest["variants"].values()} == {"fake_scores", "fake_feature"}
    scores = next(v for v in manifest["variants"].values() if v["module"] == "fake_scores")
    assert scores["resources"]["weights"]["spec"]["revision"] == "rev1"
    assert CALLS["fake_scores"] == 3
    assert CALLS["build"] == 1
    assert executor.counts[plan.variants[0].identity].computed == 3


def test_table_has_an_id_column_for_each_level_the_input_defines(env):
    add_input(
        env,
        "document_id\tparagraph_id\tsentence_id\tregion_id\ttext\tcond\n"
        "a\t1\t1\t2\thorse\tx\n"
        "a\t1\t1\t1\tThe\tx\n"
        "a\t2\t1\t1\tfell.\ty\n",
    )
    _, _, directory, _ = run(env)
    rows = read_table(directory)
    assert rows[0][:9] == [
        "document_id",
        "paragraph_id",
        "sentence_id",
        "region_id",
        "region_index",
        "text",
        "char_start",
        "char_end",
        "cond",
    ]
    assert [row[:9] for row in rows[1:]] == [
        ["a", "1", "1", "1", "1", "The", "0", "3", "x"],
        ["a", "1", "1", "2", "2", "horse", "4", "9", "x"],
        ["a", "2", "1", "1", "3", "fell.", "10", "15", "y"],
    ]


def test_second_run_is_a_no_op(env):
    add_input(env, DOCS)
    run(env)
    CALLS.clear()
    _, _, _, written = run(env)
    assert not written
    assert sum(CALLS.values()) == 0


def test_deleted_table_rebuilds_from_stored_items(env):
    add_input(env, DOCS)
    _, _, directory, _ = run(env)
    original = (directory / "doc.tsv").read_bytes()
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    CALLS.clear()
    _, _, rebuilt, written = run(env)
    assert written and rebuilt == directory
    assert (rebuilt / "doc.tsv").read_bytes() == original
    assert CALLS["fake_scores"] == 0
    assert CALLS["build"] == 0


def test_editing_one_document_recomputes_only_it(env):
    add_input(env, DOCS)
    _, _, old, _ = run(env)
    add_input(env, DOCS.replace("It fell.", "It fell down."))
    CALLS.clear()
    plan, executor, new, _ = run(env)
    assert new != old
    assert CALLS["fake_scores"] == 1
    counts = executor.counts[plan.variants[0].identity]
    assert (counts.computed, counts.reused) == (1, 2)


def test_metadata_edits_reuse_items_but_change_the_table(env):
    add_input(env, DOCS)
    _, _, old, _ = run(env)
    add_input(env, DOCS.replace("\tz\n", "\tw\n"))
    CALLS.clear()
    _, _, new, _ = run(env)
    assert new != old
    assert CALLS["fake_scores"] == 0
    assert read_table(new)[-1][5] == "w"


def test_identical_documents_are_computed_once(env):
    add_input(env, "text\nsame words\nsame words\nother\n")
    run(env)
    assert CALLS["fake_scores"] == 2


# Sweeps, resources, and identity


def test_sweeps_share_upstream_items_and_label_columns(env):
    add_input(env, DOCS)
    write(env.definitions / "configs" / "fake_feature" / "swept.yml", "offset: [0, 10]\n")
    write(env.definitions / "workflows" / "swept.yml", "fake_feature: swept\n")
    _, _, directory, _ = run(env, workflow="swept")
    header = read_table(directory)[0]
    assert header[-4:] == ["length_offset0", "double_offset0", "length_offset10", "double_offset10"]
    assert read_table(directory)[1][-4:] == ["3.0", "6.0", "13.0", "26.0"]
    assert CALLS["fake_scores"] == 3


def test_upstream_sweep_labels_downstream_columns(env):
    add_input(env, DOCS)
    write(env.definitions / "configs" / "fake_scores" / "two.yml", "scale: [1, 2]\n")
    write(env.definitions / "workflows" / "two.yml", "fake_scores: two\n")
    _, _, directory, _ = run(env, workflow="two", modules=["fake_feature"])
    assert read_table(directory)[0][-4:] == [
        "length_scale1",
        "double_scale1",
        "length_scale2",
        "double_scale2",
    ]
    assert CALLS["fake_scores"] == 6
    assert CALLS["build"] == 2


def test_loaded_resources_are_released_after_their_last_variant(env):
    add_input(env, DOCS)
    write(env.definitions / "configs" / "fake_scores" / "two.yml", "scale: [1, 2]\n")
    write(env.definitions / "workflows" / "two.yml", "fake_scores: two\n")
    _, executor, _, _ = run(env, workflow="two", modules=["fake_feature"])
    assert CALLS["load"] == 2
    assert executor._loaded == {}


def test_resolved_resource_changes_identity(env):
    add_input(env, DOCS)
    first, _, old, _ = run(env)
    STATE["resolved_revision"] = "rev2"
    CALLS.clear()
    second, _, new, _ = run(env)
    assert second.table_identity != first.table_identity
    assert new != old
    assert CALLS["build"] == 1
    assert CALLS["fake_scores"] == 3


def test_pinned_revision_ignores_resolution_changes(env):
    add_input(env, DOCS)
    write(env.definitions / "configs" / "fake_scores" / "main.yml", "revision: fixed\n")
    first, *_ = run(env)
    STATE["resolved_revision"] = "rev2"
    second = plan_request(env, definitions(env), "doc")
    assert second.table_identity == first.table_identity
    assert first.notes == ()


def test_unpinned_revision_produces_a_note(env):
    add_input(env, DOCS)
    plan = plan_request(env, definitions(env), "doc")
    assert any("revision is not pinned" in note for note in plan.notes)


def test_item_status_counts_present_and_missing(env):
    add_input(env, DOCS)
    plan = plan_request(env, definitions(env), "doc")
    (status,) = item_status(plan, EntryStore(env.cache))
    assert (status.present, status.missing) == (0, 3)
    run(env)
    (status,) = item_status(plan, EntryStore(env.cache))
    assert (status.present, status.missing) == (3, 0)


# Planning errors


def test_module_check_failures_block_planning(env):
    add_input(env, DOCS)
    write(env.definitions / "configs" / "fake_feature" / "main.yml", "mode: half\n")
    with pytest.raises(DefinitionError, match="half mode needs an even number"):
        plan_request(env, definitions(env), "doc")


def test_metadata_column_clashing_with_feature_is_rejected(env):
    add_input(env, "text\tlength\nhello\t5\n")
    with pytest.raises(DefinitionError, match="metadata column 'length' clashes"):
        plan_request(env, definitions(env), "doc")


# Failures and concurrency


def test_failure_keeps_finished_items_and_writes_no_table(env):
    add_input(env, "text\none two\nboom here\nthree four\nfive six\n")
    STATE["fail_on"] = "boom"
    with pytest.raises(RuntimeError, match="failed on 'boom here'"):
        run(env)
    plan = plan_request(env, definitions(env), "doc")
    (status,) = item_status(plan, EntryStore(env.cache))
    assert status.present == 0  # the failing batch held documents 1 and 2
    assert not (env.outputs / "doc").exists() or not any(p for p in (env.outputs / "doc").iterdir())
    assert not list(Path(env.cache).rglob("*.lock"))
    assert not [p for p in Path(env.cache).rglob(".*") if p.is_dir()]
    STATE["fail_on"] = None
    CALLS.clear()
    run(env)
    assert CALLS["fake_scores"] == 4


def test_failure_after_a_finished_batch_keeps_that_batch(env):
    add_input(env, "text\none two\nthree four\nboom here\nfive six\n")
    STATE["fail_on"] = "boom"
    with pytest.raises(RuntimeError):
        run(env)
    plan = plan_request(env, definitions(env), "doc")
    (status,) = item_status(plan, EntryStore(env.cache))
    assert status.present == 2


def test_concurrent_runs_compute_each_item_once(env):
    add_input(env, "text\n" + "".join(f"document {i}\n" for i in range(6)))
    STATE["delay"] = 0.05
    errors = []

    def worker():
        try:
            run(env)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert CALLS["fake_scores"] == 6
    assert CALLS["build"] == 1


def test_stale_lock_from_a_dead_process_is_broken(tmp_path):
    path = tmp_path / "entry.lock"
    path.write_text(json.dumps({"host": os.uname().nodename, "pid": 2**22 + 7, "started": 0}))
    assert try_lock(path)


def test_live_lock_is_respected(tmp_path):
    path = tmp_path / "entry.lock"
    assert try_lock(path)
    assert not try_lock(path)


def test_entry_without_manifest_is_absent_and_replaced(tmp_path):
    store = EntryStore(tmp_path)
    digest = "ab" + "0" * 62
    partial = store.path("kind", digest)
    partial.mkdir(parents=True)
    (partial / "junk").write_text("x")
    assert store.get("kind", digest) is None
    store.publish("kind", digest, lambda d: (d / "data").write_text("y"), {})
    assert store.get("kind", digest) == partial
    assert not (partial / "junk").exists()


# Status and cleanup


def test_status_reports_current_stale_and_broken(env):
    add_input(env, DOCS)
    _, _, old, _ = run(env)
    add_input(env, DOCS.replace("It fell.", "It rose."))
    _, _, new, _ = run(env)
    (env.outputs / "doc" / "main-broken").mkdir()
    (env.outputs / "doc" / ".main-x.tmp").mkdir()
    statuses = {
        s.path.name: s for s in table_statuses(env.outputs, replanner(env, definitions(env)))
    }
    assert statuses[new.name].state == "current"
    assert statuses[old.name].state == "stale"
    assert statuses["main-broken"].state == "broken"
    assert statuses[".main-x.tmp"].state == "broken"


def test_status_marks_tables_of_removed_inputs_stale(env):
    add_input(env, DOCS)
    _, _, directory, _ = run(env)
    (env.inputs / "doc.tsv").unlink()
    (status,) = table_statuses(env.outputs, replanner(env, definitions(env)))
    assert status.state == "stale"
    assert "no input named 'doc'" in status.reason


# Formatting


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, "NA"),
        (float("nan"), "NA"),
        (float("inf"), "Inf"),
        (-np.inf, "-Inf"),
        (True, "TRUE"),
        (np.bool_(False), "FALSE"),
        (0.1, "0.1"),
        (np.float32(0.1), "0.1"),
        (np.float64(1e-5), "1e-05"),
        (3, "3"),
        (np.int64(7), "7"),
        ("word", "word"),
    ],
)
def test_format_value(value, text):
    assert format_value(value) == text


def test_format_value_rejects_tabs():
    with pytest.raises(TxturesError, match="tab or line break"):
        format_value("a\tb")
