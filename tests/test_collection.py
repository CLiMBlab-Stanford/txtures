"""Tests for finding and removing stale and broken store entries."""

from __future__ import annotations

import json
import os
import socket
import time

from txtures.engine import collection
from txtures.ngram_text import NGRAM_TEXT_VERSION
from txtures.store import EntryStore


class Module:
    contract_version = 2


REGISTRY = {"scores": Module}


def entry(store: EntryStore, kind: str, digest: str, **manifest):
    path = store.path(kind, digest)
    path.mkdir(parents=True)
    (path / "data").write_text("x" * 10)
    (path / "manifest.json").write_text(json.dumps({"digest": digest, "kind": kind, **manifest}))
    return path


def corpus(dataset, files=("corpus.txt",), text_version=NGRAM_TEXT_VERSION):
    return {
        "version": 1,
        "spec": {
            "dataset": str(dataset),
            "files": "*.txt",
            "file_list": list(files),
            "text_version": text_version,
        },
    }


def digest(n: int) -> str:
    return f"{n:02d}" + "0" * 62


def found(garbage):
    return {(g.store, g.path.name): g.reason for g in garbage}


def test_stale_entries_cascade_and_current_ones_stay(tmp_path):
    items, resources = EntryStore(tmp_path / "cache"), EntryStore(tmp_path / "resources")
    (tmp_path / "kept").mkdir()
    (tmp_path / "kept" / "corpus.txt").write_text("a\n")
    (tmp_path / "changed").mkdir()
    (tmp_path / "changed" / "other.txt").write_text("a\n")
    entry(resources, "ngram_corpus", digest(1), **corpus(tmp_path / "kept"))
    entry(resources, "ngram_corpus", digest(2), **corpus(tmp_path / "gone"))
    entry(resources, "ngram_corpus", digest(3), **corpus(tmp_path / "changed"))
    entry(resources, "ngram_corpus", digest(4), **corpus("org/data", text_version=0))
    entry(resources, "ngram_corpus", digest(5), **{**corpus("org/data"), "version": 0})
    entry(resources, "ngram_corpus", digest(6), **{**corpus("org/data"), "version": 99})
    entry(resources, "unknown_kind", digest(7), version=0, spec={})
    model = {"version": 1, "spec": {"order": 3}}
    entry(resources, "kenlm_model", digest(8), **model, dependencies={"corpus": digest(1)})
    entry(resources, "kenlm_model", digest(9), **model, dependencies={"corpus": digest(2)})
    item = {"module": "scores", "contract_version": 2, "upstream": {}}
    entry(items, "scores", digest(10), **item, resources={"model": digest(8)})
    entry(items, "scores", digest(11), **item, resources={"model": digest(9)})
    entry(items, "scores", digest(12), **{**item, "contract_version": 1}, resources={})
    entry(items, "scores", digest(13), **{**item, "contract_version": 3}, resources={})
    entry(items, "scores", digest(14), **{**item, "upstream": {"base": digest(12)}}, resources={})
    entry(items, "removed_module", digest(15), module="removed_module", contract_version=0)

    garbage = collection.find(items, resources, REGISTRY)

    assert found(garbage) == {
        ("resources", digest(2)): f"its corpus directory {tmp_path / 'gone'} no longer exists",
        ("resources", digest(3)): (
            f"the files of its corpus directory {tmp_path / 'changed'} have changed"
        ),
        ("resources", digest(4)): f"tokenized by text version 0, now {NGRAM_TEXT_VERSION}",
        ("resources", digest(5)): "built by ngram_corpus version 0, now 1",
        ("resources", digest(9)): f"built from the stale 'corpus' {digest(2)[:12]}",
        ("cache", digest(11)): "computed with the stale resource 'model'",
        ("cache", digest(12)): "computed under contract version 1, now 2",
        ("cache", digest(14)): f"computed from the stale 'base' {digest(12)[:12]}",
    }
    assert all(g.size == len(g.path.joinpath("manifest.json").read_bytes()) + 10 for g in garbage)
    for g in garbage:
        collection.remove(g)
    assert collection.find(items, resources, REGISTRY) == []
    assert items.get("scores", digest(10)) and resources.get("kenlm_model", digest(8))
    assert not any(p.name.startswith(".") for p in tmp_path.rglob("*"))


def test_broken_paths_unless_a_live_lock_covers_them(tmp_path):
    store = EntryStore(tmp_path)
    shard = store.path("scores", digest(1)).parent
    shard.mkdir(parents=True)
    (shard / f".{digest(1)[:12]}.abc").mkdir()  # staging, no lock
    (shard / digest(2)).mkdir()  # no manifest, no lock
    (shard / f".{digest(3)[:12]}.abc").mkdir()  # staging under a live lock
    (shard / digest(3)).mkdir()
    live = {"host": socket.gethostname(), "pid": os.getpid(), "started": time.time()}
    (shard / f"{digest(3)}.lock").write_text(json.dumps(live))
    dead = {"host": socket.gethostname(), "pid": 2**22 + 12345, "started": time.time()}
    (shard / f"{digest(4)}.lock").write_text(json.dumps(dead))
    old = shard / f"{digest(5)}.lock"
    old.write_text(json.dumps({"host": "elsewhere", "pid": 1, "started": 0}))
    os.utime(old, (0, 0))

    garbage = collection.find(store, EntryStore(tmp_path / "none"), REGISTRY)

    assert found(garbage) == {
        ("cache", f".{digest(1)[:12]}.abc"): "left by a failed write",
        ("cache", digest(2)): "no valid manifest",
        ("cache", f"{digest(4)}.lock"): "lock of a process that is gone",
        ("cache", f"{digest(5)}.lock"): "lock of a process that is gone",
    }
    for g in garbage:
        collection.remove(g)
    assert sorted(p.name for p in shard.iterdir()) == sorted(
        [f".{digest(3)[:12]}.abc", digest(3), f"{digest(3)}.lock"]
    )


def test_tools_images_of_other_definitions_from_older_releases_are_stale(monkeypatch):
    from txtures.tools import ToolsImage, definition_digest

    monkeypatch.setattr("txtures.__version__", "0.6.0")
    image = ToolsImage()
    current = {"spec": {"definition_sha256": definition_digest()}, "txtures_version": "0.5.0"}
    other = {"spec": {"definition_sha256": "0" * 64}, "txtures_version": "0.5.0"}
    assert image.stale(current) is None
    assert image.stale(other) == "built from another tools definition by txtures 0.5.0"
    assert image.stale({**other, "txtures_version": "0.6.0.dev3"}) is None
    assert image.stale({**other, "txtures_version": "0.7.0"}) is None


def test_built_before():
    assert collection.built_before("0.5.0", "0.6.0")
    assert collection.built_before("0.9.1", "0.10.0")
    assert not collection.built_before("0.6.0", "0.6.0+local")
    assert not collection.built_before("0.6.0.dev1", "0.6.0")
    assert not collection.built_before(None, "0.6.0")
