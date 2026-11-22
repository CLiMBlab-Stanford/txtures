"""Tests for the `gcg_treebank` resource.

`tests/data/gcg/expected.gcg15` was written by the ModelBlocks pipeline from
the hand-written trees in `trees.mrg`; the reannotation itself is tested in
prsr.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "gcg"


def fake_ptb(root: Path, nested: bool = True) -> Path:
    """A stand-in Treebank-3: the hand-written trees as WSJ section 02, split in two files."""
    wsj = (root / "treebank_3" if nested else root) / "parsed" / "mrg" / "wsj" / "02"
    wsj.mkdir(parents=True)
    trees = (DATA / "trees.mrg").read_text().splitlines()
    (wsj / "wsj_0201.mrg").write_text("\n".join(trees[:3]) + "\n")
    (wsj / "wsj_0202.mrg").write_text("\n".join(trees[3:]) + "\n")
    return root


def test_treebank_is_identified_by_contents_not_path(tmp_path):
    from txtures.engine.planning import plan_resource
    from txtures.gcg.treebank import GcgTreebank
    from txtures.modules.base import RunContext

    first = fake_ptb(tmp_path / "a")
    second = fake_ptb(tmp_path / "b", nested=False)
    spec = {"sections": [2]}
    one = plan_resource("t", GcgTreebank(spec), RunContext(corpora={"ptb3": first}))
    two = plan_resource("t", GcgTreebank(spec), RunContext(corpora={"ptb3": second}))
    assert one.identity == two.identity
    assert one.resolved["files"] == ["02/wsj_0201.mrg", "02/wsj_0202.mrg"]
    (second / "parsed/mrg/wsj/02/wsj_0202.mrg").write_text("( (S (NN x)) )\n")
    three = plan_resource("t", GcgTreebank(spec), RunContext(corpora={"ptb3": second}))
    assert three.identity != one.identity


def test_treebank_without_the_setting_explains_itself():
    from txtures.errors import TxturesError
    from txtures.gcg.treebank import GcgTreebank
    from txtures.modules.base import RunContext

    with pytest.raises(TxturesError, match="LDC99T42.*corpora.ptb3"):
        GcgTreebank({}).resolve(RunContext())


def test_treebank_build_matches_modelblocks(tmp_path, capsys, monkeypatch):
    from txtures.cli import main
    from txtures.gcg.treebank import TREES_FILE, properties, read_trees
    from txtures.store import EntryStore

    for key in ("inputs", "outputs", "resources", "cache", "definitions"):
        monkeypatch.setenv(f"TXTURES_{key.upper()}", str(tmp_path / key))
    monkeypatch.setenv("TXTURES_CORPORA_PTB3", str(fake_ptb(tmp_path / "ptb")))
    monkeypatch.setenv("TXTURES_EXECUTION_EXECUTOR", "local")
    assert main(["resources", "build", "--kind", "gcg_treebank", "--spec", "sections=[2]"]) == 0
    assert "built resource gcg_treebank" in capsys.readouterr().out
    store = EntryStore(tmp_path / "resources" / "store")
    [entry] = list((tmp_path / "resources" / "store" / "gcg_treebank").glob("*/*"))
    expected = (DATA / "expected.gcg15").read_text().splitlines()
    trees = [line for line in expected if line != "!ARTICLE"]
    assert list(read_trees(entry / TREES_FILE)) == trees
    assert properties(entry) == {"trees": 8, "fail_trees": 1}
    assert store.get("gcg_treebank", entry.name) == entry
