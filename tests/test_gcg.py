"""Tests for the gcg15 reannotation port.

`tests/data/gcg/expected.gcg15` was written by the ModelBlocks pipeline
(editabletrees2linetrees.pl, make-trees-fixed.pl, annotate-gcg15.pl,
killUnaries.pl -l, makefailtrees.py, and the Makefile's tag edits) from the
hand-written trees in `trees.mrg`. Exact agreement on all of WSJ is recorded
in `plans/evidence/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from txtures.gcg import ARTICLE, fail_tree, kill_unaries, ptb_lines, reannotate
from txtures.gcg.perl import PerlString, interpolate, replacement_pieces

DATA = Path(__file__).parent / "data" / "gcg"


def test_reannotation_matches_modelblocks():
    lines = [ARTICLE, *ptb_lines((DATA / "trees.mrg").read_text())]
    expected = (DATA / "expected.gcg15").read_text().splitlines()
    assert [reannotate(line) for line in lines] == expected


def test_ptb_lines_joins_and_strips_the_outer_bracket():
    text = "( (S\n    (NP (DT A) )\n    (VP (VBZ b) )) )\n\n*x* header\n((NP (NN c)))\n"
    assert ptb_lines(text) == ["(S (NP (DT A)) (VP (VBZ b)))", "(NP (NN c))"]


def test_kill_unaries_collapses_chains_and_drops_empties():
    # checked against killUnaries.pl -l
    assert kill_unaries("(S (NP (N (NN a))) (NP (-NONE- *T*-1)))\n") == "(S (NN a))\n"


def test_fail_tree():
    assert fail_tree("(S-lS (NP a) (V-aN b))\n") == "(FAIL (FAIL a) (FAIL b))\n"
    assert fail_tree("(S-lS (N-aD a) (V-aN b))\n") == "(S-lS (N-aD a) (V-aN b))\n"


@pytest.mark.parametrize(
    ("pattern", "replacement", "g", "text", "result"),
    [
        (r"\(([^ ]*) ", r"<\1>", False, "(A (B c))", "<A>(B c))"),
        (r"\(([^ ]*) ", r"<\1>", True, "(A (B c))", "<A><B>c))"),
        (r"x(y)?z", r"[\1]", False, "xz", "[]"),  # an unmatched group inserts nothing
        (r"a(b)", r"$1\^\n", False, "ab", "b^\n"),  # $1, an escaped ^, a newline
        (r"a$", "b", False, "a\n", "b\n"),  # $ matches before the final newline
    ],
)
def test_substitution_semantics(pattern, replacement, g, text, result):
    line = PerlString(text)
    assert line.sub(pattern, replacement, g=g)
    assert line.text == result


def test_variables_are_interpolated():
    variables = {"CODA": "-[fl]x", "NAME": "N"}
    assert interpolate(r"^${NAME}($CODA)", variables) == "^N(-[fl]x)"
    assert replacement_pieces(r"\1$NAME-\2", variables) == [1, "N-", 2]


def test_scalars_and_marking():
    line = PerlString("^S [NP [DT a]] [VP b]^")
    assert line.set("j", 0.5) and not line.set("k", 0)
    assert line.get("j") == 0.5 and line.get("missing") == 0
    line.mark_children()
    assert line.text == "^S <NP [DT a]> <VP b>^"


# The gcg_treebank resource


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
