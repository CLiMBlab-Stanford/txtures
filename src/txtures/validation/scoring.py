"""Bracket scores and per-word log-likelihood of parses, with ModelBlocks' evalb as the judge.

Both systems' trees are rebuilt from their best states with
`prsr.evaluation.tree_from_states` (equal to ModelBlocks'
`tokdecs2linetrees.py`), normalized alike (`for_scoring`), and scored by
ModelBlocks' copy of evalb with `resource-general/srcmodel/new.prm`, labeled
and with every label `X` (unlabeled). A failed parse is a flat tree of
`UNF` nodes, which `new.prm` deletes, so it counts against recall.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prsr.evaluation import for_scoring, gold_for_scoring, tree_from_states, unlabeled
from prsr.trees import Tree

from txtures.validation import Paths, run


def _flat(words: list[str]) -> Tree:
    return Tree("UNF", [Tree("UNF", [Tree(w)]) for w in words])


def read_tokdecs_states(path: Path, lengths: list[int]):
    """synproc's output as words, states (with string symbols), and surprisal, per sentence."""
    symbols: list[str] = []
    index: dict[str, int] = {}

    def symbol(x: str) -> int:
        if x not in index:
            index[x] = len(symbols)
            symbols.append(x)
        return index[x]

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("word pos"):
            continue
        f = line.split(" ")
        pairs = tuple(
            (symbol(a), symbol(b)) for a, b in (x.split("/") for x in f[4].split(";") if x)
        )
        rows.append((f[0], (symbol(f[1]), int(f[2]), int(f[3]), pairs), float(f[-1])))
    sentences, position = [], 0
    for n in lengths:
        chunk = rows[position : position + n]
        sentences.append(([r[0] for r in chunk], [r[1] for r in chunk], [r[2] for r in chunk]))
        position += n
    return symbols, sentences


def _evalb(paths: Paths, gold: Path, test: Path) -> dict[str, float]:
    out = test.with_suffix(test.suffix + ".evalb")
    prm = paths.modelblocks / "resource-general" / "srcmodel" / "new.prm"
    run(f"{paths.mb / 'evalb'} -p {prm} {gold} {test} > {out}")
    text = out.read_text(encoding="utf-8", errors="replace")
    summary = text[text.index("-- All --") :]
    found = {}
    for key, name in (
        ("Bracketing Recall", "recall"),
        ("Bracketing Precision", "precision"),
        ("Bracketing FMeasure", "f1"),
        ("Complete match", "complete"),
        ("Tagging accuracy", "tagging"),
    ):
        match = re.search(rf"{key}\s*=\s*([0-9.]+)", summary)
        found[name] = float(match.group(1)) if match else math.nan
    return found


def score(paths: Paths, name: str, parse: str) -> dict:
    """Score a parse of `NAME.linetoks` against `NAME.gold`.

    `parse` is a synproc `.tokdecs` file or a txtures `.json` file (from
    `python -m txtures.validation parse`) in the ModelBlocks work directory.
    """
    gold_lines = (paths.mb / f"{name}.gold").read_text(encoding="utf-8").splitlines()
    sentences = [
        s.split() for s in (paths.mb / f"{name}.linetoks").read_text(encoding="utf-8").splitlines()
    ]
    source = paths.mb / parse
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        symbols = data["symbols"]
        shifted = False
        parsed = [
            (
                s["words"],
                [
                    None if x is None else (x[0], x[1], x[2], tuple(map(tuple, x[3])))
                    for x in s["states"]
                ],
                s["surprisal"],
            )
            for s in data["sentences"]
        ]
    else:
        symbols, parsed = read_tokdecs_states(source, [len(s) for s in sentences])
    shifted = source.suffix != ".json"  # synproc's output pairs words with the next store
    gold_trees, test_trees, failed, surprisal, words = [], [], 0, 0.0, 0
    for gold, (ws, states, bits) in zip(gold_lines, parsed, strict=True):
        tree = tree_from_states(ws, states, symbols, shifted)
        if tree is None:
            failed += 1
            tree = _flat(ws)
        gold_trees.append(gold_for_scoring(gold))
        test_trees.append(for_scoring(tree))
        finite = [b for b in bits if not math.isnan(b)]
        surprisal += sum(finite)
        words += len(ws)
    stem = paths.mb / f"{name}.{source.stem}"
    results = {"sentences": len(gold_trees), "failed": failed, "bits_per_word": surprisal / words}
    for kind, transform in (("labeled", lambda t: t), ("unlabeled", unlabeled)):
        gold_file = Path(f"{stem}.gold.{kind}")
        test_file = Path(f"{stem}.test.{kind}")
        gold_file.write_text("".join(f"{transform(t)}\n" for t in gold_trees), encoding="utf-8")
        test_file.write_text("".join(f"{transform(t)}\n" for t in test_trees), encoding="utf-8")
        results[kind] = _evalb(paths, gold_file, test_file)
    print(json.dumps(results, indent=1))
    return results
