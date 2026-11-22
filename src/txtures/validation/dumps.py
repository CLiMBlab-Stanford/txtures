"""Reading the state that the Java harnesses (`OneStep`, `SplitMergeStep`) dump, and comparing it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from prsr.em import Corpus, Rules, index_trees
from prsr.lexicon import Lexicon
from prsr.trees import training_tree


@dataclass
class Dump:
    """A Berkeley grammar and lexicon as `OneStep.dump` writes them."""

    categories: dict[str, int]
    substates: dict[int, int]
    binary: dict[tuple[int, int, int], np.ndarray]
    unary: dict[tuple[int, int], np.ndarray]
    counts: dict[str, dict] = field(default_factory=dict)  # L, LU, T, TU, W
    totals: tuple[float, float] = (0.0, 0.0)
    log_likelihood: float | None = None
    grammar_smoother: dict[int, np.ndarray] | None = None
    lexicon_smoother: dict[int, np.ndarray] | None = None
    split_trees: dict[int, object] = field(default_factory=dict)
    merge_weights: dict[int, np.ndarray] = field(default_factory=dict)
    deltas: dict[int, np.ndarray] = field(default_factory=dict)
    pairs: dict[int, np.ndarray] = field(default_factory=dict)

    def grammar(self, unary_keys=None) -> Rules:
        unary = (
            self.unary
            if unary_keys is None
            else {k: v for k, v in self.unary.items() if k in unary_keys}
        )
        return Rules(self.substates, self.binary, unary, self.grammar_smoother)

    def lexicon(self) -> Lexicon:
        c = self.counts
        return Lexicon(
            self.substates,
            self.lexicon_smoother,
            c["L"],
            c["LU"],
            c["T"],
            c["TU"],
            c["W"],
            *self.totals,
        )


def parse_split_tree(text: str):
    """Berkeley's `Tree<Short>` text: `(label child ...)` with leaves as numbers."""
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    position = 0

    def node():
        nonlocal position
        if tokens[position] != "(":
            value = int(tokens[position])
            position += 1
            return value
        position += 1
        label = int(tokens[position])
        position += 1
        children = []
        while tokens[position] != ")":
            children.append(node())
        position += 1
        return [label, children]

    return node()


def load(path: Path, extra: Path | None = None) -> Dump:
    """Read a dump, and optionally a `SplitMergeStep` `.extra` file beside it."""
    dump = Dump({}, {}, {}, {})
    counts = {"L": {}, "LU": {}, "T": {}, "TU": {}, "W": {}}
    gs, ls = {}, {}
    sub = dump.substates
    for line in path.open(encoding="utf-8"):
        f = line.rstrip("\n").split(" ")
        kind = f[0]
        if kind == "LL":
            dump.log_likelihood = float(f[1])
        elif kind == "TAG":
            dump.categories[f[2]] = int(f[1])
            sub[int(f[1])] = int(f[3])
        elif kind == "B":
            p, left, right = map(int, f[1:4])
            values = np.array(f[4:], float)
            dump.binary[(p, left, right)] = values.reshape(sub[left], sub[right], sub[p]).transpose(
                2, 0, 1
            )
        elif kind == "U":
            p, c = map(int, f[1:3])
            dump.unary[(p, c)] = np.array(f[3:], float).reshape(sub[c], sub[p]).T
        elif kind in ("T", "TU"):
            counts[kind][int(f[1])] = np.array(f[2:], float)
        elif kind in ("L", "LU"):
            counts[kind].setdefault(int(f[1]), {})[f[2]] = np.array(f[3:], float)
        elif kind == "W":
            counts["W"][" ".join(f[1:-1])] = float(f[-1])
        elif kind == "TOT":
            dump.totals = (float(f[1]), float(f[2]))
        elif kind in ("GS", "LS"):
            values = np.array(f[2:], float)
            size = int(round(len(values) ** 0.5))
            (gs if kind == "GS" else ls)[int(f[1])] = values.reshape(size, size)
        elif kind == "ST":
            dump.split_trees[int(f[1])] = parse_split_tree(line.split(" ", 2)[2].strip())
    dump.counts = counts
    dump.grammar_smoother = gs or None
    dump.lexicon_smoother = ls or None
    if extra is not None:
        for line in extra.open(encoding="utf-8"):
            f = line.rstrip("\n").split(" ", 2)
            values = np.array(f[2].split() if len(f) > 2 else [], float)
            if f[0] == "MW":
                dump.merge_weights[int(f[1])] = values
            elif f[0] == "D":
                dump.deltas[int(f[1])] = values
            elif f[0] == "P":
                dump.pairs[int(f[1])] = values
            elif f[0] == "ST":
                dump.split_trees[int(f[1])] = parse_split_tree(f[2].strip())
    return dump


def training_corpus(path: Path, categories: dict[str, int]) -> Corpus:
    """The trainer's corpus from ModelBlocks' prepared trees (`.nol.prtrm.senttrees`)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("(")]
    return index_trees([training_tree(line) for line in lines], categories)


def compare(mine: dict, reference: dict, what: str) -> float:
    """Print and return the largest difference between two tables of arrays or numbers.

    Each value's difference is relative to the largest entry of the reference
    value, so rounding-level noise next to large entries does not count.
    """
    keys = set(mine) | set(reference)
    worst, unmatched = 0.0, 0
    for key in keys:
        a, r = mine.get(key), reference.get(key)
        if a is None or r is None:
            if np.any(np.asarray(a if a is not None else r) != 0):
                unmatched += 1
            continue
        a, r = np.asarray(a, float), np.asarray(r, float)
        if a.shape != r.shape:
            unmatched += 1
            continue
        if r.size:
            worst = max(worst, float(np.max(np.abs(a - r)) / max(float(np.abs(r).max()), 1e-300)))
    print(
        f"{what}: {len(keys)} entries, max relative difference {worst:.2e}, unmatched {unmatched}"
    )
    return worst


def flat(table: dict) -> dict:
    """`{tag: {word: v}}` as `{(tag, word): v}`."""
    return {(outer, inner): v for outer, values in table.items() for inner, v in values.items()}
