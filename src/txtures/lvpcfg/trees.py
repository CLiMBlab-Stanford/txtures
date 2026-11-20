"""Training trees for the latent-variable PCFG, prepared as the Berkeley trainer prepares them.

ModelBlocks trains its synproc grammars with the Berkeley parser's
`GrammarTrainer` on GCG trees (`resource-lvpcfg/Makefile`). Before training,
the trees go through these steps, reproduced here:

1. `escape` (ModelBlocks' `.extrparplusops` step): wrap each tree in an
   unlabeled root bracket and turn `-` before a lowercase letter, and
   `-PRTRM`, into `+`, so that Berkeley's label normalization keeps GCG
   operators.
2. `parse`: read the tree; the unlabeled root is labeled `ROOT`.
3. `normalize` (Berkeley's `StandardTreeNormalizer`): cut each nonterminal
   label at its first `-`, `=`, or `^` after the first character, remove
   `-NONE-` nodes and nodes left empty, and collapse unary chains of
   identical labels.
4. `annotate` (`TreeAnnotations.processTree` with vertical Markovization 1,
   horizontal 0, and RIGHT binarization): add `^g` to every nonterminal
   below the root that is not a preterminal; binarize nodes with more than
   two children into a left-branching chain of `@LABEL` nodes.

These steps depend only on the bracketing and labels, so they serve any CFG
treebank, not only GCG.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

ROOT = "ROOT"


@dataclass
class Tree:
    """A tree node: a label and its children; a leaf (a word) has none."""

    label: str
    children: list[Tree] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_preterminal(self) -> bool:
        return len(self.children) == 1 and self.children[0].is_leaf

    def __str__(self) -> str:
        if self.is_leaf:
            return self.label
        return "(" + self.label + " " + " ".join(str(c) for c in self.children) + ")"

    def leaves(self) -> Iterator[str]:
        if self.is_leaf:
            yield self.label
        for child in self.children:
            yield from child.leaves()


def escape(line: str) -> str:
    """ModelBlocks' `.extrparplusops` step for one tree line."""
    line = f"({line})"
    line = re.sub(r"\-(?=[a-z][^\)]* )", "+", line)
    return line.replace("-PRTRM", "+PRTRM")


_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def parse(text: str) -> Tree:
    """Read one bracketed tree, as Berkeley's `PennTreeReader` does.

    An unlabeled bracket is labeled `ROOT`.
    """
    tokens = _TOKEN.findall(text)
    position = 0

    def node() -> Tree:
        nonlocal position
        if tokens[position] != "(":
            raise ValueError(f"expected ( at token {position} of {text!r}")
        position += 1
        label = ROOT
        if tokens[position] not in "()":
            label = tokens[position]
            position += 1
        tree = Tree(label)
        while tokens[position] != ")":
            if tokens[position] == "(":
                tree.children.append(node())
            else:
                tree.children.append(Tree(tokens[position]))
                position += 1
        position += 1
        return tree

    tree = node()
    if position != len(tokens):
        raise ValueError(f"text after the tree in {text!r}")
    return tree


def _strip_function(label: str) -> str:
    cut = label.find("-")
    cut2 = label.find("=")
    cut3 = label.find("^")
    if cut3 > 0 and (cut3 < cut2 or cut2 == -1):
        cut2 = cut3
    if cut2 > 0 and (cut2 < cut or cut <= 0):
        cut = cut2
    return label[:cut] if cut > 0 else label


def normalize(tree: Tree) -> Tree | None:
    """Berkeley's `StandardTreeNormalizer`; `None` if nothing is left."""

    def strip(t: Tree) -> Tree:
        if t.is_leaf:
            return Tree(t.label)
        return Tree(_strip_function(t.label), [strip(c) for c in t.children])

    def empty(t: Tree) -> Tree | None:
        if t.label == "-NONE-":
            return None
        if t.is_leaf:
            return Tree(t.label)
        kept = [c for c in (empty(c) for c in t.children) if c is not None]
        return Tree(t.label, kept) if kept else None

    def x_over_x(t: Tree) -> Tree:
        children = t.children
        while len(children) == 1 and not children[0].is_leaf and children[0].label == t.label:
            children = children[0].children
        return Tree(t.label, [x_over_x(c) for c in children])

    stripped = empty(strip(tree))
    return None if stripped is None else x_over_x(stripped)


def annotate(tree: Tree) -> Tree:
    """`TreeAnnotations.processTree` with vertical 1, horizontal 0, RIGHT binarization."""

    def mark(t: Tree, suffix: str) -> Tree:
        if t.is_preterminal:
            return t
        return Tree(t.label + suffix, [mark(c, "^g") for c in t.children])

    def binarize(t: Tree) -> Tree:
        if t.is_leaf:
            return Tree(t.label)
        if len(t.children) == 1:
            return Tree(t.label, [binarize(t.children[0])])
        return Tree(t.label, _chain(t, len(t.children) - 1, "@" + t.label + "->").children)

    def _chain(t: Tree, left: int, label: str) -> Tree:
        right = t.children[left]
        children = []
        if left == 1:
            children.append(binarize(t.children[0]))
        elif left > 1:
            children.append(_chain(t, left - 1, label + "_" + right.label))
        children.append(binarize(right))
        return Tree(label, children)

    def forget(t: Tree) -> Tree:
        if t.is_leaf:
            return Tree(t.label)
        label = t.label
        cut = label.find(">") - 1
        if cut > 0:
            label = label[:cut]
        return Tree(label, [forget(c) for c in t.children])

    return forget(binarize(mark(tree, "")))


def training_tree(line: str) -> Tree | None:
    """A GCG tree line as the Berkeley trainer sees it (all four steps)."""
    normalized = normalize(parse(escape(line)))
    return None if normalized is None else annotate(normalized)
