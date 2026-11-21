# This file is part of txtures. `tree_from_states` follows ModelBlocks
# (resource-incrsem/scripts/tokdecs2linetrees.py), Copyright 2009, ModelBlocks
# developers, reimplemented for txtures in 2026.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""Trees from synproc's best sequences, and trees normalized for bracket scoring.

`tree_from_states` rebuilds a sentence's tree from the hidden state of each
word, as ModelBlocks' `tokdecs2linetrees.py` does for synproc's output: a
word with fork 1 starts a new fragment `(p w)`, one with fork 0 attaches
`(p w)` under the base of the last fragment; unless the store is then empty,
its last element `a/b` joins the last fragment: with join 1 the pair
`[fragment, (b)]` attaches under the base of the fragment before, with join
0 the fragment is replaced by `(a fragment (b))`. Split-category suffixes
(`_3`) are removed.

`for_scoring` normalizes gold and parsed trees alike before bracket
scoring: operator and morphology tags are removed as ModelBlocks' `nol`
step does, so is the `-PRTRM` mark, and unary chains are collapsed to their
top label, since the compiled grammar folds them and the parser cannot
output them.
"""

from __future__ import annotations

import re

from txtures.gcg import remove_operators
from txtures.lvpcfg.trees import Tree, parse

_SPLIT = re.compile(r"_[0-9]+$")


def _base(label: str) -> str:
    return _SPLIT.sub("", label)


def _attach_base(tree: Tree, children: list[Tree]) -> Tree:
    """Attach `children` at the base (rightmost leaf category) of a derivation fragment."""
    node = tree
    while node.children:
        node = node.children[-1]
    node.children = children
    return tree


def tree_from_states(
    words: list[str], states: list, symbols: list[str], shifted: bool = False
) -> Tree | None:
    """The tree of a parse: `states[t] = (preterminal, fork, join, store)` with symbol indices.

    The parser's state for word t holds the store before that word's join.
    synproc's output pairs word t's decisions with the next word's store
    instead, which records that join, and the last word with an empty store;
    `shifted` says the states are already in that form (as read from a
    `.tokdecs` file). Returns `None` if the states do not form one tree (as
    `tokdecs2linetrees.py` reports "not a valid tree").
    """
    if any(state is None for state in states):
        return None
    if not shifted:
        stores = [state[3] for state in states[1:]] + [()]
        states = [(*state[:3], store) for state, store in zip(states, stores, strict=True)]
    fragments: list[Tree] = []
    for word, state in zip(words, states, strict=True):
        pretrm, fork, join, store = state
        leaf = Tree(_base(symbols[pretrm]), [Tree(word)])
        if fork == 1:
            fragments.append(leaf)
        else:
            if not fragments:
                return None
            _attach_base(fragments[-1], [leaf])
        if not store:
            return fragments[0] if len(fragments) == 1 else None
        apex, base = (_base(symbols[x]) for x in store[-1])
        children = [fragments[-1], Tree(base)]
        if join == 1:
            if len(fragments) < 2:
                return None
            fragments = fragments[:-2] + [_attach_base(fragments[-2], children)]
        else:
            fragments = fragments[:-1] + [Tree(apex, children)]
    return fragments[0] if len(fragments) == 1 else None


def for_scoring(tree: Tree) -> Tree:
    """A tree normalized for bracket scoring (see the module's description)."""

    def label(text: str) -> str:
        text = remove_operators(f"({text} x)")[1:-3]
        return text.replace("-PRTRM", "")

    def walk(node: Tree) -> Tree:
        if node.is_leaf:
            return Tree(node.label)
        top = node
        while len(top.children) == 1 and not top.children[0].is_leaf:
            top = top.children[0]
        return Tree(label(node.label), [walk(c) for c in top.children])

    return walk(tree)


def gold_for_scoring(line: str) -> Tree:
    """A gold gcg15 tree line, ModelBlocks' `nol` step applied, normalized for scoring."""
    return for_scoring(parse(f"({remove_operators(line)})").children[0])


def unlabeled(tree: Tree) -> Tree:
    """Every nonterminal relabeled `X`, preterminals included, for unlabeled scoring."""
    if tree.is_leaf:
        return Tree(tree.label)
    return Tree("X", [unlabeled(c) for c in tree.children])
