# This file is part of txtures. The parser follows ModelBlocks
# (resource-incrsem/src/synproc.cpp and include/StoreStateSynProc.hpp,
# include/Beam.hpp, include/BerkUnkWord.hpp), Copyright 2009, ModelBlocks
# developers, reimplemented for txtures in 2026.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""synproc, the incremental left-corner parser, as a plain Python reference.

`reference_parse` follows `synproc.cpp` step for step. It is slow, and it
is the oracle for the batched GPU parser.

A beam entry is a hidden state `(p, f, j, q)`: the last word's preterminal
`p`, the fork `f` and join `j` decisions, and the store `q`, a tuple of
`(apex, base)` pairs. For each word, from each entry of the previous beam:

1. the apex `a` of the new store's last element, from
   `A(|q| + f - j, j, ancestor(f), j == 0 ? lchild : -)`;
2. its base `b`, from `B(|q| + f - j, j, a, lchild)`; the new store `q'`
   keeps the first `|q| - 1 + f - j` elements of `q` and ends with
   `(j == 0 ? a : apex of q[|q| + f - j - 1], b)`;
3. each preterminal `t` of the word, with `P(word | t)`;
4. the fork `f'`, from `F(|q'|, ancestor'(1))`, and `t` from
   `P(|q'|, f', base of q')`;
5. the join `j'`, from `J(|q'| + f', ancestor'(f'), lchild'(f', t))`.

`ancestor(f)` is the base of `q[|q| - 2 + f]`, or `T` when there is none;
`lchild(f, p)` is `p` when `f = 1` and the apex of `q`'s last element
otherwise. A word other than the last must leave `|q'| + f' - j' > 0`, the
last exactly 0. The new entry `(t, f', j', q')` scores the previous one's
log-probability plus the logs of all these factors; entries with the same
state keep the best, and the beam keeps the `beam` best. A word's
surprisal, in bits, is `log2` of the previous beam's total probability over
the current one's.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from txtures.lvpcfg import leftcorner as lc

ROOT = lc.ROOT
BOTTOM = lc.TERMINAL


def berkeley_signature(word: str) -> str:
    """`unkWordBerk`: Berkeley's level-5 signature, without position or `-KNOWNLC`."""
    parts = ["UNK"]
    caps = 0
    digit = dash = lower = False
    for char in word:
        if char.isdigit():
            digit = True
        elif char == "-":
            dash = True
        elif char.isalpha():
            if char.islower():
                lower = True
            else:
                caps += 1
    first = word[0] if word else ""
    lowered = word.lower()
    if first.isupper():
        parts.append("-INITC" if caps == 1 else "-CAPS")
    elif first and not first.isalpha() and caps > 0:
        parts.append("-CAPS")
    elif lower:
        parts.append("-LC")
    if digit:
        parts.append("-NUM")
    if dash:
        parts.append("-DASH")
    if lowered.endswith("s") and len(word) >= 3:
        if lowered[-2] not in "siu":
            parts.append("-s")
    elif len(word) >= 5 and not dash and not (digit and caps > 0):
        for suffix in ("ed", "ing", "ion", "er", "est", "ly", "ity", "y", "al"):
            if lowered.endswith(suffix):
                parts.append("-" + suffix)
                break
    return "".join(parts)


class Tables:
    """synproc's conditional tables, computed on demand from the compiled factors."""

    def __init__(self, model: lc.Compiled):
        self.model = model
        self.index = model.index
        self.symbols = model.symbols
        self.plus = model.plus.cpu().numpy()
        self.left_corner = model.left_corner.cpu().numpy()
        self.first = model.first.cpu().numpy()
        depths = lc.MAX_DEPTH + 1
        self.f = [[x.cpu().numpy() for x in lc.f_table(model, d)] for d in range(depths)]
        self.j = [[x.cpu().numpy() for x in lc.j_table(model, d)] for d in range(depths)]
        self.terminal = [
            lc._terminal_gamma(model, d, lc.LEFT).cpu().numpy() for d in range(depths + 1)
        ]
        self.b = {
            (d, s): lc.b_conditional(model, d, s)
            for d in range(depths + 1)
            for s in (lc.LEFT, lc.RIGHT)
            if lc._valid(d, s)
        }
        # P(word | preterminal), with each known word's signature mass added (synproc.cpp)
        self.lexicon: dict[str, dict[int, float]] = defaultdict(dict)
        for tag, words in model.lexicon.items():
            if tag not in self.index:
                continue
            for word, p in words.items():
                self.lexicon[word][self.index[tag]] = p
        self._merged: dict[str, dict[int, float]] = {}

    def words(self, word: str) -> dict[int, float]:
        """synproc's lexical entries of a word: known, plus or else its signature's."""
        if word in self._merged:
            return self._merged[word]
        known = self.lexicon.get(word)
        signature = self.lexicon.get(berkeley_signature(word), {})
        if known is None:
            merged = dict(signature)
        else:
            merged = dict(known)
            for tag, p in signature.items():
                merged[tag] = merged.get(tag, 0.0) + p
        self._merged[word] = merged
        return merged

    def a(self, d: int, j: int, b: int, c0: int | None) -> dict[int, float]:
        if j == 1:
            if not 0 <= d <= lc.MAX_DEPTH or not self.first[d][b].any():
                return {}
            return {b: 1.0}
        if not 1 <= d <= lc.MAX_DEPTH + 1 or c0 is None:
            return {}
        weights = self.plus[d - 1][b] * self.left_corner[d][:, c0]
        total = self.j[d - 1][0][b, c0]
        if total <= 0:
            return {}
        return {int(a): float(w / total) for a in np.nonzero(weights > 0)[0] for w in [weights[a]]}

    def b_dist(self, d: int, j: int, a: int, c0: int) -> dict[int, float]:
        table = self.b.get((d, lc.LEFT if j == 0 else lc.RIGHT))
        if table is None:
            return {}
        if j == 0 and not (1 <= d and self.plus[d - 1][:, a].any()):
            return {}
        return table.get((a, c0), {})

    def f_dist(self, d: int, b: int) -> dict[int, float]:
        if not 0 <= d <= lc.MAX_DEPTH:
            return {}
        f0, f1 = self.f[d][0][b], self.f[d][1][b]
        total = f0 + f1
        if total <= 0:
            return {}
        return {f: float(w / total) for f, w in ((0, f0), (1, f1)) if w > 0}

    def p(self, d: int, f: int, b: int, t: int) -> float:
        if not 0 <= d <= lc.MAX_DEPTH:
            return 0.0
        if f == 0:
            return 1.0 if (t == b and self.f[d][0][b] > 0) else 0.0
        f1 = self.f[d][1][b]
        return float(self.plus[d][b, t] * self.terminal[d + 1][t] / f1) if f1 > 0 else 0.0

    def j_dist(self, dd: int, b: int, c0: int) -> dict[int, float]:
        if not 1 <= dd <= lc.MAX_DEPTH + 1:
            return {}
        j0, j1 = self.j[dd - 1][0][b, c0], self.j[dd - 1][1][b, c0]
        total = j0 + j1
        if total <= 0:
            return {}
        return {j: float(w / total) for j, w in ((0, j0), (1, j1)) if w > 0}


State = tuple[int, int, int, tuple[tuple[int, int], ...]]


@dataclass
class Parse:
    """Per-word surprisal (bits) and the most likely sequence of hidden states."""

    surprisal: list[float]
    states: list[State | None]


START: State = (-1, 0, 1, ((-1, -1),))
"""synproc's start, `S/.` with preterminal `.`, `f = 0`, `j = 1`: its new store is empty.

synproc looks up the apex and base of that empty store under the unsplit
names `T` and `S`, which split-category models lack; here an empty new store
takes no apex or base step (probability 1), so the first word is predicted
from `T` at depth 0, as the compiled model intends.
"""


def reference_parse(
    tables: Tables, words: list[str], beam: int = 5000, recombine: str = "max"
) -> Parse:
    """Parse one sentence as `synproc.cpp` does, with an exact top-`beam` beam.

    `recombine="max"` keeps the best path into each state, as synproc does;
    `"sum"` adds the paths, so that with an unbounded beam the total of the
    last beam is the sentence's probability under the model.
    """
    top = tables.index[ROOT]
    beams: list[dict[State, tuple[float, State | None]]] = [{START: (0.0, None)}]
    for t, word in enumerate(words, 1):
        last = t == len(words)
        entries: dict[State, tuple[float, State]] = {}
        lexical = tables.words(word)
        for state, (score, _) in beams[-1].items():
            p, f, j, q = state
            depth = len(q) + f - j

            def ancestor(store, fork):
                k = len(store) - 2 + fork
                return store[k][1] if k >= 0 else top

            def lchild(store, fork, pretrm):
                return pretrm if fork == 1 else (store[-1][0] if store else top)

            if depth <= 0:
                expansions = [((), 1.0, 1.0)]
            else:
                c0 = lchild(q, f, p) if j == 0 else None
                expansions = []
                for a, pa in tables.a(depth, j, ancestor(q, f), c0).items():
                    for b, pb in tables.b_dist(depth, j, a, lchild(q, f, p)).items():
                        kept = q[: len(q) - (1 - f + j)]
                        apex = a if j == 0 else q[len(q) + f - j - 1][0]
                        expansions.append((kept + ((apex, b),), pa, pb))
            for new_q, pa, pb in expansions:
                for tag, pw in lexical.items():
                    brink = new_q[-1][1] if new_q else top
                    for fork, pf in tables.f_dist(len(new_q), ancestor(new_q, 1)).items():
                        pp = tables.p(len(new_q), fork, brink, tag)
                        if pp <= 0:
                            continue
                        for join, pj in tables.j_dist(
                            len(new_q) + fork, ancestor(new_q, fork), lchild(new_q, fork, tag)
                        ).items():
                            size = len(new_q) + fork - join
                            if (not last and size <= 0) or (last and size != 0):
                                continue
                            total = score + sum(math.log(x) for x in (pa, pb, pw, pf, pp, pj))
                            key = (tag, fork, join, new_q)
                            if key not in entries:
                                entries[key] = (total, state)
                            elif recombine == "sum":
                                old = entries[key][0]
                                high = max(old, total)
                                merged = high + math.log(
                                    math.exp(old - high) + math.exp(total - high)
                                )
                                entries[key] = (merged, entries[key][1])
                            elif total > entries[key][0]:
                                entries[key] = (total, state)
        best = sorted(entries.items(), key=lambda item: -item[1][0])[:beam]
        beams.append(dict(best))
    surprisal = []
    for t in range(1, len(beams)):
        if not beams[t]:
            surprisal.append(math.nan)
            continue
        reference = max(score for score, _ in beams[t - 1].values())
        previous = sum(math.exp(score - reference) for score, _ in beams[t - 1].values())
        current = sum(math.exp(score - reference) for score, _ in beams[t].values())
        surprisal.append(math.log2(previous) - math.log2(current))
    states: list[State | None] = [None] * len(words)
    if beams[-1]:
        state = max(beams[-1], key=lambda s: beams[-1][s][0])
        for t in range(len(words), 0, -1):
            states[t - 1] = state
            state = beams[t][state][1]
    return Parse(surprisal, states)


class MergedLexicon:
    """P(word | preterminal) with the unknown-word model merged in, as synproc does or once.

    synproc adds, at load, each known word's signature entries to its own
    (`synproc.cpp`, the loop after reading the model). With `merge="once"`
    each word gets its signature's entries exactly once. With
    `merge="synproc"` the result is synproc's: it visits words in its map's
    order, which is the order of first appearance going through
    preterminals in sorted order and each one's words in sorted order (the
    order in which ModelBlocks writes and reads the model); a word gets its
    signature's entries as they stand then, merged already when the
    signature came earlier; a signature that is its own signature doubles;
    and signatures that are not words are created, empty, after all words.
    An unknown word takes its signature's merged entries.
    """

    def __init__(
        self, lexicon: dict[str, dict[str, float]], index: dict[str, int], merge: str = "once"
    ):
        if merge not in ("once", "synproc"):
            raise ValueError(f"lexicon merge must be once or synproc, not {merge!r}")
        self.merge = merge
        self.plain: dict[str, dict[int, float]] = {}
        position: dict[str, int] = {}
        for tag in sorted(lexicon):
            if tag not in index:
                continue
            for word in sorted(lexicon[tag]):
                if lexicon[tag][word] <= 0:
                    continue  # ModelBlocks writes only positive entries (ccu2cc.py)
                self.plain.setdefault(word, {})[index[tag]] = lexicon[tag][word]
                position.setdefault(word, len(position))
        self.position = position
        self._merged: dict[str, dict[int, float]] = {}

    def _pos(self, word: str) -> float:
        return self.position.get(word, float("inf"))

    def merged(self, key: str) -> dict[int, float]:
        """The entries of a map key after synproc's merging (`merge="synproc"`)."""
        if key in self._merged:
            return self._merged[key]
        entries = dict(self.plain.get(key, {}))
        signature = berkeley_signature(key)
        if signature == key:
            entries = {t: 2 * p for t, p in entries.items()}
        else:
            source = (
                self.merged(signature)
                if self._pos(signature) < self._pos(key)
                else self.plain.get(signature, {})
            )
            for tag, p in source.items():
                entries[tag] = entries.get(tag, 0.0) + p
        self._merged[key] = entries
        return entries

    def entries(self, word: str) -> dict[int, float]:
        """synproc's lexical entries of a word: its own and its signature's."""
        signature = berkeley_signature(word)
        if self.merge == "synproc":
            return self.merged(word) if word in self.plain else self.merged(signature)
        if word not in self.plain:
            return dict(self.plain.get(signature, {}))
        merged = dict(self.plain[word])
        for tag, p in self.plain.get(signature, {}).items():
            merged[tag] = merged.get(tag, 0.0) + p
        return merged
