# This file is part of txtures. The computation follows ModelBlocks
# (resource-lcparse/scripts/berkgrammar2ckygr.py, berklexicon2ckylex.py,
# ccu2cc.py, and resource-incrsem/src/ccmodel2synproccptmodel.cpp),
# Copyright 2009, ModelBlocks developers, reimplemented for txtures in 2026.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""Compile a PCFG over split categories into synproc's depth-bounded left-corner model.

The steps follow ModelBlocks' chain from a Berkeley grammar to a
`.synprocmodel`:

1. `Pcfg.from_berkeley_text`: rules and lexical entries over split
   categories `CAT_k` (`berkgrammar2ckygr.py`, `berklexicon2ckylex.py`):
   binary rules, unary rules (Berkeley's text dump writes the Viterbi
   closure of its unaries), the root's unary rules, and lexical scores. Each
   preterminal gets a terminal rule `CAT_k -> - -` with weight 1.
2. `Pcfg.without_unaries` (`ccu2cc.py`): the sum over unary chains (19
   compositions from the identity) is folded into the binary rules, the
   terminal rules, and the lexicon below each category.
3. `compile_model` (`ccmodel2synproccptmodel`): with depth bound 4,
   - `delta[d, s, c]`, the probability that `c` expands within the bound at
     depth `d` as a left (`s = 0`) or right (`s = 1`) child, by 21
     fixed-point iterations: terminal weight plus, for each binary rule,
     its weight times `delta` of the left child (at `d` for a left parent,
     `d + 1` for a right one) and of the right child at `d` (right);
   - `gamma`, each rule's weight renormalized by those `delta`s;
   - the first left-corner step `M0[d]` from a right child at `d` and the
     left-corner step `L[d]` within a left child at `d`, and their
     truncated closure `plus[d] = M0[d] (I + L[d+1] + ... + L[d+1]^20)`.

   synproc's tables follow: `F` (fork: whether the next word is the
   expected category itself, `f = 0`, or a left descendant, `f = 1`), `P`
   (the preterminal), `J` (join), `A` (the apex), and `B` (the base), each
   normalized; `compiled.f_table` and the other accessors give them.

The root is `T`, with rules `T -> c T` from the root's unary rules and
`delta[0, right, T] = 1`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

MAX_DEPTH = 4
ITERATIONS = 20
UNARY_ITERATIONS = 20
TERMINAL = "-"
ROOT = "T"
LEFT, RIGHT = 0, 1


@dataclass
class Pcfg:
    """Rules over split-category symbols (strings), as ModelBlocks' `x-ccu`/`x-cc` models."""

    binary: dict[str, dict[tuple[str, str], float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    unary: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    root: dict[str, float] = field(default_factory=dict)
    lexicon: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    terminal: dict[str, float] = field(default_factory=dict)  # weight of CAT_k -> - -

    @classmethod
    def from_berkeley_text(cls, grammar: Iterable[str], lexicon: Iterable[str]) -> Pcfg:
        """Read Berkeley's `.grammar` and `.lexicon` text dumps as ModelBlocks does.

        `+` becomes `-` everywhere, as the Makefile's `sed`, and `&` becomes `-`.
        """
        pcfg = cls()
        for line in grammar:
            fields = line.replace("+", "-").rstrip("\n").split(" ")
            if len(fields) == 4:
                parent, child = fields[0].replace("&", "-"), fields[2].replace("&", "-")
                if parent == "ROOT_0":
                    if child != "ROOT_0":
                        pcfg.root[child] = float(fields[3])
                elif parent != child:
                    pcfg.unary[parent][child] = float(fields[3])
            elif len(fields) == 5:
                parent = fields[0].replace("&", "-")
                left, right = fields[2].replace("&", "-"), fields[3].replace("&", "-")
                pcfg.binary[parent][(left, right)] = float(fields[4])
        pattern = re.compile(r"([^ ]+) ([^ ]+) \[(.*)\]")
        for line in lexicon:
            match = pattern.search(line.replace("+", "-"))
            if match is None:
                continue
            tag, word, scores = match.group(1).replace("&", "-"), match.group(2), match.group(3)
            for k, value in enumerate(scores.split(",")):
                if "NaN" in value:
                    continue
                symbol = f"{tag}_{k}"
                pcfg.terminal[symbol] = 1.0
                pcfg.lexicon[symbol][word] = float(value)
        return pcfg

    def without_unaries(self) -> Pcfg:
        """`ccu2cc.py`: fold the sum over unary chains into rules and the lexicon."""
        current: dict[str, dict[str, float]] = defaultdict(dict)
        star: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for c in list(self.unary) + list(self.binary) + list(self.terminal):
            current[c][c] = 1.0
            star[c][c] = 1.0
        for _ in range(1, UNARY_ITERATIONS):
            previous, current = current, defaultdict(lambda: defaultdict(float))
            for c, row in previous.items():
                for child, weight in row.items():
                    for grandchild, p in self.unary.get(child, {}).items():
                        current[c][grandchild] += weight * p
                        star[c][grandchild] += weight * p
        out = Pcfg(root=dict(self.root))
        lexicon: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        binary: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
        terminal: dict[str, float] = defaultdict(float)
        for c, row in star.items():
            for child, weight in row.items():
                for word, p in self.lexicon.get(child, {}).items():
                    lexicon[c][word] += weight * p
                for pair, p in self.binary.get(child, {}).items():
                    binary[c][pair] += weight * p
                if child in self.terminal:
                    terminal[c] += weight * self.terminal[child]
        out.lexicon = {c: dict(m) for c, m in lexicon.items()}
        out.binary = {c: {k: v for k, v in m.items()} for c, m in binary.items()}
        out.terminal = dict(terminal)
        return out


@dataclass
class Compiled:
    """synproc's depth-bounded left-corner model as factors over symbol indices."""

    symbols: list[str]
    index: dict[str, int]
    delta: torch.Tensor  # [MAX_DEPTH + 2, 2, N]
    rule_parent: torch.Tensor  # binary rules (parent, left, right)
    rule_left: torch.Tensor
    rule_right: torch.Tensor
    gamma: torch.Tensor  # [MAX_DEPTH + 2, 2, rules]: renormalized rule weights
    gamma_valid: torch.Tensor  # [MAX_DEPTH + 2, 2, N]: where gamma is defined
    terminal: torch.Tensor  # [N]
    first: torch.Tensor  # [MAX_DEPTH + 1, N, N]: M0[d][b, c0]
    left_corner: torch.Tensor  # [MAX_DEPTH + 2, N, N]: L[d][c, c0]
    plus: torch.Tensor  # [MAX_DEPTH + 1, N, N]
    lexicon: dict[str, dict[str, float]]


def _valid(d: int, s: int) -> bool:
    return d <= MAX_DEPTH or s == LEFT


def compile_model(pcfg: Pcfg, device: str = "cpu") -> Compiled:
    """`ccmodel2synproccptmodel`: delta, gamma, and the left-corner closures."""
    symbols = sorted(
        set(pcfg.binary)
        | set(pcfg.terminal)
        | {ROOT}
        | {x for m in pcfg.binary.values() for pair in m for x in pair}
        | set(pcfg.root)
    )
    index = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)
    parents, lefts, rights, weights = [], [], [], []
    for parent, row in pcfg.binary.items():
        for (left, right), p in row.items():
            if right == TERMINAL:
                continue
            parents.append(index[parent])
            lefts.append(index[left])
            rights.append(index[right])
            weights.append(p)
    for child, p in pcfg.root.items():  # T -> c T
        parents.append(index[ROOT])
        lefts.append(index[child])
        rights.append(index[ROOT])
        weights.append(p)

    def t(values, dtype=torch.long) -> torch.Tensor:
        return torch.as_tensor(np.array(values), dtype=dtype, device=device)

    parent, left, right = t(parents), t(lefts), t(rights)
    weight = t(weights, torch.float64)
    terminal = torch.zeros(n, dtype=torch.float64, device=device)
    for symbol, p in pcfg.terminal.items():
        terminal[index[symbol]] = p
    depths = MAX_DEPTH + 2
    valid = torch.tensor(
        [[_valid(d, s) for s in (LEFT, RIGHT)] for d in range(depths)], device=device
    )
    positive = weight > 0
    delta = torch.zeros(depths, 2, n, dtype=torch.float64, device=device)
    for i in range(ITERATIONS + 1):
        new = torch.zeros_like(delta)
        for d in range(depths):
            for s in (LEFT, RIGHT):
                if not valid[d, s]:
                    continue
                row = torch.where(terminal > 0, terminal, torch.zeros_like(terminal))
                if i > 0:
                    left_depth = d if s == LEFT else d + 1
                    left_delta = (
                        delta[left_depth, LEFT] if left_depth < depths else torch.zeros_like(row)
                    )
                    product = weight * left_delta[left] * delta[d, RIGHT][right]
                    product = torch.where(
                        positive & (product > 0), product, torch.zeros_like(product)
                    )
                    row = row.index_add(0, parent, product)
                new[d, s] = row
        delta = new
    delta[0, RIGHT, index[ROOT]] = 1.0
    root = index[ROOT]
    gamma = torch.zeros(depths, 2, len(weight), dtype=torch.float64, device=device)
    gamma_valid = torch.zeros(depths, 2, n, dtype=torch.bool, device=device)
    is_root = torch.arange(n, device=device) == root
    for d in range(depths):
        for s in (LEFT, RIGHT):
            if not valid[d, s]:
                continue
            allowed = (is_root & (d == 0) & (s == RIGHT)) | (~is_root & (d != 0))
            gamma_valid[d, s] = allowed
            left_depth = d if s == LEFT else d + 1
            left_delta = (
                delta[left_depth, LEFT]
                if left_depth < depths
                else torch.zeros(n, dtype=torch.float64, device=device)
            )
            own = delta[d, s][parent]
            scaled = (
                weight
                * left_delta[left]
                * delta[d, RIGHT][right]
                / torch.where(own > 0, own, torch.ones_like(own))
            )
            g = torch.where(own > 0, scaled, weight)
            gamma[d, s] = torch.where(allowed[parent] & (g > 0), g, torch.zeros_like(g))
    first = torch.zeros(MAX_DEPTH + 1, n, n, dtype=torch.float64, device=device)
    left_corner = torch.zeros(depths, n, n, dtype=torch.float64, device=device)
    for d in range(depths):
        left_corner[d].index_put_((parent, left), gamma[d, LEFT], accumulate=True)
    for d in range(MAX_DEPTH + 1):
        first[d].index_put_((parent, left), gamma[d, RIGHT], accumulate=True)
        allowed = (is_root & (d == 0)) | (~is_root & (d != 0))
        first[d] *= allowed.unsqueeze(1)
    plus = torch.zeros_like(first)
    for d in range(MAX_DEPTH + 1):
        term = first[d]
        total = term.clone()
        for _ in range(ITERATIONS):
            term = term @ left_corner[d + 1]
            total += term
        plus[d] = total
    return Compiled(
        symbols,
        index,
        delta,
        parent,
        left,
        right,
        gamma,
        gamma_valid,
        terminal,
        first,
        left_corner,
        plus,
        pcfg.lexicon,
    )


# synproc's tables, from the factors


def _terminal_gamma(model: Compiled, d: int, s: int) -> torch.Tensor:
    """Each symbol's terminal weight where gamma is defined at (d, s), else 0."""
    if d > MAX_DEPTH + 1 or not _valid(d, s):
        return torch.zeros_like(model.terminal)
    return torch.where(model.gamma_valid[d, s], model.terminal, torch.zeros_like(model.terminal))


def _allowed(model: Compiled, d: int) -> torch.Tensor:
    """Symbols that `zero` and `plus` are defined for at depth d."""
    is_root = torch.zeros_like(model.terminal, dtype=torch.bool)
    is_root[model.index[ROOT]] = True
    return is_root if d == 0 else ~is_root


def f_table(model: Compiled, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unnormalized F weights `[N]` for f = 0 and f = 1 at depth d (`mF[(d, b)]`)."""
    f1 = model.plus[d] @ _terminal_gamma(model, d + 1, LEFT)
    f0 = _terminal_gamma(model, d, RIGHT) * _allowed(model, d)
    return f0, f1


def j_table(model: Compiled, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unnormalized J weights `[N, N]` (b, c0) at depth d + 1 for j = 0 and j = 1."""
    return model.plus[d] @ model.left_corner[d + 1], model.first[d]


def b_conditional(model: Compiled, d: int, s: int) -> dict[tuple[int, int], dict[int, float]]:
    """`B`: gamma at (d, s) renormalized given the parent and the left child."""
    weights = model.gamma[d, s].cpu().numpy()
    parents, lefts = model.rule_parent.cpu().numpy(), model.rule_left.cpu().numpy()
    rights = model.rule_right.cpu().numpy()
    table: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for w, p, left, right in zip(weights, parents, lefts, rights, strict=True):
        if w > 0:
            row = table[(int(p), int(left))]
            row[int(right)] = row.get(int(right), 0.0) + float(w)
    for row in table.values():
        total = sum(row.values())
        for key in row:
            row[key] /= total
    return table
