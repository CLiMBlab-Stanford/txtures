"""Splitting, merging, and smoothing substates, as Berkeley's `GrammarTrainer` does.

Split (`Grammar.splitAllStates`, `SophisticatedLexicon.splitAllStates`).
Every category except `ROOT` doubles its substates: substate `k` becomes
`2k` and `2k + 1`. A binary rule's weight `w` is divided among the new
child substates, `w / f + εL·(±1) + εR·(±1)`, where `f` is the number of
child substate combinations a weight is split into (4 when both children
split), `ε = w / f · r / 100 · (u - 0.5)` with `u` uniform on [0, 1) and
`r` the randomness, and the sign is + for the first new child substate and
- for the second. A fresh `εL` is drawn for each parent substate, and a
fresh `εR` for each parent substate and new left substate. Unary rules
are split the same way with one child. Lexical counts are halved.

Merge (`GrammarMerger`). Each node's posterior over its substates gives
the merge weights: the average posterior of each substate of each
category. The loss of merging sibling substates `i` and `j` sums, over the
nodes of the category, `log(S / S')`, where `S = Σ_k I_k O_k` and `S'`
replaces the pair's two terms by `(w_i I_i + w_j I_j) / (w_i + w_j) ·
(O_i + O_j)`. The losses of all sibling pairs of all categories are sorted
together and half are merged; `merge_pairs` gives the two rules for
choosing them, Berkeley's and the default, which treats negligible losses
as zero. Merged child
substates add their probabilities and merged parent substates average
theirs by merge weight; lexical counts add up.

Smoothing (`SmoothAcrossParentBits`). Each category's split tree records
how its substates descend from one another. Below its first binary split,
a substate keeps `1 - s` of its probability and gives `s / (m - 1)` to
each other substate of its branch of `m`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from txtures.lvpcfg.em import Corpus, Grammar, prepare
from txtures.lvpcfg.inside import node_posteriors
from txtures.lvpcfg.lexicon import Lexicon

SplitTree = int | list  # a substate (leaf) or a list of subtrees

ROOT_CATEGORY = 0


# Split trees


def extend_split_tree(tree: SplitTree, split: bool) -> SplitTree:
    """`Grammar.extendSplitTrees`: each leaf k gains children 2k and 2k + 1 (or k)."""
    if isinstance(tree, int):
        return [tree, [2 * tree, 2 * tree + 1] if split else [tree]]
    label, children = tree
    return [label, [extend_split_tree(c, split) for c in children]]


def _leaves(tree: SplitTree) -> list[int]:
    if isinstance(tree, int):
        return [tree]
    return [leaf for child in tree[1] for leaf in _leaves(child)]


def smoother_matrix(tree: SplitTree, smooth: float) -> np.ndarray:
    """`SmoothAcrossParentBits`: the matrix D for one category (substates in rows)."""
    leaves = _leaves(tree)
    n = max(leaves) + 1
    if n == 1:
        return np.ones((1, 1))
    weights = np.zeros((n, n))
    while not isinstance(tree, int) and len(tree[1]) == 1:
        tree = tree[1][0]
    for branch in tree[1][:2]:
        members = _leaves(branch)
        for i in members:
            for j in members:
                weights[i, j] = 1 - smooth if i == j else smooth / (len(members) - 1)
    return weights


def smoothers(trees: Mapping[int, SplitTree], smooth: float) -> dict[int, np.ndarray]:
    return {category: smoother_matrix(tree, smooth) for category, tree in trees.items()}


# Split


def split(
    grammar: Grammar,
    lexicon: Lexicon,
    trees: Mapping[int, SplitTree],
    rng: np.random.Generator,
    randomness: float = 1.0,
) -> tuple[Grammar, Lexicon, dict[int, SplitTree]]:
    """Double every category's substates (not `ROOT`'s), adding Berkeley's noise to rules."""
    old = grammar.substates
    new = {c: n if c == ROOT_CATEGORY else 2 * n for c, n in old.items()}

    def factor(c: int) -> int:
        return new[c] // old[c]

    binary = {}
    for (a, b, c), w in grammar.binary.items():
        fa, fb, fc = factor(a), factor(b), factor(c)
        out = np.zeros((new[a], new[b], new[c]))
        divisor = fb * fc
        for pa in range(old[a]):
            for p in range(fa):
                for lb in range(old[b]):
                    for rc in range(old[c]):
                        score = w[pa, lb, rc]
                        if score == 0:
                            continue  # Berkeley keeps no scores for null child pairs
                        e_left = score / divisor * randomness / 100 * (rng.random() - 0.5)
                        for i in range(fb):
                            sign_left = -1 if i == 1 else 1
                            el = 0.0 if fb == 1 else sign_left * e_left
                            e_right = score / divisor * randomness / 100 * (rng.random() - 0.5)
                            for j in range(fc):
                                sign_right = -1 if j == 1 else 1
                                er = 0.0 if fc == 1 else sign_right * e_right
                                out[fa * pa + p, fb * lb + i, fc * rc + j] = (
                                    score / divisor + el + er
                                )
        binary[(a, b, c)] = out
    unary = {}
    for (a, b), w in grammar.unary.items():
        fa, fb = factor(a), factor(b)
        out = np.zeros((new[a], new[b]))
        for pa in range(old[a]):
            for p in range(fa):
                for cb in range(old[b]):
                    score = w[pa, cb]
                    if score == 0:
                        continue
                    e = score / fb * randomness / 100 * (rng.random() - 0.5)
                    for i in range(fb):
                        sign = -1 if i == 1 else 1
                        out[fa * pa + p, fb * cb + i] = score / fb + (0.0 if fb == 1 else sign * e)
        unary[(a, b)] = out
    new_trees = {c: extend_split_tree(t, new[c] > old[c]) for c, t in trees.items()}
    return Grammar(new, binary, unary, None), split_lexicon(lexicon, new), new_trees


def _halve(values: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(values / factor, factor) if factor > 1 else values.copy()


def split_lexicon(lexicon: Lexicon, new: Mapping[int, int]) -> Lexicon:
    def f(tag: int) -> int:
        return new[tag] // lexicon.substates[tag]

    return Lexicon(
        new,
        None,
        {t: {w: _halve(v, f(t)) for w, v in m.items()} for t, m in lexicon.word_tag.items()},
        {t: {w: _halve(v, f(t)) for w, v in m.items()} for t, m in lexicon.unseen_word_tag.items()},
        {t: _halve(v, f(t)) for t, v in lexicon.tag_count.items()},
        {t: _halve(v, f(t)) for t, v in lexicon.unseen_tag_count.items()},
        dict(lexicon.word_count),
        lexicon.total,
        lexicon.total_unseen,
    )


# Merge


@dataclass
class MergeStatistics:
    weights: dict[int, np.ndarray]  # category -> merge weight of each substate
    deltas: dict[int, np.ndarray]  # category -> loss of merging (2k, 2k + 1)
    log_likelihood: float


def merge_statistics(
    corpus: Corpus, grammar: Grammar, lexicon: Lexicon, device: str = "cpu"
) -> MergeStatistics:
    """`computeMergeWeights` and `computeDeltas`, with lexical smoothing as Berkeley uses there."""
    prepared = prepare(corpus, grammar, lexicon, no_smoothing=False, device=device)
    totals, inside, outside, posterior = node_posteriors(
        prepared.encoded, prepared.binary, prepared.unary, prepared.lexical
    )
    inside, outside = inside.cpu().numpy(), outside.cpu().numpy()
    posterior = posterior.cpu().numpy()
    finite = np.isfinite(totals.cpu().numpy())
    tree_of_node = np.repeat(np.arange(len(corpus.nodes)), [len(t) for t in corpus.nodes])
    keep = finite[tree_of_node]
    categories = prepared.categories
    sums = {c: np.zeros(n) for c, n in grammar.substates.items()}
    for c in np.unique(categories):
        rows = posterior[(categories == c) & keep, : grammar.substates[c]]
        totals_c = rows.sum(axis=1, keepdims=True)
        totals_c[totals_c == 0] = 1
        sums[c] = (rows / totals_c).sum(axis=0)
    weights = {}
    for c, values in sums.items():
        total = values.sum()
        weights[c] = values / (total if total != 0 else 1)
    deltas = {}
    for c, n in grammar.substates.items():
        pairs = n // 2
        if pairs == 0:
            continue
        rows = (categories == c) & keep
        i_s, o_s = inside[rows, :n], outside[rows, :n]
        separated = (i_s * o_s).sum(axis=1)
        w = weights[c]
        delta = np.zeros(pairs)
        for k in range(pairs):
            i, j = 2 * k, 2 * k + 1
            weight_sum = w[i] + w[j] if w[i] + w[j] != 0 else 1
            merged = (i_s[:, i] * w[i] / weight_sum + i_s[:, j] * w[j] / weight_sum) * (
                o_s[:, i] + o_s[:, j]
            )
            combined = separated - i_s[:, i] * o_s[:, i] - i_s[:, j] * o_s[:, j] + merged
            ok = (merged != 0) & (separated != 0)
            delta[k] = np.log(separated[ok] / combined[ok]).sum()
        deltas[c] = delta
    return MergeStatistics(weights, deltas, float(totals[finite].sum()))


MERGE_RULES = ("negligible", "berkeley")
NEGLIGIBLE_LOSS = 1e-6
"""Merge losses at or below this (in nats, summed over the treebank) count as zero."""


def merge_pairs(
    deltas: Mapping[int, np.ndarray],
    fraction: float = 0.5,
    rule: str = "negligible",
    negligible: float = NEGLIGIBLE_LOSS,
) -> dict[int, list[int]]:
    """For each category, the sibling pairs (by k) to merge.

    `rule="negligible"` (the default): losses with magnitude at most
    `negligible` count as zero; the threshold is the loss at position
    `⌊fraction · n⌋` among all `n` sibling pairs, and every pair at or below
    it is merged. Splits that learned nothing are therefore always merged,
    whatever their rounding.

    `rule="berkeley"`: `GrammarMerger.determineMergePairs`. Losses of
    exactly 0 are left out, both of the count and of the merges; the
    threshold is the loss at position `⌊fraction · n⌋` among the others.
    When many splits learned nothing, as in later rounds on GCG trees, the
    threshold falls among losses at rounding level, and which pairs merge
    depends on rounding.
    """
    if rule not in MERGE_RULES:
        raise ValueError(f"merge rule must be one of {MERGE_RULES}, not {rule!r}")
    if rule == "berkeley":
        values = sorted(d for ds in deltas.values() for d in ds if d != 0)
        if not values:
            return {}
        threshold = values[int(len(values) * fraction)]
        return {
            c: [k for k, d in enumerate(ds) if d != 0 and d <= threshold]
            for c, ds in deltas.items()
        }
    cleaned = {c: np.where(np.abs(ds) <= negligible, 0.0, ds) for c, ds in deltas.items()}
    values = sorted(d for ds in cleaned.values() for d in ds)
    if not values:
        return {}
    threshold = values[int(len(values) * fraction)]
    return {c: [k for k, d in enumerate(ds) if d <= threshold] for c, ds in cleaned.items()}


def merge_mapping(
    substates: Mapping[int, int], pairs: Mapping[int, list[int]]
) -> dict[int, np.ndarray]:
    """For each category, the new substate of each old one after merging the pairs."""
    mapping = {}
    for c, n in substates.items():
        merged = set(pairs.get(c, ()))
        target: list[int] = []
        count = 0
        for s in range(n):
            if s % 2 == 1 and s // 2 in merged:
                target.append(target[-1])
            else:
                target.append(count)
                count += 1
        mapping[c] = np.array(target, dtype=int)
    return mapping


def merge(
    grammar: Grammar,
    lexicon: Lexicon,
    trees: Mapping[int, SplitTree],
    pairs: Mapping[int, list[int]],
    weights: Mapping[int, np.ndarray],
) -> tuple[Grammar, Lexicon, dict[int, SplitTree]]:
    """`doTheMerges`: merge the chosen sibling pairs in the grammar, lexicon, and split trees."""
    mapping = merge_mapping(grammar.substates, pairs)
    new = {c: int(m.max()) + 1 if len(m) else 0 for c, m in mapping.items()}
    new[ROOT_CATEGORY] = 1

    def children_sum(c: int) -> np.ndarray:
        m = np.zeros((grammar.substates[c], new[c]))
        m[np.arange(grammar.substates[c]), mapping[c]] = 1
        return m

    def parent_average(c: int) -> np.ndarray:
        m = np.zeros((new[c], grammar.substates[c]))
        w = weights[c]
        for s, t in enumerate(mapping[c]):
            members = np.nonzero(mapping[c] == t)[0]
            m[t, s] = 1.0 if len(members) == 1 else w[s] / w[members].sum()
        return m

    binary = {
        k: np.einsum(
            "Aa,abc,bB,cC->ABC", parent_average(k[0]), v, children_sum(k[1]), children_sum(k[2])
        )
        for k, v in grammar.binary.items()
    }
    unary = {
        k: np.einsum("Aa,ab,bB->AB", parent_average(k[0]), v, children_sum(k[1]))
        for k, v in grammar.unary.items()
    }

    def add(values: np.ndarray, tag: int) -> np.ndarray:
        return values @ children_sum(tag)

    merged_lexicon = Lexicon(
        new,
        lexicon.smoother,
        {t: {w: add(v, t) for w, v in m.items()} for t, m in lexicon.word_tag.items()},
        lexicon.unseen_word_tag,  # Berkeley does not merge these; the next pass replaces them
        {t: add(v, t) for t, v in lexicon.tag_count.items()},
        lexicon.unseen_tag_count,
        dict(lexicon.word_count),
        lexicon.total,
        lexicon.total_unseen,
    )
    new_trees = {c: prune_split_tree(t, mapping[c], pairs.get(c, ())) for c, t in trees.items()}
    return Grammar(new, binary, unary, grammar.smoother), merged_lexicon, new_trees


def prune_split_tree(tree: SplitTree, mapping: np.ndarray, merged) -> SplitTree:
    """`Grammar.pruneSplitTree`: merged leaves become one, and leaves are renumbered."""
    depth = _depth(tree)

    def walk(t: SplitTree, level: int) -> SplitTree:
        if isinstance(t, int):
            return t
        label, children = t
        if level == depth - 2:
            kept = []
            for child in children:
                leaf = child if isinstance(child, int) else child[0]
                # A leaf is kept if it is the first of its partners.
                if not (leaf % 2 == 1 and leaf // 2 in merged):
                    kept.append(int(mapping[leaf]))
            return [label, kept]
        return [label, [walk(c, level + 1) for c in children]]

    return walk(tree, 0)


def _depth(tree: SplitTree) -> int:
    if isinstance(tree, int):
        return 1
    return 1 + max(_depth(c) for c in tree[1])
