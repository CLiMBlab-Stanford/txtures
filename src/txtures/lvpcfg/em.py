"""One EM iteration of the latent-variable PCFG, as Berkeley's `GrammarTrainer` runs it.

E-step (`GrammarTrainer.doOneEStep`): inside scores of every training tree
under the previous grammar, with lexical scores from the previous lexicon
without smoothing of unknown words, and expected counts by backpropagation.
Lexical counts are added token by token, in treebank order, then rare words
are tied (`Lexicon.tie_rare_words`).

M-step (`Grammar.optimize(0)`, `SophisticatedLexicon.optimize`): each
rule's counts are divided by its parent substate's total over all rules
(unary rules from a category to itself excluded from the total), and
values below the pruning threshold are zeroed. With a smoother, the
probabilities of each rule are then mixed across the parent's substates and
normalized again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from txtures.lvpcfg.inside import BINARY, LEXICAL, UNARY, encode, expected_counts
from txtures.lvpcfg.lexicon import PRUNE, RARE, Lexicon, token_signatures, train_tokens
from txtures.lvpcfg.trees import Tree


@dataclass
class Grammar:
    """Rule probabilities by substate: `binary[(A, B, C)]` is `[a, b, c]`, `unary[(A, B)]` `[a, b]`."""

    substates: Mapping[int, int]
    binary: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    unary: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    smoother: Mapping[int, np.ndarray] | None = None
    """Per category, the matrix mixing rule probabilities across its substates."""


@dataclass
class Corpus:
    """Training trees as category numbers: nodes in post-order, and the tokens."""

    nodes: list[list[tuple[str, tuple, int, int]]]
    tokens: list[tuple[str, int, int]]  # word, tag, position in its sentence
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def signatures(self) -> list[str]:
        """Each token's unknown-word signature (the same in every training pass)."""
        if "signatures" not in self._cache:
            self._cache["signatures"] = token_signatures(self.tokens)
        return self._cache["signatures"]


def index_trees(trees: Sequence[Tree], categories: Mapping[str, int]) -> Corpus:
    """Number the trees' categories and list their rules and tokens."""
    all_nodes, tokens = [], []
    for tree in trees:
        nodes: list[tuple[str, tuple, int, int]] = []
        position = 0

        def visit(t: Tree, nodes: list = nodes) -> int:
            nonlocal position
            if t.is_preterminal:
                tokens.append((t.children[0].label, categories[t.label], position))
                position += 1
                nodes.append(("lexical", (len(tokens) - 1,), -1, -1))
                return len(nodes) - 1
            kids = [visit(c) for c in t.children]
            labels = tuple(categories[x.label] for x in (t, *t.children))
            kind = "unary" if len(kids) == 1 else "binary"
            nodes.append((kind, labels, kids[0], kids[1] if len(kids) > 1 else -1))
            return len(nodes) - 1

        visit(tree)
        all_nodes.append(nodes)
    return Corpus(all_nodes, tokens)


def _padded(table: Mapping[tuple, np.ndarray], keys: list[tuple], shape: tuple) -> torch.Tensor:
    out = np.zeros((max(len(keys), 1), *shape))
    for i, key in enumerate(keys):
        values = table.get(key)
        if values is not None:
            out[(i, *(slice(0, d) for d in values.shape))] = values
    return torch.from_numpy(out)


@dataclass
class Prepared:
    """A corpus encoded for `inside` under one grammar and lexicon."""

    encoded: object
    binary: torch.Tensor
    unary: torch.Tensor
    lexical: torch.Tensor
    binary_index: dict[tuple[int, int, int], int]
    unary_index: dict[tuple[int, int], int]
    categories: np.ndarray  # the category of each node


def prepare(
    corpus: Corpus,
    grammar: Grammar,
    lexicon: Lexicon,
    no_smoothing: bool = True,
    device: str = "cpu",
) -> Prepared:
    """Encode the trees with the grammar's rule weights and the lexicon's scores."""
    size = max(grammar.substates.values())
    key = ("encoding", str(device))
    if key not in corpus._cache:
        binary_keys = sorted({n[1] for t in corpus.nodes for n in t if n[0] == "binary"})
        unary_keys = sorted({n[1] for t in corpus.nodes for n in t if n[0] == "unary"})
        binary_index = {k: i for i, k in enumerate(binary_keys)}
        unary_index = {k: i for i, k in enumerate(unary_keys)}
        encoded_trees, categories = [], []
        for nodes in corpus.nodes:
            encoded = []
            for kind, rule, left, right in nodes:
                if kind == "lexical":
                    encoded.append((LEXICAL, rule[0], -1, -1))
                    categories.append(corpus.tokens[rule[0]][1])
                elif kind == "unary":
                    encoded.append((UNARY, unary_index[rule], left, -1))
                    categories.append(rule[0])
                else:
                    encoded.append((BINARY, binary_index[rule], left, right))
                    categories.append(rule[0])
            encoded_trees.append(encoded)
        corpus._cache[key] = (
            encode(encoded_trees, device),
            binary_keys,
            unary_keys,
            binary_index,
            unary_index,
            np.array(categories),
        )
    encoding, binary_keys, unary_keys, binary_index, unary_index, categories = corpus._cache[key]
    lexical = np.zeros((len(corpus.tokens), size))
    scores: dict[tuple[str, int, int], np.ndarray] = {}
    for i, (word, tag, position) in enumerate(corpus.tokens):
        # Without smoothing, a seen word's score does not depend on its position.
        key = (word, tag, 0 if no_smoothing else position)
        if key not in scores:
            scores[key] = lexicon.score(word, tag, position, no_smoothing=no_smoothing)
        lexical[i, : grammar.substates[tag]] = scores[key]
    return Prepared(
        encoding,
        _padded(grammar.binary, binary_keys, (size, size, size)).to(device),
        _padded(grammar.unary, unary_keys, (size, size)).to(device),
        torch.from_numpy(lexical).to(device),
        binary_index,
        unary_index,
        categories,
    )


def iterate(
    corpus: Corpus,
    grammar: Grammar,
    lexicon: Lexicon,
    device: str = "cpu",
    rare: int = RARE,
    lexicon_only: bool = False,
) -> tuple[float, Grammar, Lexicon]:
    """One EM iteration; returns the training log-likelihood and the new grammar and lexicon.

    With `lexicon_only`, only the lexicon is re-estimated and the grammar is
    returned unchanged, as after a merge.
    """
    prepared = prepare(corpus, grammar, lexicon, True, device)
    encoded, binary, unary = prepared.encoded, prepared.binary, prepared.unary
    binary_index, unary_index = prepared.binary_index, prepared.unary_index
    total, binary_counts, unary_counts, token_counts = expected_counts(
        encoded, binary, unary, prepared.lexical
    )
    # Lexicon M-step: tokens in treebank order.
    new_lexicon = Lexicon(lexicon.substates, lexicon.smoother)
    train_tokens(
        new_lexicon,
        corpus.tokens,
        corpus.signatures,
        token_counts.cpu().numpy(),
        lexicon.word_count,
        rare,
    )
    new_lexicon.tie_rare_words(rare)
    new_lexicon.prune()
    if lexicon_only:
        return float(total), grammar, new_lexicon
    # Grammar M-step.
    sub = grammar.substates
    b_counts = {
        k: binary_counts[i, : sub[k[0]], : sub[k[1]], : sub[k[2]]].cpu().numpy()
        for k, i in binary_index.items()
    }
    u_counts = {
        k: unary_counts[i, : sub[k[0]], : sub[k[1]]].cpu().numpy() for k, i in unary_index.items()
    }
    new_grammar = Grammar(grammar.substates, *normalize(b_counts, u_counts), grammar.smoother)
    if grammar.smoother is not None:
        smoothed_b = {
            k: np.einsum("ap,pbc->abc", grammar.smoother[k[0]], v)
            for k, v in new_grammar.binary.items()
        }
        smoothed_u = {k: grammar.smoother[k[0]] @ v for k, v in new_grammar.unary.items()}
        new_grammar.binary, new_grammar.unary = normalize(smoothed_b, smoothed_u)
    return float(total), new_grammar, new_lexicon


def normalize(
    binary: Mapping[tuple[int, int, int], np.ndarray],
    unary: Mapping[tuple[int, int], np.ndarray],
    threshold: float = PRUNE,
) -> tuple[dict, dict]:
    """`Grammar.normalize`: divide by parent-substate totals; zero values below `threshold`."""
    parents: dict[int, np.ndarray] = {}
    for (a, b), counts in unary.items():
        if a != b:
            parents[a] = parents.get(a, 0) + counts.sum(axis=1)
    for (a, _, _), counts in binary.items():
        parents[a] = parents.get(a, 0) + counts.sum(axis=(1, 2))

    def divide(counts: np.ndarray, parent: int) -> np.ndarray:
        total = parents[parent].reshape((-1,) + (1,) * (counts.ndim - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(total != 0, counts / total, counts)
        values[(total != 0) & (values < threshold)] = 0.0
        return values

    return (
        {k: divide(v, k[0]) for k, v in binary.items()},
        {k: divide(v, k[0]) for k, v in unary.items()},
    )
