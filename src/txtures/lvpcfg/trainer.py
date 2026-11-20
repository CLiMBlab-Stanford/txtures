"""Training a latent-variable PCFG by split, merge, and smooth, as Berkeley's `GrammarTrainer`.

`train` runs Berkeley's schedule with its defaults: a baseline grammar with
one substate per category, then `cycles` rounds of

1. split every category in two, then EM;
2. merge back half of the splits, re-estimate the lexicon, then EM;
3. turn on smoothing (0.01 for rules, 0.1 for the lexicon), then EM.

Berkeley runs 50, 20, and 10 EM iterations in these phases. Without a
validation set it keeps, at the end of each, the grammar from before the
last iteration, so the phases apply 49, 19, and 9 updates; these are the
defaults here.

Baseline (Berkeley's initialization with randomness 1): each token counts
`1 + u / 100` toward its word and tag in a first lexicon; a second lexicon
counts the same way, also counting rare words (by the first lexicon's
counts) toward their signatures; rare words are tied. Each rule's count is
its number of occurrences plus `u`. `u` is uniform on [0, 1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from txtures.lvpcfg import splitmerge as sm
from txtures.lvpcfg.em import Corpus, Grammar, normalize
from txtures.lvpcfg.lexicon import RARE, Lexicon, train_tokens

LOG = logging.getLogger(__name__)

GRAMMAR_SMOOTHING = 0.01
LEXICON_SMOOTHING = 0.1


@dataclass
class TrainingLog:
    """The training log-likelihood after each phase and of every EM iteration."""

    phases: list[dict] = field(default_factory=list)

    def add(self, round_: int, phase: str, likelihoods: Sequence[float], substates: int) -> None:
        self.phases.append(
            {
                "round": round_,
                "phase": phase,
                "log_likelihoods": [float(x) for x in likelihoods],
                "substates": substates,
            }
        )


def baseline(
    corpus: Corpus,
    categories: Mapping[str, int],
    rng: np.random.Generator,
    randomness: float = 1.0,
    rare: int = RARE,
) -> tuple[Grammar, Lexicon]:
    """The initial grammar and lexicon, with one substate per category."""
    substates = {c: 1 for c in categories.values()}
    noise = 1.0 + rng.random((len(corpus.tokens), 1)) * randomness / 100.0
    first = Lexicon(substates)
    train_tokens(first, corpus.tokens, corpus.signatures, noise, None, rare)
    noise = 1.0 + rng.random((len(corpus.tokens), 1)) * randomness / 100.0
    lexicon = Lexicon(substates)
    train_tokens(lexicon, corpus.tokens, corpus.signatures, noise, first.word_count, rare)
    lexicon.tie_rare_words(rare)
    lexicon.prune()
    binary: dict[tuple, np.ndarray] = {}
    unary: dict[tuple, np.ndarray] = {}
    for nodes in corpus.nodes:
        for kind, key, _, _ in nodes:
            if kind == "binary":
                binary[key] = binary.get(key, 0) + np.ones((1, 1, 1))
            elif kind == "unary":
                unary[key] = unary.get(key, 0) + np.ones((1, 1))
    for table in (binary, unary):
        for key in table:
            table[key] = table[key] + rng.random(table[key].shape) * randomness
    return Grammar(substates, *normalize(binary, unary)), lexicon


def train(
    corpus: Corpus,
    categories: Mapping[str, int],
    cycles: int = 3,
    seed: int = 2,
    merge_rule: str = "negligible",
    updates: tuple[int, int, int] = (49, 19, 9),
    device: str = "cpu",
    randomness: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> tuple[Grammar, Lexicon, dict[int, sm.SplitTree], TrainingLog]:
    """Run Berkeley's split-merge-smooth schedule; return the final grammar, lexicon, and log.

    The grammar and lexicon stay on `device` as tensors (`txtures.lvpcfg.fast`)
    from the baseline to the end.
    """
    import torch

    from txtures.lvpcfg import fast

    report = progress or LOG.info
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    grammar, lexicon = baseline(corpus, categories, rng, randomness)
    template_grammar, template_lexicon = grammar, lexicon
    state = fast.to_state(corpus, grammar, lexicon, device)
    trees: dict[int, sm.SplitTree] = {c: 0 for c in categories.values()}
    log = TrainingLog()
    size_of = max(categories.values()) + 1

    def em(state, count: int, label: str):
        likelihoods = []
        for i in range(count):
            ll, state = fast.iterate(corpus, state)
            likelihoods.append(ll)
            report(f"{label} iteration {i + 1}/{count}: log-likelihood {ll:.2f}")
        return state, likelihoods

    for round_ in range(1, cycles + 1):
        old = dict(state.substates)
        state = fast.split(state, generator, randomness)
        trees = {c: sm.extend_split_tree(t, state.substates[c] > old[c]) for c, t in trees.items()}
        state, lls = em(state, updates[0], f"round {round_} split")
        log.add(round_, "split", lls, sum(state.substates.values()))

        weights, deltas = fast.merge_statistics(corpus, state)
        pairs = sm.merge_pairs(deltas, rule=merge_rule)
        report(f"round {round_}: merging {sum(map(len, pairs.values()))} pairs")
        mapping = sm.merge_mapping(state.substates, pairs)
        trees = {c: sm.prune_split_tree(t, mapping[c], pairs.get(c, ())) for c, t in trees.items()}
        state = fast.merge(corpus, state, pairs, weights)
        _, state = fast.iterate(corpus, state, lexicon_only=True)
        state, lls = em(state, updates[1], f"round {round_} merge")
        log.add(round_, "merge", lls, sum(state.substates.values()))

        size = state.binary.shape[1]
        state.grammar_smoother = fast.smoother_tensor(
            sm.smoothers(trees, GRAMMAR_SMOOTHING), size_of, size, device
        )
        state.lexicon_smoother = fast.smoother_tensor(
            sm.smoothers(trees, LEXICON_SMOOTHING), size_of, size, device
        )
        state, lls = em(state, updates[2], f"round {round_} smooth")
        log.add(round_, "smooth", lls, sum(state.substates.values()))
    template_grammar = Grammar(state.substates, template_grammar.binary, template_grammar.unary)
    grammar, lexicon = fast.to_dicts(corpus, state, template_grammar, template_lexicon)
    if cycles:
        grammar.smoother = sm.smoothers(trees, GRAMMAR_SMOOTHING)
        lexicon.smoother = sm.smoothers(trees, LEXICON_SMOOTHING)
    return grammar, lexicon, trees, log
