"""The lexicon of the latent-variable PCFG: P(word | tag substate), as Berkeley estimates it.

This reimplements the behavior of the Berkeley parser's `SophisticatedLexicon`
with its defaults (unknown-word level 5, rare-word threshold 20, smoothing
parameters 0.5 and 0.1), as `GrammarTrainer` uses it.

Counts. Each training token adds its posterior over its tag's substates to
`word_tag[tag][word]`, `tag_count[tag]`, `word_count[word]`, and `total`.
When the word occurred fewer than `rare + 0.5` times under the previous
lexicon, the token also counts toward its signature: `word_count[sig]`,
`unseen_word_tag[tag][sig]`, `unseen_tag_count[tag]`, and `total_unseen`.
After a pass, `tie_rare_words` replaces the substate counts of each word
seen fewer than `rare + 0.5` times by its tag's unseen-word distribution,
scaled to the word's count, and counts below the pruning threshold are
zeroed.

Scores. For a word seen in training, `P(T_t | W)` is smoothed toward
`P(T_t | unseen)` with weight 0.0001 when the word occurred more than 100
times or when smoothing is off (as in training), and with weight
`smooth[1]` otherwise; Bayes' rule turns it into `P(W | T_t)`. For an
unseen word, `P(T_t | signature)` is smoothed toward `P(T_t | unseen)` with
weight `smooth[0]`, and `P(W)` is taken as `1 / total`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

RARE = 20
SMOOTHING = (0.5, 0.1)
SMOOTH_IN_UNKNOWNS = 100.0
PRUNE = 1.0e-30

_SUFFIXES = ("ed", "ing", "ion", "er", "est", "ly", "ity", "y", "al")


def signature(word: str, position: int, known) -> str:
    """Berkeley's unknown-word signature at level 5 (`SophisticatedLexicon.getSignature`).

    `known(word)` says whether a lowercased form is in the lexicon; it adds
    `-KNOWNLC` to a capitalized sentence-initial word.
    """
    parts = ["UNK"]
    if not word:
        return "UNK"
    caps = 0
    has_digit = has_dash = has_lower = False
    for char in word:
        if char.isdigit():
            has_digit = True
        elif char == "-":
            has_dash = True
        elif char.isalpha():
            if char.islower():
                has_lower = True
            elif char.istitle() and not char.isupper():
                has_lower = True
                caps += 1
            else:
                caps += 1
    first = word[0]
    lowered = word.lower()
    if first.isupper() or (first.istitle() and not first.islower()):
        if position == 0 and caps == 1:
            parts.append("-INITC")
            if known(lowered):
                parts.append("-KNOWNLC")
        else:
            parts.append("-CAPS")
    elif not first.isalpha() and caps > 0:
        parts.append("-CAPS")
    elif has_lower:
        parts.append("-LC")
    if has_digit:
        parts.append("-NUM")
    if has_dash:
        parts.append("-DASH")
    if lowered.endswith("s") and len(word) >= 3:
        if lowered[-2] not in "siu":
            parts.append("-s")
    elif len(word) >= 5 and not has_dash and not (has_digit and caps > 0):
        for suffix in _SUFFIXES:
            if lowered.endswith(suffix):
                parts.append("-" + suffix)
                break
    return "".join(parts)


@dataclass
class Lexicon:
    """Lexical counts by tag (an integer) and substate, with Berkeley's scoring."""

    substates: Mapping[int, int]
    smoother: Mapping[int, np.ndarray] | None = None
    """Per tag, the matrix `D` that scores are multiplied by (smoothing across substates)."""
    word_tag: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    unseen_word_tag: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    tag_count: dict[int, np.ndarray] = field(default_factory=dict)
    unseen_tag_count: dict[int, np.ndarray] = field(default_factory=dict)
    word_count: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    total_unseen: float = 0.0
    _last: tuple[str, int, str] | None = None

    def _signature(self, word: str, position: int) -> str:
        """`getCachedSignature`: the last word and position's signature is reused."""
        if self._last is not None and self._last[0] == word and self._last[1] == position:
            return self._last[2]
        found = signature(word, position, self.word_count.__contains__)
        self._last = (word, position, found)
        return found

    def _vector(self, table: dict[int, np.ndarray], tag: int) -> np.ndarray:
        if tag not in table:
            table[tag] = np.zeros(self.substates[tag])
        return table[tag]

    def train_token(
        self,
        word: str,
        tag: int,
        position: int,
        weights: np.ndarray,
        previous_counts: Mapping[str, float] | None,
        rare: int = RARE,
    ) -> None:
        """Add one token's substate posteriors (`SophisticatedLexicon.trainTree`)."""
        sig = self._signature(word, position)
        self.word_count.setdefault(sig, 0.0)
        unseen = self.unseen_word_tag.setdefault(tag, {})
        if sig not in unseen:
            unseen[sig] = np.zeros(self.substates[tag])
        seen = self.word_tag.setdefault(tag, {})
        if word not in seen:
            seen[word] = np.zeros(self.substates[tag])
        tag_count = self._vector(self.tag_count, tag)
        unseen_tag_count = self._vector(self.unseen_tag_count, tag)
        rare_word = previous_counts is not None and previous_counts.get(word, 0.0) < rare + 0.5
        for substate, weight in enumerate(weights):
            if weight == 0:
                continue
            seen[word][substate] += weight
            tag_count[substate] += weight
            self.word_count[word] = self.word_count.get(word, 0.0) + weight
            self.total += weight
            if rare_word:
                self.word_count[sig] = self.word_count.get(sig, 0.0) + weight
                unseen[sig][substate] += weight
                unseen_tag_count[substate] += weight
                self.total_unseen += weight

    def tie_rare_words(self, rare: int = RARE) -> None:
        """`tieRareWordStats`: rare words take their tag's unseen-word substate distribution."""
        for tag, words in self.word_tag.items():
            unseen = self.unseen_tag_count.get(tag)
            if unseen is None:
                continue
            unseen_total = unseen.sum()
            if unseen_total == 0:
                continue
            for word, counts in words.items():
                if self.word_count.get(word, 0.0) < rare + 0.5:
                    words[word] = unseen * counts.sum() / unseen_total

    def prune(self, threshold: float = PRUNE) -> None:
        """`removeUnlikelyTags`: zero the word-tag counts below the threshold."""
        for words in self.word_tag.values():
            for counts in words.values():
                counts[counts < threshold] = 0.0

    def score(
        self, word: str, tag: int, position: int, no_smoothing: bool, is_signature: bool = False
    ) -> np.ndarray:
        """`SophisticatedLexicon.score`: P(word | tag substate) for each substate.

        With `is_signature`, `word` is itself an unknown-word signature, scored
        by the unseen-word model as Berkeley's text dump scores signatures.
        """
        n = self.substates[tag]
        result = np.zeros(n)
        c_w = self.word_count.get(word, 0.0)
        seen_word = c_w > 0.0
        tag_count = self.tag_count.get(tag, np.zeros(n))
        unseen_tag = self.unseen_tag_count.get(tag, np.zeros(n))
        for s in range(n):
            if not is_signature and (seen_word or no_smoothing):
                c_t = tag_count[s]
                if c_t == 0:
                    continue
                c_tw = self.word_tag.get(tag, {}).get(word)
                c_tw = 0.0 if c_tw is None else c_tw[s]
                p_t_u = 1.0 if self.total_unseen == 0 else unseen_tag[s] / self.total_unseen
                if c_w > SMOOTH_IN_UNKNOWNS or no_smoothing:
                    if no_smoothing and c_w == 0:
                        p_t_w = c_tw / 1
                    else:
                        p_t_w = (c_tw + 0.0001 * p_t_u) / (c_w + 0.0001)
                else:
                    p_t_w = (c_tw + SMOOTHING[1] * p_t_u) / (c_w + SMOOTHING[1])
                if p_t_w == 0:
                    continue
                p_w_t = p_t_w * (c_w / self.total) / (c_t / self.total)
            else:
                sig = word if is_signature else self._signature(word, position)
                c_ts = self.unseen_word_tag.get(tag, {}).get(sig)
                c_ts = 0.0 if c_ts is None else c_ts[s]
                c_s = self.word_count.get(sig, 0.0)
                p_t_u = unseen_tag[s] / self.total_unseen
                p_t_s = (c_ts + SMOOTHING[0] * p_t_u) / (c_s + SMOOTHING[0])
                p_w_t = p_t_s * (1.0 / self.total) / (tag_count[s] / self.total)
            result[s] = 1e-87 if p_w_t == 0 else p_w_t
        if self.smoother is not None and tag in self.smoother:
            result = self.smoother[tag] @ result
        return result


def token_signatures(tokens) -> list[str]:
    """The signature of each `(word, tag, position)` token as a training pass computes it.

    `-KNOWNLC` depends on the words and signatures counted before each
    token in the pass, and the last signature is cached, so signatures are
    computed in order. Every pass over the same tokens gives the same ones.
    """
    keys: set[str] = set()
    last: tuple[str, int, str] | None = None
    found = []
    for word, _, position in tokens:
        if last is not None and last[0] == word and last[1] == position:
            sig = last[2]
        else:
            sig = signature(word, position, keys.__contains__)
            last = (word, position, sig)
        keys.add(sig)
        keys.add(word)
        found.append(sig)
    return found


def train_tokens(
    lexicon: Lexicon,
    tokens,
    signatures: list[str],
    posteriors: np.ndarray,
    previous_counts: Mapping[str, float] | None,
    rare: int = RARE,
) -> None:
    """`Lexicon.train_token` for every token at once (sums in a different order).

    `posteriors` is `[tokens, S]`, zero beyond each tag's substates.
    """
    words = [w for w, _, _ in tokens]
    tags = np.array([t for _, t, _ in tokens])
    weights = posteriors.sum(axis=1)
    rare_token = np.array(
        [previous_counts is not None and previous_counts.get(w, 0.0) < rare + 0.5 for w in words]
    )

    def grouped(keys: list, mask: np.ndarray) -> dict:
        index: dict = {}
        ids = np.array(
            [index.setdefault(k, len(index)) for k, m in zip(keys, mask, strict=True) if m]
        )
        sums = np.zeros((len(index), posteriors.shape[1]))
        if len(ids):
            np.add.at(sums, ids, posteriors[mask])
        return {k: sums[i] for k, i in index.items()}

    everything = np.ones(len(words), dtype=bool)
    for (tag, word), values in grouped(
        list(zip(tags.tolist(), words, strict=True)), everything
    ).items():
        n = lexicon.substates[tag]
        table = lexicon.word_tag.setdefault(tag, {})
        table[word] = table.get(word, np.zeros(n)) + values[:n]
    for tag, sig in set(zip(tags.tolist(), signatures, strict=True)):
        # every token creates its signature's entry, rare or not
        lexicon.unseen_word_tag.setdefault(tag, {}).setdefault(
            sig, np.zeros(lexicon.substates[tag])
        )
    for (tag, sig), values in grouped(
        list(zip(tags.tolist(), signatures, strict=True)), rare_token
    ).items():
        n = lexicon.substates[tag]
        lexicon.unseen_word_tag[tag][sig] = lexicon.unseen_word_tag[tag][sig] + values[:n]
    for tag in np.unique(tags):
        n = lexicon.substates[int(tag)]
        rows = tags == tag
        lexicon.tag_count[int(tag)] = lexicon.tag_count.get(int(tag), np.zeros(n)) + posteriors[
            rows, :n
        ].sum(axis=0)
        unseen = posteriors[rows & rare_token, :n].sum(axis=0)
        lexicon.unseen_tag_count[int(tag)] = (
            lexicon.unseen_tag_count.get(int(tag), np.zeros(n)) + unseen
        )
    for sig in signatures:
        lexicon.word_count.setdefault(sig, 0.0)
    for word, weight in zip(words, weights, strict=True):
        lexicon.word_count[word] = lexicon.word_count.get(word, 0.0) + weight
    for sig, weight in zip(np.array(signatures)[rare_token], weights[rare_token], strict=True):
        lexicon.word_count[sig] = lexicon.word_count.get(sig, 0.0) + weight
    lexicon.total += float(weights.sum())
    lexicon.total_unseen += float(weights[rare_token].sum())
