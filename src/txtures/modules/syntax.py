"""Helpers shared by the modules that read `parse` results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from txtures.sud import analyze_sentence


def sentence_ranges(parse: Mapping[str, np.ndarray]) -> list[tuple[int, int]]:
    """Return the `(first, stop)` word range of each parsed sentence, in order."""
    sentence = parse["sentence"]
    ranges = []
    for index in range(len(parse["sentence_start"])):
        positions = np.flatnonzero(sentence == index)
        if len(positions):
            ranges.append((int(positions[0]), int(positions[-1]) + 1))
    return ranges


def sud_tree(parse: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Convert a stored UD parse to SUD.

    Returns each word's SUD head as a word index in the document (-1 for a
    sentence root) and its SUD relation.
    """
    heads, labels, _ = sud_analysis(parse)
    return heads, labels


def sud_analysis(parse: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str], list]:
    """Like `sud_tree`, and also return each word's `Shared` value (`Yes`, `No`, or `None`)."""
    heads = np.full(len(parse["form"]), -1, dtype=np.int64)
    labels = [""] * len(parse["form"])
    shared: list = [None] * len(parse["form"])
    for first, stop in sentence_ranges(parse):
        ud_heads = [int(h) - first + 1 if h >= 0 else 0 for h in parse["head"][first:stop]]
        sentence_heads, sentence_labels, sentence_shared = analyze_sentence(
            parse["form"][first:stop].tolist(),
            parse["lemma"][first:stop].tolist(),
            parse["upos"][first:stop].tolist(),
            parse["feats"][first:stop].tolist(),
            ud_heads,
            parse["deprel"][first:stop].tolist(),
        )
        for offset, head in enumerate(sentence_heads):
            heads[first + offset] = first + head - 1 if head else -1
            labels[first + offset] = sentence_labels[offset]
            shared[first + offset] = sentence_shared[offset]
    return heads, labels, shared


def region_heads(parse: Mapping[str, np.ndarray], heads: np.ndarray, regions: int) -> np.ndarray:
    """Return the word that heads each region, or -1 for a region without words.

    A region's head is its word whose head lies outside the region. When
    several do, the first that is not punctuation is taken, or the first.
    """
    region = parse["region"]
    result = np.full(regions, -1, dtype=np.int64)
    for index in range(regions):
        words = np.flatnonzero(region == index)
        if not len(words):
            continue
        outside = [int(w) for w in words if heads[w] < 0 or region[heads[w]] != index]
        candidates = outside or [int(w) for w in words]
        content = [w for w in candidates if parse["upos"][w] != "PUNCT"]
        result[index] = (content or candidates)[0]
    return result
