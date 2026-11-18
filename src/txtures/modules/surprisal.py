"""Region surprisal aggregated from language-model token scores.

A token belongs to the first region whose end lies after the token's last
character. So a token inside a region belongs to that region, and a
whitespace token between two regions belongs to the following one.

With `correction: word_boundary`, the surprisal of a region spanning tokens
t₁…tₖ after context c is

    −log P(t₁…tₖ | c) − log P(B | c, t₁…tₖ) + log P(B | c)

(Pimentel and Meister 2024; Oh and Schuler 2024), where *B* is the set of
tokens that begin a new word. For the first region of a context, c holds only
the document-start tokens, so no earlier word can continue and P(B | c) is 1.
"""

from __future__ import annotations

import bisect
import math
from typing import Literal

import numpy as np

from txtures.corpus import Document
from txtures.errors import TxturesError
from txtures.modules.base import Item, Module, ModuleConfig, RunContext


class SurprisalConfig(ModuleConfig):
    """Parameters for aggregating token scores into region surprisal."""

    correction: Literal["word_boundary", "none"] = "word_boundary"
    """Whether to apply the word-boundary correction to each region."""

    units: Literal["bits", "nats"] = "bits"
    """Logarithm base of the reported surprisal: 2 for bits, e for nats."""


def region_tokens(document: Document, char_start, char_end) -> list[tuple[int, int]]:
    """Return the `(first, stop)` token range of each region of `document`.

    Raises:
        TxturesError: A token's non-whitespace text spans more than one
            region, or tokens are out of order.
    """
    ends = [end for _, end in document.regions]
    assigned = []
    for start, end in zip(char_start.tolist(), char_end.tolist(), strict=True):
        last = end - 1 if end > start else start
        region = bisect.bisect_right(ends, last)
        text = document.text[start:end]
        if region < len(ends) and text.strip():
            first = start + len(text) - len(text.lstrip())
            if first < document.regions[region][0]:
                raise TxturesError(
                    f"document {document.doc_id!r}: token {text!r} spans more than one region"
                )
        assigned.append(region)
    if any(b < a for a, b in zip(assigned, assigned[1:], strict=False)):
        raise TxturesError(f"document {document.doc_id!r}: tokens are out of order")
    ranges = []
    for region in range(len(ends)):
        ranges.append((bisect.bisect_left(assigned, region), bisect.bisect_right(assigned, region)))
    return ranges


def region_surprisal(
    document: Document, scores: dict[str, np.ndarray], correction: str
) -> np.ndarray:
    """Return each region's surprisal in nats. Regions with no tokens get NaN."""
    logprob = scores["logprob"].astype(np.float64)
    boundary = scores["boundary_logprob"].astype(np.float64)
    context = scores["context"]
    if correction == "word_boundary" and len(boundary) and np.isnan(boundary).all():
        raise TxturesError(
            "lm_scores could not detect word boundaries for this model's tokenizer; "
            "use correction: none"
        )
    result = np.full(len(document.regions), np.nan)
    ranges = region_tokens(document, scores["char_start"], scores["char_end"])
    for region, (first, stop) in enumerate(ranges):
        if first == stop:
            continue
        value = -logprob[first:stop].sum()
        if correction == "word_boundary":
            value -= boundary[stop - 1]
            if first > 0 and context[first - 1] == context[first]:
                value += boundary[first - 1]
        result[region] = value
    return result


class Surprisal(Module):
    """Compute the surprisal of each region from `lm_scores` output."""

    name = "surprisal"
    contract_version = 1
    Config = SurprisalConfig
    requires = ("lm_scores",)
    features = ("surprisal",)
    sweeps = {
        "correction": ("correction",),
        "units": ("units",),
    }
    persist = False

    def run(self, config: SurprisalConfig, item: Item, context: RunContext) -> np.ndarray:
        nats = region_surprisal(item.document, item.upstream["lm_scores"], config.correction)
        return nats / math.log(2) if config.units == "bits" else nats

    def columns(self, config: SurprisalConfig, document: Document, result: np.ndarray):
        return {"surprisal": result}
