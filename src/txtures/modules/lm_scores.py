"""Token-level scores from a causal language model.

For every token of a document, `lm_scores` stores its ID, its character span
in the document text, the context it was scored in, and two natural-log
probabilities:

- `logprob`: the probability of the token given the tokens before it in its
  context;
- `boundary_logprob`: the probability that the next token belongs to the
  boundary set *B* (begins a new word), given the context through this token.
  For the last token of a context it comes from the model's final-position
  logits.

Each context starts with the document-start tokens that the `hf_model`
resource resolved. Contexts longer than the window are scored with a sliding
window, in which every token is predicted from at least `window - stride`
tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from txtures.corpus import Corpus, Document
from txtures.errors import TxturesError
from txtures.modules.base import Item, Module, ModuleConfig, RunContext, Workload
from txtures.modules.hf_model import HFModel, LanguageModel, is_local

LOGITS_BUDGET_BYTES = 1 << 30
"""Approximate size of the float32 log-probabilities computed in one forward batch."""


class LMScoresConfig(ModuleConfig):
    """Parameters for scoring tokens with a causal language model."""

    model: str = Field(default="gpt2", min_length=1)
    """Hugging Face repository, or the absolute path of a local model directory."""

    revision: str | None = Field(default=None, min_length=1)
    """Commit hash, branch, or tag to load. When unset, the default branch.
    Branches and tags are resolved to a commit hash, which is recorded."""

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Model weight precision. Log-probabilities are always computed in float32."""

    window: int | None = Field(default=None, gt=1)
    """Tokens per forward pass. When unset, the model's maximum context length."""

    stride: int | None = Field(default=None, gt=0)
    """Tokens advanced between windows. When unset, half the window."""

    context: Literal["document", "sentence"] = "document"
    """Whether each document or each sentence is scored as one context."""

    document_start: Literal["auto", "bos", "eos", "none"] = "auto"
    """How the start of a context is marked for the model. `auto` follows the
    model's training convention, read from its tokenizer; see
    `hf_model.auto_document_start`."""

    @model_validator(mode="after")
    def _stride_fits_window(self) -> Self:
        if self.window is not None and self.stride is not None and self.stride >= self.window:
            raise ValueError("stride must be smaller than window")
        return self


@dataclass(frozen=True)
class Window:
    """A forward pass over `sequence[start:stop]` whose outputs at positions
    `own:stop` are kept."""

    sequence: int
    start: int
    stop: int
    own: int


def windows(length: int, window: int, stride: int) -> list[tuple[int, int, int]]:
    """Divide positions `0..length-1` into `(start, stop, own)` windows.

    The first window keeps all of its positions; each later window keeps only
    the positions the previous one did not reach, so every position is kept
    exactly once and has at least `window - stride` positions before it in
    its window, or all earlier positions.
    """
    spans = [(0, min(window, length), 0)]
    start = 0
    while start + window < length:
        start += stride
        spans.append((start, min(start + window, length), start - stride + window))
    return spans


def score_sequences(
    lm: LanguageModel, sequences: Sequence[Sequence[int]], window: int, stride: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Score token sequences, returning `(next_logprob, boundary_logprob)` for each.

    For position `p` of a sequence, `next_logprob[p]` is the log-probability
    of token `p + 1` given tokens `0..p` (NaN for the last position) and
    `boundary_logprob[p]` is the log-probability that token `p + 1` is in
    *B* (NaN when the tokenizer's boundaries were not detected).
    """
    import torch

    plan = [
        Window(index, start, stop, own)
        for index, sequence in enumerate(sequences)
        for start, stop, own in windows(len(sequence), window, stride)
    ]
    outputs = [
        (np.full(len(s), np.nan, dtype=np.float32), np.full(len(s), np.nan, dtype=np.float32))
        for s in sequences
    ]
    vocabulary = lm.boundary.shape[0]
    per_batch = max(1, LOGITS_BUDGET_BYTES // (window * vocabulary * 4))
    pad = lm.tokenizer.pad_token_id
    if pad is None:
        pad = lm.tokenizer.eos_token_id if lm.tokenizer.eos_token_id is not None else 0
    with torch.inference_mode():
        for first in range(0, len(plan), per_batch):
            batch = plan[first : first + per_batch]
            width = max(w.stop - w.start for w in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, w in enumerate(batch):
                chunk = sequences[w.sequence][w.start : w.stop]
                ids[row, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
                mask[row, : len(chunk)] = 1
            logits = lm.model(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device)).logits
            for row, w in enumerate(batch):
                sequence = sequences[w.sequence]
                kept = logits[row, w.own - w.start : w.stop - w.start].float()
                if kept.shape[-1] != vocabulary:
                    raise TxturesError(
                        f"model returned {kept.shape[-1]} logits per position, but its "
                        f"boundary set covers {vocabulary} token IDs"
                    )
                logprobs = torch.log_softmax(kept, dim=-1)
                next_logprob, boundary_logprob = outputs[w.sequence]
                targets = range(w.own + 1, min(w.stop + 1, len(sequence)))
                if len(targets):
                    target_ids = torch.tensor(
                        [sequence[t] for t in targets], dtype=torch.long, device=logprobs.device
                    )
                    chosen = logprobs[: len(targets)].gather(1, target_ids[:, None])[:, 0]
                    next_logprob[w.own : w.own + len(targets)] = chosen.cpu().numpy()
                if lm.boundaries_detected:
                    in_boundary = logprobs.masked_fill(~lm.boundary, -math.inf)
                    boundary_logprob[w.own : w.stop] = (
                        torch.logsumexp(in_boundary, dim=-1).cpu().numpy()
                    )
            del logits
    return outputs


def contexts(document: Document, context: str) -> list[tuple[int, int]]:
    """Character spans of the contexts a document is scored in."""
    if context == "sentence" and document.sentences:
        return [
            (document.regions[s.start][0], document.regions[s.stop - 1][1])
            for s in document.sentences
        ]
    return [(0, len(document.text))]


class LMScores(Module):
    """Score every token of each context with a causal language model."""

    name = "lm_scores"
    contract_version = 1
    Config = LMScoresConfig
    identifying = ("model",)
    sweeps = {
        "model": ("model", "revision"),
        "window": ("window", "stride"),
        "context": ("context",),
    }
    batch_size = 8

    def workload(self, config: LMScoresConfig) -> Workload:
        return Workload(slurm=True)

    def resources(self, config: LMScoresConfig):
        return {
            "model": HFModel(
                {
                    "model": config.model,
                    "revision": config.revision,
                    "dtype": config.dtype,
                    "document_start": config.document_start,
                }
            )
        }

    def check(self, config: LMScoresConfig, corpus: Corpus) -> list[str]:
        if config.context == "sentence" and not any(d.sentences for d in corpus.documents):
            return [f"context: sentence needs sentences, but input {corpus.name} defines none"]
        return []

    def notes(self, config: LMScoresConfig) -> list[str]:
        if config.revision is None and not is_local(config.model):
            return [
                f"model {config.model} has no pinned revision; its current commit is "
                "resolved and recorded on every run"
            ]
        return []

    def run_batch(
        self, config: LMScoresConfig, items: Sequence[Item], context: RunContext
    ) -> list[dict[str, np.ndarray]]:
        if not items:
            return []
        lm: LanguageModel = items[0].resources["model"]
        window = config.window or lm.max_length
        if window is None:
            raise TxturesError(
                f"model {config.model} does not declare a maximum context length; set window"
            )
        if lm.max_length is not None and window > lm.max_length:
            raise TxturesError(
                f"window {window} exceeds the maximum context length {lm.max_length} "
                f"of model {config.model}"
            )
        stride = config.stride or max(1, window // 2)
        if stride >= window:
            raise TxturesError(f"stride {stride} must be smaller than window {window}")

        sequences: list[list[int]] = []
        pieces: list[tuple[int, int, list[int], list[tuple[int, int]]]] = []
        for number, item in enumerate(items):
            for index, (start, end) in enumerate(contexts(item.document, config.context)):
                encoded = lm.tokenizer(
                    item.document.text[start:end],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                ids = list(encoded["input_ids"])
                spans = [(start + a, start + b) for a, b in encoded["offset_mapping"]]
                pieces.append((number, index, ids, spans))
                sequences.append([*lm.start_ids, *ids])
        scores = score_sequences(lm, sequences, window, stride)

        offset = len(lm.start_ids)
        parts: list[list[dict[str, np.ndarray]]] = [[] for _ in items]
        for (number, index, ids, spans), (next_logprob, boundary_logprob) in zip(
            pieces, scores, strict=True
        ):
            logprob = np.full(len(ids), np.nan, dtype=np.float32)
            if offset:
                logprob[:] = next_logprob[offset - 1 : -1]
            else:
                logprob[1:] = next_logprob[:-1]
            parts[number].append(
                {
                    "token_id": np.asarray(ids, dtype=np.int64),
                    "char_start": np.asarray([a for a, _ in spans], dtype=np.int64),
                    "char_end": np.asarray([b for _, b in spans], dtype=np.int64),
                    "context": np.full(len(ids), index, dtype=np.int32),
                    "logprob": logprob,
                    "boundary_logprob": boundary_logprob[offset:],
                }
            )
        return [
            {key: np.concatenate([part[key] for part in document]) for key in document[0]}
            for document in parts
        ]
