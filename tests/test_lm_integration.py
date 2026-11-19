"""Tests with real tokenizers and models from the Hugging Face hub.

They download files into the Hugging Face cache and run only with
`pytest -m integration`.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from txtures.corpus import Document
from txtures.errors import TxturesError
from txtures.modules import Item, RunContext
from txtures.modules.hf_model import (
    HFModel,
    auto_document_start,
    boundaries_detected,
    boundary_mask,
    load_tokenizer,
    resolve_revision,
)
from txtures.modules.lm_scores import LMScores, LMScoresConfig
from txtures.modules.surprisal import Surprisal, SurprisalConfig, region_tokens

pytestmark = pytest.mark.integration

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PYTHIA_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"

TEXTS = [
    "The cat  sat.\nHé said “ok”, didn't he?",
    "Naïve café-goers—1,234 of them—ate 🍰 in Zürich.",
    'It rained. Then (oddly) it   snowed; "really" snowed!',
]

MERGES_NEWLINES = {"Qwen/Qwen2.5-0.5B"}

TOKENIZERS = {
    "gpt2": "eos",
    "EleutherAI/pythia-70m": "eos",
    "hf-internal-testing/llama-tokenizer": "bos",
    "allenai/OLMo-1B-hf": "eos",
    "Qwen/Qwen2.5-0.5B": "eos",
}


def words(text: str, document_id="d") -> Document:
    return Document(document_id, text, tuple(m.span() for m in re.finditer(r"\S+", text)))


@pytest.mark.parametrize(("model", "expected_start"), TOKENIZERS.items())
def test_tokenizer_matrix(model, expected_start):
    resolved = {"model": model, "revision": resolve_revision(model, None, None)}
    tokenizer = load_tokenizer(resolved, None)
    assert auto_document_start(tokenizer) == expected_start
    mask = boundary_mask(tokenizer, len(tokenizer))
    assert boundaries_detected(tokenizer, mask)
    assert mask[tokenizer.eos_token_id]
    for text in TEXTS:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, spans = encoded["input_ids"], encoded["offset_mapping"]
        # After the first token, a token is in B exactly when its text starts
        # with whitespace. (A SentencePiece tokenizer marks the first word as
        # word-initial itself.)
        for token, (start, _) in list(zip(ids, spans, strict=True))[1:]:
            assert mask[token] == text[start].isspace(), tokenizer.convert_ids_to_tokens(token)
        doc = words(text)
        starts, ends = np.array([a for a, _ in spans]), np.array([b for _, b in spans])
        if model in MERGES_NEWLINES and "\n" in text:
            # These tokenizers join punctuation with a following newline
            # (".\n"), so a token spans two regions; txtures refuses rather
            # than misattribute it. Inputs contain newlines only when a
            # sidecar sets a newline joiner.
            with pytest.raises(TxturesError, match="spans more than one region"):
                region_tokens(doc, starts, ends)
            continue
        ranges = region_tokens(doc, starts, ends)
        assert all(stop > first for first, stop in ranges)
        assert ranges[-1][1] == len(ids)


def load(model: str, revision: str, tmp_path: Path, document_start: str = "auto"):
    spec = {
        "model": model,
        "revision": revision,
        "dtype": "float32",
        "document_start": document_start,
    }
    resource = HFModel(spec)
    context = RunContext(device="cpu")
    resolved = resource.resolve(context)
    resource.build(resolved, tmp_path, context)
    return resolved, resource.load(resolved, tmp_path, context)


def surprisal(lm, doc, correction="word_boundary", **config):
    (scores,) = LMScores().run_batch(
        LMScoresConfig(**config), [Item(doc, {}, {"model": lm})], RunContext("cpu")
    )
    item = Item(doc, {"lm_scores": scores}, {})
    return scores, Surprisal().run(SurprisalConfig(correction=correction), item, None)


def test_pythia_matches_a_direct_forward_pass(tmp_path):
    resolved, lm = load("EleutherAI/pythia-70m", PYTHIA_REVISION, tmp_path)
    assert resolved["document_start"] == "eos"
    doc = words(TEXTS[0])
    scores, uncorrected = surprisal(lm, doc, correction="none")
    ids = [lm.tokenizer.eos_token_id, *lm.tokenizer.encode(doc.text, add_special_tokens=False)]
    with torch.inference_mode():
        logits = lm.model(input_ids=torch.tensor([ids])).logits[0].float()
    logprobs = torch.log_softmax(logits, -1)
    direct = -sum(logprobs[p, ids[p + 1]].item() for p in range(len(ids) - 1)) / math.log(2)
    assert uncorrected.sum() == pytest.approx(direct, rel=1e-5)
    _, corrected = surprisal(lm, doc)
    assert np.isfinite(corrected).all() and (corrected > 0).all()


def test_gpt2_sliding_window_and_sentence_context(tmp_path):
    _, lm = load("gpt2", GPT2_REVISION, tmp_path)
    doc = words(" ".join(["The old man the boat."] * 40))
    _, full = surprisal(lm, doc)
    _, windowed = surprisal(lm, doc, window=64, stride=32)
    np.testing.assert_allclose(windowed[:10], full[:10], rtol=1e-4)
    assert np.isfinite(windowed).all()


def byte_level_boundaries(tokenizer) -> torch.Tensor:
    """B for a GPT-2 byte-level tokenizer, built from its byte-to-character
    table without the tokenizer's decoder: tokens whose bytes begin with the
    UTF-8 encoding of a whitespace character, and EOS."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    byte_of, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            byte_of[chr(byte)] = byte
        else:
            byte_of[chr(256 + extra)] = byte
            extra += 1
    mask = []
    for token in tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))):
        raw = bytes(byte_of[char] for char in token) if all(c in byte_of for c in token) else b""
        mask.append(raw.decode("utf-8", errors="replace")[:1].isspace())
    mask[tokenizer.eos_token_id] = True
    return torch.tensor(mask)


def test_gpt2_matches_byte_level_reference(tmp_path):
    _, lm = load("gpt2", GPT2_REVISION, tmp_path)
    reference = byte_level_boundaries(lm.tokenizer)
    assert torch.equal(reference, lm.boundary)
    for text in GOLDEN:
        ids = [lm.tokenizer.eos_token_id, *lm.tokenizer.encode(text)]
        with torch.inference_mode():
            logits = lm.model(input_ids=torch.tensor([ids])).logits[0].float()
        logprobs = torch.log_softmax(logits, -1)
        boundary = torch.logsumexp(logprobs.masked_fill(~reference, -math.inf), -1)
        groups: list[list[int]] = []
        for position in range(1, len(ids)):
            if not groups or reference[ids[position]]:
                groups.append([])
            groups[-1].append(position)
        expected = []
        for group in groups:
            value = -sum(logprobs[p - 1, ids[p]].item() for p in group)
            value -= boundary[group[-1]].item()
            if group[0] > 1:
                value += boundary[group[0] - 1].item()
            expected.append(value / math.log(2))
        _, values = surprisal(lm, words(text))
        # Same model, same float32 log-probabilities; only summation order differs.
        np.testing.assert_allclose(values, expected, atol=1e-5)


GOLDEN = {
    "The horse raced past the barn fell.": [
        4.73479,
        13.651452,
        10.626697,
        6.446749,
        1.914255,
        6.842625,
        23.242634,
    ],
    "After the student moved the chair broke.": [
        9.310087,
        3.713225,
        12.609502,
        10.234115,
        7.393403,
        9.245433,
        21.181077,
    ],
    "I like bananas, but\nnot bananas-flavored candy.": [
        6.15114,
        6.859798,
        16.160805,
        2.350765,
        31.775247,
        21.461651,
        7.758431,
    ],
}


def test_gpt2_golden_values(tmp_path):
    _, lm = load("gpt2", GPT2_REVISION, tmp_path)
    for text, expected in GOLDEN.items():
        _, values = surprisal(lm, words(text))
        # Golden values were computed on CPU in float32; other devices and
        # library versions differ by far less than 1e-3 bits.
        np.testing.assert_allclose(values, expected, atol=1e-3)


def test_compute_through_the_api_matches_golden_values(tmp_path, monkeypatch):
    from huggingface_hub import constants

    import txtures as tx

    # Share the Hugging Face cache the other tests use, so GPT-2 is not
    # downloaded again into the temporary resources directory.
    hub = Path(constants.HF_HUB_CACHE)
    hub.mkdir(parents=True, exist_ok=True)
    (tmp_path / "resources" / "huggingface").mkdir(parents=True)
    (tmp_path / "resources" / "huggingface" / "hub").symlink_to(hub)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    monkeypatch.setenv("TXTURES_HOME", str(tmp_path))

    config = {"lm_scores": {"model": "gpt2", "revision": GPT2_REVISION}}
    table = tx.compute("surprisal", list(GOLDEN), config=config, device="cpu")
    for document_id, expected in enumerate(GOLDEN.values(), start=1):
        values = table.loc[table["document_id"] == str(document_id), "surprisal"]
        np.testing.assert_allclose(values, expected, atol=1e-3)
    assert (tmp_path / "cache" / "lm_scores").is_dir()
