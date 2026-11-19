"""Tests of `hf_model`, `lm_scores`, and `surprisal` with a toy language model."""

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
import toy_lm
from conftest import write

from txtures.corpus import Document, Segment
from txtures.definitions import Definitions
from txtures.engine import Executor, plan_request, run_request
from txtures.errors import DefinitionError, TxturesError
from txtures.modules import REGISTRY, Item, RunContext
from txtures.modules.hf_model import HFModel, directory_digest
from txtures.modules.lm_scores import LMScores, LMScoresConfig, score_sequences, windows
from txtures.modules.surprisal import Surprisal, SurprisalConfig, region_surprisal
from txtures.settings import resolve_settings

LN2 = math.log(2)


@pytest.fixture(scope="session")
def toy_models(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("models")
    styles = ("eos", "bos", "no_bos", "quiet_bos", "bare")
    return {style: toy_lm.build(root / style, style) for style in styles}


@pytest.fixture(scope="session")
def lms(toy_models, tmp_path_factory):
    """Loaded `hf_model` resources for each toy model, keyed by (style, document_start)."""
    loaded = {}
    root = tmp_path_factory.mktemp("resources")

    def get(style: str, document_start: str = "auto"):
        key = (style, document_start)
        if key not in loaded:
            resource = HFModel(
                {
                    "model": str(toy_models[style]),
                    "revision": None,
                    "dtype": "float32",
                    "document_start": document_start,
                }
            )
            context = RunContext(device="cpu")
            resolved = resource.resolve(context)
            directory = root / f"{style}-{document_start}"
            directory.mkdir()
            resource.build(resolved, directory, context)
            loaded[key] = resource.load(resolved, directory, context)
        return loaded[key]

    return get


def document(words: list[str], sentences: list[int] | None = None, document_id="d") -> Document:
    """A document whose regions are `words` joined by single spaces.

    `sentences` lists sentence lengths in regions.
    """
    regions, position = [], 0
    for word in words:
        regions.append((position, position + len(word)))
        position += len(word) + 1
    spans = ()
    if sentences:
        starts = np.cumsum([0, *sentences])
        spans = tuple(
            Segment(str(n + 1), int(a), int(b))
            for n, (a, b) in enumerate(zip(starts, starts[1:], strict=False))
        )
    return Document(document_id, " ".join(words), tuple(regions), spans)


def scores(lm, documents, **config):
    module = LMScores()
    items = [Item(d, {}, {"model": lm}) for d in documents]
    return module.run_batch(LMScoresConfig(**config), items, RunContext(device="cpu"))


def surprisal(lm, doc, correction="word_boundary", units="bits", **config):
    (result,) = scores(lm, [doc], **config)
    item = Item(doc, {"lm_scores": result}, {})
    return Surprisal().run(SurprisalConfig(correction=correction, units=units), item, None)


# Brute-force reference


class StringProbabilities:
    """Probabilities of text prefixes under a toy model, by enumerating tokens.

    `event(prefix)` is the probability that the generated text begins with
    `prefix` and that the next character is whitespace or the text ends. It
    sums over token sequences by their decoded strings, without using token
    spans, region assignment, or the boundary set.
    """

    def __init__(self, lm):
        self.lm = lm

    @cache  # noqa: B019 - instances live for one test
    def next_logprobs(self, prefix: tuple[int, ...]) -> np.ndarray:
        ids = torch.tensor([[*self.lm.start_ids, *prefix]])
        with torch.inference_mode():
            logits = self.lm.model(input_ids=ids).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1).double().numpy()

    def event(self, target: str) -> float:
        return self._search((), "", target)

    def _search(self, prefix: tuple[int, ...], text: str, target: str) -> float:
        if len(text) > len(target):
            return 1.0 if text[len(target)].isspace() else 0.0
        probabilities = np.exp(self.next_logprobs(prefix))
        total = 0.0
        for token, string in toy_lm.STRINGS.items():
            if string is None:
                total += probabilities[token] if text == target else 0.0
                continue
            extended = text + string
            if text == target:
                total += probabilities[token] if string[0].isspace() else 0.0
            elif target.startswith(extended) or extended.startswith(target):
                total += probabilities[token] * self._search((*prefix, token), extended, target)
        return total


def brute_force_surprisal(lm, doc: Document) -> list[float]:
    probabilities = StringProbabilities(lm)
    values, previous = [], 1.0
    for _, end in doc.regions:
        current = probabilities.event(doc.text[:end])
        values.append(-math.log2(current / previous))
        previous = current
    return values


@pytest.mark.parametrize("style", ["eos", "bos"])
@pytest.mark.parametrize("words", [["ab", "ba", "b"], ["a"], ["b", "a", "ab", "a"]])
def test_corrected_surprisal_matches_brute_force(lms, style, words):
    lm = lms(style)
    doc = document(words)
    expected = brute_force_surprisal(lm, doc)
    # Both sides use float32 log-probabilities from the same model; 1e-4 bits
    # is far above float32 rounding of sums of a few terms and far below any
    # error the correction could introduce.
    np.testing.assert_allclose(surprisal(lm, doc), expected, atol=1e-4)


def test_correction_changes_values_and_none_sums_token_surprisal(lms):
    lm = lms("eos")
    doc = document(["ab", "ba", "b", "aa"])
    (result,) = scores(lm, [doc])
    uncorrected = surprisal(lm, doc, correction="none", units="nats")
    np.testing.assert_allclose(uncorrected.sum(), -result["logprob"].astype(float).sum())
    corrected = surprisal(lm, doc, units="nats")
    assert not np.allclose(corrected, uncorrected)
    # The corrections telescope: their sum is the boundary term of the last region.
    np.testing.assert_allclose(
        corrected.sum() - uncorrected.sum(), -float(result["boundary_logprob"][-1]), rtol=1e-6
    )


def test_units(lms):
    lm = lms("eos")
    doc = document(["ab", "b"])
    np.testing.assert_allclose(
        surprisal(lm, doc, units="nats"), surprisal(lm, doc, units="bits") * LN2
    )


def test_first_region_has_no_denominator(lms):
    lm = lms("eos")
    doc = document(["ab", "b"])
    (result,) = scores(lm, [doc])
    expected = -(result["logprob"][:2].astype(float).sum() + result["boundary_logprob"][1])
    assert surprisal(lm, doc, units="nats")[0] == pytest.approx(expected)


# Token scores


def test_token_scores_have_spans_and_contexts(lms):
    lm = lms("eos")
    doc = document(["ab", "ba", "b"], sentences=[2, 1])
    (result,) = scores(lm, [doc])
    assert result["token_id"].tolist() == [1, 2, 4, 1, 4]
    assert list(zip(result["char_start"], result["char_end"], strict=True)) == [
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 5),
        (5, 7),
    ]
    assert result["context"].tolist() == [0] * 5
    (by_sentence,) = scores(lm, [doc], context="sentence")
    assert by_sentence["context"].tolist() == [0, 0, 0, 0, 1]
    # The second sentence starts afresh, without a leading space.
    assert by_sentence["token_id"].tolist() == [1, 2, 4, 1, 2]


def test_token_logprob_matches_direct_forward_pass(lms):
    lm = lms("eos")
    doc = document(["ab", "ba", "b"])
    (result,) = scores(lm, [doc])
    ids = torch.tensor([[0, 1, 2, 4, 1, 4]])
    with torch.inference_mode():
        logprobs = torch.log_softmax(lm.model(input_ids=ids).logits[0].float(), -1)
    expected = [logprobs[p, ids[0, p + 1]].item() for p in range(5)]
    np.testing.assert_allclose(result["logprob"], expected, rtol=1e-6)
    boundary = torch.logsumexp(logprobs[:, [0, 3, 4]], -1)[1:]
    np.testing.assert_allclose(result["boundary_logprob"], boundary.numpy(), rtol=1e-6)


def test_document_start_none_leaves_first_token_unscored(lms):
    lm = lms("eos", "none")
    assert lm.start_ids == []
    doc = document(["ab", "b"])
    (result,) = scores(lm, [doc], document_start="none")
    assert np.isnan(result["logprob"][0]) and not np.isnan(result["logprob"][1:]).any()
    values = surprisal(lm, doc, document_start="none")
    assert np.isnan(values[0]) and not np.isnan(values[1])


def test_one_sentence_document_scores_the_same_by_sentence(lms):
    lm = lms("eos")
    doc = document(["ab", "ba", "b"], sentences=[3])
    (by_document,) = scores(lm, [doc])
    (by_sentence,) = scores(lm, [doc], context="sentence")
    for key in by_document:
        np.testing.assert_array_equal(by_document[key], by_sentence[key])


def test_sentence_context_restarts_each_sentence(lms):
    lm = lms("eos")
    two = document(["ab", "ba", "b", "a"], sentences=[2, 2])
    alone = document(["b", "a"])
    expected = surprisal(lm, alone)
    np.testing.assert_allclose(surprisal(lm, two, context="sentence")[2:], expected, rtol=1e-5)


def test_batches_match_single_documents(lms):
    lm = lms("eos")
    docs = [document(["ab", "ba", "b"]), document(["a"]), document(["b", "b", "ab", "a", "a"])]
    together = scores(lm, docs)
    for doc, result in zip(docs, together, strict=True):
        (alone,) = scores(lm, [doc])
        for key in alone:
            np.testing.assert_allclose(result[key], alone[key], rtol=1e-5)


# Sliding windows


@pytest.mark.parametrize(
    ("length", "window", "stride"), [(1, 4, 2), (4, 4, 2), (5, 4, 2), (11, 4, 3), (12, 5, 1)]
)
def test_windows_keep_each_position_once_with_enough_context(length, window, stride):
    spans = windows(length, window, stride)
    kept = [p for start, stop, own in spans for p in range(own, stop)]
    assert kept == list(range(length))
    for start, stop, own in spans:
        assert stop - start <= window
        assert start == 0 or own - start >= window - stride


def test_sliding_window_equals_full_context_when_it_fits(lms):
    lm = lms("eos")
    doc = document(["ab", "ba", "b", "a"])
    (full,) = scores(lm, [doc])
    (windowed,) = scores(lm, [doc], window=16, stride=3)
    for key in full:
        np.testing.assert_array_equal(full[key], windowed[key])


def test_sliding_window_scores_come_from_their_window(lms):
    lm = lms("eos")
    sequence = [0, *([1, 2, 4, 3] * 5)]
    window, stride = 6, 4
    ((next_logprob, boundary),) = score_sequences(lm, [sequence], window, stride)
    for start, stop, own in windows(len(sequence), window, stride):
        ids = torch.tensor([sequence[start:stop]])
        with torch.inference_mode():
            logprobs = torch.log_softmax(lm.model(input_ids=ids).logits[0].float(), -1)
        for p in range(own, stop):
            if p + 1 < len(sequence):
                expected = logprobs[p - start, sequence[p + 1]].item()
                assert next_logprob[p] == pytest.approx(expected, rel=1e-5)
            expected = torch.logsumexp(logprobs[p - start, [0, 3, 4]], -1).item()
            assert boundary[p] == pytest.approx(expected, rel=1e-5)
    assert np.isnan(next_logprob[-1]) and not np.isnan(next_logprob[:-1]).any()


def test_window_longer_than_model_is_rejected(lms):
    with pytest.raises(TxturesError, match="exceeds the maximum context length"):
        scores(lms("eos"), [document(["a"])], window=32)


def test_long_document_uses_sliding_window_by_default(lms):
    lm = lms("eos")
    doc = document(["ab", "ba"] * 10)
    (result,) = scores(lm, [doc])
    assert len(result["token_id"]) == 40
    assert not np.isnan(result["logprob"]).any()


# Resource resolution


def test_local_model_resolves_to_content_digest(toy_models):
    path = str(toy_models["eos"])
    spec = {"model": path, "revision": None, "dtype": "float32", "document_start": "auto"}
    resolved = HFModel(spec).resolve(RunContext())
    assert resolved == {
        "model": path,
        "revision": "sha256:" + directory_digest(toy_models["eos"]),
        "dtype": "float32",
        "document_start": "eos",
    }


@pytest.mark.parametrize(
    ("style", "choice", "ids"),
    [("eos", "eos", [0]), ("bos", "bos", [5]), ("no_bos", "eos", [0]), ("quiet_bos", "bos", [5])],
)
def test_document_start_auto_follows_the_tokenizer(lms, style, choice, ids):
    assert lms(style).document_start == choice
    assert lms(style).start_ids == ids


def test_document_start_without_special_tokens_is_refused_unless_none(toy_models, lms):
    spec = {
        "model": str(toy_models["bare"]),
        "revision": None,
        "dtype": "float32",
        "document_start": "auto",
    }
    with pytest.raises(TxturesError, match="declares neither a BOS nor an EOS"):
        HFModel(spec).resolve(RunContext())
    with pytest.raises(TxturesError, match="no BOS token"):
        HFModel({**spec, "document_start": "bos"}).resolve(RunContext())
    assert lms("bare", "none").start_ids == []


def test_revision_cannot_be_set_for_local_model(toy_models):
    spec = {
        "model": str(toy_models["eos"]),
        "revision": "main",
        "dtype": "float32",
        "document_start": "auto",
    }
    with pytest.raises(TxturesError, match="local model directory"):
        HFModel(spec).resolve(RunContext())


def test_boundary_set(lms):
    assert lms("eos").boundary.tolist() == [True, False, False, True, True]
    assert lms("bos").boundary.tolist() == [True, False, False, True, True, False]


# Region assignment


def test_token_spanning_regions_is_rejected():
    doc = Document("d", "ab b", ((0, 1), (1, 2), (3, 4)))
    tokens = {
        "char_start": np.array([0, 2]),
        "char_end": np.array([2, 4]),
        "logprob": np.zeros(2),
        "boundary_logprob": np.zeros(2),
        "context": np.zeros(2, dtype=np.int32),
    }
    with pytest.raises(TxturesError, match="spans more than one region"):
        region_surprisal(doc, tokens, "none")


def test_whitespace_tokens_join_the_following_region():
    # Tokens: "a", " ", "b" — the lone space belongs to region "b".
    doc = Document("d", "a b", ((0, 1), (2, 3)))
    tokens = {
        "char_start": np.array([0, 1, 2]),
        "char_end": np.array([1, 2, 3]),
        "logprob": np.log(np.array([0.5, 0.25, 0.125])),
        "boundary_logprob": np.log(np.array([0.5, 0.5, 0.5])),
        "context": np.zeros(3, dtype=np.int32),
    }
    values = region_surprisal(doc, tokens, "none") / LN2
    np.testing.assert_allclose(values, [1.0, 5.0])


def test_undetected_boundaries_refuse_correction():
    doc = Document("d", "a", ((0, 1),))
    tokens = {
        "char_start": np.array([0]),
        "char_end": np.array([1]),
        "logprob": np.zeros(1),
        "boundary_logprob": np.full(1, np.nan),
        "context": np.zeros(1, dtype=np.int32),
    }
    with pytest.raises(TxturesError, match="correction: none"):
        region_surprisal(doc, tokens, "word_boundary")
    assert region_surprisal(doc, tokens, "none").tolist() == [0.0]


# Through the engine


@pytest.fixture
def env(tmp_path: Path, toy_models):
    locations = {
        key: str(tmp_path / key)
        for key in ("definitions", "inputs", "outputs", "resources", "cache")
    }
    write(
        tmp_path / "definitions" / "configs" / "lm_scores" / "toy.yml",
        f"extends: main\nmodel: {toy_models['eos']}\n",
    )
    write(
        tmp_path / "definitions" / "configs" / "lm_scores" / "bare.yml",
        f"extends: main\nmodel: {toy_models['bare']}\n",
    )
    write(tmp_path / "definitions" / "workflows" / "toy.yml", "lm_scores: toy\nsurprisal: main\n")
    write(
        tmp_path / "definitions" / "workflows" / "bare.yml",
        "lm_scores: bare\nsurprisal: main\n",
    )
    write(
        tmp_path / "definitions" / "workflows" / "units.yml",
        "lm_scores: toy\nsurprisal: units\n",
    )
    write(
        tmp_path / "definitions" / "configs" / "surprisal" / "units.yml",
        "extends: main\nunits: [bits, nats]\n",
    )
    write(tmp_path / "inputs" / "doc.tsv", "document_id\ttext\nx\tab ba b\ny\ta\n")
    return resolve_settings(
        {**locations, "execution.device": "cpu"},
        env={},
        user_home=tmp_path / "home",
    )


def run(settings, workflow):
    from txtures.store import EntryStore

    definitions = Definitions.load(settings.definitions, REGISTRY)
    plan = plan_request(settings, definitions, "doc", workflow)
    executor = Executor(
        plan, EntryStore(settings.cache), EntryStore(settings.resource_store), RunContext("cpu")
    )
    return plan, executor, run_request(plan, executor, None)


def test_surprisal_table_from_a_local_model(env, lms):
    plan, executor, (directory, written) = run(env, "toy")
    assert written
    rows = [line.split("\t") for line in (directory / "doc.tsv").read_text().splitlines()]
    assert rows[0] == ["document_id", "region_index", "text", "char_start", "char_end", "surprisal"]
    assert [row[2] for row in rows[1:]] == ["ab", "ba", "b", "a"]
    expected = [
        *surprisal(lms("eos"), document(["ab", "ba", "b"])),
        *surprisal(lms("eos"), document(["a"])),
    ]
    np.testing.assert_allclose([float(row[5]) for row in rows[1:]], expected, rtol=1e-6)
    (lm_variant,) = [p for p in plan.variants if p.name == "lm_scores"]
    assert executor.counts[lm_variant.identity].computed == 2
    _, executor, (_, written) = run(env, "toy")
    assert not written


def test_swept_units_reuse_one_lm_scores_variant(env):
    plan, executor, (directory, _) = run(env, "units")
    header = (directory / "doc.tsv").read_text().splitlines()[0].split("\t")
    assert header[-2:] == ["surprisal_bits", "surprisal_nats"]
    assert sum(1 for p in plan.variants if p.name == "lm_scores") == 1


def test_model_without_special_tokens_is_a_planning_problem(env):
    with pytest.raises(DefinitionError, match="declares neither"):
        run(env, "bare")
