"""Tests of the n-gram tokenizer, corpus, unigram table, and surprisal modules."""

import gzip
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from txtures.errors import TxturesError
from txtures.inputs import InputSettings, read_texts
from txtures.modules import Item, RunContext, ngram
from txtures.modules.ngram import (
    NgramCorpus,
    NgramSurprisal,
    NgramSurprisalConfig,
    UnigramLookup,
    UnigramSurprisal,
    UnigramSurprisalConfig,
    contexts,
    parse_query,
    region_scores,
    write_table,
)
from txtures.modules.sentences import sentences_result
from txtures.ngram_text import sentences, tokenize


def upstream(doc, spans):
    return {"sentences": sentences_result(spans)}


def document(text, sentence_delimiter=None):
    settings = InputSettings(sentence_delimiter=sentence_delimiter)
    (doc,) = read_texts(text, settings).corpus.documents
    return doc


# Tokens and sentences


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Don’t stop, Mr. Smith!", ["do", "n't", "stop", ",", "mr", ".", "smith", "!"]),
        ("It cost $3,500.75 in 2020.", ["it", "cost", "$", "3,500.75", "in", "2020", "."]),
        ("They're well-known.", ["they", "'re", "well-known", "."]),
        ("I'd've", ["i", "'d", "'ve"]),
        ("can't", ["ca", "n't"]),
        ("Café", ["café"]),
    ],
)
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


def test_tokenize_keeps_case_when_asked():
    assert tokenize("New York", lowercase=False) == ["New", "York"]


def test_sentences_respect_abbreviations_initials_and_lines():
    text = 'Mr. Smith saw J. Doe at 5 p.m. today. "Wow!" Then 3 dogs ran.\nA new line'
    assert sentences(text) == [
        "Mr. Smith saw J. Doe at 5 p.m. today.",
        '"Wow!"',
        "Then 3 dogs ran.",
        "A new line",
    ]


# Contexts and region sums


def test_contexts_follow_the_given_sentences():
    doc = document("The cat sat. It ran away.")
    assert contexts(doc, lowercase=True, spans=[(0, 3), (3, 6)]) == [
        (["the", "cat", "sat", "."], [0, 1, 2, 2]),
        (["it", "ran", "away", "."], [3, 4, 5, 5]),
    ]
    assert contexts(doc, lowercase=True) == [
        (["the", "cat", "sat", ".", "it", "ran", "away", "."], [0, 1, 2, 2, 3, 4, 5, 5])
    ]


def test_region_scores_sum_tokens_and_count_unknown_ones():
    doc = document("don't go")
    found = contexts(doc, lowercase=True)
    scores = [[(-1.0, False), (-0.5, True), (-2.0, False)]]
    result = region_scores(doc, found, scores)
    assert result["nats"] == pytest.approx([1.5 * math.log(10), 2.0 * math.log(10)])
    assert result["oov"].tolist() == [1, 0]


def test_parse_query_reads_kenlm_output():
    output = (
        "This binary file contains trie.\n"
        "the=9 2 -0.34827152\tcat=7 2 -0.5466951\t</s>=5 1 -1.0009534\t"
        "zebra=0 1 -1.6812413\t</s>=5 1 -1.0009534\t"
    )
    assert parse_query(output, [["the", "cat"], ["zebra"]]) == [
        [(-0.34827152, False), (-0.5466951, False)],
        [(-1.6812413, True)],
    ]
    with pytest.raises(TxturesError, match="does not match"):
        parse_query(output, [["the", "dog"], ["zebra"]])


# The unigram table


ARPA = """\\data\\
ngram 1=5

\\1-grams:
-1.5\t<unk>
-99\t<s>
-0.5\tthe
-1.0\tcat
-0.75\t</s>

\\end\\
"""


def test_unigram_table_looks_up_words_and_unknowns(tmp_path):
    (tmp_path / "u.arpa").write_text(ARPA)
    properties = write_table(tmp_path / "u.arpa", tmp_path)
    assert properties == {"vocabulary": 3, "unknown_logprob": -1.5}
    table = UnigramLookup(tmp_path)
    assert table.score("the") == (-0.5, False)
    assert table.score("cat") == (-1.0, False)
    assert table.score("zebra") == (-1.5, True)
    assert table.score("<s>") == (-1.5, True)


# The corpus


def test_local_corpus_resolves_to_a_digest_of_its_files(tmp_path):
    (tmp_path / "a.txt").write_text("One doc.\n")
    spec = {
        "dataset": str(tmp_path),
        "revision": None,
        "files": "*.txt",
        "text_field": "text",
        "max_documents": None,
        "lowercase": True,
        "text_version": 1,
    }
    first = NgramCorpus(spec).resolve(RunContext())
    assert first["file_list"] == ["a.txt"] and first["revision"].startswith("sha256:")
    (tmp_path / "a.txt").write_text("Another doc.\n")
    assert NgramCorpus(spec).resolve(RunContext())["revision"] != first["revision"]
    with pytest.raises(TxturesError, match="no files"):
        NgramCorpus({**spec, "files": "*.parquet"}).resolve(RunContext())


@pytest.mark.parametrize("cores", [1, 2])
def test_corpus_build_tokenizes_sentences_and_limits_documents(tmp_path, monkeypatch, cores):
    monkeypatch.setattr(ngram, "_cores", lambda: cores)
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(pa.table({"text": ["The cat sat. It ran.", "A dog!"]}), source / "a.parquet")
    pq.write_table(pa.table({"text": ["Third one.", "Fourth."]}), source / "b.parquet")
    spec = {
        "dataset": str(source),
        "revision": None,
        "files": "*.parquet",
        "text_field": "text",
        "max_documents": 3,
        "lowercase": True,
        "text_version": 1,
    }
    resource = NgramCorpus(spec)
    resolved = resource.resolve(RunContext())
    built = tmp_path / "built"
    built.mkdir()
    resource.build(resolved, built, RunContext())
    with gzip.open(built / ngram.CORPUS_FILE, "rt") as handle:
        lines = handle.read().splitlines()
    assert lines == ["the cat sat .", "it ran .", "a dog !", "third one ."]
    properties = (built / "properties.json").read_text()
    assert '"documents": 3' in properties and '"tokens": 13' in properties
    assert not (built / "parts").exists()


def test_document_corpus_puts_each_document_on_one_line(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("The cat sat. It ran.\nA dog!\n")
    spec = {
        "dataset": str(source),
        "revision": None,
        "files": "*.txt",
        "text_field": "text",
        "max_documents": None,
        "lowercase": True,
        "text_version": 1,
        "context": "document",
    }
    resource = NgramCorpus(spec)
    built = tmp_path / "built"
    built.mkdir()
    resource.build(resource.resolve(RunContext()), built, RunContext())
    with gzip.open(built / ngram.CORPUS_FILE, "rt") as handle:
        assert handle.read().splitlines() == ["the cat sat . it ran .", "a dog !"]


def test_ngram_context_selects_the_corpus_and_the_scored_contexts():
    sentence = NgramSurprisal().resources(NgramSurprisalConfig())["model"].spec["corpus"]
    assert "context" not in sentence
    config = NgramSurprisalConfig(context="document")
    assert NgramSurprisal().resources(config)["model"].spec["corpus"]["context"] == "document"
    model = FakeModel()
    doc = document("The cat sat. The dog sat.")
    item = Item(doc, upstream(doc, [(0, 3), (3, 6)]), {"model": model})
    NgramSurprisal().run_batch(config, [item], RunContext())
    assert model.calls == [[["the", "cat", "sat", ".", "the", "dog", "sat", "."]]]


# Configs and modules


def test_prune_thresholds_are_checked_and_cut_to_the_order():
    with pytest.raises(ValueError, match="cannot drop unigrams"):
        NgramSurprisalConfig(prune=(1, 1))
    with pytest.raises(ValueError, match="must not decrease"):
        NgramSurprisalConfig(prune=(0, 2, 1))
    model = NgramSurprisal().resources(NgramSurprisalConfig(order=2))["model"]
    assert model.spec["prune"] == [0, 0]
    assert NgramSurprisal().resources(NgramSurprisalConfig())["model"].spec["prune"] == [0, 0, 1]


class FakeModel:
    """Scores every token by a fixed table; unknown tokens get -3."""

    def __init__(self):
        self.calls = []

    def score(self, sentences_):
        self.calls.append(sentences_)
        table = {"the": -0.2, "cat": -1.0, "sat": -0.7, ".": -0.1}
        return [[(table.get(t, -3.0), t not in table) for t in tokens] for tokens in sentences_]


def test_ngram_module_scores_every_sentence_of_a_batch_in_one_call():
    model = FakeModel()
    docs = [document("The cat sat. The dog sat."), document("A cat.")]
    spans = [[(0, 3), (3, 6)], [(0, 2)]]
    items = [Item(d, upstream(d, s), {"model": model}) for d, s in zip(docs, spans, strict=True)]
    results = NgramSurprisal().run_batch(NgramSurprisalConfig(), items, RunContext())
    assert len(model.calls) == 1 and len(model.calls[0]) == 3
    first = NgramSurprisal().columns(NgramSurprisalConfig(), docs[0], results[0])
    assert first["ngram_surprisal"] == pytest.approx(
        np.array([0.2, 1.0, 0.8, 0.2, 3.0, 0.8]) * math.log2(10)
    )
    assert first["ngram_oov"].tolist() == [0, 0, 0, 0, 1, 0]


def test_unigram_module_reads_the_table(tmp_path):
    (tmp_path / "u.arpa").write_text(ARPA)
    write_table(tmp_path / "u.arpa", tmp_path)
    doc = document("The cat. The zebra")
    item = Item(doc, {}, {"table": UnigramLookup(tmp_path)})
    config = UnigramSurprisalConfig(units="nats")
    columns = UnigramSurprisal().columns(config, doc, UnigramSurprisal().run(config, item, None))
    ln10 = math.log(10)
    # `cat.` holds `cat` and `.`, which this table does not know.
    expected = [0.5 * ln10, (1.0 + 1.5) * ln10, 0.5 * ln10, 1.5 * ln10]
    assert columns["unigram_surprisal"] == pytest.approx(expected)
    assert columns["unigram_oov"].tolist() == [0, 1, 0, 1]


def test_columns_are_labeled_by_the_corpus():
    from txtures.definitions import Definitions

    expansion = Definitions.packaged().expand("main", ["ngram_surprisal", "unigram_surprisal"])
    assert [c.name for c in expansion.columns()] == [
        "ngram_surprisal_openwebtext",
        "ngram_oov_openwebtext",
        "unigram_surprisal_openwebtext",
        "unigram_oov_openwebtext",
    ]


# Against KenLM's models, with the tools image


TINY_CORPUS = Path(__file__).with_name("data") / "tiny_corpus"
"""A small, fixed corpus. A local corpus is identified by its path and files, so a
fixed one is built into the resource store once, not on every run."""


def tiny_corpus():
    return {"dataset": str(TINY_CORPUS), "files": "*.txt", "discount_fallback": True}


def read_arpa(path):
    """Unigram log10 probabilities and backoffs, and bigram log10 probabilities."""
    unigrams, backoffs, bigrams, order = {}, {}, {}, 0
    for line in path.read_text().splitlines():
        if line.startswith("\\") and line.endswith("-grams:"):
            order = int(line[1])
            continue
        fields = line.split("\t")
        if order == 1 and len(fields) >= 2:
            unigrams[fields[1]] = float(fields[0])
            backoffs[fields[1]] = float(fields[2]) if len(fields) > 2 else 0.0
        elif order == 2 and len(fields) >= 2:
            bigrams[tuple(fields[1].split(" "))] = float(fields[0])
    return unigrams, backoffs, bigrams


def reference_bigram(arpa, tokens):
    """log10 p of each token after `<s>` by the ARPA backoff rule, without KenLM's code."""
    unigrams, backoffs, bigrams = arpa
    values, previous = [], "<s>"
    for token in tokens:
        word = token if token in unigrams else "<unk>"
        if (previous, word) in bigrams:
            values.append(bigrams[(previous, word)])
        else:
            values.append(backoffs.get(previous, 0.0) + unigrams[word])
        previous = word
    return values


@pytest.mark.heavy
@pytest.mark.tools
def test_kenlm_bigram_surprisal_matches_the_arpa_model(tmp_path):
    import dataclasses

    from txtures.engine.execution import build_resource, load_resource
    from txtures.engine.planning import plan_resource
    from txtures.environment import run_context
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    settings = resolve_settings({})
    context = dataclasses.replace(run_context(settings), device="cpu")
    store = EntryStore(settings.resource_store)
    config = NgramSurprisalConfig(order=2, **tiny_corpus())
    planned = plan_resource("model", NgramSurprisal().resources(config)["model"], context)
    build_resource(store, planned, context, lambda message: None)
    model = load_resource(store, planned, context)

    tools = load_resource(store, planned.dependencies["tools"], context)
    corpus = load_resource(store, planned.dependencies["corpus"], context)
    arpa = tmp_path / "reference.arpa"
    result = tools.run(
        ["lmplz", "-o", "2", "-S", "100M", "-T", str(tmp_path), "--discount_fallback",
         "--prune", "0", "0", "--text", str(corpus), "--arpa", str(arpa)],
        binds=[tmp_path, corpus.parent], capture_output=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    reference = read_arpa(arpa)

    doc = document("The cat saw the zebra. The horse raced past the barn.")
    spans = [(0, 5), (5, 11)]
    item = Item(doc, upstream(doc, spans), {"model": model})
    (scored,) = NgramSurprisal().run_batch(config, [item], context)
    expected = np.zeros(len(doc.regions))
    for tokens, regions in contexts(doc, lowercase=True, spans=spans):
        for region, value in zip(regions, reference_bigram(reference, tokens), strict=True):
            expected[region] += -value * math.log(10)
    assert scored["nats"] == pytest.approx(expected, rel=1e-5)
    assert scored["oov"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


@pytest.mark.heavy
@pytest.mark.tools
def test_unigram_table_matches_the_order_one_arpa_model(tmp_path):
    import dataclasses

    from txtures.engine.execution import build_resource, load_resource
    from txtures.engine.planning import plan_resource
    from txtures.environment import run_context
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    settings = resolve_settings({})
    context = dataclasses.replace(run_context(settings), device="cpu")
    store = EntryStore(settings.resource_store)
    config = UnigramSurprisalConfig(**tiny_corpus())
    planned = plan_resource("table", UnigramSurprisal().resources(config)["table"], context)
    build_resource(store, planned, context, lambda message: None)
    table = load_resource(store, planned, context)

    tools = load_resource(store, planned.dependencies["tools"], context)
    corpus = load_resource(store, planned.dependencies["corpus"], context)
    arpa = tmp_path / "reference.arpa"
    result = tools.run(
        ["lmplz", "-o", "1", "-S", "100M", "-T", str(tmp_path), "--discount_fallback",
         "--text", str(corpus), "--arpa", str(arpa)],
        binds=[tmp_path, corpus.parent], capture_output=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    unigrams, _, _ = read_arpa(arpa)
    for word, logprob in unigrams.items():
        if word not in ("<s>", "<unk>"):
            assert table.score(word) == (pytest.approx(logprob), False)
    assert table.score("zebra") == (pytest.approx(unigrams["<unk>"]), True)
