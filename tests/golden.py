"""Golden outputs: fixed inputs and the values txtures computes from them.

Each target is something whose results are versioned: a module's output
contract (`contract_version`), a resource (`version`), or the input reader
(`READER_VERSION`). `tests/golden/TARGET.json` records the version and the
values computed under it. `test_golden.py` recomputes every target and
fails when

- the values differ while the version is unchanged: a change altered
  computed results, so increment the version in the same change; or
- the version differs from the recorded one: regenerate the golden values;
- the external inputs differ from the recorded ones, such as the Stanza
  models a new Stanza release resolves to: regenerate the golden values.

Regenerate with

    txtures-install/env/bin/python tests/golden.py --update [TARGET ...]

which refuses to record changed values under an unchanged version, so the
golden files cannot be updated around a missing version bump.

Most targets run without downloads. The toy language model uses
`weights="formula"`, so its weights do not depend on PyTorch's random
number generator. The `parse` and `stanza_pipeline` targets use Stanza's
English models, built on demand into the configured resource store
(`resources/store` in the txtures home), so they download only the first
time on a site. Targets in `HEAVY` load neural models and are skipped by
`pytest --fast`.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GOLDEN = Path(__file__).with_name("golden")
RTOL = 1e-5
"""Relative tolerance for floats. Language-model scores are float32, and a
different CPU or BLAS can reorder sums, which moves values by about 1e-7
relative; a real change to a computation moves them far more."""
ATOL = 1e-6


@dataclass(frozen=True)
class Target:
    """One versioned thing: how to read its version and compute its values."""

    version: Callable[[], int]
    compute: Callable[[], Any]
    bump: str
    """Where the version lives, for messages."""
    inputs: Callable[[], Any] | None = None
    """What the values depend on outside txtures' versions, such as resolved
    model revisions; a change asks for regeneration, not a version bump."""


# Values


def plain(value: Any) -> Any:
    """Convert computed values to JSON: arrays to lists, NaN to null."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def differences(expected: Any, actual: Any, where: str = "") -> list[str]:
    """Where `actual` differs from `expected`, with floats compared within the tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                found.append(f"{where}/{key}: present in only one")
            else:
                found += differences(expected[key], actual[key], f"{where}/{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(expected)} != {len(actual)}"]
        found = []
        for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
            found += differences(a, b, f"{where}[{index}]")
        return found
    numbers = (int, float)
    if (
        isinstance(expected, numbers)
        and isinstance(actual, numbers)
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        if math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
            return []
        return [f"{where}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{where}: {expected!r} != {actual!r}"]


# Inputs


def toy_model(directory: Path, style: str = "eos") -> Path:
    import toy_lm

    return toy_lm.build(directory / style, style, weights="formula")


def loaded_model(directory: Path, style: str = "eos"):
    from txtures.modules import RunContext
    from txtures.modules.hf_model import HFModel

    resource = HFModel(
        {
            "model": str(toy_model(directory, style)),
            "revision": None,
            "dtype": "float32",
            "document_start": "auto",
        }
    )
    context = RunContext(device="cpu")
    resolved = resource.resolve(context)
    built = directory / f"built-{style}"
    built.mkdir()
    resource.build(resolved, built, context)
    return resource, resolved, built, resource.load(resolved, built, context)


def toy_documents():
    from test_lm import document

    return {
        "three_words": document(["ab", "ba", "b"]),
        "two_sentences": document(["a", "ab", "b", "ba", "a"], sentences=[2, 3]),
        "long": document(["ab", "ba", "bab", "a", "b", "ab", "ba"]),
    }


def parsed(text: str, rows: str):
    """A one-sentence parse from rows of `form upos head deprel [feats [ner]]`."""
    from test_parse import document

    from txtures.modules.parse import ParsedWord, align

    doc = document(text)
    words, position = [], 0
    for token, line in enumerate(rows.strip().splitlines()):
        form, upos, head, deprel, *rest = line.split()
        feats = rest[0] if rest else "_"
        ner = rest[1] if len(rest) > 1 else "O"
        start = doc.text.index(form, position)
        position = start + len(form)
        words.append(
            ParsedWord(start, position, token, form, form.lower(), upos, upos, feats, int(head),
                       deprel, ner)
        )  # fmt: skip
    return doc, align(doc, [(0, len(doc.regions))], [words])


FIN = "VerbForm=Fin"
PARSES = {
    "object_relative": (
        "The reporter who the senator attacked admitted the error .",
        f"""
        The DET 2 det
        reporter NOUN 7 nsubj
        who PRON 6 obj PronType=Rel
        the DET 5 det
        senator NOUN 6 nsubj
        attacked VERB 2 acl:relcl {FIN}
        admitted VERB 0 root {FIN}
        the DET 9 det
        error NOUN 7 obj
        . PUNCT 7 punct
        """,
    ),
    "coordination": (
        "The police arrested and questioned the student who had stolen the car and the bike .",
        f"""
        The DET 2 det
        police NOUN 3 nsubj
        arrested VERB 0 root {FIN}
        and CCONJ 5 cc
        questioned VERB 3 conj {FIN}
        the DET 7 det
        student NOUN 3 obj
        who PRON 10 nsubj PronType=Rel
        had AUX 10 aux {FIN}
        stolen VERB 7 acl:relcl VerbForm=Part
        the DET 12 det
        car NOUN 10 obj
        and CCONJ 15 cc
        the DET 15 det
        bike NOUN 12 conj
        . PUNCT 3 punct
        """,
    ),
    "pied_piping": (
        "I don't think the house in which I live is in New York .",
        f"""
        I PRON 4 nsubj _
        do AUX 4 aux {FIN}
        n't PART 4 advmod
        think VERB 0 root VerbForm=Inf
        the DET 6 det
        house NOUN 14 nsubj
        in ADP 8 case
        which PRON 10 obl PronType=Rel
        I PRON 10 nsubj
        live VERB 6 acl:relcl {FIN}
        is AUX 14 cop {FIN}
        in ADP 14 case
        New PROPN 14 compound _ B-GPE
        York PROPN 4 ccomp _ E-GPE
        . PUNCT 4 punct
        """,
    ),
    "names_and_dates": (
        "Quickly , Mary Smith left on Monday .",
        f"""
        Quickly ADV 5 advmod
        , PUNCT 5 punct
        Mary PROPN 5 nsubj _ B-PERSON
        Smith PROPN 3 flat _ E-PERSON
        left VERB 0 root {FIN}
        on ADP 7 case
        Monday PROPN 5 obl _ S-DATE
        . PUNCT 5 punct
        """,
    ),
}


def parses():
    return {name: parsed(text, rows) for name, (text, rows) in PARSES.items()}


# Targets


def reader_values() -> dict:
    from txtures.inputs import load_input

    files = {
        "lines.txt": "The horse raced past the barn fell.\nCafé au  lait.\n",
        "regions.tsv": (
            "document_id\tsentence_id\tregion_id\ttext\tcondition\n"
            "d1\t1\t1\tThe\ta\nd1\t1\t2\thorse.\ta\nd1\t2\t1\tIt\tb\nd1\t2\t2\tfell.\tb\n"
        ),
        "delimited.txt": "One | two <s> three | four\n",
        "delimited.json": json.dumps({"region_delimiter": "|", "sentence_delimiter": "<s>"}),
    }
    values = {}
    with tempfile.TemporaryDirectory() as root:
        for name, text in files.items():
            (Path(root) / name).write_text(text, encoding="utf-8")
        for name in ("lines", "regions", "delimited"):
            corpus = load_input(Path(root), name).corpus
            values[name] = {
                "layout": corpus.layout,
                "documents": [
                    {
                        "id": d.document_id,
                        "text": d.text,
                        "regions": d.regions,
                        "sentences": [[s.id, s.start, s.stop] for s in d.sentences],
                        "region_ids": d.region_ids,
                        "metadata": [dict(m) for m in d.region_metadata],
                        "digest": d.digest,
                    }
                    for d in corpus.documents
                ],
            }
    return values


def hf_model_values() -> dict:
    values = {}
    with tempfile.TemporaryDirectory() as root:
        for style in ("eos", "bos", "no_bos"):
            _, resolved, built, _ = loaded_model(Path(root) / style, style)
            properties = json.loads((built / "properties.json").read_text())
            for key in ("transformers_version", "tokenizers_version"):
                properties.pop(key, None)
            values[style] = {
                "document_start": resolved["document_start"],
                "properties": properties,
                "boundary": np.load(built / "boundary.npy").astype(int),
            }
    return values


def lm_scores_values() -> dict:
    from txtures.modules import Item, RunContext
    from txtures.modules.lm_scores import LMScores, LMScoresConfig

    configs = {
        "document": {},
        "sentence": {"context": "sentence"},
        "window": {"window": 4, "stride": 2},
    }
    values = {}
    with tempfile.TemporaryDirectory() as root:
        *_, lm = loaded_model(Path(root))
        for config_name, config in configs.items():
            for doc_name, doc in toy_documents().items():
                if config_name == "sentence" and not doc.sentences:
                    continue
                (result,) = LMScores().run_batch(
                    LMScoresConfig(**config), [Item(doc, {}, {"model": lm})], RunContext("cpu")
                )
                values[f"{config_name}/{doc_name}"] = dict(result)
    return values


def surprisal_values() -> dict:
    from txtures.modules import Item, RunContext
    from txtures.modules.lm_scores import LMScores, LMScoresConfig
    from txtures.modules.surprisal import Surprisal, SurprisalConfig

    values = {}
    with tempfile.TemporaryDirectory() as root:
        *_, lm = loaded_model(Path(root))
        for doc_name, doc in toy_documents().items():
            (scores,) = LMScores().run_batch(
                LMScoresConfig(), [Item(doc, {}, {"model": lm})], RunContext("cpu")
            )
            for correction in ("word_boundary", "none"):
                for units in ("bits", "nats"):
                    config = SurprisalConfig(correction=correction, units=units)
                    item = Item(doc, {"lm_scores": scores}, {})
                    values[f"{doc_name}/{correction}/{units}"] = Surprisal().run(config, item, None)
    return values


def word_length_values() -> dict:
    from txtures.inputs import read_texts
    from txtures.modules.word_length import region_lengths

    texts = {
        "plain": "The horse, it fell.",
        "accents": "Café au lait",
        "regions": ["New York", "is", "big."],
    }  # noqa: E501
    corpus = read_texts(texts).corpus
    return {d.document_id: region_lengths(d) for d in corpus.documents}


def pos_values() -> dict:
    from txtures.modules import Item, RunContext
    from txtures.modules.pos import Pos, PosConfig

    values = {}
    for name, (doc, parse) in parses().items():
        for tagset in ("upos", "xpos"):
            item = Item(doc, {"parse": parse}, {})
            values[f"{name}/{tagset}"] = list(
                Pos().run(PosConfig(tagset=tagset), item, RunContext())
            )
    return values


def named_entity_values() -> dict:
    from txtures.modules import Item, RunContext
    from txtures.modules.named_entity import NamedEntity, NamedEntityConfig

    values = {}
    for name, (doc, parse) in parses().items():
        item = Item(doc, {"parse": parse}, {})
        values[f"{name}/names"] = NamedEntity().run(NamedEntityConfig(), item, RunContext())
        everything = NamedEntityConfig(types=("PERSON", "GPE", "DATE"))
        values[f"{name}/with_dates"] = NamedEntity().run(everything, item, RunContext())
    return values


def dependency_locality_values() -> dict:
    import itertools

    from txtures.modules.dependency_locality import DependencyLocalityConfig, region_costs

    values = {}
    grid = itertools.product(
        (("words", False), ("referents", False), ("referents", True)),
        (False, True),
        (False, True),
        ("gibson", "sud"),
        ("conjunction", "conjunct"),
    )
    for (distance, verbs), coordination, modifiers, relative, head in grid:
        config = DependencyLocalityConfig(
            distance=distance,
            verbs=verbs,
            coordination=coordination,
            modifiers=modifiers,
            relative_clauses=relative,
            coordination_head=head,
        )
        key = f"{distance}/v{int(verbs)}c{int(coordination)}m{int(modifiers)}/{relative}/{head}"
        for name, (doc, parse) in parses().items():
            values[f"{key}/{name}"] = region_costs(doc, parse, config)
    for name, (doc, parse) in parses().items():
        values[f"log/{name}"] = region_costs(doc, parse, DependencyLocalityConfig(transform="log"))
    return values


PARSE_TEXTS = {
    "split_here": "The reporter who the senator attacked admitted the error. I don't live in New York.",
    "given": ["Mr. Smith went to Washington.", "It rained and the river rose."],
}


@functools.cache
def stanza() -> tuple[Any, Any, Any]:
    """Stanza's English models, built into the configured store if missing: the
    planned resource, the loaded models, and the run context (on CPU)."""
    import dataclasses

    from txtures.engine.execution import build_resource
    from txtures.engine.planning import plan_resource
    from txtures.environment import run_context
    from txtures.modules.parse import Parse, ParseConfig
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    settings = resolve_settings({})
    context = dataclasses.replace(run_context(settings), device="cpu")
    resource = Parse().resources(ParseConfig())["stanza"]
    planned = plan_resource("stanza", resource, context)
    store = EntryStore(settings.resource_store)
    build_resource(store, planned, context, lambda message: print(message, file=sys.stderr))
    directory = store.get(planned.kind, planned.identity)
    return planned, resource.load(planned.resolved, directory, context), context


def stanza_inputs() -> dict:
    planned, _, _ = stanza()
    return dict(planned.resolved)


def stanza_pipeline_values() -> dict:
    from txtures.engine.planning import resource_record
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    planned, _, _ = stanza()
    directory = EntryStore(resolve_settings({}).resource_store).get(planned.kind, planned.identity)
    files = json.loads((directory / "files.json").read_text())
    return {"files": files["files"], "kind": resource_record(planned)["kind"]}


def parse_values() -> dict:
    from txtures.inputs import InputSettings, read_texts
    from txtures.modules import Item
    from txtures.modules.parse import Parse, ParseConfig

    _, models, context = stanza()
    values = {}
    for name, text in PARSE_TEXTS.items():
        settings = InputSettings(sentence_delimiter="<s>") if isinstance(text, list) else None
        texts = " <s> ".join(text) if isinstance(text, list) else text
        corpus = read_texts({name: texts}, settings or InputSettings()).corpus
        (document,) = corpus.documents
        (result,) = Parse().run_batch(
            ParseConfig(), [Item(document, {}, {"stanza": models})], context
        )
        values[name] = dict(result)
    return values


NGRAM_TEXTS = [
    "Don\u2019t stop, Mr. Smith! It cost $3,500.75 in 2020 -- really?",
    "They're well-known; I'd've gone. \"Wow.\" Then J. Doe left at 5 p.m. today.",
    "Caf\u0065\u0301 au lait\nA second line",
]
NGRAM_DOCUMENTS = {
    "split_here": "The cat saw the zebra. The horse raced past the barn.",
    "given": "The dog chased the man. <s> A bird liked the barn!",
}


def ngram_text_values() -> dict:
    from txtures.ngram_text import sentences, tokenize

    return {
        text: {
            "tokens": tokenize(text),
            "cased": tokenize(text, lowercase=False),
            "sentences": sentences(text),
        }
        for text in NGRAM_TEXTS
    }


def _corpus_config(**extra):
    corpus = Path(__file__).with_name("data") / "tiny_corpus"
    return {"dataset": str(corpus), "files": "*.txt", "discount_fallback": True, **extra}


def ngram_corpus_values() -> dict:
    import gzip

    from txtures.modules import RunContext
    from txtures.modules.ngram import NgramCorpus, UnigramSurprisalConfig

    resource = NgramCorpus(UnigramSurprisalConfig(**_corpus_config()).corpus_spec())
    resolved = resource.resolve(RunContext())
    with tempfile.TemporaryDirectory() as root:
        resource.build(resolved, Path(root), RunContext())
        with gzip.open(Path(root) / "corpus.txt.gz", "rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        properties = json.loads((Path(root) / "properties.json").read_text())
    return {"first_lines": lines[:20], "lines": len(lines), "properties": properties}


@functools.cache
def _built(module_name: str, **config):
    """Build an n-gram module's resource into the configured store; return it loaded."""
    import dataclasses

    from txtures.engine.execution import build_resource, load_resource
    from txtures.engine.planning import plan_resource
    from txtures.environment import run_context
    from txtures.modules import REGISTRY
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    settings = resolve_settings({})
    context = dataclasses.replace(run_context(settings), device="cpu")
    module = REGISTRY[module_name]()
    parsed = module.Config(**_corpus_config(**config))
    (resource,) = module.resources(parsed).values()
    planned = plan_resource("resource", resource, context)
    store = EntryStore(settings.resource_store)
    build_resource(store, planned, context, lambda message: print(message, file=sys.stderr))
    directory = store.get(planned.kind, planned.identity)
    return module, parsed, load_resource(store, planned, context), directory, context


def tools_inputs() -> dict:
    from txtures.tools import definition_digest

    return {"tools_definition_sha256": definition_digest()}


def _ngram_documents():
    from txtures.inputs import InputSettings, read_texts

    return {
        name: read_texts({name: text}, InputSettings(sentence_delimiter="<s>")).corpus.documents[0]
        for name, text in NGRAM_DOCUMENTS.items()
    }


WORDS = ["the", "cat", "barn", "raced", "past", ".", "zebra", "</s>"]


def unigram_table_values() -> dict:
    _, _, table, directory, _ = _built("unigram_surprisal")
    properties = json.loads((directory / "properties.json").read_text())
    return {"properties": properties, "scores": {w: list(table.score(w)) for w in WORDS}}


def kenlm_model_values() -> dict:
    _, _, model, directory, _ = _built("ngram_surprisal", order=3)
    properties = json.loads((directory / "properties.json").read_text())
    sentences_ = [["the", "cat", "saw", "the", "zebra", "."], ["the", "horse", "raced", "past"]]
    return {"properties": properties, "scores": model.score(sentences_)}


def _surprisal_values(module_name: str, **config) -> dict:
    from txtures.modules import Item

    module, parsed, loaded, _, context = _built(module_name, **config)
    name = "table" if module_name == "unigram_surprisal" else "model"
    values = {}
    for doc_name, doc in _ngram_documents().items():
        (result,) = module.run_batch(parsed, [Item(doc, {}, {name: loaded})], context)
        values[doc_name] = dict(module.columns(parsed, doc, result))
    return values


def unigram_surprisal_values() -> dict:
    return _surprisal_values("unigram_surprisal")


def ngram_surprisal_values() -> dict:
    return _surprisal_values("ngram_surprisal", order=3)


def _class_version(path: str) -> Callable[[], int]:
    def version() -> int:
        import importlib

        module, name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module), name).version

    return version


def _ngram_text_version() -> int:
    from txtures.ngram_text import NGRAM_TEXT_VERSION

    return NGRAM_TEXT_VERSION


def _stanza_version() -> int:
    from txtures.modules.stanza_pipeline import StanzaPipeline

    return StanzaPipeline.version


def _module_version(name: str) -> Callable[[], int]:
    def version() -> int:
        from txtures.modules import REGISTRY

        return REGISTRY[name].contract_version

    return version


def _hf_model_version() -> int:
    from txtures.modules.hf_model import HFModel

    return HFModel.version


def _reader_version() -> int:
    from txtures.inputs import READER_VERSION

    return READER_VERSION


TARGETS: dict[str, Target] = {
    "reader": Target(_reader_version, reader_values, "READER_VERSION in txtures/inputs.py"),
    "hf_model": Target(_hf_model_version, hf_model_values, "HFModel.version"),
    "stanza_pipeline": Target(
        _stanza_version, stanza_pipeline_values, "StanzaPipeline.version", stanza_inputs
    ),
    "parse": Target(
        _module_version("parse"), parse_values, "parse's contract_version", stanza_inputs
    ),
    "ngram_text": Target(
        _ngram_text_version, ngram_text_values, "NGRAM_TEXT_VERSION in txtures/ngram_text.py"
    ),
    "ngram_corpus": Target(
        _class_version("txtures.modules.ngram.NgramCorpus"),
        ngram_corpus_values,
        "NgramCorpus.version",
    ),
    "unigram_table": Target(
        _class_version("txtures.modules.ngram.UnigramTable"),
        unigram_table_values,
        "UnigramTable.version",
        tools_inputs,
    ),
    "kenlm_model": Target(
        _class_version("txtures.modules.ngram.KenLMModel"),
        kenlm_model_values,
        "KenLMModel.version",
        tools_inputs,
    ),
    "unigram_surprisal": Target(
        _module_version("unigram_surprisal"),
        unigram_surprisal_values,
        "unigram_surprisal's contract_version",
        tools_inputs,
    ),
    "ngram_surprisal": Target(
        _module_version("ngram_surprisal"),
        ngram_surprisal_values,
        "ngram_surprisal's contract_version",
        tools_inputs,
    ),
    **{
        name: Target(_module_version(name), compute, f"{name}'s contract_version")
        for name, compute in (
            ("lm_scores", lm_scores_values),
            ("surprisal", surprisal_values),
            ("word_length", word_length_values),
            ("pos", pos_values),
            ("named_entity", named_entity_values),
            ("dependency_locality", dependency_locality_values),
        )
    },
}


TOOLS = frozenset({"unigram_table", "kenlm_model", "unigram_surprisal", "ngram_surprisal"})
"""Targets that run programs in the tools image."""
HEAVY = frozenset({"hf_model", "lm_scores", "surprisal", "stanza_pipeline", "parse"}) | TOOLS
"""Targets that load neural models or run the tools image; `pytest --fast` skips them."""


def path(name: str) -> Path:
    return GOLDEN / f"{name}.json"


def compute(name: str) -> dict:
    target = TARGETS[name]
    record = {"target": name, "version": target.version(), "values": plain(target.compute())}
    if target.inputs is not None:
        record["inputs"] = plain(target.inputs())
    return record


def update(names: list[str]) -> int:
    """Record golden values, refusing changed values under an unchanged version."""
    status = 0
    GOLDEN.mkdir(exist_ok=True)
    for name in names:
        current = compute(name)
        if path(name).exists():
            recorded = json.loads(path(name).read_text(encoding="utf-8"))
            changed = differences(recorded["values"], current["values"])
            same_inputs = recorded.get("inputs") == current.get("inputs")
            if changed and recorded["version"] == current["version"] and same_inputs:
                print(
                    f"{name}: values changed but the version is still {current['version']}; "
                    f"increment {TARGETS[name].bump} first:\n  " + "\n  ".join(changed[:10]),
                    file=sys.stderr,
                )
                status = 1
                continue
            if not changed and recorded["version"] == current["version"] and same_inputs:
                print(f"{name}: unchanged")
                continue
        path(name).write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"{name}: recorded version {current['version']}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or record golden outputs.")
    parser.add_argument("--update", action="store_true", help="record the current values")
    parser.add_argument("targets", nargs="*", help="targets (default: all)")
    args = parser.parse_args(argv)
    names = args.targets or list(TARGETS)
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        parser.error(f"unknown targets {', '.join(unknown)}; choose from {', '.join(TARGETS)}")
    if args.update:
        return update(names)
    parser.error("pass --update to record values; run the tests to check them")
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
