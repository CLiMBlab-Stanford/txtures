"""Unigram and n-gram surprisal from KenLM models trained on a corpus.

A corpus (`ngram_corpus`) is a Hugging Face dataset of Parquet files at a
pinned commit, or a local directory of Parquet or text files, tokenized and
split into sentences by `txtures.ngram_text`, one sentence per line. KenLM's
`lmplz`, run from the tools image, estimates modified Kneser-Ney models on
it:

- `unigram_table`: an order-1 model, whose probabilities come from raw
  counts with modified Kneser-Ney discounts, the discounted mass spread over
  the vocabulary and `<unk>`. KenLM cannot query an order-1 model, so the
  build stores its probabilities as a sorted table that `unigram_surprisal`
  reads directly.
- `kenlm_model`: a model of order 2 or more, pruned as configured, stored in
  KenLM's binary trie format and queried with KenLM's `query`.

The target text is tokenized the same way. Each sentence from the
`sentences` module (the input's, or Stanza's split moved to region
boundaries) is scored as its own context, starting from `<s>`; the
probability of `</s>` is not counted. A region's surprisal is the sum of
its tokens' surprisals, and its out-of-vocabulary count the number of its
tokens that the model does not know, which all get the probability of
`<unk>`. A region without tokens has no surprisal.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from txtures.corpus import Document
from txtures.errors import TxturesError
from txtures.modules.base import Item, Module, ModuleConfig, Resource, RunContext, Workload
from txtures.modules.sentences import sentence_spans
from txtures.ngram_text import NGRAM_TEXT_VERSION, sentences, tokenize
from txtures.store import write_json

CORPUS_FILE = "corpus.txt.gz"
MODEL_FILE = "model.binary"
PROPERTIES_FILE = "properties.json"
LN10 = math.log(10)


def is_local(dataset: str) -> bool:
    return Path(dataset).is_absolute()


def _scratch() -> Path:
    for name in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    return Path(tempfile.gettempdir())


def _memory() -> str:
    """`lmplz -S`: 80% of the Slurm job's memory, or of the machine's outside Slurm."""
    allocated = os.environ.get("SLURM_MEM_PER_NODE")
    if allocated and allocated.isdigit():
        return f"{int(int(allocated) * 0.8)}M"
    return "80%"


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# The corpus


def _local_files(directory: Path, pattern: str) -> list[str]:
    if not directory.is_dir():
        raise TxturesError(f"corpus directory {directory} does not exist")
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and fnmatch.fnmatch(p.relative_to(directory).as_posix(), pattern)
    )


def _files_digest(directory: Path, files: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for name in files:
        hasher.update(name.encode() + b"\0")
        with (directory / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _documents(path: Path, text_field: str):
    """Yield the documents of one corpus file: rows of a Parquet file, or lines of a text file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=[text_field], batch_size=1024):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line


def _count_documents(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _prepare_file(job: tuple) -> tuple[int, int, int]:
    """Tokenize one corpus file into a gzipped part. Returns documents, lines, tokens.

    A line holds one sentence, or with `context` `document` one document's
    sentences joined.
    """
    source, text_field, limit, lowercase, context, output = job
    if isinstance(source, tuple):
        from huggingface_hub import hf_hub_download

        repository, filename, revision, cache, token = source
        source = hf_hub_download(
            repository, filename, repo_type="dataset", revision=revision,
            cache_dir=cache, token=token,
        )  # fmt: skip
    counts = [0, 0, 0]
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=1) as out:
        for document in _documents(Path(source), text_field):
            if limit is not None and counts[0] >= limit:
                break
            counts[0] += 1
            lines = [tokenize(sentence, lowercase) for sentence in sentences(document)]
            if context == "document":
                lines = [[token for line in lines for token in line]]
            for tokens in lines:
                if tokens:
                    out.write(" ".join(tokens) + "\n")
                    counts[1] += 1
                    counts[2] += len(tokens)
    return counts[0], counts[1], counts[2]


class NgramCorpus(Resource):
    """A tokenized corpus, one sentence or one document per line, in `corpus.txt.gz`.

    Spec keys: `dataset` (a Hugging Face dataset, or the absolute path of a
    local directory), `revision`, `files` (a glob of the files to use),
    `text_field` (the Parquet column), `max_documents` (the first so many
    documents, in file order; unset for all), `lowercase`, `text_version`,
    and `context`: `document` for one document per line. Without it each
    line is a sentence, so corpora built before the key existed keep their
    identity. Resolving fixes the revision (a commit, or a digest of a
    local directory's files) and the list of files.
    """

    kind = "ngram_corpus"
    version = 1

    def stale(self, manifest: Mapping[str, Any]) -> str | None:
        """A corpus tokenized by an older `NGRAM_TEXT_VERSION`, or a local
        directory that is gone or no longer has the same files.

        Changes to the contents of files that remain are not detected, as that
        would read the whole corpus.
        """
        spec = manifest["spec"]
        recorded = spec.get("text_version")
        if isinstance(recorded, int) and recorded < NGRAM_TEXT_VERSION:
            return f"tokenized by text version {recorded}, now {NGRAM_TEXT_VERSION}"
        if not is_local(spec["dataset"]):
            return None
        directory = Path(spec["dataset"])
        if not directory.is_dir():
            return f"its corpus directory {directory} no longer exists"
        if _local_files(directory, spec["files"]) != list(spec.get("file_list", [])):
            return f"the files of its corpus directory {directory} have changed"
        return None

    def resolve(self, context: RunContext) -> dict[str, Any]:
        spec = dict(self.spec)
        if is_local(spec["dataset"]):
            if spec.get("revision") is not None:
                raise TxturesError("revision cannot be set for a local corpus directory")
            files = _local_files(Path(spec["dataset"]), spec["files"])
            revision = "sha256:" + _files_digest(Path(spec["dataset"]), files)
        else:
            from huggingface_hub import HfApi

            try:
                info = HfApi().dataset_info(
                    spec["dataset"], revision=spec.get("revision"), token=context.hf_token
                )
            except Exception as error:
                raise TxturesError(
                    f"cannot resolve dataset {spec['dataset']!r} on the Hugging Face hub: {error}"
                ) from error
            revision = info.sha
            names = [s.rfilename for s in info.siblings or []]
            files = sorted(n for n in names if fnmatch.fnmatch(n, spec["files"]))
        if not files:
            raise TxturesError(f"no files of {spec['dataset']} match {spec['files']!r}")
        return {**spec, "revision": revision, "file_list": files}

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=16, memory_gb=32, time_hours=24, ngpus=0)

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        dataset, files = resolved["dataset"], resolved["file_list"]
        if is_local(dataset):
            sources: list[Any] = [str(Path(dataset) / name) for name in files]
            limits = _limits([Path(s) for s in sources], resolved.get("max_documents"))
        else:
            sources = [
                (dataset, name, resolved["revision"], context.hf_cache, context.hf_token)
                for name in files
            ]
            limits = [None] * len(files)
            if resolved.get("max_documents") is not None:
                paths = [Path(_prepare_source(s)) for s in sources]
                sources = [str(p) for p in paths]
                limits = _limits(paths, resolved["max_documents"])
        parts = directory / "parts"
        parts.mkdir()
        jobs = [
            (
                source,
                resolved["text_field"],
                limit,
                resolved["lowercase"],
                resolved.get("context", "sentence"),
                parts / f"{i:05d}.gz",
            )
            for i, (source, limit) in enumerate(zip(sources, limits, strict=True))
            if limit is None or limit > 0
        ]
        counts = _run_jobs(jobs, max(1, min(_cores(), len(jobs))), parts)
        with (directory / CORPUS_FILE).open("wb") as out:
            # Concatenated gzip members are one valid gzip stream.
            for job in jobs:
                with Path(job[-1]).open("rb") as part:
                    shutil.copyfileobj(part, out, 1 << 20)
        shutil.rmtree(parts)
        documents, sentence_count, tokens = (sum(c[i] for c in counts) for i in range(3))
        write_json(
            directory / PROPERTIES_FILE,
            {"documents": documents, "sentences": sentence_count, "tokens": tokens},
        )

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Path:
        return directory / CORPUS_FILE


def _run_jobs(jobs: list[tuple], workers: int, parts: Path) -> list[tuple[int, int, int]]:
    """Prepare corpus files in `workers` processes, each taking every `workers`-th job.

    Workers are separate `python -m txtures.ngram_worker` processes, so they
    work however the calling program was started (a script, a notebook,
    standard input), unlike `multiprocessing`'s spawned workers.
    """
    if workers == 1:
        return [_prepare_file(job) for job in jobs]
    listing = parts / "jobs.json"
    listing.write_text(json.dumps([[str(j[-1]), *_portable(j)] for j in jobs]), encoding="utf-8")
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "txtures.ngram_worker", str(listing), str(k), str(workers)]
        )
        for k in range(workers)
    ]
    failed = [p.args for p in processes if p.wait() != 0]
    if failed:
        raise TxturesError(f"{len(failed)} of {workers} corpus workers failed; see their output")
    return [tuple(json.loads(Path(f"{job[-1]}.json").read_text())) for job in jobs]


def _portable(job: tuple) -> list[Any]:
    source, text_field, limit, lowercase, context, _ = job
    source = list(source) if isinstance(source, tuple) else source
    if isinstance(source, list):
        source = [str(v) if isinstance(v, Path) else v for v in source]
    return [source, text_field, limit, lowercase, context]


def _worker(listing: Path, index: int, workers: int) -> None:
    """Prepare this worker's share of the jobs in `listing`, recording each part's counts."""
    for position, (output, source, text_field, limit, lowercase, context) in enumerate(
        json.loads(listing.read_text(encoding="utf-8"))
    ):
        if position % workers != index:
            continue
        source = tuple(source) if isinstance(source, list) else source
        job = (source, text_field, limit, lowercase, context, Path(output))
        counts = _prepare_file(job)
        Path(f"{output}.json").write_text(json.dumps(counts), encoding="utf-8")


def _prepare_source(source: tuple) -> str:
    from huggingface_hub import hf_hub_download

    repository, filename, revision, cache, token = source
    return hf_hub_download(
        repository, filename, repo_type="dataset", revision=revision, cache_dir=cache, token=token
    )


def _limits(paths: Sequence[Path], total: int | None) -> list[int | None]:
    """How many documents to take from each file to take the first `total` in file order."""
    if total is None:
        return [None] * len(paths)
    limits, left = [], total
    for path in paths:
        take = min(left, _count_documents(path))
        limits.append(take)
        left -= take
    return limits


# KenLM models


def _check(result: subprocess.CompletedProcess, what: str, log: Path) -> None:
    if result.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-20:] if log.exists() else []
        raise TxturesError(f"{what} failed ({result.returncode}):\n" + "\n".join(tail))


def _lmplz(tools, corpus: Path, order: int, arpa: Path, log: Path, resolved, work: Path) -> None:
    argv = ["lmplz", "-o", str(order), "-S", _memory(), "-T", str(work)]
    argv += ["--text", str(corpus), "--arpa", str(arpa)]
    prune = list(resolved.get("prune") or [])
    if prune:
        argv += ["--prune", *(str(p) for p in prune)]
    if resolved.get("discount_fallback"):
        argv.append("--discount_fallback")
    with log.open("w", encoding="utf-8") as handle:
        result = tools.run(
            argv, binds=[corpus.parent, work], stdout=handle, stderr=subprocess.STDOUT
        )
    _check(result, "lmplz", log)


def arpa_counts(arpa: Path) -> dict[str, int]:
    """The n-gram counts in an ARPA file's header, by order."""
    counts = {}
    with arpa.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.match(r"ngram (\d+)=(\d+)", line)
            if match:
                counts[match.group(1)] = int(match.group(2))
            elif line.startswith("\\1-grams:"):
                break
    return counts


@dataclass
class KenLMQuery:
    """A built `kenlm_model`, scored with KenLM's `query` in the tools image."""

    tools: Any
    model: Path

    def score(self, sentences_: Sequence[Sequence[str]]) -> list[list[tuple[float, bool]]]:
        """Return `(log10 probability, out of vocabulary)` of each token of each sentence."""
        if not sentences_:
            return []
        text = "".join(" ".join(tokens) + "\n" for tokens in sentences_)
        result = self.tools.run(
            ["query", "-v", "word", "-l", "lazy", str(self.model)],
            binds=[self.model.parent],
            input=text,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TxturesError(f"KenLM query failed: {result.stderr.strip()[-2000:]}")
        return parse_query(result.stdout, sentences_)


_ENTRY = re.compile(r"(\S+)=(\d+) (\d+) (-?(?:inf|[0-9.]+(?:[eE][-+]?\d+)?))")


def parse_query(output: str, sentences_: Sequence[Sequence[str]]) -> list[list[tuple[float, bool]]]:
    """Read `query -v word` output: per token `word=id order log10prob`, `</s>` ending each
    sentence. Returns each sentence's tokens without `</s>`.

    Raises:
        TxturesError: The output does not match the sentences.
    """
    entries = _ENTRY.findall(output)
    scored, position = [], 0
    for tokens in sentences_:
        chunk = entries[position : position + len(tokens) + 1]
        position += len(tokens) + 1
        words = [entry[0] for entry in chunk]
        if words != [*tokens, "</s>"]:
            raise TxturesError(f"KenLM query output does not match the sentence {list(tokens)}")
        scored.append([(float(entry[3]), entry[1] == "0") for entry in chunk[:-1]])
    if position != len(entries):
        raise TxturesError("KenLM query output has more tokens than the sentences")
    return scored


class KenLMModel(Resource):
    """A KenLM model of order 2 or more, in binary trie format.

    Spec keys: `corpus` (an `ngram_corpus` spec), `order`, `prune` (the
    pruning thresholds of orders 1, 2, ..., as `lmplz --prune` takes them),
    and `discount_fallback`. Depends on the tools image and the corpus.
    """

    kind = "kenlm_model"
    version = 1

    def dependencies(self) -> Mapping[str, Resource]:
        from txtures.tools import ToolsImage

        return {"tools": ToolsImage(), "corpus": NgramCorpus(self.spec["corpus"])}

    def resolve(self, context: RunContext) -> dict[str, Any]:
        spec = dict(self.spec)
        spec.pop("corpus")
        return spec

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=8, memory_gb=100, time_hours=48, ngpus=0)

    def build(self, resolved, directory: Path, context: RunContext, dependencies) -> None:
        tools, corpus = dependencies["tools"], dependencies["corpus"]
        with tempfile.TemporaryDirectory(dir=_scratch(), prefix="txtures-kenlm-") as work:
            work = Path(work)
            arpa = work / "model.arpa"
            _lmplz(tools, corpus, resolved["order"], arpa, directory / "lmplz.log", resolved, work)
            log = directory / "build_binary.log"
            with log.open("w", encoding="utf-8") as handle:
                result = tools.run(
                    ["build_binary", "trie", str(arpa), str(directory / MODEL_FILE)],
                    binds=[work, directory],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            _check(result, "build_binary", log)
            write_json(directory / PROPERTIES_FILE, {"ngrams": arpa_counts(arpa)})

    def load(self, resolved, directory: Path, context: RunContext, dependencies) -> KenLMQuery:
        return KenLMQuery(dependencies["tools"], directory / MODEL_FILE)


# The unigram table


def write_table(arpa: Path, directory: Path) -> dict[str, Any]:
    """Store an order-1 ARPA model as a sorted table: `words.bin` (UTF-8 words,
    concatenated), `offsets.npy` (where each word starts, and the end), and
    `logprobs.npy` (log10 probabilities). Returns its properties."""
    entries: list[tuple[bytes, float]] = []
    unknown = None
    with arpa.open(encoding="utf-8") as handle:
        inside = False
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("\\1-grams:"):
                inside = True
                continue
            if inside and (not line or line.startswith("\\")):
                break
            if inside:
                logprob, word = line.split("\t")[:2]
                if word == "<unk>":
                    unknown = float(logprob)
                elif word != "<s>":
                    entries.append((word.encode("utf-8"), float(logprob)))
    if unknown is None:
        raise TxturesError(f"{arpa} has no <unk> entry")
    entries.sort()
    offsets = np.zeros(len(entries) + 1, dtype=np.int64)
    with (directory / "words.bin").open("wb") as out:
        for index, (word, _) in enumerate(entries):
            out.write(word)
            offsets[index + 1] = offsets[index] + len(word)
    np.save(directory / "offsets.npy", offsets, allow_pickle=False)
    logprobs = np.array([p for _, p in entries], dtype=np.float64)
    np.save(directory / "logprobs.npy", logprobs, allow_pickle=False)
    properties = {"vocabulary": len(entries), "unknown_logprob": unknown}
    write_json(directory / PROPERTIES_FILE, properties)
    return properties


class UnigramLookup:
    """A built `unigram_table`: log10 probabilities by binary search over the sorted words."""

    def __init__(self, directory: Path):
        self.words = np.memmap(directory / "words.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r", allow_pickle=False)
        self.logprobs = np.load(directory / "logprobs.npy", mmap_mode="r", allow_pickle=False)
        properties = json.loads((directory / PROPERTIES_FILE).read_text())
        self.unknown = float(properties["unknown_logprob"])

    def _word(self, index: int) -> bytes:
        return bytes(self.words[self.offsets[index] : self.offsets[index + 1]])

    def score(self, word: str) -> tuple[float, bool]:
        """Return the word's `(log10 probability, out of vocabulary)`."""
        target = word.encode("utf-8")
        low, high = 0, len(self.offsets) - 1
        while low < high:
            middle = (low + high) // 2
            if self._word(middle) < target:
                low = middle + 1
            else:
                high = middle
        if low < len(self.offsets) - 1 and self._word(low) == target:
            return float(self.logprobs[low]), False
        return self.unknown, True


class UnigramTable(Resource):
    """An order-1 KenLM model of a corpus, as a table. Spec keys: `corpus` and
    `discount_fallback`. Depends on the tools image and the corpus."""

    kind = "unigram_table"
    version = 1

    def dependencies(self) -> Mapping[str, Resource]:
        from txtures.tools import ToolsImage

        return {"tools": ToolsImage(), "corpus": NgramCorpus(self.spec["corpus"])}

    def resolve(self, context: RunContext) -> dict[str, Any]:
        spec = dict(self.spec)
        spec.pop("corpus")
        return spec

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=4, memory_gb=32, time_hours=12, ngpus=0)

    def build(self, resolved, directory: Path, context: RunContext, dependencies) -> None:
        tools, corpus = dependencies["tools"], dependencies["corpus"]
        with tempfile.TemporaryDirectory(dir=_scratch(), prefix="txtures-unigram-") as work:
            work = Path(work)
            arpa = work / "unigram.arpa"
            _lmplz(tools, corpus, 1, arpa, directory / "lmplz.log", resolved, work)
            write_table(arpa, directory)

    def load(self, resolved, directory: Path, context: RunContext, dependencies) -> UnigramLookup:
        return UnigramLookup(directory)


# Region scores


def contexts(
    document: Document, lowercase: bool, spans: Sequence[tuple[int, int]] | None = None
) -> list[tuple[list[str], list[int]]]:
    """The token contexts of a document: for each sentence, its tokens and each token's region.

    `spans` are the sentences' region ranges, as the `sentences` module gives
    them. Without them the document is one context, as for `context:
    document` and for unigram scores, which do not depend on context.
    """
    if spans is None:
        spans = [(0, len(document.regions))]
    found = []
    for start, stop in spans:
        tokens, regions = [], []
        for region in range(start, stop):
            for token in tokenize(document.region_text(region), lowercase):
                tokens.append(token)
                regions.append(region)
        if tokens:
            found.append((tokens, regions))
    return found


def region_scores(
    document: Document,
    found: Sequence[tuple[list[str], list[int]]],
    scored: Sequence[Sequence[tuple[float, bool]]],
) -> dict[str, np.ndarray]:
    """Sum token scores into regions: `nats` (surprisal) and `oov` (unknown tokens)."""
    nats = np.full(len(document.regions), np.nan)
    oov = np.zeros(len(document.regions), dtype=np.int64)
    for (_, regions), scores in zip(found, scored, strict=True):
        for region, (log10, unknown) in zip(regions, scores, strict=True):
            value = -log10 * LN10
            nats[region] = value if math.isnan(nats[region]) else nats[region] + value
            oov[region] += int(unknown)
    return {"nats": nats, "oov": oov}


# Modules


class CorpusConfig(ModuleConfig):
    """Parameters shared by the n-gram modules: the corpus and the output units."""

    dataset: str = Field(default="Skylion007/openwebtext", min_length=1)
    """Hugging Face dataset of Parquet files, or the absolute path of a local
    directory of Parquet files (one document per row) or text files (one per line)."""

    revision: str | None = Field(default=None, min_length=1)
    """Commit, branch, or tag of the dataset; resolved to a commit and recorded."""

    files: str = Field(default="plain_text/train-*.parquet", min_length=1)
    """Glob of the dataset's files to use, relative to the dataset."""

    text_field: str = Field(default="text", min_length=1)
    """The Parquet column that holds each document's text."""

    max_documents: int | None = Field(default=None, gt=0)
    """Use only the first so many documents, in file order; unset uses all."""

    lowercase: bool = True
    """Lowercase the corpus and the text before tokenizing."""

    discount_fallback: bool = False
    """Pass `--discount_fallback` to lmplz, which small or deduplicated corpora need."""

    units: Literal["bits", "nats"] = "bits"
    """Logarithm base of the reported surprisal: 2 for bits, e for nats."""

    def corpus_spec(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "files": self.files,
            "text_field": self.text_field,
            "max_documents": self.max_documents,
            "lowercase": self.lowercase,
            "text_version": NGRAM_TEXT_VERSION,
        }


class UnigramSurprisalConfig(CorpusConfig):
    """Parameters for unigram surprisal."""


class NgramSurprisalConfig(CorpusConfig):
    """Parameters for n-gram surprisal."""

    order: int = Field(default=5, ge=2)
    """The n-gram order."""

    context: Literal["sentence", "document"] = "sentence"
    """What one context is, in the training corpus and in the scored text:
    each sentence, starting after `<s>`, or each whole document. The model is
    trained on lines of the same kind, so its n-grams cross sentence
    boundaries only with `document`."""

    prune: tuple[int, ...] = (0, 0, 1)
    """Pruning thresholds of orders 1, 2, ...: n-grams seen this many times or
    fewer are dropped, and the last threshold applies to higher orders, as in
    `lmplz --prune`. Unigrams cannot be pruned."""

    @model_validator(mode="after")
    def _valid_prune(self) -> Self:
        if self.prune and self.prune[0] != 0:
            raise ValueError("prune cannot drop unigrams; its first threshold must be 0")
        if any(b < a for a, b in zip(self.prune, self.prune[1:], strict=False)):
            raise ValueError("prune thresholds must not decrease with the order")
        if any(p < 0 for p in self.prune):
            raise ValueError("prune thresholds must not be negative")
        return self


CORPUS_SWEEP = ("dataset", "revision", "files", "text_field", "max_documents")


def _units(nats: np.ndarray, units: str) -> np.ndarray:
    return nats / math.log(2) if units == "bits" else nats


class UnigramSurprisal(Module):
    """Compute each region's unigram surprisal under a corpus's order-1 KenLM model."""

    name = "unigram_surprisal"
    contract_version = 1
    Config = UnigramSurprisalConfig
    features = ("unigram_surprisal", "unigram_oov")
    sweeps = {"dataset": CORPUS_SWEEP, "units": ("units",)}
    identifying = ("dataset",)
    persist = False

    def resources(self, config: UnigramSurprisalConfig):
        spec = {"corpus": config.corpus_spec(), "discount_fallback": config.discount_fallback}
        return {"table": UnigramTable(spec)}

    def run(self, config: UnigramSurprisalConfig, item: Item, context: RunContext):
        table: UnigramLookup = item.resources["table"]
        found = contexts(item.document, config.lowercase)
        scored = [[table.score(token) for token in tokens] for tokens, _ in found]
        return region_scores(item.document, found, scored)

    def columns(self, config: UnigramSurprisalConfig, document: Document, result):
        return {
            "unigram_surprisal": _units(result["nats"], config.units),
            "unigram_oov": result["oov"],
        }


class NgramSurprisal(Module):
    """Compute each region's n-gram surprisal under a corpus's KenLM model."""

    name = "ngram_surprisal"
    contract_version = 2
    Config = NgramSurprisalConfig
    requires = ("sentences",)
    features = ("ngram_surprisal", "ngram_oov")
    sweeps = {
        "dataset": CORPUS_SWEEP,
        "order": ("order",),
        "context": ("context",),
        "units": ("units",),
    }
    identifying = ("dataset",)
    persist = True
    batch_size = 256

    def workload(self, config: NgramSurprisalConfig) -> Workload:
        return Workload(slurm=True, ngpus=0)

    def resources(self, config: NgramSurprisalConfig):
        corpus = config.corpus_spec()
        if config.context == "document":
            corpus["context"] = "document"
        spec = {
            "corpus": corpus,
            "order": config.order,
            "prune": list(config.prune[: config.order]),
            "discount_fallback": config.discount_fallback,
        }
        return {"model": KenLMModel(spec)}

    def run_batch(self, config: NgramSurprisalConfig, items: Sequence[Item], context: RunContext):
        if not items:
            return []
        model: KenLMQuery = items[0].resources["model"]
        found = [
            contexts(
                item.document,
                config.lowercase,
                None
                if config.context == "document"
                else sentence_spans(item.upstream["sentences"]),
            )
            for item in items
        ]
        scored = model.score([tokens for document in found for tokens, _ in document])
        results, position = [], 0
        for item, document in zip(items, found, strict=True):
            mine = scored[position : position + len(document)]
            position += len(document)
            results.append(region_scores(item.document, document, mine))
        return results

    def columns(self, config: NgramSurprisalConfig, document: Document, result):
        return {
            "ngram_surprisal": _units(np.asarray(result["nats"], dtype=float), config.units),
            "ngram_oov": result["oov"],
        }
