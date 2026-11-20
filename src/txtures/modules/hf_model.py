"""The `hf_model` resource: a causal language model and its tokenizer.

A model is named either by a Hugging Face repository or by the absolute path
of a local directory written by `save_pretrained`. Resolving a repository
fixes its revision to a commit hash; resolving a local directory fixes it to
a digest of the directory's files. Resolution also fixes how the start of a
context is marked, so that choice is part of the resource's identity.

Building the resource records the tokenizer properties that scoring needs:
the model's maximum context length and the boundary set *B*, the token IDs
that begin a new word. It also downloads the weights into the Hugging Face
cache, which keeps them outside the resource store, so that a GPU job never
waits for a download.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from txtures.errors import TxturesError
from txtures.modules.base import Resource, RunContext

COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
BOUNDARY_FILE = "boundary.npy"
PROPERTIES_FILE = "properties.json"
_LENGTH_KEYS = ("n_positions", "max_position_embeddings", "n_ctx", "max_sequence_length")


def is_local(model: str) -> bool:
    """Whether `model` names a local directory rather than a Hugging Face repository."""
    return Path(model).is_absolute()


def directory_digest(directory: Path) -> str:
    """Digest of every file under `directory`: relative paths and contents."""
    if not directory.is_dir():
        raise TxturesError(f"model directory {directory} does not exist")
    hasher = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        hasher.update(path.relative_to(directory).as_posix().encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def resolve_revision(
    model: str, revision: str | None, cache: Path | None, token: str | None = None
) -> str:
    """Return the commit hash that `revision` (or the default branch) names now.

    A commit hash is returned unchanged. Otherwise the hub is asked; when it
    cannot be reached, the hash recorded in the local cache is used.

    Raises:
        TxturesError: The repository or revision does not exist, or the hub
            cannot be reached and the cache does not record the revision.
    """
    if revision is not None and COMMIT_HASH.fullmatch(revision):
        return revision
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        info = HfApi().model_info(model, revision=revision, token=token)
        if info.sha is None:
            raise TxturesError(f"the hub reports no commit for model {model!r}")
        return info.sha
    except GatedRepoError as error:
        raise TxturesError(f"model {model!r} is gated; log in with `txtures hf login`") from error
    except RevisionNotFoundError as error:
        raise TxturesError(f"model {model!r} has no revision {revision!r}") from error
    except RepositoryNotFoundError as error:
        raise TxturesError(f"model {model!r} was not found on the Hugging Face hub") from error
    except Exception as hub_error:  # network failures and offline mode
        try:
            path = snapshot_download(
                model, revision=revision, cache_dir=cache, local_files_only=True
            )
        except Exception:
            raise TxturesError(
                f"cannot resolve the revision of model {model!r}: the hub could not be "
                f"reached ({type(hub_error).__name__}) and the cache does not record it"
            ) from hub_error
        return Path(path).name


def _source(
    resolved: Mapping[str, Any], cache: Path | None, token: str | None = None
) -> dict[str, Any]:
    """Arguments for `from_pretrained` that load exactly the resolved model."""
    if is_local(resolved["model"]):
        return {"pretrained_model_name_or_path": resolved["model"]}
    source = {
        "pretrained_model_name_or_path": resolved["model"],
        "revision": resolved["revision"],
        "cache_dir": cache,
    }
    if token is not None:
        source["token"] = token
    return source


@lru_cache(maxsize=16)
def _tokenizer(model: str, revision: str, cache: Path | None, token: str | None):
    from transformers import AutoTokenizer

    resolved = {"model": model, "revision": revision}
    try:
        tokenizer = AutoTokenizer.from_pretrained(**_source(resolved, cache, token))
    except Exception as error:
        raise TxturesError(f"cannot load the tokenizer of model {model!r}: {error}") from error
    if not getattr(tokenizer, "is_fast", True):
        raise TxturesError(f"model {model!r} has no fast tokenizer, which character spans need")
    return tokenizer


def load_tokenizer(resolved: Mapping[str, Any], cache: Path | None, token: str | None = None):
    return _tokenizer(resolved["model"], resolved["revision"], cache, token)


def adds_bos(tokenizer) -> bool:
    """Whether the tokenizer puts its BOS token before text by default."""
    bos = tokenizer.bos_token_id
    if bos is None:
        return False
    plain = tokenizer("a", add_special_tokens=False)["input_ids"]
    default = tokenizer("a")["input_ids"]
    return len(default) > len(plain) and default[0] == bos


def auto_document_start(tokenizer) -> str | None:
    """Choose how a context start is marked: `bos`, `eos`, or `None`.

    The choice follows how the model saw document starts in training:

    1. A tokenizer that adds BOS by default (Llama, Mistral, Gemma) gets `bos`.
    2. One whose BOS and EOS are the same token (GPT-2, Pythia) gets `eos`:
       such models were trained on documents separated by that token.
    3. One that declares no BOS but has an EOS (OLMo, Qwen, Falcon) also gets
       `eos`, for the same reason.
    4. One that declares a distinct BOS without adding it gets `bos`, since a
       model declares a BOS only when it was trained with one.

    `None` means the tokenizer declares neither token.
    """
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if adds_bos(tokenizer):
        return "bos"
    if eos is not None and (bos is None or bos == eos):
        return "eos"
    if bos is not None:
        return "bos"
    return None


def start_ids(tokenizer, document_start: str) -> list[int]:
    """Token IDs placed before every context."""
    if document_start == "none":
        return []
    token = tokenizer.bos_token_id if document_start == "bos" else tokenizer.eos_token_id
    if token is None:
        raise TxturesError(f"the tokenizer has no {document_start.upper()} token")
    return [token]


def boundary_mask(tokenizer, size: int) -> np.ndarray:
    """Return a Boolean mask over token IDs marking the boundary set *B*.

    A token belongs to *B* when decoding it after a fixed word yields text
    that starts with whitespace. EOS also belongs to *B*. IDs at or beyond
    the tokenizer's vocabulary (padding rows of the output layer) do not.
    """
    prefix = tokenizer.encode("a", add_special_tokens=False)
    base = tokenizer.decode(prefix, clean_up_tokenization_spaces=False)
    count = min(len(tokenizer), size)
    texts = tokenizer.batch_decode(
        [[*prefix, token] for token in range(count)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    mask = np.zeros(size, dtype=bool)
    for token, text in enumerate(texts):
        mask[token] = text.startswith(base) and text[len(base) : len(base) + 1].isspace()
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < size:
        mask[tokenizer.eos_token_id] = True
    return mask


def boundaries_detected(tokenizer, mask: np.ndarray) -> bool:
    """Check that *B* separates words on a test text.

    In `a bbbbbbbbbbbbbbbb`, the second word must take at least two tokens, of
    which only the first is in *B*.
    """
    prefix = tokenizer.encode("a", add_special_tokens=False)
    ids = tokenizer.encode("a " + "b" * 16, add_special_tokens=False)
    if ids[: len(prefix)] != prefix:
        return False
    word = [token < len(mask) and bool(mask[token]) for token in ids[len(prefix) :]]
    return len(word) >= 2 and word[0] and not any(word[1:])


def max_length(config) -> int | None:
    """The longest context the model accepts, from its config."""
    for key in _LENGTH_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def device(name: str) -> str:
    """Resolve the `device` setting; `auto` prefers CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prefetch_weights(
    resolved: Mapping[str, Any], cache: Path | None, token: str | None = None
) -> None:
    """Download a hub model's files at its resolved revision into the Hugging Face cache.

    Safetensors weights are taken when the repository has them, and PyTorch
    `.bin` weights otherwise, so a repository with both is not downloaded
    twice. A local model directory needs nothing.

    Raises:
        TxturesError: The files cannot be downloaded.
    """
    if is_local(resolved["model"]):
        return
    from huggingface_hub import HfApi, snapshot_download

    try:
        files = HfApi().list_repo_files(
            resolved["model"], revision=resolved["revision"], token=token
        )
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        snapshot_download(
            resolved["model"],
            revision=resolved["revision"],
            cache_dir=cache,
            token=token,
            allow_patterns=[weights, "*.json", "*.model", "*.txt", "*.tiktoken"],
        )
    except Exception as error:
        raise TxturesError(
            f"cannot download the files of model {resolved['model']!r}: {error}"
        ) from error


@dataclass
class LanguageModel:
    """A loaded `hf_model` resource, as modules receive it."""

    model: Any
    tokenizer: Any
    device: str
    start_ids: list[int]
    boundary: Any
    """Boolean tensor over token IDs on `device`, marking *B*."""
    boundaries_detected: bool
    max_length: int | None
    document_start: str


class HFModel(Resource):
    """A causal language model from the Hugging Face hub or a local directory.

    Spec keys: `model`, `revision`, `dtype`, and `document_start`. The
    resolved spec has the same keys, with `revision` fixed and
    `document_start` one of `bos`, `eos`, or `none`.
    """

    kind = "hf_model"
    version = 1

    def resolve(self, context: RunContext) -> dict[str, Any]:
        model = self.spec["model"]
        revision = self.spec.get("revision")
        if is_local(model):
            if revision is not None:
                raise TxturesError("revision cannot be set for a local model directory")
            revision = "sha256:" + directory_digest(Path(model))
        else:
            revision = resolve_revision(model, revision, context.hf_cache, context.hf_token)
        resolved = {"model": model, "revision": revision, "dtype": self.spec["dtype"]}
        tokenizer = load_tokenizer(resolved, context.hf_cache, context.hf_token)
        choice = self.spec["document_start"]
        if choice == "auto":
            choice = auto_document_start(tokenizer)
            if choice is None:
                raise TxturesError(
                    f"model {model!r} declares neither a BOS nor an EOS token, so no token "
                    "can mark a document start; set document_start: none to score it "
                    "without one (its first word then has no surprisal)"
                )
        start_ids(tokenizer, choice)
        return {**resolved, "document_start": choice}

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        import tokenizers
        import transformers
        from transformers import AutoConfig

        prefetch_weights(resolved, context.hf_cache, context.hf_token)
        tokenizer = load_tokenizer(resolved, context.hf_cache, context.hf_token)
        try:
            config = AutoConfig.from_pretrained(
                **_source(resolved, context.hf_cache, context.hf_token)
            )
        except Exception as error:
            raise TxturesError(
                f"cannot load the config of model {resolved['model']!r}: {error}"
            ) from error
        size = int(getattr(config, "vocab_size", 0) or 0) or len(tokenizer)
        mask = boundary_mask(tokenizer, size)
        np.save(directory / BOUNDARY_FILE, mask, allow_pickle=False)
        properties = {
            "max_length": max_length(config),
            "vocab_size": size,
            "boundary_tokens": int(mask.sum()),
            "boundaries_detected": boundaries_detected(tokenizer, mask),
            "start_ids": start_ids(tokenizer, resolved["document_start"]),
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
        }
        (directory / PROPERTIES_FILE).write_text(
            json.dumps(properties, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging

        properties = json.loads((directory / PROPERTIES_FILE).read_text(encoding="utf-8"))
        mask = np.load(directory / BOUNDARY_FILE, allow_pickle=False)
        target = device(context.device)
        logging.disable_progress_bar()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                **_source(resolved, context.hf_cache, context.hf_token),
                dtype=getattr(torch, resolved["dtype"]),
            )
        except Exception as error:
            raise TxturesError(f"cannot load model {resolved['model']!r}: {error}") from error
        model.to(target)
        model.eval()
        return LanguageModel(
            model=model,
            tokenizer=load_tokenizer(resolved, context.hf_cache, context.hf_token),
            device=target,
            start_ids=list(properties["start_ids"]),
            boundary=torch.from_numpy(mask).to(target),
            boundaries_detected=bool(properties["boundaries_detected"]),
            max_length=properties["max_length"],
            document_start=resolved["document_start"],
        )
