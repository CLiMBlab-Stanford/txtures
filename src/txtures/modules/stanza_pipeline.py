"""The `stanza_pipeline` resource: Stanza models for one language.

Stanza publishes its models in the Hugging Face repository
`stanfordnlp/stanza-LANG`, tagged `vVERSION` for each resources version, and
lists which model files each processor needs in a `resources.json` file for
that version. Resolving the resource fixes the resources version (the one the
installed Stanza uses) and resolves the tag to a commit hash, so the model
files are fixed by the resource's identity. Building downloads those files
into the resource directory, checking each against the MD5 sum that
`resources.json` records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from txtures.errors import TxturesError
from txtures.modules.base import Resource, RunContext
from txtures.modules.hf_model import device, resolve_revision

PROCESSORS = ("tokenize", "mwt", "pos", "lemma", "depparse", "ner")
"""The Stanza processors the `parse` module runs, in pipeline order."""

FILES_FILE = "files.json"


def repository(language: str) -> str:
    """The Hugging Face repository that holds Stanza's models for `language`."""
    return f"stanfordnlp/stanza-{language}"


def model_url(resolved: Mapping[str, Any]) -> str:
    """Stanza's model URL template, pinned to the resolved commit."""
    return (
        f"https://huggingface.co/{repository(resolved['language'])}/resolve/"
        f"{resolved['commit']}/models/{{filename}}"
    )


@dataclass
class StanzaModels:
    """A loaded `stanza_pipeline` resource, as `parse` receives it.

    `parser` runs every processor on text that is already one sentence.
    `splitter` runs only tokenization with sentence splitting, and is loaded
    when first used.
    """

    directory: Path
    language: str
    package: str
    device: str
    parser: Any
    _splitter: Any = field(default=None, repr=False)

    @property
    def splitter(self) -> Any:
        if self._splitter is None:
            self._splitter = _pipeline(self, "tokenize", tokenize_no_ssplit=False)
        return self._splitter


def _pipeline(models: StanzaModels, processors: str, **options) -> Any:
    import stanza

    try:
        return stanza.Pipeline(
            models.language,
            dir=str(models.directory),
            package=models.package,
            processors=processors,
            download_method=None,
            device=models.device,
            logging_level="WARN",
            **options,
        )
    except Exception as error:
        raise TxturesError(f"cannot load the Stanza {models.language} models: {error}") from error


class StanzaPipeline(Resource):
    """Stanza tokenization, tagging, lemmatization, parsing, and NER models.

    Spec keys: `language` (a Stanza language code such as `en`) and
    `package` (a Stanza package name; `default` is the language's default).
    The resolved spec adds `resources_version` and `commit`.
    """

    kind = "stanza_pipeline"
    version = 1

    def resolve(self, context: RunContext) -> dict[str, Any]:
        from stanza.resources.common import DEFAULT_RESOURCES_VERSION

        language = self.spec["language"]
        version = DEFAULT_RESOURCES_VERSION
        try:
            commit = resolve_revision(
                repository(language), f"v{version}", context.hf_cache, context.hf_token
            )
        except TxturesError as error:
            raise TxturesError(f"Stanza models for language {language!r}: {error}") from error
        return {
            "language": language,
            "package": self.spec["package"],
            "processors": list(PROCESSORS),
            "resources_version": version,
            "commit": commit,
        }

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        import stanza

        try:
            stanza.download(
                resolved["language"],
                model_dir=str(directory),
                package=resolved["package"],
                processors=",".join(resolved["processors"]),
                resources_version=resolved["resources_version"],
                model_url=model_url(resolved),
                logging_level="WARN",
            )
        except Exception as error:
            raise TxturesError(
                f"cannot download the Stanza {resolved['language']} models: {error}"
            ) from error
        files = sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*.pt")
            if path.is_file()
        )
        if not files:
            raise TxturesError(f"Stanza downloaded no models for {resolved['language']!r}")
        (directory / FILES_FILE).write_text(
            json.dumps({"files": files, "stanza_version": stanza.__version__}, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Any:
        models = StanzaModels(
            directory=directory,
            language=resolved["language"],
            package=resolved["package"],
            device=device(context.device),
            parser=None,
        )
        models.parser = _pipeline(models, ",".join(resolved["processors"]), tokenize_no_ssplit=True)
        return models
