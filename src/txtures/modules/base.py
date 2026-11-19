"""The module contract and the registry of available modules."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from txtures.corpus import Corpus, Document

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class ModuleConfig(BaseModel):
    """Base class for module configs.

    Unknown keys are rejected, and validated configs are immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


@dataclass(frozen=True)
class RunContext:
    """Execution settings passed to modules and resources. They never affect results.

    `hf_token` authenticates with the Hugging Face hub; `None` lets the hub
    library find a token itself.
    """

    device: str = "auto"
    hf_cache: Path | None = None
    hf_token: str | None = field(default=None, repr=False)


class Resource:
    """A text-independent input to a module, such as a model, stored in the resource store.

    A resource is described by a JSON-compatible `spec`. Before planning,
    `resolve` turns the spec into a fully determined form, for example by
    replacing a missing model revision with the current commit hash. The
    resource's identity is the digest of its kind, version, and resolved spec.

    Attributes:
        kind: Unique resource kind, used as a directory name in the store.
        version: Increment whenever the same resolved spec would build or
            load differently.
    """

    kind: ClassVar[str]
    version: ClassVar[int]

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    def resolve(self, context: RunContext) -> dict[str, Any]:
        """Return the fully determined spec. The default returns `spec` unchanged."""
        return dict(self.spec)

    def build(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> None:
        """Write the resource's files into `directory`. The default writes nothing."""

    def load(self, resolved: Mapping[str, Any], directory: Path, context: RunContext) -> Any:
        """Return the in-memory object that modules receive."""
        raise NotImplementedError


@dataclass(frozen=True)
class Item:
    """What a module receives to compute one document's result."""

    document: Document
    upstream: Mapping[str, Any]
    """Results of the upstream modules for the same document, by module name."""
    resources: Mapping[str, Any]
    """Loaded resources, by the names the module gave them."""


class Module:
    """A unit of computation that txtures can plan, cache, and run.

    Subclasses set the class attributes below, implement `run` or
    `run_batch`, and are added to a `Registry`. Every module processes one
    document at a time; its result for a document may be any value its
    `save` and `load` can store.

    Attributes:
        name: Unique module name, used as a directory name in definitions
            stores and as the key in workflows.
        contract_version: Output-contract version. Increment it whenever a
            change can alter what the module computes or stores.
        Config: The module's config schema. Every field needs a default.
        requires: Names of upstream modules whose outputs this module reads.
            The workflow chooses which config of each one is used.
        features: Names of the feature columns the module adds to output
            tables. Modules that only feed other modules have none.
        sweeps: Sweepable parameters. Each key is a parameter that may be
            given as a list or labeled mapping of values. Its tuple lists the
            config fields one sweep entry may set, starting with the key
            itself; for example, a `model` entry may also set `revision`.
            Order determines the order in which sweeps expand.
        persist: Whether results are kept in the cache. Cheap modules
            set this to false and are recomputed whenever they are needed.
        batch_size: Maximum number of documents passed to one `run_batch` call.
    """

    name: ClassVar[str]
    contract_version: ClassVar[int]
    Config: ClassVar[type[ModuleConfig]]
    requires: ClassVar[tuple[str, ...]] = ()
    features: ClassVar[tuple[str, ...]] = ()
    sweeps: ClassVar[Mapping[str, tuple[str, ...]]] = {}
    persist: ClassVar[bool] = True
    batch_size: ClassVar[int] = 1

    def resources(self, config: ModuleConfig) -> Mapping[str, Resource]:
        """Return the resources a config needs, by name. The default needs none."""
        return {}

    def check(self, config: ModuleConfig, corpus: Corpus) -> list[str]:
        """Return reasons the config cannot process `corpus`. The default finds none."""
        return []

    def notes(self, config: ModuleConfig) -> list[str]:
        """Return warnings about a config that does not prevent running it."""
        return []

    def run(self, config: ModuleConfig, item: Item, context: RunContext) -> Any:
        """Compute the result for one document."""
        raise NotImplementedError(f"module {self.name} implements neither run nor run_batch")

    def run_batch(
        self, config: ModuleConfig, items: Sequence[Item], context: RunContext
    ) -> list[Any]:
        """Compute results for several documents. The default calls `run` for each."""
        return [self.run(config, item, context) for item in items]

    def save(self, result: Any, directory: Path) -> None:
        """Store a result in `directory`.

        The default stores a mapping of names to arrays as `data.npz`, without
        pickling.
        """
        arrays = {name: np.asarray(value) for name, value in result.items()}
        np.savez(directory / "data.npz", **arrays)

    def load(self, directory: Path) -> Any:
        """Read a result written by `save`."""
        with np.load(directory / "data.npz", allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def columns(self, config: ModuleConfig, document: Document, result: Any) -> Mapping[str, Any]:
        """Return one sequence per feature, each with one value per region of `document`."""
        return {}


class Registry:
    """The set of modules available to definitions and workflows."""

    def __init__(self, modules: Iterable[type[Module]] = ()):
        self._modules: dict[str, type[Module]] = {}
        for module in modules:
            self.register(module)

    def register(self, module: type[Module]) -> type[Module]:
        """Add a module class, checking its declarations. Usable as a decorator.

        Raises:
            ValueError: The module's declarations are inconsistent or its name
                is already registered.
        """
        name = getattr(module, "name", None)
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError(f"module name {name!r} must match {NAME_PATTERN.pattern}")
        if name in self._modules:
            raise ValueError(f"module {name!r} is already registered")
        if not isinstance(getattr(module, "contract_version", None), int):
            raise ValueError(f"module {name!r} must set an integer contract_version")
        config = getattr(module, "Config", None)
        if not (isinstance(config, type) and issubclass(config, ModuleConfig)):
            raise ValueError(f"module {name!r} must set Config to a ModuleConfig subclass")
        fields = config.model_fields
        for field_name, info in fields.items():
            if info.is_required():
                raise ValueError(f"module {name!r} config field {field_name!r} needs a default")
        grouped: set[str] = set()
        for key, group in module.sweeps.items():
            if not group or group[0] != key:
                raise ValueError(f"module {name!r} sweep {key!r} must list {key!r} first")
            for field_name in group:
                if field_name not in fields:
                    raise ValueError(
                        f"module {name!r} sweep {key!r} names unknown field {field_name!r}"
                    )
                if field_name in grouped:
                    raise ValueError(
                        f"module {name!r} field {field_name!r} belongs to more than one sweep"
                    )
                grouped.add(field_name)
        if not isinstance(module.batch_size, int) or module.batch_size < 1:
            raise ValueError(f"module {name!r} batch_size must be a positive integer")
        for feature in module.features:
            if not NAME_PATTERN.fullmatch(feature):
                raise ValueError(
                    f"module {name!r} feature {feature!r} must match {NAME_PATTERN.pattern}"
                )
        self._modules[name] = module
        return module

    def __contains__(self, name: object) -> bool:
        return name in self._modules

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __getitem__(self, name: str) -> type[Module]:
        return self._modules[name]

    def closure(self, names: Iterable[str]) -> list[str]:
        """Return `names` and every module they require, upstream before downstream.

        Raises:
            KeyError: A name, or a module it requires, is not registered.
            ValueError: The requirements contain a cycle.
        """
        order: list[str] = []
        state: dict[str, str] = {}

        def visit(name: str, path: tuple[str, ...]) -> None:
            if state.get(name) == "done":
                return
            if state.get(name) == "visiting":
                cycle = " -> ".join((*path, name))
                raise ValueError(f"module requirements form a cycle: {cycle}")
            if name not in self._modules:
                raise KeyError(name)
            state[name] = "visiting"
            for upstream in self._modules[name].requires:
                visit(upstream, (*path, name))
            state[name] = "done"
            order.append(name)

        for name in names:
            visit(name, ())
        return order
