"""The module contract and the registry of available modules."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class ModuleConfig(BaseModel):
    """Base class for module configs.

    Unknown keys are rejected, and validated configs are immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


class Module:
    """A unit of computation that textures can plan, cache, and run.

    Subclasses set the class attributes below and are added to a `Registry`.

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
    """

    name: ClassVar[str]
    contract_version: ClassVar[int]
    Config: ClassVar[type[ModuleConfig]]
    requires: ClassVar[tuple[str, ...]] = ()
    features: ClassVar[tuple[str, ...]] = ()
    sweeps: ClassVar[Mapping[str, tuple[str, ...]]] = {}


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
        for field_name, field in fields.items():
            if field.is_required():
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
