"""Resolution of configs and workflows into variants.

Resolution happens in three stages:

1. `extends` chains are flattened. A config or workflow without `extends`
   starts from the packaged defaults: a module's schema defaults, or, for a
   workflow, the `main` config of every registered module. `extends: main` starts from the store's
   `main` definition, which is itself a partial override of the packaged one.
2. Each config body is split into fixed values and sweeps, and every
   combination is validated against the module's schema.
3. A workflow request is expanded into variants: one fully resolved,
   sweep-free config per combination of upstream variants, config IDs, and
   sweep values. Each variant has a digest and a tuple of labels that name it
   in output columns.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from txtures.definitions.store import (
    ID_PATTERN,
    Definition,
    StoreContents,
    read_store,
)
from txtures.errors import DefinitionError, Problem
from txtures.identity import digest
from txtures.modules import REGISTRY, Module, ModuleConfig, Registry

LABEL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""Pattern for variant labels, which become parts of column names."""

MAIN = "main"
EXTENDS = "extends"


# Labels and sweeps


@dataclass(frozen=True)
class Label:
    """The label of one swept value in a variant's lineage.

    `source` is the swept parameter, or `"config"` when the label is a config
    ID chosen from a workflow list.
    """

    module: str
    source: str
    value: str


def derive_label(key: str, value: Any) -> str:
    """Derive a column label from a swept value.

    Strings keep their last `/`-separated component with characters other than
    letters and digits removed; the parameter name is prefixed when the result
    would not start with a letter. Numbers, Booleans, and null become the
    parameter name followed by the value with `.` and `-` removed.

    Raises:
        ValueError: No valid label can be derived.
    """
    if isinstance(value, str):
        label = re.sub(r"[^A-Za-z0-9]", "", value.rsplit("/", 1)[-1])
        if not label[:1].isalpha():
            label = f"{key}{label}"
    elif value is None or isinstance(value, bool | int | float):
        label = f"{key}{value}".replace(".", "").replace("-", "")
    else:
        raise ValueError(f"cannot derive a label for {key} value {value!r}; give explicit labels")
    if not LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"cannot derive a label for {key} value {value!r}; give explicit labels")
    return label


@dataclass(frozen=True)
class Sweep:
    """One swept parameter: labeled entries, each setting fields of its group."""

    key: str
    entries: tuple[tuple[str, Mapping[str, Any]], ...]


def _sweep_entry(key: str, group: Sequence[str], value: Any) -> dict[str, Any]:
    """Turn one sweep entry into the fields it sets."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(group))
        if unknown:
            raise ValueError(
                f"{key} entries may set only {', '.join(group)}; got {', '.join(unknown)}"
            )
        if key not in value:
            raise ValueError(f"{key} entry {dict(value)!r} must set {key}")
        return dict(value)
    if isinstance(value, list):
        raise ValueError(f"{key} entries cannot be lists")
    return {key: value}


def parse_sweep(key: str, group: Sequence[str], value: list | Mapping) -> Sweep:
    """Parse a list or labeled mapping given for a sweepable parameter.

    Raises:
        ValueError: The sweep is empty, an entry is malformed, or labels are
            invalid or repeated.
    """
    if not value:
        raise ValueError(f"{key} sweep is empty")
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        for label, entry in value.items():
            if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label):
                raise ValueError(f"{key} label {label!r} must match {LABEL_PATTERN.pattern}")
            entries.append((label, _sweep_entry(key, group, entry)))
    else:
        for entry in value:
            fields = _sweep_entry(key, group, entry)
            entries.append((derive_label(key, fields[key]), fields))
    labels = [label for label, _ in entries]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ValueError(
            f"{key} sweep repeats label {', '.join(repeated)}; give explicit labels as a mapping"
        )
    return Sweep(key, tuple(entries))


@dataclass(frozen=True)
class ConfigVariant:
    """One combination of a config's sweep values, validated against its schema."""

    config: ModuleConfig
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class ResolvedConfig:
    """A config with `extends` flattened, split into fixed values and sweeps."""

    module: type[Module]
    id: str
    fixed: Mapping[str, Any]
    sweeps: tuple[Sweep, ...]
    variants: tuple[ConfigVariant, ...]


def _format_validation_error(error: ValidationError) -> list[str]:
    messages = []
    for detail in error.errors():
        where = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{where}: {detail['msg']}" if where else detail["msg"])
    return messages


def _unsweepable_hints(module: type[Module], fixed: Mapping[str, Any], location: str):
    """Explain list values that failed validation because their parameter is not sweepable."""
    sweepable = ", ".join(module.sweeps) or "none"
    return [
        Problem(location, f"{key} cannot be swept (sweepable parameters: {sweepable})")
        for key, value in fixed.items()
        if isinstance(value, list) and key not in module.sweeps
    ]


def parse_config(module: type[Module], config_id: str, body: Mapping[str, Any], location: str):
    """Split a flattened config body into fixed values and sweeps, and validate every variant.

    Raises:
        DefinitionError: The body is invalid for the module's schema.
    """
    problems: list[Problem] = []
    fixed: dict[str, Any] = {}
    sweeps: dict[str, Sweep] = {}
    for key, value in body.items():
        if key in module.sweeps and isinstance(value, list | Mapping):
            try:
                sweeps[key] = parse_sweep(key, module.sweeps[key], value)
            except ValueError as error:
                problems.append(Problem(location, str(error)))
        else:
            fixed[key] = value
    for key in sweeps:
        conflicts = [name for name in module.sweeps[key][1:] if name in fixed]
        if conflicts:
            problems.append(
                Problem(
                    location,
                    f"{', '.join(conflicts)} cannot be set alongside a {key} sweep; "
                    f"set it in each {key} entry",
                )
            )
    if problems:
        raise DefinitionError(problems)

    ordered = [sweeps[key] for key in module.sweeps if key in sweeps]
    variants: list[ConfigVariant] = []
    for combination in itertools.product(*(sweep.entries for sweep in ordered)):
        values = dict(fixed)
        labels: list[Label] = []
        for sweep, (label, fields) in zip(ordered, combination, strict=True):
            values.update(fields)
            if len(sweep.entries) > 1:
                labels.append(Label(module.name, sweep.key, label))
        try:
            config = module.Config.model_validate(values)
        except ValidationError as error:
            suffix = f" (variant {', '.join(label for label, _ in combination)})" if ordered else ""
            for message in _format_validation_error(error):
                problems.append(Problem(location, f"{message}{suffix}"))
            problems.extend(_unsweepable_hints(module, fixed, location))
            continue
        variants.append(ConfigVariant(config, tuple(labels)))
    if problems:
        raise DefinitionError(problems)
    return ResolvedConfig(module, config_id, fixed, tuple(ordered), tuple(variants))


# Workflow expansion


@dataclass(frozen=True)
class Variant:
    """One fully resolved, sweep-free run of a module within a workflow request.

    `digest` identifies the variant's computation: the module, its contract
    version, its resolved config, and the digests of its upstream variants.
    Config IDs and labels do not enter it.
    """

    module: str
    config_id: str
    config: ModuleConfig
    labels: tuple[Label, ...]
    upstream: Mapping[str, Variant]
    digest: str
    lineage: Mapping[str, str]
    """Digest of this variant and of every ancestor variant, by module."""


@dataclass(frozen=True)
class Column:
    """One feature column of an output table and the variant that produces it."""

    name: str
    feature: str
    variant: Variant


@dataclass(frozen=True)
class Expansion:
    """The variants a workflow request needs, by module in dependency order."""

    workflow: str
    requested: tuple[str, ...]
    variants: Mapping[str, tuple[Variant, ...]]

    def columns(self, registry: Registry = REGISTRY) -> tuple[Column, ...]:
        """Return the feature columns of the requested modules, in request order."""
        columns = []
        for module in self.requested:
            for variant in self.variants[module]:
                suffix = "".join(f"_{label.value}" for label in variant.labels)
                for feature in registry[module].features:
                    columns.append(Column(f"{feature}{suffix}", feature, variant))
        return tuple(columns)


def variant_digest(module: type[Module], config: ModuleConfig, upstream: Mapping[str, Variant]):
    """Compute the digest that identifies one variant's computation."""
    return digest(
        {
            "module": module.name,
            "contract_version": module.contract_version,
            "config": config.model_dump(mode="json"),
            "upstream": {name: variant.digest for name, variant in upstream.items()},
        }
    )


def _merge_lineage(variants: Iterable[Variant]) -> dict[str, str] | None:
    """Merge ancestor digests, or return `None` if two variants disagree on an ancestor."""
    merged: dict[str, str] = {}
    for variant in variants:
        for module, value in variant.lineage.items():
            if merged.setdefault(module, value) != value:
                return None
    return merged


def _merge_labels(groups: Iterable[Iterable[Label]]) -> tuple[Label, ...]:
    """Concatenate labels in order, keeping the first label from each source."""
    seen: dict[tuple[str, str], Label] = {}
    for group in groups:
        for label in group:
            seen.setdefault((label.module, label.source), label)
    return tuple(seen.values())


# Definitions


class Definitions:
    """Configs and workflows from a definitions store, merged with packaged defaults."""

    def __init__(self, contents: StoreContents, registry: Registry = REGISTRY):
        self.contents = contents
        self.registry = registry

    @classmethod
    def load(cls, root: Path, registry: Registry = REGISTRY) -> Definitions:
        """Read the store at `root`. Layout problems are kept for `validate`.

        Raises:
            TxturesError: `root` is not a directory.
        """
        return cls(read_store(root, registry), registry)

    @classmethod
    def packaged(cls, registry: Registry = REGISTRY) -> Definitions:
        """Return the packaged definitions alone, without a store."""
        return cls(StoreContents(None), registry)

    # Flattening

    def _flatten(
        self,
        kind: str,
        definitions: Mapping[str, Definition],
        packaged: Mapping[str, Any],
        definition_id: str,
        chain: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Flatten the `extends` chain of one definition into a single body."""
        if definition_id in chain:
            cycle = " -> ".join((*chain, definition_id))
            raise DefinitionError(
                [Problem(f"{kind} {chain[0]}", f"extends forms a cycle: {cycle}")]
            )
        definition = definitions.get(definition_id)
        if definition_id == MAIN:
            body = dict(definition.body) if definition else {}
            if EXTENDS in body:
                raise DefinitionError([Problem(definition.location, "main cannot use extends")])
            return {**packaged, **body}
        if definition is None:
            where = f"{kind} {chain[-1]}" if chain else kind
            raise DefinitionError([Problem(where, f"no {kind} named {definition_id!r}")])
        body = dict(definition.body)
        parent = body.pop(EXTENDS, None)
        if parent is None:
            return {**packaged, **body}
        if not isinstance(parent, str):
            raise DefinitionError(
                [Problem(definition.location, "extends must name one definition")]
            )
        return {
            **self._flatten(kind, definitions, packaged, parent, (*chain, definition_id)),
            **body,
        }

    def _definition_location(self, definitions: Mapping[str, Definition], key: str, kind: str):
        definition = definitions.get(key)
        return definition.location if definition else f"{kind} {key}"

    def config(self, module_name: str, config_id: str) -> ResolvedConfig:
        """Resolve one config of one module.

        Raises:
            DefinitionError: The module is unknown or the config is missing or invalid.
        """
        if module_name not in self.registry:
            raise DefinitionError(
                [Problem(f"config {config_id}", f"unknown module {module_name!r}")]
            )
        module = self.registry[module_name]
        definitions = self.contents.configs.get(module_name, {})
        kind = f"{module_name} config"
        body = self._flatten(kind, definitions, {}, config_id)
        location = self._definition_location(definitions, config_id, kind)
        return parse_config(module, config_id, body, location)

    def workflow(self, workflow_id: str) -> dict[str, tuple[str, ...]]:
        """Resolve a workflow into the config IDs it selects for each module.

        Raises:
            DefinitionError: The workflow is missing or malformed, or names an
                unknown module.
        """
        packaged = {name: MAIN for name in self.registry}
        body = self._flatten("workflow", self.contents.workflows, packaged, workflow_id)
        location = self._definition_location(self.contents.workflows, workflow_id, "workflow")
        problems: list[Problem] = []
        selected: dict[str, tuple[str, ...]] = {}
        for module_name, value in body.items():
            if module_name not in self.registry:
                problems.append(Problem(location, f"unknown module {module_name!r}"))
                continue
            ids = [value] if isinstance(value, str) else value
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(item, str) and ID_PATTERN.fullmatch(item) for item in ids)
            ):
                problems.append(
                    Problem(
                        location, f"{module_name} must name a config ID or a list of config IDs"
                    )
                )
                continue
            if len(set(ids)) != len(ids):
                problems.append(Problem(location, f"{module_name} lists a config ID twice"))
                continue
            selected[module_name] = tuple(ids)
        if problems:
            raise DefinitionError(problems)
        return selected

    # Expansion

    def expand(
        self,
        workflow_id: str = MAIN,
        modules: Sequence[str] | None = None,
        *,
        max_variants: int = 64,
    ) -> Expansion:
        """Expand a workflow request into variants.

        `modules` names the feature modules whose columns are requested. When
        it is `None`, every feature module the workflow selects is requested.

        Variants of one module are ordered with upstream variants varying
        slowest, then the workflow's config IDs, then each sweep in the order
        the module declares its sweepable parameters.

        Raises:
            DefinitionError: A definition is invalid, the workflow does not
                select a config for a needed module, a module expands to more
                than `max_variants` variants, or two columns would share a name.
        """
        selected = self.workflow(workflow_id)
        location = f"workflow {workflow_id}"
        if modules is None:
            requested = [name for name in selected if self.registry[name].features]
        else:
            unknown = [name for name in modules if name not in self.registry]
            if unknown:
                raise DefinitionError(
                    [Problem(location, f"unknown module {name!r}") for name in unknown]
                )
            no_features = [name for name in modules if not self.registry[name].features]
            if no_features:
                raise DefinitionError(
                    [
                        Problem(location, f"module {name!r} has no feature columns")
                        for name in no_features
                    ]
                )
            requested = list(dict.fromkeys(modules))
        try:
            order = self.registry.closure(requested)
        except ValueError as error:
            raise DefinitionError([Problem(location, str(error))]) from error
        missing = [name for name in order if name not in selected]
        if missing:
            raise DefinitionError(
                [
                    Problem(location, f"does not select a config for {name}, which is needed")
                    for name in missing
                ]
            )

        problems: list[Problem] = []
        configs: dict[str, list[ResolvedConfig]] = {}
        for name in order:
            configs[name] = []
            for config_id in selected[name]:
                if config_id != MAIN and config_id not in self.contents.configs.get(name, {}):
                    workflow_location = self._definition_location(
                        self.contents.workflows, workflow_id, "workflow"
                    )
                    problems.append(
                        Problem(workflow_location, f"no {name} config named {config_id!r}")
                    )
                    continue
                try:
                    configs[name].append(self.config(name, config_id))
                except DefinitionError as error:
                    problems.extend(error.problems)
        if problems:
            raise DefinitionError(problems)

        variants: dict[str, tuple[Variant, ...]] = {}
        for name in order:
            module = self.registry[name]
            label_ids = len(configs[name]) > 1
            upstream_choices = [variants[upstream] for upstream in module.requires]
            found: list[Variant] = []
            for upstream in itertools.product(*upstream_choices):
                lineage = _merge_lineage(upstream)
                if lineage is None:
                    continue
                by_module = {variant.module: variant for variant in upstream}
                for resolved in configs[name]:
                    id_label = (Label(name, "config", resolved.id),) if label_ids else ()
                    for config_variant in resolved.variants:
                        value = variant_digest(module, config_variant.config, by_module)
                        labels = _merge_labels(
                            [
                                *(variant.labels for variant in upstream),
                                id_label,
                                config_variant.labels,
                            ]
                        )
                        found.append(
                            Variant(
                                module=name,
                                config_id=resolved.id,
                                config=config_variant.config,
                                labels=labels,
                                upstream=by_module,
                                digest=value,
                                lineage={**lineage, name: value},
                            )
                        )
            if len(found) > max_variants:
                problems.append(
                    Problem(
                        location,
                        f"{name} expands to {len(found)} variants, more than "
                        f"execution.max_variants ({max_variants})",
                    )
                )
            variants[name] = tuple(found)
        if problems:
            raise DefinitionError(problems)

        expansion = Expansion(workflow_id, tuple(requested), variants)
        names = [column.name for column in expansion.columns(self.registry)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise DefinitionError(
                [
                    Problem(location, f"two variants produce column {name!r}; give distinct labels")
                    for name in repeated
                ]
            )
        return expansion

    # Validation

    def validate(self, *, max_variants: int = 64) -> list[Problem]:
        """Check every definition, including unused ones, and return all problems found."""
        problems = list(self.contents.problems)
        for module_name in self.registry:
            ids = {MAIN, *self.contents.configs.get(module_name, {})}
            for config_id in sorted(ids):
                try:
                    self.config(module_name, config_id)
                except DefinitionError as error:
                    problems.extend(error.problems)
        workflow_ids = {MAIN, *self.contents.workflows}
        for workflow_id in sorted(workflow_ids):
            try:
                self.expand(workflow_id, max_variants=max_variants)
            except DefinitionError as error:
                problems.extend(error.problems)
        return list(dict.fromkeys(problems))
