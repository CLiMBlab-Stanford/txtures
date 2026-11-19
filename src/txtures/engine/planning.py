"""Planning: from a request to a fixed set of identities and missing work.

A request names an input, a workflow, and optionally the feature modules
whose columns the output table should hold. Planning reads the input, expands
the workflow into variants, resolves each variant's resources, and computes
the identity of every variant, item, and the output table. Nothing is
computed while planning, apart from resolving resources (which may contact a
model hub).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from txtures.corpus import Corpus, Document
from txtures.definitions import Column, Definitions, Expansion, Variant
from txtures.environment import run_context
from txtures.errors import DefinitionError, InputError, Problem, TxturesError
from txtures.identity import digest
from txtures.inputs import load_input
from txtures.modules import Module, Resource
from txtures.settings import Settings
from txtures.store import EntryStore


@dataclass(frozen=True)
class PlannedResource:
    """A resource with its resolved spec and identity."""

    name: str
    resource: Resource
    resolved: Mapping[str, Any]
    identity: str

    @property
    def kind(self) -> str:
        return self.resource.kind


@dataclass(frozen=True)
class PlannedVariant:
    """A variant with its module instance, resources, and full identity.

    `identity` covers the variant's config-level digest, its resolved
    resources, and the identities of its upstream variants, so it changes
    whenever anything that affects the variant's results changes.
    """

    variant: Variant
    module: Module
    resources: Mapping[str, PlannedResource]
    upstream: Mapping[str, PlannedVariant]
    identity: str

    @property
    def name(self) -> str:
        return self.variant.module

    def item_identity(self, document: Document) -> str:
        """Identity of this variant's result for one document."""
        return digest({"variant": self.identity, "document": document.digest})


@dataclass(frozen=True)
class Plan:
    """Everything a request needs, with identities fixed before execution.

    `variants` is in dependency order, upstream first. `notes` holds warnings
    that do not prevent running.
    """

    input_name: str
    corpus: Corpus
    workflow: str
    requested: tuple[str, ...]
    expansion: Expansion
    variants: tuple[PlannedVariant, ...]
    columns: tuple[Column, ...]
    table_identity: str
    table_directory: Path
    notes: tuple[str, ...] = field(default=())

    def planned(self, variant: Variant) -> PlannedVariant:
        for planned in self.variants:
            if planned.variant.digest == variant.digest:
                return planned
        raise KeyError(variant.digest)

    def column_variants(self) -> list[tuple[Column, PlannedVariant]]:
        return [(column, self.planned(column.variant)) for column in self.columns]

    def resources(self) -> list[PlannedResource]:
        """Return each distinct resource the plan needs, in first-use order."""
        found: dict[str, PlannedResource] = {}
        for planned in self.variants:
            for resource in planned.resources.values():
                found.setdefault(resource.identity, resource)
        return list(found.values())


def item_kind(module_name: str) -> str:
    """Store kind for a module's items, so each module has its own directory."""
    return module_name


def resource_identity(resource: Resource, resolved: Mapping[str, Any]) -> str:
    return digest({"kind": resource.kind, "version": resource.version, "spec": dict(resolved)})


def table_identity(corpus: Corpus, columns: Sequence[tuple[str, str]]) -> str:
    """Identity of an output table: the input and each column's name and variant identity."""
    return digest({"input": corpus.digest, "columns": [list(pair) for pair in columns]})


def plan_request(
    settings: Settings,
    definitions: Definitions,
    input_name: str,
    workflow: str = "main",
    modules: Sequence[str] | None = None,
    *,
    corpus: Corpus | None = None,
) -> Plan:
    """Plan a request without computing anything.

    `corpus` may be given directly, as `tx.compute` does for in-memory text;
    otherwise the input named `input_name` is read from the inputs directory.

    Raises:
        InputError: The input is invalid.
        DefinitionError: A definition is invalid, or a module cannot process
            the input with its config.
    """
    if corpus is None:
        report = load_input(settings.inputs, input_name)
        if report.corpus is None:
            raise InputError(report.problems)
        corpus = report.corpus
    expansion = definitions.expand(workflow, modules, max_variants=settings.max_variants)
    context = run_context(settings)
    registry = definitions.registry

    planned: dict[str, PlannedVariant] = {}
    resolved_resources: dict[str, PlannedResource] = {}
    problems: list[Problem] = []
    notes: list[str] = []
    ordered: list[PlannedVariant] = []
    for module_name, variants in expansion.variants.items():
        module = registry[module_name]()
        for variant in variants:
            where = f"{module_name} variant {variant.config_id}"
            if variant.labels:
                where += f" ({', '.join(label.value for label in variant.labels)})"
            for message in module.check(variant.config, corpus):
                problems.append(Problem(where, message))
            for message in module.notes(variant.config):
                notes.append(f"{where}: {message}")
            resources: dict[str, PlannedResource] = {}
            for name, resource in module.resources(variant.config).items():
                key = digest({"kind": resource.kind, "spec": resource.spec})
                if key not in resolved_resources:
                    try:
                        resolved = resource.resolve(context)
                    except TxturesError as error:
                        problems.append(Problem(where, str(error)))
                        continue
                    resolved_resources[key] = PlannedResource(
                        name, resource, resolved, resource_identity(resource, resolved)
                    )
                resources[name] = resolved_resources[key]
            upstream = {name: planned[value.digest] for name, value in variant.upstream.items()}
            identity = digest(
                {
                    "variant": variant.digest,
                    "resources": {name: value.identity for name, value in resources.items()},
                    "upstream": {name: value.identity for name, value in upstream.items()},
                }
            )
            entry = PlannedVariant(variant, module, resources, upstream, identity)
            planned[variant.digest] = entry
            ordered.append(entry)
    if problems:
        raise DefinitionError(problems)

    columns = expansion.columns(registry)
    taken = set(corpus.metadata_columns)
    clashes = sorted({column.name for column in columns} & taken)
    if clashes:
        raise DefinitionError(
            [
                Problem(
                    f"input {corpus.name}",
                    f"metadata column {name!r} clashes with a feature column",
                )
                for name in clashes
            ]
        )
    identity = table_identity(
        corpus, [(column.name, planned[column.variant.digest].identity) for column in columns]
    )
    directory = settings.outputs / corpus.name / f"{workflow}-{identity[:12]}"
    return Plan(
        input_name=corpus.name,
        corpus=corpus,
        workflow=workflow,
        requested=expansion.requested,
        expansion=expansion,
        variants=tuple(ordered),
        columns=columns,
        table_identity=identity,
        table_directory=directory,
        notes=tuple(dict.fromkeys(notes)),
    )


@dataclass(frozen=True)
class VariantStatus:
    """How many of a persisted variant's items exist in the cache."""

    variant: PlannedVariant
    present: int
    missing: int


def item_status(plan: Plan, items: EntryStore) -> list[VariantStatus]:
    """Count stored and missing items for every persisted variant in the plan."""
    statuses = []
    for planned in plan.variants:
        if not planned.module.persist:
            continue
        present = sum(
            1
            for document in plan.corpus.documents
            if items.get(item_kind(planned.name), planned.item_identity(document)) is not None
        )
        statuses.append(VariantStatus(planned, present, len(plan.corpus.documents) - present))
    return statuses
