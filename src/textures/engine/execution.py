"""Local execution of a plan.

Execution builds missing resources, then computes missing items for every
persisted variant in dependency order, in batches of the module's
`batch_size`. Each item is locked while it is computed, so concurrent runs
that need the same item compute it once. Results of modules that are not
persisted are computed when they are first needed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from textures.corpus import Document
from textures.engine.planning import Plan, PlannedResource, PlannedVariant, item_kind
from textures.errors import TexturesError
from textures.modules import Item, RunContext
from textures.store import EntryStore

Progress = Callable[[str], None]


@dataclass
class VariantCounts:
    """How many items of one variant were computed and how many were reused."""

    computed: int = 0
    reused: int = 0


class Executor:
    """Runs a plan against the item and resource stores on this machine."""

    def __init__(
        self,
        plan: Plan,
        items: EntryStore,
        resources: EntryStore,
        context: RunContext,
        progress: Progress | None = None,
    ):
        self.plan = plan
        self.items = items
        self.resource_store = resources
        self.context = context
        self.progress = progress or (lambda message: None)
        self.counts: dict[str, VariantCounts] = {
            planned.identity: VariantCounts() for planned in plan.variants
        }
        self._loaded: dict[str, Any] = {}
        self._memo: dict[tuple[str, str], Any] = {}

    # Resources

    def prepare_resources(self) -> None:
        """Build every resource the plan needs that the resource store lacks."""
        for resource in self.plan.resources():
            if self.resource_store.get(resource.kind, resource.identity) is not None:
                continue
            with self.resource_store.lock(resource.kind, resource.identity):
                if self.resource_store.get(resource.kind, resource.identity) is not None:
                    continue
                self.progress(f"building resource {resource.kind} {resource.identity[:12]}")
                self.resource_store.publish(
                    resource.kind,
                    resource.identity,
                    lambda directory, r=resource: r.resource.build(
                        r.resolved, directory, self.context
                    ),
                    {"spec": dict(resource.resolved), "version": resource.resource.version},
                )

    def _resource(self, resource: PlannedResource) -> Any:
        if resource.identity not in self._loaded:
            directory = self.resource_store.get(resource.kind, resource.identity)
            if directory is None:
                raise TexturesError(f"resource {resource.kind} {resource.identity} is missing")
            self._loaded[resource.identity] = resource.resource.load(
                resource.resolved, directory, self.context
            )
        return self._loaded[resource.identity]

    # Items

    def _item(self, planned: PlannedVariant, document: Document) -> Item:
        return Item(
            document=document,
            upstream={
                name: self.result(upstream, document) for name, upstream in planned.upstream.items()
            },
            resources={name: self._resource(r) for name, r in planned.resources.items()},
        )

    def _compute(self, planned: PlannedVariant, documents: Sequence[Document]) -> list[Any]:
        items = [self._item(planned, document) for document in documents]
        results = planned.module.run_batch(planned.variant.config, items, self.context)
        if len(results) != len(documents):
            raise TexturesError(
                f"module {planned.name} returned {len(results)} results for "
                f"{len(documents)} documents"
            )
        return list(results)

    def _publish(self, planned: PlannedVariant, document: Document, result: Any) -> None:
        identity = planned.item_identity(document)
        manifest = {
            "module": planned.name,
            "contract_version": planned.module.contract_version,
            "variant": planned.identity,
            "config": planned.variant.config.model_dump(mode="json"),
            "document": document.digest,
            "resources": {name: r.identity for name, r in planned.resources.items()},
            "upstream": {
                name: upstream.item_identity(document)
                for name, upstream in planned.upstream.items()
            },
        }
        self.items.publish(
            item_kind(planned.name),
            identity,
            lambda directory: planned.module.save(result, directory),
            manifest,
        )

    def _compute_and_publish(self, planned: PlannedVariant, documents: Sequence[Document]) -> None:
        for document, result in zip(documents, self._compute(planned, documents), strict=True):
            self._publish(planned, document, result)
        self.counts[planned.identity].computed += len(documents)
        self._memo.clear()

    def compute_items(self) -> None:
        """Compute every missing item of every persisted variant.

        A loaded resource, such as a model in GPU memory, is released once no
        later variant needs it.
        """
        persisted = [planned for planned in self.plan.variants if planned.module.persist]
        for index, planned in enumerate(persisted):
            self._compute_variant(planned)
            needed = {
                r.identity for later in persisted[index + 1 :] for r in later.resources.values()
            }
            for identity in list(self._loaded):
                if identity not in needed:
                    del self._loaded[identity]

    def _compute_variant(self, planned: PlannedVariant) -> None:
        documents = self.plan.corpus.documents
        kind = item_kind(planned.name)
        counts = self.counts[planned.identity]
        unique: dict[str, Document] = {}
        for document in documents:
            unique.setdefault(planned.item_identity(document), document)
        missing = [d for i, d in unique.items() if self.items.get(kind, i) is None]
        counts.reused += len(unique) - len(missing)
        if not missing:
            return
        self.progress(f"{_describe(planned)}: computing {len(missing)} of {len(unique)} items")
        locks = _HeldLocks(self.items, kind)
        waiting = [d for d in missing if not locks.acquire(planned.item_identity(d))]
        try:
            todo = []
            for document in missing:
                identity = planned.item_identity(document)
                if not locks.holds(identity):
                    continue
                if self.items.get(kind, identity) is None:
                    todo.append(document)
                else:
                    counts.reused += 1
                    locks.release(identity)
            size = planned.module.batch_size
            for start in range(0, len(todo), size):
                batch = todo[start : start + size]
                self._compute_and_publish(planned, batch)
                for document in batch:
                    locks.release(planned.item_identity(document))
        finally:
            locks.release_all()
        for document in waiting:
            identity = planned.item_identity(document)
            with self.items.lock(kind, identity):
                if self.items.get(kind, identity) is None:
                    self._compute_and_publish(planned, [document])
                else:
                    counts.reused += 1

    def result(self, planned: PlannedVariant, document: Document) -> Any:
        """Return a variant's result for one document, loading or computing it."""
        key = (planned.identity, document.digest)
        if key in self._memo:
            return self._memo[key]
        if planned.module.persist:
            kind = item_kind(planned.name)
            identity = planned.item_identity(document)
            directory = self.items.get(kind, identity)
            if directory is None:
                with self.items.lock(kind, identity):
                    if self.items.get(kind, identity) is None:
                        self._compute_and_publish(planned, [document])
                directory = self.items.get(kind, identity)
                if directory is None:
                    raise TexturesError(f"item {identity} of {planned.name} is missing")
            value = planned.module.load(directory)
        else:
            (value,) = self._compute(planned, [document])
        self._memo[key] = value
        return value

    def forget(self) -> None:
        """Drop results held in memory; stored results stay in the item store."""
        self._memo.clear()


class _HeldLocks:
    """Item locks held by this run. Each is released exactly once, so a lock that
    another process takes after this run releases it is never removed."""

    def __init__(self, items: EntryStore, kind: str):
        self.items = items
        self.kind = kind
        self._held: set[str] = set()

    def acquire(self, identity: str) -> bool:
        if self.items.try_lock(self.kind, identity):
            self._held.add(identity)
            return True
        return False

    def holds(self, identity: str) -> bool:
        return identity in self._held

    def release(self, identity: str) -> None:
        if identity in self._held:
            self.items.unlock(self.kind, identity)
            self._held.discard(identity)

    def release_all(self) -> None:
        for identity in list(self._held):
            self.release(identity)


def _describe(planned: PlannedVariant) -> str:
    labels = ", ".join(label.value for label in planned.variant.labels)
    return f"{planned.name} [{labels or planned.variant.config_id}]"
