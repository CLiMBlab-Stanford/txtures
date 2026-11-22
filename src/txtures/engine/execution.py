"""Local execution of a plan.

Execution builds missing resources, then computes missing items for every
persisted variant in dependency order, in batches of the module's
`batch_size`. Each item is locked while it is computed, so concurrent runs
that need the same item compute it once. Results of modules that are not
persisted are computed when they are first needed.

Several workers can share one plan: worker `k` of `n` first computes the
items whose position among the variant's distinct documents is `k` modulo
`n`, then waits for or computes the rest, so a worker that fails does not
leave items missing.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from txtures.corpus import Document
from txtures.engine.planning import Plan, PlannedResource, PlannedVariant, item_kind
from txtures.errors import TxturesError
from txtures.modules import Item, RunContext
from txtures.store import EntryStore

Progress = Callable[[str], None]


@dataclass
class VariantCounts:
    """How many items of one variant were computed and how many were reused."""

    computed: int = 0
    reused: int = 0


def load_resource(
    store: EntryStore,
    resource: PlannedResource,
    context: RunContext,
    loaded: dict[str, Any] | None = None,
) -> Any:
    """Load a built resource, and first its dependencies, passed to its `load`
    as `dependencies` when it has any. `loaded` caches loaded resources by
    identity.

    Raises:
        TxturesError: The resource or a dependency is not in the store.
    """
    loaded = {} if loaded is None else loaded
    if resource.identity in loaded:
        return loaded[resource.identity]
    dependencies = {
        name: load_resource(store, dependency, context, loaded)
        for name, dependency in resource.dependencies.items()
    }
    directory = store.get(resource.kind, resource.identity)
    if directory is None:
        raise TxturesError(f"resource {resource.kind} {resource.identity} is missing")
    if dependencies:
        value = resource.resource.load(
            resource.resolved, directory, context, dependencies=dependencies
        )
    else:
        value = resource.resource.load(resource.resolved, directory, context)
    loaded[resource.identity] = value
    return value


def build_resource(
    store: EntryStore,
    resource: PlannedResource,
    context: RunContext,
    progress: Callable[[str], None],
) -> bool:
    """Build `resource` into `store` unless it is there; return whether this call built it.

    Missing dependencies are built first. The build holds the entry's lock,
    so concurrent callers build it once.
    """
    for dependency in resource.dependencies.values():
        build_resource(store, dependency, context, progress)
    if store.get(resource.kind, resource.identity) is not None:
        return False
    with store.lock(resource.kind, resource.identity):
        if store.get(resource.kind, resource.identity) is not None:
            return False
        progress(f"building resource {resource.kind} {resource.identity[:12]}")
        loaded = {
            name: load_resource(store, dependency, context)
            for name, dependency in resource.dependencies.items()
        }

        def build(directory) -> None:
            if loaded:
                resource.resource.build(resource.resolved, directory, context, dependencies=loaded)
            else:
                resource.resource.build(resource.resolved, directory, context)

        manifest = {"spec": dict(resource.resolved), "version": resource.resource.version}
        if resource.dependencies:
            manifest["dependencies"] = {n: d.identity for n, d in resource.dependencies.items()}
        store.publish(resource.kind, resource.identity, build, manifest)
    return True


class Executor:
    """Runs a plan against the item and resource stores on this machine.

    `loaded` may be a dictionary shared by the executors of several plans,
    so that a model loaded for one is reused by the next, and `keep` holds
    the identities of resources a later plan needs, which are not released
    when this plan is done with them.
    """

    def __init__(
        self,
        plan: Plan,
        items: EntryStore,
        resources: EntryStore,
        context: RunContext,
        progress: Progress | None = None,
        shard: tuple[int, int] = (0, 1),
        loaded: dict[str, Any] | None = None,
        keep: Collection[str] = (),
    ):
        index, count = shard
        if not 0 <= index < count:
            raise ValueError(f"shard {index} of {count} is out of range")
        self.shard = shard
        self.plan = plan
        self.items = items
        self.resource_store = resources
        self.context = context
        self.progress = progress or (lambda message: None)
        self.counts: dict[str, VariantCounts] = {
            planned.identity: VariantCounts() for planned in plan.variants
        }
        self._loaded: dict[str, Any] = {} if loaded is None else loaded
        self._keep = frozenset(keep)
        self._memo: dict[tuple[str, str], Any] = {}

    # Resources

    def prepare_resources(self) -> None:
        """Build every resource the plan needs that the resource store lacks."""
        for resource in self.plan.resources():
            build_resource(self.resource_store, resource, self.context, self.progress)

    def _resource(self, resource: PlannedResource) -> Any:
        return load_resource(self.resource_store, resource, self.context, self._loaded)

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
            raise TxturesError(
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

    def compute_items(
        self,
        only: Collection[str] | None = None,
        skip: Collection[str] = (),
        wait: bool = True,
    ) -> None:
        """Compute every missing item of every persisted variant.

        `only`, if given, limits this to the variants with these identities,
        and `skip` leaves out others. With `wait` false, items another
        process holds are left to it instead of waited for, so a worker
        stops as soon as nothing is left for it to take.

        A loaded resource, such as a model in GPU memory, is released once no
        later variant needs it.
        """
        persisted = [
            planned
            for planned in self.plan.variants
            if planned.module.persist
            and (only is None or planned.identity in only)
            and planned.identity not in skip
        ]
        for index, planned in enumerate(persisted):
            self._compute_variant(planned, wait)
            needed = self._keep | {
                r.identity for later in persisted[index + 1 :] for r in later.resources.values()
            }
            for identity in list(self._loaded):
                if identity not in needed:
                    del self._loaded[identity]

    def _compute_variant(self, planned: PlannedVariant, wait: bool = True) -> None:
        kind = item_kind(planned.name)
        counts = self.counts[planned.identity]
        unique: dict[str, Document] = {}
        for document in self.plan.corpus.documents:
            unique.setdefault(planned.item_identity(document), document)
        missing = {i: d for i, d in unique.items() if self.items.get(kind, i) is None}
        counts.reused += len(unique) - len(missing)
        if not missing:
            return
        self.progress(f"{_describe(planned)}: computing {len(missing)} of {len(unique)} items")
        index, count = self.shard
        own = {i for position, i in enumerate(unique) if position % count == index}
        ordered = [i for i in missing if i in own] + [i for i in missing if i not in own]
        held_elsewhere = self._compute_unlocked(planned, [missing[i] for i in ordered])
        if not wait:
            return
        for document in held_elsewhere:
            identity = planned.item_identity(document)
            with self.items.lock(kind, identity):
                if self.items.get(kind, identity) is None:
                    self._compute_and_publish(planned, [document])
                else:
                    counts.reused += 1

    def _compute_unlocked(
        self, planned: PlannedVariant, documents: Sequence[Document]
    ) -> list[Document]:
        """Compute missing items that no one else holds, in batches.

        Locks are taken one batch at a time, so another worker can take over
        items this one has not reached. Returns the documents whose items
        another process was computing.
        """
        kind = item_kind(planned.name)
        counts = self.counts[planned.identity]
        locks = _HeldLocks(self.items, kind)
        held_elsewhere: list[Document] = []
        batch: list[Document] = []

        def flush() -> None:
            self._compute_and_publish(planned, batch)
            for document in batch:
                locks.release(planned.item_identity(document))
            batch.clear()

        try:
            for document in documents:
                identity = planned.item_identity(document)
                if self.items.get(kind, identity) is not None:
                    counts.reused += 1
                elif not locks.acquire(identity):
                    held_elsewhere.append(document)
                elif self.items.get(kind, identity) is not None:
                    counts.reused += 1
                    locks.release(identity)
                else:
                    batch.append(document)
                    if len(batch) == planned.module.batch_size:
                        flush()
            if batch:
                flush()
        finally:
            locks.release_all()
        return held_elsewhere

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
                    raise TxturesError(f"item {identity} of {planned.name} is missing")
            value = planned.module.load(directory)
        else:
            (value,) = self._compute(planned, [document])
        self._memo[key] = value
        return value

    def forget(self) -> None:
        """Drop results held in memory; stored results stay in the cache."""
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
