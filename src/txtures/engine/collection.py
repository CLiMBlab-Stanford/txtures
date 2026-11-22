"""Find entries of the item and resource stores that no request can use.

An entry is garbage when it is broken or stale. Broken: a directory left by
a failed write (a staging directory or an entry without a valid manifest)
or a lock whose holder is gone, unless a live lock still covers it. Stale:
a valid entry that this txtures will never read again, because

- it was written under an older contract version of its module, or an older
  version of its resource kind;
- its resource says so (`Resource.stale`), for example a local corpus whose
  directory was removed; or
- it was built from, or computed with, an entry that is itself stale.

Entries of modules or resource kinds this txtures does not know, and
entries of newer versions, are left alone: a newer txtures sharing the
store may still use them.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtures.store import LOCK_SUFFIX, MANIFEST, EntryStore, _is_stale, read_json


@dataclass(frozen=True)
class Garbage:
    """A removable path in a store, and why it can be removed."""

    store: str
    kind: str
    path: Path
    reason: str

    @property
    def size(self) -> int:
        """The bytes the path takes, not following symbolic links."""
        return _size(self.path)


def _size(path: Path) -> int:
    try:
        if path.is_symlink() or not path.is_dir():
            return path.lstat().st_size
    except FileNotFoundError:
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                pass
    return total


@dataclass(frozen=True)
class _Entry:
    kind: str
    digest: str
    path: Path
    manifest: dict[str, Any]


def _locked(live: set[str], prefix: str) -> bool:
    return any(digest.startswith(prefix) for digest in live)


def _scan(store: EntryStore, name: str) -> tuple[list[_Entry], list[Garbage]]:
    """The valid entries of `store`, and its broken paths."""
    entries, broken = [], []
    if not store.root.is_dir():
        return entries, broken
    for kind_dir in sorted(p for p in store.root.iterdir() if p.is_dir()):
        kind = kind_dir.name
        for shard in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            children = sorted(shard.iterdir())
            live = {
                p.name[: -len(LOCK_SUFFIX)]
                for p in children
                if p.name.endswith(LOCK_SUFFIX) and not _is_stale(p, store.stale_seconds)
            }

            for path in children:
                if path.name.endswith(LOCK_SUFFIX):
                    if path.name[: -len(LOCK_SUFFIX)] not in live:
                        broken.append(Garbage(name, kind, path, "lock of a process that is gone"))
                elif path.name.startswith("."):
                    # A staging directory, named `.DIGEST12.RANDOM`.
                    if not _locked(live, path.name[1:].split(".", 1)[0]):
                        broken.append(Garbage(name, kind, path, "left by a failed write"))
                elif store.get(kind, path.name) is None:
                    if not _locked(live, path.name):
                        broken.append(Garbage(name, kind, path, "no valid manifest"))
                else:
                    entries.append(_Entry(kind, path.name, path, read_json(path / MANIFEST)))
    return entries, broken


def _release(version: object) -> tuple[int, ...] | None:
    match = re.match(r"(\d+(?:\.\d+)*)", version) if isinstance(version, str) else None
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def built_before(recorded: object, current: str) -> bool:
    """Whether txtures version `recorded` is an earlier release than `current`.

    Only the leading release numbers count, so a development build of a
    release is not earlier than the release. An unreadable version is not
    earlier.
    """
    old, new = _release(recorded), _release(current)
    return old is not None and new is not None and old < new


def _older(manifest: Mapping[str, Any], field: str, current: int) -> bool:
    recorded = manifest.get(field)
    return isinstance(recorded, int) and recorded < current


def _stale_resources(entries: list[_Entry]) -> dict[str, str]:
    """Stale resource entries: digest → reason."""
    from txtures.modules import resource_class

    stale: dict[str, str] = {}
    for entry in entries:
        try:
            cls = resource_class(entry.kind)
        except KeyError:
            continue
        if _older(entry.manifest, "version", cls.version):
            stale[entry.digest] = (
                f"built by {entry.kind} version {entry.manifest['version']}, now {cls.version}"
            )
            continue
        spec = entry.manifest.get("spec")
        reason = cls(spec).stale(entry.manifest) if isinstance(spec, dict) else None
        if reason:
            stale[entry.digest] = reason
    _cascade(entries, stale, "dependencies", "built from")
    return stale


def _stale_items(entries: list[_Entry], registry, removed_resources: Mapping[str, str]) -> dict:
    stale: dict[str, str] = {}
    for entry in entries:
        module = entry.manifest.get("module", entry.kind)
        if module not in registry:
            continue
        current = registry[module].contract_version
        if _older(entry.manifest, "contract_version", current):
            stale[entry.digest] = (
                f"computed under contract version {entry.manifest['contract_version']}, "
                f"now {current}"
            )
            continue
        resources = entry.manifest.get("resources") or {}
        gone = [n for n, digest in resources.items() if digest in removed_resources]
        if gone:
            stale[entry.digest] = f"computed with the stale resource {gone[0]!r}"
    _cascade(entries, stale, "upstream", "computed from")
    return stale


def _cascade(entries: list[_Entry], stale: dict[str, str], field: str, verb: str) -> None:
    """Mark entries that reference a stale entry through `field`, until none changes."""
    changed = True
    while changed:
        changed = False
        for entry in entries:
            if entry.digest in stale:
                continue
            references = entry.manifest.get(field) or {}
            for name, digest in references.items():
                if digest in stale:
                    stale[entry.digest] = f"{verb} the stale {name!r} {digest[:12]}"
                    changed = True
                    break


def find(items: EntryStore, resources: EntryStore, registry) -> list[Garbage]:
    """Broken and stale paths of the item and resource stores, resources first."""
    resource_entries, garbage = _scan(resources, "resources")
    stale_resources = _stale_resources(resource_entries)
    garbage += [
        Garbage("resources", e.kind, e.path, stale_resources[e.digest])
        for e in resource_entries
        if e.digest in stale_resources
    ]
    item_entries, broken_items = _scan(items, "cache")
    stale_items = _stale_items(item_entries, registry, stale_resources)
    garbage += broken_items
    garbage += [
        Garbage("cache", e.kind, e.path, stale_items[e.digest])
        for e in item_entries
        if e.digest in stale_items
    ]
    return garbage


def remove(garbage: Garbage) -> None:
    """Remove a path found by `find`.

    An entry is first renamed to a staging name, so that no reader sees it
    half removed; a later collection removes what an interruption leaves.
    """
    path = garbage.path
    if path.is_dir() and not path.name.startswith("."):
        hidden = path.with_name(f".{path.name[:12]}.gc{os.getpid()}")
        try:
            os.rename(path, hidden)
        except FileNotFoundError:
            return
        path = hidden
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
