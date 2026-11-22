"""Content-addressed stores for resources and item intermediates.

An entry lives at `ROOT/KIND/AB/DIGEST/`, where `AB` is the first two
characters of the digest. It is valid only when its `manifest.json` exists
and records the same digest. Entries are written in a staging directory
beside their final location and published with a rename, and the manifest is
written last, so readers never see a partial entry.

Producers of the same entry coordinate through a lock file created with
`O_CREAT | O_EXCL`. While a process holds locks, a background heartbeat
refreshes their modification times every minute. A lock is broken when its
process is known to be gone (on the same host) or when it has not been
refreshed within the stale limit, so a holder that dies on another host
delays others by minutes, not hours.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from txtures import __version__
from txtures.errors import TxturesError

MANIFEST = "manifest.json"
LOCK_SUFFIX = ".lock"
STALE_LOCK_SECONDS = 15 * 60
HEARTBEAT_SECONDS = 60
POLL_SECONDS = 0.2


def make_shared_dir(path: Path) -> None:
    """Create `path` and missing parents, group-writable with the setgid bit.

    Directories that already exist are left unchanged. The group and other
    permission bits otherwise follow the process umask.
    """
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        _share(directory)


def _share(path: Path) -> None:
    """Add group read and write (and search and setgid for directories) to `path`.

    Symbolic links are left alone: changing one's mode would change its
    target, which may lie outside the entry or not exist on this host.
    """
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            return
        extra = stat.S_IRGRP | stat.S_IWGRP
        if stat.S_ISDIR(mode):
            extra |= stat.S_IXGRP | stat.S_ISGID
        os.chmod(path, stat.S_IMODE(mode) | extra)
    except PermissionError:
        pass


def _share_tree(root: Path) -> None:
    _share(root)
    for directory, subdirectories, files in os.walk(root):
        for name in (*subdirectories, *files):
            _share(Path(directory) / name)


def write_json(path: Path, value: Any) -> None:
    """Write `value` as indented JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    """Return the current UTC time in ISO 8601 form."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# Locks


@dataclass(frozen=True)
class LockOwner:
    host: str
    pid: int
    started: float

    @classmethod
    def current(cls) -> LockOwner:
        return cls(socket.gethostname(), os.getpid(), time.time())


class Heartbeat:
    """Refreshes the modification time of held lock files from a daemon thread."""

    def __init__(self, interval: float = HEARTBEAT_SECONDS):
        self.interval = interval
        self._paths: set[Path] = set()
        self._guard = threading.Lock()
        self._thread: threading.Thread | None = None

    def add(self, path: Path) -> None:
        with self._guard:
            self._paths.add(path)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="txtures-lock-heartbeat", daemon=True
                )
                self._thread.start()

    def discard(self, path: Path) -> None:
        with self._guard:
            self._paths.discard(path)

    def beat(self) -> None:
        with self._guard:
            paths = list(self._paths)
        for path in paths:
            try:
                os.utime(path)
            except OSError:
                pass

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            self.beat()


HEARTBEAT = Heartbeat()
"""The process-wide heartbeat for every lock this process holds."""


def _read_owner(path: Path) -> LockOwner | None:
    try:
        data = read_json(path)
        return LockOwner(str(data["host"]), int(data["pid"]), float(data["started"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_stale(path: Path, stale_seconds: float) -> bool:
    owner = _read_owner(path)
    if owner is not None and owner.host == socket.gethostname() and not _process_alive(owner.pid):
        return True
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > stale_seconds


def try_lock(path: Path, stale_seconds: float = STALE_LOCK_SECONDS) -> bool:
    """Try once to create the lock file at `path`, breaking a stale lock first."""
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            if not _is_stale(path, stale_seconds):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        owner = LockOwner.current()
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"host": owner.host, "pid": owner.pid, "started": owner.started}, handle)
        _share(path)
        HEARTBEAT.add(path)
        return True
    return False


@contextmanager
def lock(path: Path, stale_seconds: float = STALE_LOCK_SECONDS) -> Iterator[None]:
    """Hold the lock file at `path`, waiting while another live process holds it."""
    while not try_lock(path, stale_seconds):
        time.sleep(POLL_SECONDS)
    try:
        yield
    finally:
        unlock(path)


def unlock(path: Path) -> None:
    HEARTBEAT.discard(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# Stores


class EntryStore:
    """A content-addressed directory of entries, such as the `items` or `resources` store."""

    def __init__(self, root: Path, stale_seconds: float = STALE_LOCK_SECONDS):
        self.root = root
        self.stale_seconds = stale_seconds

    def path(self, kind: str, digest: str) -> Path:
        return self.root / kind / digest[:2] / digest

    def lock_path(self, kind: str, digest: str) -> Path:
        entry = self.path(kind, digest)
        return entry.with_name(entry.name + LOCK_SUFFIX)

    def get(self, kind: str, digest: str) -> Path | None:
        """Return the entry's directory if a valid entry exists, otherwise `None`."""
        entry = self.path(kind, digest)
        try:
            manifest = read_json(entry / MANIFEST)
        except (OSError, ValueError):
            return None
        return entry if isinstance(manifest, dict) and manifest.get("digest") == digest else None

    def manifest(self, kind: str, digest: str) -> dict[str, Any]:
        return read_json(self.path(kind, digest) / MANIFEST)

    def try_lock(self, kind: str, digest: str) -> bool:
        make_shared_dir(self.path(kind, digest).parent)
        return try_lock(self.lock_path(kind, digest), self.stale_seconds)

    @contextmanager
    def lock(self, kind: str, digest: str) -> Iterator[None]:
        make_shared_dir(self.path(kind, digest).parent)
        with lock(self.lock_path(kind, digest), self.stale_seconds):
            yield

    def unlock(self, kind: str, digest: str) -> None:
        unlock(self.lock_path(kind, digest))

    def publish(
        self,
        kind: str,
        digest: str,
        write: Callable[[Path], None],
        manifest: Mapping[str, Any],
    ) -> Path:
        """Write an entry and publish it atomically. The caller should hold its lock.

        `write` fills a staging directory. The manifest, with `digest`,
        `kind`, `created`, and `txtures_version` added, is written last. If a
        valid entry already exists, the staged copy is discarded.

        Raises:
            TxturesError: A different, invalid directory occupies the entry's path.
        """
        final = self.path(kind, digest)
        make_shared_dir(final.parent)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=final.parent))
        try:
            write(staging)
            record = {
                **manifest,
                "digest": digest,
                "kind": kind,
                "created": now(),
                "txtures_version": __version__,
            }
            write_json(staging / MANIFEST, record)
            _share_tree(staging)
            if self.get(kind, digest) is not None:
                shutil.rmtree(staging)
                return final
            if final.exists():
                # An entry without a valid manifest was left by an earlier failure.
                shutil.rmtree(final)
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if self.get(kind, digest) is None:
            raise TxturesError(f"failed to publish store entry {final}")
        return final
