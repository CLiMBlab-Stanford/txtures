"""Run the test suite from a copy of the code and environment on local disk.

On a network file system, Python's imports are slow: libraries such as
transformers import most of their modules lazily, each one a series of file
lookups over the network. This script copies the interpreter and the
installation's environment to local disk once per host, refreshes a copy of
`src` and `tests` there on every run, and runs pytest from that copy:

    txtures-install/env/bin/python tests/local.py [PYTEST OPTIONS]

The environment copy is kept between runs, under
`$TXTURES_TEST_LOCAL` (default: the temporary directory), and is identified
by `uv.lock`, the installation record, and the Python version, so it is
rebuilt when any of them changes. Measured on this lab's NFS, the suite ran
in 18 s from the copy and 30 s in place.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parent.parent
MARKER = ".complete"
CODE = ("src", "tests", "pyproject.toml", "install", ".python-version", "uv.lock")


def key() -> str:
    """What identifies the environment: the lock file, install record, and interpreter."""
    hasher = hashlib.sha256()
    for path in (CHECKOUT / "uv.lock", Path(sys.prefix).parent / "install.json"):
        hasher.update(path.read_bytes() if path.exists() else b"-")
    hasher.update(sys.version.encode())
    hasher.update(sys.prefix.encode())
    return hasher.hexdigest()[:16]


def root() -> Path:
    base = os.environ.get("TXTURES_TEST_LOCAL") or tempfile.gettempdir()
    return Path(base) / getpass.getuser() / f"txtures-tests-{key()}"


def _copy(source: Path, target: Path) -> None:
    def copy(src, dst, *, follow_symlinks=True):
        try:
            return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        except PermissionError:
            return shutil.copy(src, dst, follow_symlinks=follow_symlinks)

    shutil.copytree(source, target, symlinks=True, copy_function=copy)


def environment(local: Path) -> Path:
    """Copy the interpreter and environment to `local` unless done; return its Python."""
    python = local / "env" / "bin" / "python"
    if (local / MARKER).exists():
        return python
    if local.exists():
        shutil.rmtree(local)
    staging = local.with_name(local.name + f".{os.getpid()}.tmp")
    print(f"copying the environment to {local} (once per host)", file=sys.stderr)
    base, prefix = Path(sys.base_prefix), Path(sys.prefix)
    _copy(base, staging / "interpreter")
    _copy(prefix, staging / "env")
    # Point the environment at the copied interpreter and the copied source.
    interpreter = local / "interpreter" / Path(sys.executable).resolve().relative_to(base)
    link = staging / "env" / "bin" / "python"
    link.unlink()
    link.symlink_to(interpreter)
    config = staging / "env" / "pyvenv.cfg"
    lines = [
        f"home = {interpreter.parent}" if line.startswith("home = ") else line
        for line in config.read_text().splitlines()
    ]
    config.write_text("\n".join(lines) + "\n")
    site = next((staging / "env" / "lib").glob("python3*/site-packages"))
    for pth in site.glob("__editable__.txtures-*.pth"):
        pth.write_text(f"{local / 'src'}\n")
    staging.rename(local)
    (local / MARKER).write_text("")
    return python


def refresh(local: Path) -> None:
    """Replace the copy's code with the checkout's."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for name in CODE:
        source, target = CHECKOUT / name, local / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        else:
            shutil.copy2(source, target)


def main(argv: list[str]) -> int:
    local = root()
    python = environment(local)
    refresh(local)
    command = [str(python), "-m", "pytest", "-p", "no:cacheprovider", *argv]
    return subprocess.run(command, cwd=local).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
