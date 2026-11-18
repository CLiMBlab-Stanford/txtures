"""Reject a proposed main release whose package version did not advance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _version import parse_release_version, project_version


def main(argv: list[str]) -> int:
    """Compare the worktree version with the pyproject at one base commit."""
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} BASE_COMMIT")
    base_document = subprocess.run(
        ["git", "show", f"{argv[1]}:pyproject.toml"],
        check=True,
        capture_output=True,
    ).stdout
    previous = project_version(base_document)
    proposed = project_version(Path("pyproject.toml").read_bytes())
    if parse_release_version(proposed) <= parse_release_version(previous):
        raise SystemExit(f"main release version must advance: {previous} -> {proposed}")
    print(f"main release version advances: {previous} -> {proposed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
