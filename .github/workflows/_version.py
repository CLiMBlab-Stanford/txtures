"""Release-version parsing shared by the repository's release checks."""

from __future__ import annotations

import re
import tomllib

RELEASE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def parse_release_version(value: object) -> tuple[int, int, int]:
    """Parse a MAJOR.MINOR.PATCH string, rejecting pre-release and build suffixes."""
    if not isinstance(value, str) or not RELEASE_VERSION.fullmatch(value):
        raise ValueError(f"expected a MAJOR.MINOR.PATCH release version, got {value!r}")
    major, minor, patch = (int(part) for part in value.split("."))
    return major, minor, patch


def project_version(document: bytes | str) -> str:
    """Read and validate the project version from pyproject.toml content."""
    if isinstance(document, bytes):
        document = document.decode()
    value = tomllib.loads(document).get("project", {}).get("version")
    parse_release_version(value)
    return value
