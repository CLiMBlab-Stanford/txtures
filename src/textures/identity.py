"""Content digests that identify configs, stored entries, and output tables."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with sorted keys and no insignificant whitespace.

    Raises:
        ValueError: The value contains NaN or infinity, which have no stable
            JSON form.
        TypeError: The value contains something JSON cannot represent.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    """Return the hexadecimal SHA-256 digest of the canonical JSON form of `value`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
