"""Strict YAML reading for definitions and settings documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from txtures.errors import TxturesError


class YamlError(TxturesError):
    """A YAML document cannot be read as a mapping."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping with string keys.

    An empty file reads as an empty mapping.

    Raises:
        YamlError: The file cannot be read or parsed, has duplicate keys, or
            its top level is not a mapping with string keys.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise YamlError(f"{path}: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise YamlError(f"{path}: expected a mapping with string keys at the top level")
    return data
