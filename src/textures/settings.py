"""Site settings: storage locations and execution limits.

Settings never affect computed values, so they never enter a digest. Each
setting resolves from the first source that provides it, in this order:
command-line flag, environment variable, personal settings, the definitions
store's site settings, built-in default. `definitions` cannot come from site
settings, since it selects the store that holds them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from textures.errors import SettingsError
from textures.yamlio import YamlError, load_mapping

LAB_ROOT = Path("/juice6/u/nlp/climblab")
"""Shared lab storage. Defaults point here when it is accessible."""

SITE_DOCUMENT = Path("site") / "site.yml"
"""Location of the site settings document inside a definitions store."""

Kind = Literal["path", "text", "positive_int"]


@dataclass(frozen=True)
class SettingSpec:
    """Declaration of one setting: its dotted key, value kind, and environment variable."""

    key: str
    kind: Kind
    env: str | None = None


SPECS: tuple[SettingSpec, ...] = (
    SettingSpec("definitions", "path", "TEXTURES_DEFINITIONS"),
    SettingSpec("inputs", "path", "TEXTURES_INPUTS"),
    SettingSpec("derivatives", "path", "TEXTURES_DERIVATIVES"),
    SettingSpec("resources", "path", "TEXTURES_RESOURCES"),
    SettingSpec("items", "path", "TEXTURES_ITEMS"),
    SettingSpec("hf_cache", "path", "TEXTURES_HF_CACHE"),
    SettingSpec("execution.device", "text"),
    SettingSpec("execution.max_variants", "positive_int"),
)
SPEC_BY_KEY = {spec.key: spec for spec in SPECS}


@dataclass(frozen=True)
class Setting:
    """A resolved setting value and a description of where it came from."""

    key: str
    value: Any
    source: str


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Index by dotted key or use the named properties."""

    values: Mapping[str, Setting]

    def __getitem__(self, key: str) -> Any:
        return self.values[key].value

    @property
    def definitions(self) -> Path:
        return self["definitions"]

    @property
    def inputs(self) -> Path:
        return self["inputs"]

    @property
    def derivatives(self) -> Path:
        return self["derivatives"]

    @property
    def resources(self) -> Path:
        return self["resources"]

    @property
    def items(self) -> Path:
        return self["items"]

    @property
    def hf_cache(self) -> Path | None:
        """Hugging Face cache location, or `None` to use the Hugging Face default."""
        return self["hf_cache"]

    @property
    def device(self) -> str:
        return self["execution.device"]

    @property
    def max_variants(self) -> int:
        return self["execution.max_variants"]


def personal_settings_path(env: Mapping[str, str], home: Path) -> Path:
    """Return the personal settings path, honoring `XDG_CONFIG_HOME`."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else home / ".config"
    return base / "textures" / "settings.yml"


def _writable(path: Path) -> bool:
    """Report whether `path`, or its nearest existing ancestor, is writable."""
    while not path.exists():
        if path.parent == path:
            return False
        path = path.parent
    return os.access(path, os.W_OK | os.X_OK)


def default_values(home: Path, lab_root: Path) -> dict[str, Any]:
    """Return built-in defaults, which depend on whether lab storage is accessible."""
    lab_readable = lab_root.is_dir() and os.access(lab_root, os.R_OK | os.X_OK)
    shared = lab_root / "textures-store"
    shared_writable = lab_readable and _writable(shared)
    personal = home / "textures-store"
    return {
        "definitions": lab_root / "textures-definitions"
        if lab_readable
        else home / "textures-definitions",
        "inputs": personal / "inputs",
        "derivatives": personal / "derivatives",
        "resources": (shared if shared_writable else personal) / "resources",
        "items": (shared if shared_writable else personal) / "items",
        "hf_cache": None,
        "execution.device": "auto",
        "execution.max_variants": 64,
    }


def _flatten(document: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys, keeping only known groups nested."""
    flat: dict[str, Any] = {}
    for key, value in document.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and any(k.startswith(f"{dotted}.") for k in SPEC_BY_KEY):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _coerce(spec: SettingSpec, value: Any, base: Path, location: str) -> Any:
    """Check a raw value against its kind. Relative paths resolve against `base`."""
    if spec.kind == "path":
        if not isinstance(value, str) or not value:
            raise SettingsError(f"{location}: {spec.key} must be a non-empty path")
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base / path)
    if spec.kind == "text":
        if not isinstance(value, str) or not value:
            raise SettingsError(f"{location}: {spec.key} must be non-empty text")
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SettingsError(f"{location}: {spec.key} must be a positive integer")
    return value


def _read_document(path: Path, *, allow_definitions: bool) -> dict[str, Any]:
    """Read a settings document into checked, dotted-key values."""
    try:
        raw = load_mapping(path)
    except YamlError as error:
        raise SettingsError(str(error)) from error
    values: dict[str, Any] = {}
    for key, value in _flatten(raw).items():
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"{path}: unknown setting {key!r}")
        if spec.key == "definitions" and not allow_definitions:
            raise SettingsError(
                f"{path}: 'definitions' cannot be set in site settings; set it with "
                "--definitions, TEXTURES_DEFINITIONS, or personal settings"
            )
        values[key] = _coerce(spec, value, path.parent, str(path))
    return values


def resolve_settings(
    flags: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    lab_root: Path = LAB_ROOT,
) -> Settings:
    """Resolve every setting from flags, environment, settings documents, and defaults.

    `flags` maps dotted keys to values given on the command line; `None` values
    are ignored. Relative paths from flags and environment variables resolve
    against `cwd`. Relative paths in a settings document resolve against that
    document's directory. A missing personal or site document is not an error.

    Raises:
        SettingsError: A settings document or value is invalid.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    cwd = Path.cwd() if cwd is None else cwd
    flags = {key: value for key, value in (flags or {}).items() if value is not None}
    for key in flags:
        if key not in SPEC_BY_KEY:
            raise SettingsError(f"unknown setting {key!r}")

    personal_path = personal_settings_path(env, home)
    personal = (
        _read_document(personal_path, allow_definitions=True) if personal_path.is_file() else {}
    )
    defaults = default_values(home, lab_root)
    resolved: dict[str, Setting] = {}

    def from_layers(spec: SettingSpec, site: Mapping[str, Any], site_path: Path) -> Setting:
        if spec.key in flags:
            value = _coerce(spec, flags[spec.key], cwd, "command line")
            return Setting(spec.key, value, "command line")
        if spec.env and env.get(spec.env):
            value = _coerce(spec, env[spec.env], cwd, spec.env)
            return Setting(spec.key, value, f"environment ({spec.env})")
        if spec.key in personal:
            return Setting(spec.key, personal[spec.key], f"personal settings ({personal_path})")
        if spec.key in site:
            return Setting(spec.key, site[spec.key], f"site settings ({site_path})")
        return Setting(spec.key, defaults[spec.key], "default")

    definitions = from_layers(SPEC_BY_KEY["definitions"], {}, Path())
    resolved["definitions"] = definitions
    site_path = definitions.value / SITE_DOCUMENT
    site = _read_document(site_path, allow_definitions=False) if site_path.is_file() else {}
    for spec in SPECS:
        if spec.key != "definitions":
            resolved[spec.key] = from_layers(spec, site, site_path)
    return Settings(resolved)
