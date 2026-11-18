"""Site settings: storage locations and execution limits.

Settings never affect computed values, so they never enter a digest. Each
setting resolves from the first source that provides it, in this order:
command-line flag, environment variable, the installation record written by
`./install`, personal settings, the definitions store's site settings,
built-in default. `definitions` cannot come from site settings, since it
selects the store that holds them.

Built-in defaults are directories in the home directory. A lab shares its
stores by giving members the path of its definitions store, whose site
settings name the rest; `txtures site use PATH` records it in personal
settings.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from txtures.errors import SettingsError
from txtures.yamlio import YamlError, load_mapping

SITE_DOCUMENT = Path("site") / "site.yml"
"""Location of the site settings document inside a definitions store."""

Kind = Literal["path", "text", "choice", "positive_int", "count", "text_list"]


@dataclass(frozen=True)
class SettingSpec:
    """Declaration of one setting: its dotted key, value kind, meaning, and environment variable.

    A `choice` setting accepts only the values in `choices`.
    """

    key: str
    kind: Kind
    description: str
    env: str | None = None
    choices: tuple[str, ...] = ()


SPECS: tuple[SettingSpec, ...] = (
    SettingSpec("definitions", "path", "the definitions store", "TXTURES_DEFINITIONS"),
    SettingSpec("inputs", "path", "directory of input files", "TXTURES_INPUTS"),
    SettingSpec("derivatives", "path", "directory of output tables", "TXTURES_DERIVATIVES"),
    SettingSpec("items", "path", "store of per-document results", "TXTURES_ITEMS"),
    SettingSpec("resources", "path", "models and the Hugging Face cache", "TXTURES_RESOURCES"),
    SettingSpec("execution.device", "text", "auto, cpu, cuda, cuda:N, or mps"),
    SettingSpec(
        "execution.executor", "choice", "local, or slurm to submit jobs", choices=("local", "slurm")
    ),
    SettingSpec("execution.max_variants", "positive_int", "most variants one module may expand to"),
    SettingSpec("slurm.partition", "text", "partition to submit to; required for slurm"),
    SettingSpec("slurm.account", "text", "account to charge"),
    SettingSpec("slurm.time_hours", "positive_int", "time limit of each task, in hours"),
    SettingSpec("slurm.memory_gb", "positive_int", "memory of each task, in GB"),
    SettingSpec("slurm.cpus", "positive_int", "CPUs of each task"),
    SettingSpec("slurm.gpus", "count", "GPUs of each task; 0 for none"),
    SettingSpec("slurm.workers", "positive_int", "array tasks that share a run"),
    SettingSpec("slurm.options", "text_list", "further sbatch options"),
)
LOCATIONS = ("definitions", "items", "resources")
"""Settings that decide whether results are shared; see `uses_default_locations`."""
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
    def items(self) -> Path:
        return self["items"]

    @property
    def resources(self) -> Path:
        """The shared directory of text-independent resources, such as models."""
        return self["resources"]

    @property
    def resource_store(self) -> Path:
        """The content-addressed resource store, inside `resources`."""
        return self.resources / "store"

    @property
    def hf_home(self) -> Path:
        """The Hugging Face home directory, inside `resources`."""
        return self.resources / "huggingface"

    @property
    def hf_cache(self) -> Path:
        """The Hugging Face hub cache, where model files are downloaded."""
        return self.hf_home / "hub"

    @property
    def executor(self) -> str:
        return self["execution.executor"]

    def group(self, prefix: str) -> dict[str, Any]:
        """Settings under `prefix.`, keyed without the prefix."""
        return {
            key.removeprefix(f"{prefix}."): setting.value
            for key, setting in self.values.items()
            if key.startswith(f"{prefix}.")
        }

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
    return base / "txtures" / "settings.yml"


def default_values(home: Path) -> dict[str, Any]:
    """Return built-in defaults: directories in the home directory.

    `txtures-definitions`, `txtures-store`, and `txtures-resources` are
    siblings, apart from any code.
    """
    store = home / "txtures-store"
    return {
        "definitions": home / "txtures-definitions",
        "inputs": store / "inputs",
        "derivatives": store / "derivatives",
        "items": store / "items",
        "resources": home / "txtures-resources",
        "execution.device": "auto",
        "execution.executor": "local",
        "execution.max_variants": 64,
        "slurm.partition": None,
        "slurm.account": None,
        "slurm.time_hours": 24,
        "slurm.memory_gb": 32,
        "slurm.cpus": 4,
        "slurm.gpus": 1,
        "slurm.workers": 1,
        "slurm.options": [],
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
    if spec.kind == "choice":
        if value not in spec.choices:
            raise SettingsError(f"{location}: {spec.key} must be one of {', '.join(spec.choices)}")
        return value
    if spec.kind == "text_list":
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise SettingsError(f"{location}: {spec.key} must be a list of non-empty text")
        return list(value)
    if isinstance(value, str) and value.isdigit():
        value = int(value)  # from a flag or environment variable
    if spec.kind == "count":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SettingsError(f"{location}: {spec.key} must be a non-negative integer")
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
                "--definitions, TXTURES_DEFINITIONS, or personal settings"
            )
        values[key] = _coerce(spec, value, path.parent, str(path))
    return values


def resolve_settings(
    flags: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
    installation: Mapping[str, Any] | None = None,
    installation_source: str = "installation",
) -> Settings:
    """Resolve every setting from flags, environment, settings documents, and defaults.

    `flags` maps dotted keys to values given on the command line; `None` values
    are ignored. Relative paths from flags and environment variables resolve
    against `cwd`. Relative paths in a settings document resolve against that
    document's directory. A missing personal or site document is not an error.
    `installation` holds the settings an installation record fixes, which
    are absolute.

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

    installed: dict[str, Any] = {}
    for key, value in (installation or {}).items():
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"{installation_source}: unknown setting {key!r}")
        installed[key] = _coerce(spec, value, Path("/"), installation_source)

    personal_path = personal_settings_path(env, home)
    personal = (
        _read_document(personal_path, allow_definitions=True) if personal_path.is_file() else {}
    )
    defaults = default_values(home)
    resolved: dict[str, Setting] = {}

    def from_layers(spec: SettingSpec, site: Mapping[str, Any], site_path: Path) -> Setting:
        if spec.key in flags:
            value = _coerce(spec, flags[spec.key], cwd, "command line")
            return Setting(spec.key, value, "command line")
        if spec.env and env.get(spec.env):
            value = _coerce(spec, env[spec.env], cwd, spec.env)
            return Setting(spec.key, value, f"environment ({spec.env})")
        if spec.key in installed:
            return Setting(spec.key, installed[spec.key], installation_source)
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


def uses_default_locations(settings: Settings) -> bool:
    """Report whether the definitions store, item store, and resources all use defaults.

    Such a configuration shares no results with anyone, which is usually an
    oversight on a machine with lab storage.
    """
    return all(settings.values[key].source == "default" for key in LOCATIONS)


# Editing personal settings


def _personal_document(path: Path) -> dict[str, Any]:
    return _read_document(path, allow_definitions=True) if path.is_file() else {}


def _write_personal(path: Path, values: Mapping[str, Any]) -> None:
    """Write dotted-key values as a nested YAML document, atomically."""
    import yaml

    nested: dict[str, Any] = {}
    for key in (spec.key for spec in SPECS if spec.key in values):
        value = values[key]
        value = str(value) if isinstance(value, Path) else value
        *groups, name = key.split(".")
        target = nested
        for group in groups:
            target = target.setdefault(group, {})
        target[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Personal txtures settings, written by `txtures site`.\n"
    if nested:
        text += yaml.safe_dump(nested, sort_keys=False, allow_unicode=True)
    descriptor, staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def set_personal(
    key: str,
    values: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Set one setting in personal settings and return the document's path.

    `values` holds one value, or any number for a list setting. A relative
    path is made absolute against `cwd`. The document is rewritten, so
    comments in it are not kept.

    Raises:
        SettingsError: The key is unknown or the value is invalid.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    cwd = Path.cwd() if cwd is None else cwd
    spec = SPEC_BY_KEY.get(key)
    if spec is None:
        raise SettingsError(f"unknown setting {key!r}; see `txtures site ls`")
    if spec.kind == "text_list":
        raw: Any = list(values)
    elif len(values) != 1:
        raise SettingsError(f"{key} takes one value")
    else:
        raw = values[0]
    value = _coerce(spec, raw, cwd, key)
    path = personal_settings_path(env, home)
    document = _personal_document(path)
    document[key] = value.absolute() if isinstance(value, Path) else value
    _write_personal(path, document)
    return path


def unset_personal(
    key: str, *, env: Mapping[str, str] | None = None, home: Path | None = None
) -> tuple[Path, bool]:
    """Remove one setting from personal settings.

    Returns the document's path and whether the setting was there.

    Raises:
        SettingsError: The key is unknown.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    if key not in SPEC_BY_KEY:
        raise SettingsError(f"unknown setting {key!r}; see `txtures site ls`")
    path = personal_settings_path(env, home)
    document = _personal_document(path)
    if key not in document:
        return path, False
    del document[key]
    _write_personal(path, document)
    return path, True
