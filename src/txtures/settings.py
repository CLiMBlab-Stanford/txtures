"""Settings: the txtures home, its directories, and execution limits.

Settings never affect computed values, so they never enter a digest.

Everything txtures keeps lives in one directory, the txtures home, which is
also the site::

    HOME/site.yml       site settings: execution and Slurm limits
    HOME/definitions/   the definitions store
    HOME/inputs/        input files
    HOME/outputs/       output tables
    HOME/cache/         stored per-document results
    HOME/resources/     models and the Hugging Face cache

The home comes from a flag, `TXTURES_HOME`, or the home-path file, a plain
text file holding only its path, which `txtures site use PATH` writes;
otherwise it is `~/txtures-home`. Each directory comes from a flag or its
environment variable, such as `TXTURES_CACHE`, and otherwise from the
layout above, so the site cannot move it. Every other setting comes from a
flag, its environment variable, the home's `site.yml`, or a built-in
default, in that order. `txtures site set` and `txtures site unset` edit
`site.yml`; `txtures env` keeps environment variables for a user's own
departures from the site.
"""

from __future__ import annotations

import os
import shlex
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from txtures.errors import SettingsError
from txtures.yamlio import YamlError, load_mapping

SITE_DOCUMENT = "site.yml"
"""Name of the site settings document in the home."""

OLD_SITE_DOCUMENT = Path("site") / "site.yml"
"""Where earlier versions kept site settings, inside the definitions store."""

Kind = Literal["path", "text", "choice", "positive_int", "count", "text_list"]


@dataclass(frozen=True)
class SettingSpec:
    """Declaration of one setting: its dotted key, value kind, and meaning.

    A `choice` setting accepts only the values in `choices`. A `location` is
    a directory of the home: it is never a site setting, and by default it is
    the home's subdirectory of the same name.
    """

    key: str
    kind: Kind
    description: str
    choices: tuple[str, ...] = ()
    location: bool = False

    @property
    def env(self) -> str:
        """The environment variable for this setting, such as `TXTURES_SLURM_PARTITION`."""
        return "TXTURES_" + self.key.upper().replace(".", "_")


SPECS: tuple[SettingSpec, ...] = (
    SettingSpec("home", "path", "the txtures home: site settings and the directories below"),
    SettingSpec("definitions", "path", "the definitions store", location=True),
    SettingSpec("inputs", "path", "input files", location=True),
    SettingSpec("outputs", "path", "output tables", location=True),
    SettingSpec("cache", "path", "stored per-document results", location=True),
    SettingSpec("resources", "path", "models and the Hugging Face cache", location=True),
    SettingSpec("execution.device", "text", "auto, cpu, cuda, cuda:N, or mps"),
    SettingSpec(
        "execution.executor",
        "choice",
        "auto submits intensive work to Slurm when it is set up; local or slurm always",
        choices=("local", "slurm", "auto"),
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
SPEC_BY_KEY = {spec.key: spec for spec in SPECS}
LOCATION_KEYS = tuple(spec.key for spec in SPECS if spec.location)
SHARED = ("definitions", "cache", "resources")
"""Locations that decide whether results are shared; see `uses_default_locations`."""
RENAMED = {"derivatives": "outputs", "items": "cache"}
"""Settings that earlier versions named differently, by their old names."""
DEFAULT_HOME = "txtures-home"
"""The default home, in the user's home directory."""


@dataclass(frozen=True)
class Setting:
    """A resolved setting value and a description of where it came from."""

    key: str
    value: Any
    source: str


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Index by dotted key or use the named properties.

    `legacy` describes settings that earlier versions kept in other places;
    it is set only when resolution was asked to tolerate them.
    """

    values: Mapping[str, Setting]
    legacy: Legacy | None = None

    def __getitem__(self, key: str) -> Any:
        return self.values[key].value

    @property
    def home(self) -> Path:
        return self["home"]

    @property
    def site_document(self) -> Path:
        """The site settings document, in the home."""
        return self.home / SITE_DOCUMENT

    @property
    def definitions(self) -> Path:
        return self["definitions"]

    @property
    def inputs(self) -> Path:
        return self["inputs"]

    @property
    def outputs(self) -> Path:
        return self["outputs"]

    @property
    def cache(self) -> Path:
        return self["cache"]

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


def config_directory(env: Mapping[str, str], user_home: Path) -> Path:
    """The user's txtures configuration directory, honoring `XDG_CONFIG_HOME`."""
    config_home = env.get("XDG_CONFIG_HOME")
    return (Path(config_home) if config_home else user_home / ".config") / "txtures"


def home_path_file(env: Mapping[str, str], user_home: Path) -> Path:
    """The home-path file, which holds the path of the user's txtures home."""
    return config_directory(env, user_home) / "home-path"


def legacy_settings_path(env: Mapping[str, str], user_home: Path) -> Path:
    """The personal settings document of earlier versions, which txtures no longer reads."""
    return config_directory(env, user_home) / "settings.yml"


def default_values() -> dict[str, Any]:
    """Return the built-in defaults of the settings that are not locations."""
    return {
        "execution.device": "auto",
        "execution.executor": "auto",
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


def _unknown(key: str, where: str) -> SettingsError:
    if key in RENAMED:
        return SettingsError(f"{where}: {key!r} is now called {RENAMED[key]!r}")
    return SettingsError(f"{where}: unknown setting {key!r}; see `txtures site ls`")


def _read_raw(path: Path, *, old_names: bool = False) -> dict[str, Any]:
    """Read a settings document into dotted-key values as written, or `{}` if it is absent.

    With `old_names`, the names of earlier versions are accepted too.
    """
    if not path.is_file():
        return {}
    try:
        raw = load_mapping(path)
    except YamlError as error:
        raise SettingsError(str(error)) from error
    flat = _flatten(raw)
    for key in flat:
        if key not in SPEC_BY_KEY and not (old_names and key in RENAMED):
            raise _unknown(key, str(path))
    return flat


def _read_site(path: Path) -> dict[str, Any]:
    """Read the site settings document into checked values."""
    values: dict[str, Any] = {}
    for key, value in _read_raw(path).items():
        spec = SPEC_BY_KEY[key]
        if key == "home" or spec.location:
            raise SettingsError(
                f"{path}: {key!r} cannot be a site setting, since the home's layout fixes it; "
                "remove it from the file"
            )
        values[key] = _coerce(spec, value, path.parent, str(path))
    return values


def _read_home_path(path: Path) -> Path | None:
    """Read the home-path file, or return `None` if it is absent."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text or "\n" in text:
        raise SettingsError(
            f"{path} must hold one line, the path of a txtures home; "
            "rewrite it with `txtures site use PATH`"
        )
    return _coerce(SPEC_BY_KEY["home"], text, path.parent, str(path))


def _split_words(text: str, location: str) -> list[str]:
    """Split a list setting's environment variable into words, as a shell would."""
    try:
        return shlex.split(text)
    except ValueError as error:
        raise SettingsError(f"{location}: {error}") from None


# Settings kept by earlier versions


@dataclass(frozen=True)
class Legacy:
    """Settings that earlier versions kept outside the home's `site.yml`.

    `documents` maps each such document to the settings it holds, under
    their current names; paths are absolute. `home` is the suggested home.
    """

    documents: Mapping[Path, Mapping[str, Any]]
    home: Path

    def message(self) -> str:
        """Explain how to move these settings into a txtures home."""
        values: dict[str, Any] = {}
        for document in reversed(list(self.documents.values())):
            values.update(document)
        home = self.home
        lines = [
            f"{', '.join(map(str, self.documents))} "
            f"{'is' if len(self.documents) == 1 else 'are'} from an earlier txtures, which "
            "kept settings there. txtures now keeps everything in one home directory, "
            f"with site settings in HOME/{SITE_DOCUMENT} and the directories "
            f"{', '.join(LOCATION_KEYS)} inside it. To move to a home at {home}:"
        ]
        step = 0
        moves = [
            (values[key], home / key)
            for key in LOCATION_KEYS
            if key in values and values[key] != home / key
        ]
        for old, new in moves:
            step += 1
            lines.append(f"  {step}. move or symlink {old} to {new}")
        personal = [d for d in self.documents if d.name == "settings.yml"]
        step += 1
        lines.append(
            f"  {step}. txtures site use {shlex.quote(str(home))}"
            + (f"  (this also removes {personal[0]})" if personal else "")
        )
        pairs = []
        for key, value in values.items():
            if key == "home" or SPEC_BY_KEY[key].location:
                continue
            for item in value if isinstance(value, list) else [value]:
                pairs.append(shlex.quote(f"{key}={item}"))
        if pairs:
            step += 1
            lines.append(f"  {step}. txtures site set {' '.join(pairs)}")
        for document in self.documents:
            if document in personal:
                continue
            store = document.parent.parent
            for old, new in moves:
                if store == old:
                    store = new
            step += 1
            lines.append(f"  {step}. remove {store / OLD_SITE_DOCUMENT.parent}")
        return "\n".join(lines)


def _old_document(path: Path) -> dict[str, Any]:
    """Read a settings document of earlier versions under current names, with absolute paths."""
    values = {}
    for key, value in _read_raw(path, old_names=True).items():
        key = RENAMED.get(key, key)
        values[key] = _coerce(SPEC_BY_KEY[key], value, path.parent, str(path))
    return values


def _find_legacy(
    personal: Path, definitions: Path, user_home: Path, home_chosen: bool
) -> Legacy | None:
    documents: dict[Path, Mapping[str, Any]] = {}
    if personal.is_file():
        documents[personal] = _old_document(personal)
    old_store = documents.get(personal, {}).get("definitions")
    for store in dict.fromkeys(s for s in (old_store, definitions) if s is not None):
        if (store / OLD_SITE_DOCUMENT).is_file():
            documents[store / OLD_SITE_DOCUMENT] = _old_document(store / OLD_SITE_DOCUMENT)
    if not documents:
        return None
    if old_store is not None and not home_chosen:
        home = old_store.parent
    else:
        home = definitions.parent if home_chosen else user_home / DEFAULT_HOME
    return Legacy(documents, home)


def resolve_settings(
    flags: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    user_home: Path | None = None,
    cwd: Path | None = None,
    tolerate_stale: bool = False,
) -> Settings:
    """Resolve every setting from flags, the environment, the home, and defaults.

    `flags` maps dotted keys to values given on the command line; `None` values
    are ignored. Relative paths from flags and environment variables resolve
    against `cwd`, and relative paths in `site.yml` against the home. A
    missing home-path file, home, or `site.yml` is not an error.

    Settings kept where earlier versions kept them, in personal settings or
    in the definitions store's `site/site.yml`, are an error whose message
    explains how to move them. With `tolerate_stale`, they are ignored and
    described in `Settings.legacy` instead, so that the commands that move
    them can still run.

    Raises:
        SettingsError: A settings document or value is invalid, a setting
            uses a name of earlier versions, or settings remain where
            earlier versions kept them.
    """
    env = os.environ if env is None else env
    user_home = Path.home() if user_home is None else user_home
    cwd = Path.cwd() if cwd is None else cwd
    flags = {key: value for key, value in (flags or {}).items() if value is not None}
    for key in flags:
        if key not in SPEC_BY_KEY:
            raise _unknown(key, "command line")
    for old, new in RENAMED.items():
        name = f"TXTURES_{old.upper()}"
        if env.get(name):
            raise SettingsError(f"{name} is now called {SPEC_BY_KEY[new].env}; rename it")

    def given(spec: SettingSpec) -> Setting | None:
        if spec.key in flags:
            return Setting(
                spec.key, _coerce(spec, flags[spec.key], cwd, "command line"), "command line"
            )
        if env.get(spec.env):
            raw: Any = env[spec.env]
            if spec.kind == "text_list":
                raw = _split_words(raw, spec.env)
            return Setting(spec.key, _coerce(spec, raw, cwd, spec.env), f"environment ({spec.env})")
        return None

    pointer = home_path_file(env, user_home)
    home = given(SPEC_BY_KEY["home"])
    if home is None:
        chosen = _read_home_path(pointer)
        if chosen is not None:
            home = Setting("home", chosen, f"home-path ({pointer})")
        else:
            home = Setting("home", user_home / DEFAULT_HOME, "default")
    resolved: dict[str, Setting] = {"home": home}
    for key in LOCATION_KEYS:
        spec = SPEC_BY_KEY[key]
        layout = "default" if home.source == "default" else "home"
        resolved[key] = given(spec) or Setting(key, home.value / key, layout)

    legacy = _find_legacy(
        legacy_settings_path(env, user_home),
        resolved["definitions"].value,
        user_home,
        home.source != "default",
    )
    if legacy is not None and not tolerate_stale:
        raise SettingsError(legacy.message())

    site_path = home.value / SITE_DOCUMENT
    site = _read_site(site_path)
    defaults = default_values()
    for spec in SPECS:
        if spec.key in resolved:
            continue
        if (setting := given(spec)) is None:
            if spec.key in site:
                setting = Setting(spec.key, site[spec.key], f"site settings ({site_path})")
            else:
                setting = Setting(spec.key, defaults[spec.key], "default")
        resolved[spec.key] = setting
    return Settings(resolved, legacy)


def uses_default_locations(settings: Settings) -> bool:
    """Report whether the definitions store, cache, and resources all use defaults.

    Such a configuration shares no results with anyone, which is usually an
    oversight on a machine with lab storage.
    """
    return all(settings.values[key].source == "default" for key in SHARED)


# Editing settings documents


def _write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def _write_document(path: Path, values: Mapping[str, Any], header: str) -> None:
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
    text = header
    if nested:
        text += yaml.safe_dump(nested, sort_keys=False, allow_unicode=True)
    _write_text(path, text)


SITE_HEADER = (
    "# Site settings for txtures, edited by `txtures site set` and `txtures site unset`.\n"
    "# They set execution and Slurm limits and never change computed values.\n"
    "# `txtures site ls` lists every setting.\n"
)


def use_site(
    home: Path, *, env: Mapping[str, str] | None = None, user_home: Path | None = None
) -> tuple[Path, dict[str, Any] | None]:
    """Choose `home` as the user's txtures home, the site whose settings apply.

    Writes the home-path file and removes the personal settings document of
    earlier versions. Returns the file written and the settings the removed
    document held, or `None` if there was none.

    Raises:
        SettingsError: The document of earlier versions is unreadable.
    """
    env = os.environ if env is None else env
    user_home = Path.home() if user_home is None else user_home
    path = home_path_file(env, user_home)
    legacy_path = legacy_settings_path(env, user_home)
    legacy = _old_document(legacy_path) if legacy_path.is_file() else None
    _write_text(path, f"{home.expanduser().absolute()}\n")
    if legacy is not None:
        legacy_path.unlink()
    return path, legacy


def _site_path(home: Path) -> Path:
    if not home.is_dir():
        raise SettingsError(
            f"the txtures home {home} does not exist; create it with `txtures site init` "
            "or choose one with `txtures site use PATH`"
        )
    return home / SITE_DOCUMENT


def _known_key(key: str) -> SettingSpec:
    spec = SPEC_BY_KEY.get(key)
    if spec is None:
        raise _unknown(key, "settings")
    return spec


def _check_key(key: str) -> SettingSpec:
    spec = _known_key(key)
    if key == "home":
        raise SettingsError("home chooses the site; choose it with `txtures site use PATH`")
    if spec.location:
        raise SettingsError(
            f"{key} is always HOME/{key}, so it is not a site setting; to put it elsewhere, "
            f"make HOME/{key} a symbolic link, or override it for yourself with "
            f"`txtures env set {key}=PATH`"
        )
    return spec


def check_assignments(
    assignments: Sequence[tuple[str, Sequence[str]]],
    *,
    cwd: Path | None = None,
    check_key: Callable[[str], SettingSpec] | None = None,
) -> dict[str, Any]:
    """Check settings given as keys and their values; return each key's value.

    Each assignment is a key and its values: one value, or any number for a
    list setting. A list setting may be named more than once, and its values
    are collected in order; any other setting may be named once. Relative
    paths are made absolute against `cwd`. `check_key` returns the spec of an
    allowed key, or raises; by default every setting is allowed.

    Raises:
        SettingsError: A key is unknown, not allowed, or repeated, or a value
            is invalid.
    """
    cwd = Path.cwd() if cwd is None else cwd
    check_key = check_key or _known_key
    collected: dict[str, list[str]] = {}
    for key, values in assignments:
        spec = check_key(key)
        if key in collected and spec.kind != "text_list":
            raise SettingsError(f"{key} is given more than once")
        collected.setdefault(key, []).extend(values)
    coerced = {}
    for key, values in collected.items():
        spec = SPEC_BY_KEY[key]
        if spec.kind == "text_list":
            raw: Any = values
        elif len(values) != 1:
            raise SettingsError(f"{key} takes one value")
        else:
            raw = values[0]
        value = _coerce(spec, raw, cwd, key)
        coerced[key] = value.absolute() if isinstance(value, Path) else value
    return coerced


def set_site(home: Path, assignments: Sequence[tuple[str, Sequence[str]]]) -> Path:
    """Set settings in the `site.yml` of the txtures home `home`; return the document's path.

    Assignments are checked as `check_assignments` checks them, so an
    invalid one changes nothing; `home` and the locations are refused, since
    the home's layout fixes them. Other settings keep the values written in
    the document, but comments are not kept.

    Raises:
        SettingsError: The home does not exist, a key is unknown, repeated,
            or not a site setting, a value is invalid, or the document is
            invalid.
    """
    coerced = check_assignments(assignments, check_key=_check_key)
    path = _site_path(home)
    _read_site(path)
    document = _read_raw(path)
    document.update(coerced)
    _write_document(path, document, SITE_HEADER)
    return path


def unset_site(home: Path, keys: Sequence[str]) -> tuple[Path, list[str]]:
    """Remove settings from the `site.yml` of the txtures home `home`.

    Returns the document's path and the keys that were there to remove.

    Raises:
        SettingsError: The home does not exist, a key is unknown or not a
            site setting, or the document is invalid.
    """
    for key in keys:
        _check_key(key)
    path = _site_path(home)
    _read_site(path)
    document = _read_raw(path)
    removed = [key for key in dict.fromkeys(keys) if key in document]
    if removed:
        for key in removed:
            del document[key]
        _write_document(path, document, SITE_HEADER)
    return path, removed
