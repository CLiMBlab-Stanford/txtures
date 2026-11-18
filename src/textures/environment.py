"""The process environment that textures controls.

Libraries that textures uses read configuration from environment variables:
Hugging Face libraries choose caches, tokens, and offline behavior from
`HF_*`, `HUGGINGFACE_*`, and `TRANSFORMERS_*` variables. textures removes any
such variables it inherits and sets its own, so results and cache locations
do not depend on the caller's shell.

`apply` must run before the Hugging Face libraries are imported, because they
read these variables at import time. textures imports them only when a model
is first used, after the command line has called `apply`.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from textures.settings import Settings, personal_settings_path

MANAGED_PREFIXES = ("HF_", "HUGGINGFACE_", "TRANSFORMERS_")


def token_path(env: MutableMapping[str, str], home: Path) -> Path:
    """Where the user's Hugging Face token is kept: beside personal settings.

    Tokens are personal, so they never live in the shared cache.
    """
    return personal_settings_path(env, home).with_name("hf-token")


def apply(
    settings: Settings,
    env: MutableMapping[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, str]:
    """Replace the managed variables in `env` (the process environment by default).

    Returns the variables that were set.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    for key in [key for key in env if key.startswith(MANAGED_PREFIXES)]:
        del env[key]
    values = {
        "HF_HOME": str(settings.hf_home),
        "HF_HUB_CACHE": str(settings.hf_cache),
        "HF_TOKEN_PATH": str(token_path(env, home)),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env.update(values)
    return values
