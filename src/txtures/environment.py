"""The process environment that txtures controls, and Hugging Face access.

Libraries that txtures uses read configuration from environment variables:
Hugging Face libraries choose caches, tokens, and offline behavior from
`HF_*`, `HUGGINGFACE_*`, and `TRANSFORMERS_*` variables. In processes that
txtures owns (the `txtures` command and Slurm tasks), `apply` removes any
such variables they inherit and sets its own, so results and cache locations
do not depend on the caller's shell. `apply` must run before the Hugging Face
libraries are imported, because they read these variables at import time.

The Python API leaves the environment of its host process alone. txtures
does not need it: `run_context` passes the cache and the token to the
Hugging Face libraries directly.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from txtures.modules import RunContext
from txtures.settings import Settings, config_directory

MANAGED_PREFIXES = ("HF_", "HUGGINGFACE_", "TRANSFORMERS_")


def token_path(env: MutableMapping[str, str], home: Path) -> Path:
    """Where the user's Hugging Face token is kept: in the txtures configuration directory.

    Tokens are personal, so they never live in the shared cache.
    """
    return config_directory(env, home) / "hf-token"


def saved_token(
    env: MutableMapping[str, str] | None = None, home: Path | None = None
) -> str | None:
    """Return the token saved by `txtures hf login`, or `None` if there is none."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    try:
        token = token_path(env, home).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def run_context(settings: Settings) -> RunContext:
    """The execution context for modules: device, Hugging Face cache, and token.

    Without a saved token, the Hugging Face libraries look for one as they
    usually do, so a host process's own login still works through the API.
    """
    return RunContext(settings.device, settings.hf_cache, saved_token())


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
