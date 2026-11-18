# Settings

Settings choose where textures reads and writes and how it computes. They
never change computed values; anything that does is a config parameter (see
[Definitions](definitions.md)). `textures site show` prints every setting
with its value and source, and `--json` prints the same as JSON.

## Settings and sources

| Key | Default | Meaning |
| --- | --- | --- |
| `definitions` | see below | the definitions store |
| `inputs` | `~/textures-store/inputs` | input files |
| `derivatives` | `~/textures-store/derivatives` | output tables |
| `items` | see below | stored per-document results |
| `resources` | see below | the resource store and the Hugging Face cache |
| `execution.device` | `auto` | `auto` (CUDA, then Apple MPS, then CPU), or a PyTorch device such as `cpu` or `cuda:1` |
| `execution.executor` | `local` | `local` or `slurm` (see [Slurm](slurm.md)) |
| `execution.max_variants` | 64 | the most variants one module may expand to |

The `slurm` settings are listed in [Slurm](slurm.md#settings).

Where lab storage at `/juice6/u/nlp/climblab` is readable, `definitions` is
`textures-definitions` there; otherwise it is `~/textures-definitions`.
`items` is `textures-store/items` and `resources` is `textures-resources` in
lab storage when those locations are writable, and under the home directory
otherwise. The three are siblings, kept apart from code.

Each setting takes the first value found in these sources:

1. a command-line flag: `--definitions`, `--inputs`, `--derivatives`,
   `--items`, or `--resources`, given before the command;
2. an environment variable: `TEXTURES_DEFINITIONS`, `TEXTURES_INPUTS`,
   `TEXTURES_DERIVATIVES`, `TEXTURES_ITEMS`, or `TEXTURES_RESOURCES`;
3. the installation record, which holds `resources` and
   `execution.executor` as chosen by `./install`;
4. personal settings at `$XDG_CONFIG_HOME/textures/settings.yml`
   (by default `~/.config/textures/settings.yml`);
5. site settings at `site/site.yml` in the definitions store;
6. the built-in default.

Settings documents use nested YAML:

```yaml
inputs: /path/to/inputs
execution:
  device: cuda
slurm:
  partition: gpu
```

Relative paths in a document resolve against the document's directory, and
relative paths from flags and environment variables against the working
directory. Unknown keys are errors. Site settings cannot set `definitions`,
since the definitions store holds them; personal settings can.

Personal settings let one user keep a private inputs directory or store
without editing the shared site document. To keep studies apart, point
`inputs` and `derivatives` at different directories.

## Installation

`./install` in a checkout creates a self-contained installation in
`CHECKOUT/textures-install/`, with a pinned uv, a uv-managed CPython, and the
locked dependencies from `uv.lock`. The installation holds software only;
shared data lives in `resources`.

| Option | Meaning |
| --- | --- |
| `--prefix DIR` | install into `DIR` instead of the checkout |
| `--compute cuda\|cpu` | the PyTorch build; default `cuda` on Linux and `cpu` on macOS |
| `--resources DIR` | the `resources` directory to record; an existing one is shared |
| `--executor local\|slurm` | the `execution.executor` to record |
| `--bin-dir DIR` | where to put the `textures` command; default `~/.local/bin` |
| `--no-launcher` | do not create the `textures` command |
| `--default` | make the `textures` command run this installation, replacing one for another installation |
| `--dev` | also install the development tools |

Running `./install` again updates the installation and keeps every recorded
choice that is not given again. The first installation records the default
`resources` directory, so a later change to the defaults does not move it.
The `textures` command in `--bin-dir` is replaced only if textures wrote it.

Every command runs the installation's Python in isolated mode, so
`PYTHONPATH`, user site packages, and the working directory cannot change
which code runs.

## Environment

Hugging Face libraries read their cache locations, token, and offline mode
from environment variables. Every textures command removes the `HF_*`,
`HUGGINGFACE_*`, and `TRANSFORMERS_*` variables it inherits and sets its own,
so the caller's shell cannot change where models come from:

| Variable | Value |
| --- | --- |
| `HF_HOME` | `RESOURCES/huggingface` |
| `HF_HUB_CACHE` | `RESOURCES/huggingface/hub` |
| `HF_TOKEN_PATH` | `hf-token` beside personal settings |
| `HF_HUB_DISABLE_TELEMETRY` | `1` |
| `TOKENIZERS_PARALLELISM` | `false` |

## Gated models

Some models on the hub require accepting a license and authenticating. Save
your token once:

```bash
textures hf login      # reads the token from the terminal, or from stdin
textures hf whoami     # shows the account the saved token belongs to
textures hf logout     # deletes the saved token
```

`login` checks the token with the hub and saves it beside your personal
settings, readable only by you. Tokens are personal and never stored in the
shared `resources` directory.
