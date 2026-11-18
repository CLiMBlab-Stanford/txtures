# Settings

Settings choose where textures reads and writes and how it computes. They
never change computed values; anything that does is a config parameter (see
[Definitions](definitions.md)).

## Changing settings

```bash
textures site ls                        # every setting: value, source, and meaning
textures site use /path/to/definitions  # use a lab's definitions store
textures site set slurm.partition gpu   # set one setting
textures site unset slurm.partition     # remove it again
```

`use`, `set`, and `unset` edit your personal settings file (see below),
which every textures you run reads, however it was installed. `use` sets
`definitions`, and the store's `site/site.yml` then supplies the lab's other
locations. `set` makes relative paths absolute. A list setting takes several
values, given after `--` when they start with a dash:
`textures site set slurm.options -- --exclude=node1`. When a higher-priority
source overrides what was written, the command says so. `site ls --json`
prints the same listing as JSON, and `site show` is another name for `ls`.

When `definitions`, `items`, and `resources` all come from built-in
defaults, textures shares no results with anyone, which on a lab machine is
usually an oversight. The command line then prints a note, and the Python API
issues a `DefaultLocationsWarning`, once per process. Setting any of the three
in any way silences it.

## Settings and sources

| Key | Default | Meaning |
| --- | --- | --- |
| `definitions` | `~/textures-definitions` | the definitions store |
| `inputs` | `~/textures-store/inputs` | input files |
| `derivatives` | `~/textures-store/derivatives` | output tables |
| `items` | `~/textures-store/items` | stored per-document results |
| `resources` | `~/textures-resources` | the resource store and the Hugging Face cache |
| `execution.device` | `auto` | `auto` (CUDA, then Apple MPS, then CPU), or a PyTorch device such as `cpu` or `cuda:1` |
| `execution.executor` | `local` | `local` or `slurm` (see [Slurm](slurm.md)) |
| `execution.max_variants` | 64 | the most variants one module may expand to |

The `slurm` settings are listed in [Slurm](slurm.md#settings).

The defaults keep data in the home directory, apart from code. A lab that
shares its stores keeps their locations in the `site/site.yml` of its
definitions store, and each member runs `textures site use` once.

Each setting takes the first value found in these sources:

1. a command-line flag: `--definitions`, `--inputs`, `--derivatives`,
   `--items`, or `--resources`, given before the command;
2. an environment variable: `TEXTURES_DEFINITIONS`, `TEXTURES_INPUTS`,
   `TEXTURES_DERIVATIVES`, `TEXTURES_ITEMS`, or `TEXTURES_RESOURCES`;
3. the installation record, which holds `resources` or
   `execution.executor` when they were given to `./install`;
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
| `--site DIR` | use the definitions store `DIR`, as `textures site use DIR` does |
| `--resources DIR` | fix this installation's `resources`; an existing directory is shared |
| `--executor local\|slurm` | fix this installation's `execution.executor` |
| `--bin-dir DIR` | where to put the `textures` command; default `~/.local/bin` |
| `--no-launcher` | do not create the `textures` command |
| `--default` | make the `textures` command run this installation, replacing one for another installation |
| `--dev` | also install the development tools |

Running `./install` again updates the installation and keeps every recorded
choice that is not given again. `--resources` and `--executor` are recorded
in the installation and outrank personal and site settings, so give them only
to fix a value for this installation; otherwise leave them to settings. The
`textures` command in `--bin-dir` is replaced only if textures wrote it.

textures can also be installed with pip or uv as a dependency of other
software, for example with
`uv add "textures[cuda] @ git+https://github.com/CLiMBlab-Stanford/textures"`,
or without an extra alongside a PyTorch you install yourself. The `cpu`
extra selects PyTorch's CPU-only build only within `./install`; elsewhere,
install that build yourself. Such an installation has no record and uses
settings alone. It still provides
the `textures` command in its environment, so `textures site use` works
there too.

Every command runs the installation's Python in isolated mode, so
`PYTHONPATH`, user site packages, and the working directory cannot change
which code runs.

## Environment

Hugging Face libraries read their cache locations, token, and offline mode
from environment variables. Every `textures` command, and every Slurm task,
removes the `HF_*`, `HUGGINGFACE_*`, and `TRANSFORMERS_*` variables it
inherits and sets its own, so the caller's shell cannot change where models
come from:

| Variable | Value |
| --- | --- |
| `HF_HOME` | `RESOURCES/huggingface` |
| `HF_HUB_CACHE` | `RESOURCES/huggingface/hub` |
| `HF_TOKEN_PATH` | `hf-token` beside personal settings |
| `HF_HUB_DISABLE_TELEMETRY` | `1` |
| `TOKENIZERS_PARALLELISM` | `false` |

The Python API runs inside other programs, so it changes no environment
variables. It passes the cache and the saved token to the Hugging Face
libraries directly.

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
shared `resources` directory. Without a saved token, the Python API lets the
Hugging Face libraries find one as they usually do, such as from `HF_TOKEN`.
