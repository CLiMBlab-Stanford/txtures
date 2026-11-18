# Settings

Settings choose where txtures reads and writes and how it computes. They
never change computed values; anything that does is a config parameter (see
[Definitions](definitions.md)).

## Changing settings

```bash
txtures site ls                        # every setting: value, source, and meaning
txtures site use /path/to/definitions  # use a lab's definitions store
txtures site set slurm.partition gpu   # set one setting
txtures site unset slurm.partition     # remove it again
```

`use`, `set`, and `unset` edit your personal settings file (see below),
which every txtures you run reads, however it was installed. `use` sets
`definitions`, and the store's `site/site.yml` then supplies the lab's other
locations. `set` makes relative paths absolute. A list setting takes several
values, given after `--` when they start with a dash:
`txtures site set slurm.options -- --exclude=node1`. When a higher-priority
source overrides what was written, the command says so. `site ls --json`
prints the same listing as JSON, and `site show` is another name for `ls`.

When `definitions`, `items`, and `resources` all come from built-in
defaults, txtures shares no results with anyone, which on a lab machine is
usually an oversight. The command line then prints a note, and the Python API
issues a `DefaultLocationsWarning`, once per process. Setting any of the three
in any way silences it.

## Settings and sources

| Key | Default | Meaning |
| --- | --- | --- |
| `definitions` | `~/txtures-definitions` | the definitions store |
| `inputs` | `~/txtures-store/inputs` | input files |
| `derivatives` | `~/txtures-store/derivatives` | output tables |
| `items` | `~/txtures-store/items` | stored per-document results |
| `resources` | `~/txtures-resources` | the resource store and the Hugging Face cache |
| `execution.device` | `auto` | `auto` (CUDA, then Apple MPS, then CPU), or a PyTorch device such as `cpu` or `cuda:1` |
| `execution.executor` | `local` | `local` or `slurm` (see [Slurm](slurm.md)) |
| `execution.max_variants` | 64 | the most variants one module may expand to |

The `slurm` settings are listed in [Slurm](slurm.md#settings).

The defaults keep data in the home directory, apart from code. A lab that
shares its stores keeps their locations in the `site/site.yml` of its
definitions store, and each member runs `txtures site use` once.

Each setting takes the first value found in these sources:

1. a command-line flag: `--definitions`, `--inputs`, `--derivatives`,
   `--items`, or `--resources`, given before the command;
2. an environment variable: `TXTURES_DEFINITIONS`, `TXTURES_INPUTS`,
   `TXTURES_DERIVATIVES`, `TXTURES_ITEMS`, or `TXTURES_RESOURCES`;
3. the installation record, which holds `resources` or
   `execution.executor` when they were given to `./install`;
4. personal settings at `$XDG_CONFIG_HOME/txtures/settings.yml`
   (by default `~/.config/txtures/settings.yml`);
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
`CHECKOUT/txtures-install/`, with a pinned uv, a uv-managed CPython, and the
locked dependencies from `uv.lock`. The installation holds software only;
shared data lives in `resources`.

| Option | Meaning |
| --- | --- |
| `--prefix DIR` | install into `DIR` instead of the checkout |
| `--compute cuda\|cpu` | the PyTorch build; default `cuda` on Linux and `cpu` on macOS |
| `--site DIR` | use the definitions store `DIR`, as `txtures site use DIR` does |
| `--resources DIR` | fix this installation's `resources`; an existing directory is shared |
| `--executor local\|slurm` | fix this installation's `execution.executor` |
| `--bin-dir DIR` | where to put the `txtures` command; default `~/.local/bin` |
| `--no-launcher` | do not create the `txtures` command |
| `--default` | make the `txtures` command run this installation, replacing one for another installation |
| `--dev` | also install the development tools |

Running `./install` again updates the installation and keeps every recorded
choice that is not given again. `--resources` and `--executor` are recorded
in the installation and outrank personal and site settings, so give them only
to fix a value for this installation; otherwise leave them to settings. The
`txtures` command in `--bin-dir` is replaced only if txtures wrote it.

txtures can also be installed with pip or uv as a dependency of other
software, for example with `uv add "txtures[cuda]"` or
`pip install "txtures[cuda]"`, or without an extra alongside a PyTorch you
install yourself. The `cpu`
extra selects PyTorch's CPU-only build only within `./install`; elsewhere,
install that build yourself. Such an installation has no record and uses
settings alone. It still provides
the `txtures` command in its environment, so `txtures site use` works
there too.

Every command runs the installation's Python in isolated mode, so
`PYTHONPATH`, user site packages, and the working directory cannot change
which code runs.

## Environment

Hugging Face libraries read their cache locations, token, and offline mode
from environment variables. Every `txtures` command, and every Slurm task,
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
txtures hf login      # reads the token from the terminal, or from stdin
txtures hf whoami     # shows the account the saved token belongs to
txtures hf logout     # deletes the saved token
```

`login` checks the token with the hub and saves it beside your personal
settings, readable only by you. Tokens are personal and never stored in the
shared `resources` directory. Without a saved token, the Python API lets the
Hugging Face libraries find one as they usually do, such as from `HF_TOKEN`.
