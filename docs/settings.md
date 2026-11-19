# Settings

Settings choose where txtures reads and writes and how it computes. They
never change computed values; anything that does is a config parameter (see
[Definitions](definitions.md)).

## The txtures home

Everything txtures keeps lives in one directory, the txtures home, which is
also your site:

```text
HOME/
├── site.yml       site settings: execution and Slurm limits
├── definitions/   configs and workflows (see Definitions)
├── inputs/        input files
├── outputs/       output tables
├── cache/         computed per-document results, reused by every run
└── resources/     models and the Hugging Face cache
```

```bash
txtures site init /path/to/txtures-home   # create a home
txtures site use /path/to/txtures-home    # choose your home
txtures site ls                           # every setting: value, source, and meaning
```

`site init` creates the layout with an example config and workflow, and
makes the home a Git repository unless given `--no-git`, so a lab's changes
to its settings and definitions can be reviewed and undone. Its
`.gitignore` admits only `site.yml`, `README.md`, `definitions/`, and
itself: the data directories, and anything else you keep in the home, are
never tracked. The repository is shared with the group when the home's
parent directory is group-writable; `site init` makes no commit. Without a
path, it creates the home in use, by default `~/txtures-home`.

The path may be an existing directory, as long as it has none of the
home's entries (`site.yml`, `README.md`, `.gitignore`, and the directories
above). Other contents are left as they are. A checkout of txtures can
live in the home, for example as `HOME/code`: the home's `.gitignore`
leaves it out, so the two repositories stay independent, and Git commands
run inside the checkout act on the checkout. An existing repository at the
path is left as it is.

`site use` records the home's path in your home-path file,
`$XDG_CONFIG_HOME/txtures/home-path`, by default
`~/.config/txtures/home-path`, a plain text file holding only that path.
Every use of txtures reads it: the command line, the Python API, and every
installation or program that depends on txtures, so one `site use`
configures all of them.

The layout is fixed, so a home's settings never move its directories. To
keep a directory elsewhere, such as `resources` on a larger filesystem,
make it a symbolic link. Settings never enter a result's identity, so
moving or relinking a directory never invalidates stored results.

When the home is the default and nothing overrides `definitions`, `cache`,
or `resources`, txtures shares no results with anyone, which on a lab
machine is usually an oversight. The command line then prints a note, and
the Python API issues a `DefaultLocationsWarning`, once per process.

## Changing site settings

`site.yml` holds the settings that are not locations:

| Key | Default | Meaning |
| --- | --- | --- |
| `execution.device` | `auto` | `auto` (CUDA, then Apple MPS, then CPU), or a PyTorch device such as `cpu` or `cuda:1` |
| `execution.executor` | `auto` | `auto` uses Slurm for intensive work where it is set up; `slurm` or `local` always (see [Slurm](slurm.md#choosing-local-or-slurm)) |
| `execution.max_variants` | 64 | the most variants one module may expand to |

The `slurm` settings are listed in [Slurm](slurm.md#settings).

```bash
txtures site set slurm.partition=gpu slurm.account=mylab   # change settings
txtures site unset slurm.partition                         # return to the default
```

`set` takes any number of `KEY=VALUE` pairs and checks them all before
writing, so an invalid one changes nothing. A value may itself contain `=`.
For a list setting, repeat its pair:
`txtures site set slurm.options=--exclude=node1 slurm.options=--qos=low`.
`set` also accepts one setting as `KEY VALUE...`, with the values given
after `--` when they start with a dash:
`txtures site set slurm.options -- --exclude=node1`. `set` keeps other
values as written but does not keep comments. `unset` takes any number of
keys. When a flag or environment variable overrides what was written, the
command says so. `site ls --json` prints the same listing as JSON, and
`site show` is another name for `ls`.

When several people share a home, a change to `site.yml` applies to all of
them.

`site.yml` uses nested YAML, and unknown keys are errors:

```yaml
execution:
  device: cuda
slurm:
  partition: gpu
```

## Departing from your site

Every setting has an environment variable: the key in capitals, with dots
as underscores and `TXTURES_` in front, such as `TXTURES_HOME`,
`TXTURES_CACHE`, and `TXTURES_SLURM_PARTITION`. A list variable such as
`TXTURES_SLURM_OPTIONS` holds its values as shell words, as in
`TXTURES_SLURM_OPTIONS="--qos=low --exclude=node1"`.

Each setting takes the first value found in these sources:

1. a command-line flag, given before the command: `--home`,
   `--definitions`, `--inputs`, `--outputs`, `--cache`, or `--resources`;
2. its environment variable;
3. for `home`, the home-path file; for a location, the home's
   subdirectory of the same name; for any other setting, `site.yml`;
4. the built-in default: `~/txtures-home` for `home`.

Relative paths from flags and environment variables resolve against the
working directory.

To depart from a shared site for yourself without changing it, keep such
variables in your shell startup file with `txtures env`, which takes keys
and values as `txtures site` does:

```bash
txtures env set outputs=~/my-outputs slurm.partition=debug   # override the site
txtures env ls                                               # list the variables and whether this shell has them
txtures env unset slurm.partition                            # follow the site again
```

For example, `env set outputs=PATH` writes your output tables privately
while still sharing the lab's cache and models.

`set` and `unset` edit a block of `export` lines marked
`# >>> txtures env >>>` in your shell startup file and leave the rest of the
file unchanged. The file is `~/.zshrc` when `SHELL` is zsh, `~/.bashrc` when
it is bash, and `~/.profile` otherwise; `--file PATH` chooses another. A
change applies to shells started afterwards, or after `source ~/.bashrc`.
`set` makes relative paths absolute. `txtures site ls` shows which settings
come from the environment.

## Moving from earlier versions

Earlier versions kept settings in `~/.config/txtures/settings.yml`, in
`site/site.yml` inside the definitions store, and in the installation
record, and named `outputs` `derivatives` and `cache` `items`. txtures
refuses to run while any of these remain. Its message lists the steps for
your files: which directories to move or link into the home, the
`txtures site use` and `txtures site set` commands that record your home
and its settings, and which old files to remove. `txtures site use` removes
`~/.config/txtures/settings.yml`, and `txtures install` removes settings from
the record. The `site` commands still run meanwhile, with a warning, so the
steps can be carried out.

## Installation

`./install` in a checkout creates a self-contained installation in
`CHECKOUT/txtures-install/`, with a pinned uv, a uv-managed CPython, and the
locked dependencies from `uv.lock`. The installation holds software only.
Its record, `install.json`, notes the checkout, the environment, the PyTorch
build, whether the development tools are installed, where the `txtures`
command was put, and the versions installed. It holds no settings.

| Option | Meaning |
| --- | --- |
| `--prefix DIR` | install into `DIR` instead of the checkout |
| `--compute cuda\|cpu` | the PyTorch build; default `cuda` on Linux and `cpu` on macOS |
| `--site DIR` | use the txtures home `DIR`, as `txtures site use DIR` does |
| `--bin-dir DIR` | where to put the `txtures` command; default `~/.local/bin` |
| `--no-launcher` | do not create the `txtures` command |
| `--default` | make the `txtures` command run this installation, replacing one for another installation |
| `--dev` | also install the development tools |
| `--upgrade` | first update the checkout to the newest release, then install it |

Running `./install` again updates the installation and keeps the recorded
PyTorch build unless `--compute` is given. The `txtures` command in
`--bin-dir` is replaced only if txtures wrote it.

### Updating an installation

`txtures install` runs the checkout's `./install` again for the
installation it belongs to. It keeps the prefix, the PyTorch build, the
development tools, and the location of the `txtures` command, so you do not
have to repeat the options of the first install:

```bash
txtures install --upgrade   # move to the newest release and install it
txtures install             # reinstall the current checkout
```

| Option | Meaning |
| --- | --- |
| `--upgrade` | first update the checkout to the newest release, as `./install --upgrade` does |
| `--compute cuda\|cpu` | switch the PyTorch build |
| `--dev`, `--no-dev` | add or remove the development tools |
| `--default` | make the `txtures` command run this installation |

`txtures install` works only in an installation made by `./install`, and
only while its checkout is still where it was. It works even while your
settings are invalid. Installations made before 0.5.0 did not record the
development tools or the command location. For those, it keeps the
development tools if they are installed and puts the command in
`~/.local/bin`, without replacing one that runs another installation.

To move to a new release, run `txtures install --upgrade`, or
`./install --upgrade` with any other options you use. It fetches the release tags from the checkout's `origin` and moves
the checkout to the newest release: a fast-forward of `main`, or a checkout
of the tag when the checkout is already at a release. It then runs that
release's `./install`, since the script itself may have changed. It refuses
when the checkout has uncommitted changes or is on a branch other than
`main`, and changes nothing when the checkout already includes the newest
release.

txtures can also be installed with pip or uv as a dependency of other
software, for example with `uv add "txtures[cuda]"` or
`pip install "txtures[cuda]"`, or without an extra alongside a PyTorch you
install yourself. The `cpu`
extra selects PyTorch's CPU-only build only within `./install`; elsewhere,
install that build yourself. Such an installation has no record. It still provides
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
| `HF_TOKEN_PATH` | `hf-token` beside the home-path file |
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

`login` checks the token with the hub and saves it beside your
home-path file, readable only by you. Tokens are personal and never stored in the
shared `resources` directory. Without a saved token, the Python API lets the
Hugging Face libraries find one as they usually do, such as from `HF_TOKEN`.
