# textures

textures computes word- and region-level features of text for psycholinguistic
research. Its first feature is surprisal from causal language models on the
Hugging Face hub, with word-boundary correction and user-defined regions.

textures is in early development. Its interfaces will change during the 0.x
series.

## Installation

From a checkout on Linux or macOS:

```bash
./install
```

This downloads a pinned uv and Python and installs textures with its locked
dependencies into `textures-install/` in the checkout, without using or
changing any Python already on the machine. It puts a `textures` command in
`~/.local/bin`. Useful options:

- `--compute cpu` installs PyTorch without CUDA (the default on Linux is CUDA).
- `--resources DIR` chooses the shared directory for downloaded models and
  other resources. Point it at an existing directory to share it.
- `--prefix DIR` installs somewhere other than the checkout.

On a Slurm cluster, `./install --executor slurm` makes `textures run` submit
its computation as a job and return at once; set `slurm.partition` (and
optionally `account`, `gpus`, `memory_gb`, `time_hours`, `workers`) in your
settings, and follow the job with `textures status`.

Run `./install --help` for every option and `textures site show` to see the
resulting settings. For gated models, run `textures hf login` once.

## Use

Put a file of sentences, one per line, in the inputs directory as
`gardenpath.txt`, then:

```bash
textures run gardenpath
```

This writes a table with one row per word and its surprisal under GPT-2 to
the derivatives directory. From Python:

```python
import textures as tx

table = tx.compute("surprisal", ["The horse raced past the barn fell."])
```

See the [documentation](docs/README.md), starting with
[Getting started](docs/getting-started.md).

## Development

```bash
./install --dev
textures-install/env/bin/python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution, attribution, and
release rules, and [WRITING_POLICY.md](WRITING_POLICY.md) for documentation
style.

## License

MIT. See [LICENSE](LICENSE).
