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
- `--site DIR` uses a lab's definitions store, whose site settings name the
  shared stores of models and results. `textures site use DIR` does the same
  later.
- `--prefix DIR` installs somewhere other than the checkout.

On a Slurm cluster, `./install --executor slurm` makes `textures run` submit
its computation as a job and return at once; set the partition with
`textures site set slurm.partition NAME` (and optionally `slurm.account`,
`slurm.gpus`, and others), and follow the job with `textures status`.

Run `./install --help` for every option and `textures site ls` to see the
resulting settings. For gated models, run `textures hf login` once.

textures can also be installed with pip or uv as a dependency of other
software; see [Settings](docs/settings.md#installation).

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
