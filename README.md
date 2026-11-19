# txtures

txtures computes word- and region-level features of text for psycholinguistic
research. Its first feature is surprisal from causal language models on the
Hugging Face hub, with word-boundary correction and user-defined regions.

txtures is in early development. Its interfaces will change during the 0.x
series.

## Installation

From a checkout on Linux or macOS:

```bash
./install
```

This downloads a pinned uv and Python and installs txtures with its locked
dependencies into `txtures-install/` in the checkout, without using or
changing any Python already on the machine. It puts a `txtures` command in
`~/.local/bin`. Useful options:

- `--compute cpu` installs PyTorch without CUDA (the default on Linux is CUDA).
- `--site DIR` uses a lab's txtures home, which holds its definitions,
  stored results, and models. `txtures site use DIR` does the same later.
- `--prefix DIR` installs somewhere other than the checkout.

On a Slurm cluster, `txtures site set execution.executor=slurm
slurm.partition=NAME` makes `txtures run` submit its computation as a job
and return at once (`slurm.account`, `slurm.gpus`, and others are optional).
Follow the job with `txtures status`.

Run `./install --help` for every option and `txtures site ls` to see the
resulting settings. For gated models, run `txtures hf login` once.

txtures can also be installed with pip or uv as a dependency of other
software; see
[Settings](https://txtures.readthedocs.io/en/latest/settings/#installation).

## Use

Put a file of sentences, one per line, in the inputs directory as
`gardenpath.txt`, then:

```bash
txtures run gardenpath
```

This writes a table with one row per word and its surprisal under GPT-2 to
the outputs directory. From Python:

```python
import txtures as tx

table = tx.compute("surprisal", ["The horse raced past the barn fell."])
```

See the [documentation](https://txtures.readthedocs.io), starting with
[Getting started](https://txtures.readthedocs.io/en/latest/getting-started/).

## Development

```bash
./install --dev
txtures-install/env/bin/python -m pytest -q
```

See [CONTRIBUTING.md](https://github.com/CLiMBlab-Stanford/txtures/blob/main/CONTRIBUTING.md)
for contribution, attribution, and release rules, and
[WRITING_POLICY.md](https://github.com/CLiMBlab-Stanford/txtures/blob/main/WRITING_POLICY.md)
for documentation style.

## License

MIT. See [LICENSE](https://github.com/CLiMBlab-Stanford/txtures/blob/main/LICENSE).
