# txtures documentation

- [Getting started](getting-started.md): install txtures, compute surprisal
  for a first input, and read the result.
- [Inputs](inputs.md): the file layouts txtures reads, sidecar settings, and
  the checks every input passes.
- [Definitions](definitions.md): configs, workflows, sweeps, and the column
  names that sweeps produce.
- [Surprisal](surprisal.md): the `lm_scores` and `surprisal` modules, how
  text is tokenized, and the word-boundary correction.
- [Output tables](outputs.md): what `txtures run` writes, how txtures
  decides what is current, and the shared stores.
- [Settings](settings.md): storage locations, execution settings, the
  installation, and Hugging Face access.
- [Slurm](slurm.md): submitting computation to a Slurm cluster.
- [Python API](python.md): `tx.compute`, `tx.run`, and `tx.read_table`.
