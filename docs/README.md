# txtures documentation

- [Getting started](getting-started.md): install txtures, compute surprisal
  for a first input, and read the result.
- [Inputs](inputs.md): the file layouts txtures reads, sidecar settings, and
  the checks every input passes.
- [Definitions](definitions.md): configs, workflows, sweeps, and the column
  names that sweeps produce.
- [Surprisal](surprisal.md): the `lm_scores` and `surprisal` modules, how
  text is tokenized, and the word-boundary correction.
- [N-gram surprisal](ngram.md): unigram and n-gram surprisal from KenLM
  models trained on a corpus.
- [Word length](word-length.md): the `word_length` module.
- [Parsing](parsing.md): the `parse`, `pos`, and `named_entity` modules.
- [Dependency locality](dependency-locality.md): DLT integration cost, its
  variants, and the analysis of relative clauses.
- [GCG treebank](gcg.md): Penn Treebank trees reannotated into generalized
  categorial grammar.
- [Output tables](outputs.md): what `txtures run` writes, how txtures
  decides what is current, and the shared stores.
- [Settings](settings.md): storage locations, execution settings, the
  installation, and Hugging Face access.
- [Slurm](slurm.md): submitting computation to a Slurm cluster.
- [Tools](tools.md): the Apptainer image that holds non-Python programs
  such as KenLM.
- [Python API](python.md): `tx.compute`, `tx.run`, and `tx.read_table`.
