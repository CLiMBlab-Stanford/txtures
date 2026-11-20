# Python API

```python
import txtures as tx
```

The API runs the same modules, stores, and settings as the command line.
Where Slurm is set up, `tx.compute` and `tx.run` submit work that runs
language models as a Slurm job and wait for it (see [Slurm](#slurm)).

## `tx.compute`

`tx.compute(modules, texts, ...)` computes features of text given directly
and returns a pandas DataFrame with the columns of an output table. It writes
no table, but stores and reuses per-document results like a run, so calling
it again on the same text with the same configs is fast.

```python
table = tx.compute(
    "surprisal",
    ["The horse raced past the barn fell.", "The old man the boat."],
    config={
        "lm_scores": {"model": "gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
        "surprisal": {"correction": ["word_boundary", "none"]},
    },
)
```

```text
  document_id  region_index   text  char_start  char_end  surprisal_wordboundary  surprisal_none
0           1             1    The           0         3                4.734790        4.729301
1           1             2  horse           4         9               13.651452       13.264466
```

`modules` names one feature module or a list of them. `texts` is one
document, a list of documents (numbered from 1), or a dictionary from
document ID to document. Each document is a string, divided into regions
like a line of a `.txt` input, or a list of region strings:

```python
tx.compute("surprisal", {"item1": ["The horse", "raced", "past the barn", "fell."]})
```

`region_delimiter`, `sentence_delimiter`, `region_joiner`, and
`sentence_joiner` divide and join string documents as an input's sidecar
does (see [Inputs](inputs.md#sidecar-settings)).

Configs come from one of two places:

- `config` maps module names to config bodies. Each body overrides that
  module's packaged defaults and may sweep parameters as a config file can.
  The definitions store is not read, so the call alone determines the
  result. Modules not named use their defaults.
- Without `config`, the definitions store's workflow (`workflow`, default
  `main`) chooses the configs, as for `txtures run`.

Giving both `config` and `workflow` is an error.

## `tx.run`

```python
table = tx.run("gardenpath", workflow="pythia", modules=["surprisal"])
```

`tx.run(name, workflow="main", modules=None)` is
`txtures run NAME --workflow WORKFLOW --module ...`. It computes what is
missing, writes the output table to the outputs directory unless it is
current, and returns the table as read by `tx.read_table`.

## `tx.read_table`

```python
table = tx.read_table("outputs/gardenpath/main-68ce5fbefa0d")
```

`tx.read_table(path)` reads an output table, given its directory or its
`NAME.tsv`. Location and metadata columns stay text, so a region such as `NA`
stays a string. Feature columns, as listed in `NAME.columns.tsv`, become
numbers, with `NA` read as NaN and `Inf` as infinity, or Booleans.

## Slurm

With the default `execution.executor`, `auto`, `tx.compute` and `tx.run`
compute what is missing with a Slurm job when Slurm work, such as
`lm_scores` results, is missing and Slurm is set up on this machine, and block until
the job ends. Otherwise they compute here. `slurm` submits every request
that has anything missing (see [Slurm](slurm.md#choosing-local-or-slurm)). They then
return the same table a local call would. If nothing is missing, nothing is
submitted. `slurm={"ngpus": 0, "memory_gb": 64}` sets the size of the jobs a
call submits (see [Job sizes](slurm.md#job-sizes)).

```python
table = tx.compute("surprisal", texts, config=config, nworkers=4)
```

```text
submitted job 81234 (4 tasks); logs: .../outputs/.txtures-jobs/tx.compute/main-3f9a0c1d2e4b/logs
job 81234: 4 PENDING
job 81234: 4 RUNNING
job 81234: 4 COMPLETED
```

- `nworkers` sets the number of array tasks (default `slurm.workers`;
  `workers` is an older name for it), and
  `local=True` computes in the calling process instead.
- `tx.compute` writes its text and configs to the job directory, so the
  tasks can read them. They must therefore be JSON values: strings,
  numbers, lists, and dictionaries. The tasks store their results in the
  cache, and the call builds the table from there.
- If a job for the same request is already pending or running, the call
  waits for that job instead of submitting another.
- Interrupting the call, for example with Ctrl-C, cancels the job it
  submitted. Results that tasks already stored stay in the cache. A job the
  call only waited for keeps running.
- If the job ends without computing everything, the call raises
  `TxturesError` and names the log directory.

Without `progress`, these job messages go to standard error. [Slurm](slurm.md)
describes the settings, how tasks share work, and what compute nodes need
to reach.

## Common options

`tx.compute` and `tx.run` accept keyword arguments that override settings for
one call: `home`, `definitions`, `cache`, `resources`, and `device`, and
for `tx.run` also `inputs` and `outputs`. `local` and `nworkers` choose how
the call computes (see [Slurm](#slurm)). `progress`, if given, is called with
each progress message, so `progress=print` shows what a long call is doing.

Planning notes, such as an unpinned model revision, and input warnings are
issued as Python warnings. Invalid input, definitions, or settings raise
`txtures.errors.TxturesError` or one of its subclasses. `InputError` and
`DefinitionError` list every problem in their `problems` attribute.

## Settings and the environment

The API reads the same settings as the command line, so `txtures site use`,
`txtures site set`, and `txtures env set` configure both. Software that depends on
txtures can instead pass locations as arguments or set the `TXTURES_*`
environment variables. When every shared location comes from built-in
defaults, the first call issues a `txtures.api.DefaultLocationsWarning`.

The API does not change the process environment. It passes its Hugging Face
cache and your saved token to the Hugging Face libraries directly, so a host
program's own use of those libraries is unaffected.
