# Getting started

## Install

From a checkout on Linux or macOS:

```bash
./install
```

The script downloads a pinned uv and Python, installs txtures with the
locked dependencies into `txtures-install/` in the checkout, and puts a
`txtures` command in `~/.local/bin`. It does not use or change any Python,
conda environment, or uv configuration already on the machine. Add
`--compute cpu` on a machine without an NVIDIA GPU. [Settings](settings.md#installation)
lists every option.

If your lab shares a definitions store, point txtures at it once:

```bash
txtures site use /path/to/txtures-definitions
```

Its site settings then choose the shared locations, so you reuse models and
results that others have computed. Without one, txtures keeps everything in
your home directory and says so with a note. Check the result:

```bash
txtures site ls
```

This lists every setting, its value, and where the value came from. The
locations that matter first are `inputs`, where txtures reads your files,
and `derivatives`, where it writes output tables.

## Compute surprisal

Put one document per line in a `.txt` file in the inputs directory:

```text
The horse raced past the barn fell.
After the student moved the chair broke.
```

Saved as `gardenpath.txt`, it is the input `gardenpath`. Check it, then see
what a run would do:

```console
$ txtures inputs check gardenpath
gardenpath: plain-text
  documents: 2
  regions: 14
  digest: 96ce0d0d8e50662523ccaeddbb31b4540d30f3c4be74f6dc3027551d8b9adfde
$ txtures plan gardenpath
input gardenpath: plain-text, 2 documents, 14 regions
workflow main
table .../derivatives/gardenpath/main-68ce5fbefa0d (to be written)
resources:
  hf_model 81d9ad54cbc5: missing
variants:
  lm_scores [main]: 0 stored, 2 missing
  surprisal [main]: computed when the table is written
columns: surprisal
note: lm_scores variant main: model gpt2 has no pinned revision; ...
```

The packaged `main` workflow scores text with GPT-2 and reports surprisal in
bits with word-boundary correction. Run it:

```console
$ txtures run gardenpath
building resource hf_model 81d9ad54cbc5
lm_scores [main]: computing 2 of 2 items
lm_scores [main]: 2 computed, 0 reused
wrote .../derivatives/gardenpath/main-68ce5fbefa0d
```

The table has one row per region. By default a region is a
whitespace-delimited word:

```text
doc_id  region_index  region  char_start  char_end  surprisal
1       1             The     0           3         4.734790383357864
1       2             horse   4           9         13.651452087029186
1       3             raced   10          15        10.626696552055186
```

Running the same command again finds the table current and does nothing.
After you edit one line of the input, a run rescores only that document.

## Choose a model

Models and other scientific parameters live in a definitions store, not on
the command line. Create one, then add a config and a workflow that uses it:

```bash
txtures def init
```

```yaml
# configs/lm_scores/pythia410m.yml
extends: main
model: EleutherAI/pythia-410m
revision: 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
```

```yaml
# workflows/pythia.yml
extends: main
lm_scores: pythia410m
```

```bash
txtures def validate
txtures run gardenpath --workflow pythia
```

Pin `revision` to a commit hash so that later runs use the same weights.
[Definitions](definitions.md) explains configs, workflows, and comparing
several models in one table.

## From Python

```python
import txtures as tx

table = tx.compute(
    "surprisal",
    ["The horse raced past the barn fell."],
    config={"lm_scores": {"model": "gpt2"}},
)
```

`table` is a pandas DataFrame with the same columns as an output table. See
[Python API](python.md).
