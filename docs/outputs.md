# Output tables

`txtures run NAME` writes one table directory for the request:

```text
DERIVATIVES/NAME/WORKFLOW-DIGEST12/
├── NAME.tsv           one row per region
├── NAME.columns.tsv   one row per feature column
└── manifest.json      what produced the table
```

`DIGEST12` is the first 12 hexadecimal digits of the table's identity, which
covers the input and, for each feature column, its name and everything that
determines its values: the resolved configs of its module and every module
upstream, the resolved model revisions, and each module's output-contract
version. Different requests for one input therefore write different
directories.

## The table

`NAME.tsv` has these columns, in order:

| Column | Meaning |
| --- | --- |
| `document_id` | the document's ID |
| `section_id`, `paragraph_id`, `sentence_id` | the ID of the region's section, paragraph, and sentence; each is present only when the input defines that level |
| `region_id` | the region's ID; present only when the input has a `region_id` column |
| `region_index` | the region's position in its document, from 1 |
| `text` | the region's text |
| `char_start`, `char_end` | the region's character span in the text the model read |
| metadata | the input's metadata columns, in file order |
| features | one group per requested module, in workflow order |

Values are written so that R and pandas read them back exactly. Missing
values and NaN are `NA`, infinities are `Inf` and `-Inf`, Booleans are
`TRUE` and `FALSE`, and floats use the shortest text that round-trips.

Read tables as text first. A region can be a word such as `NA`, `null`, or
`None`. By default pandas reads all three as missing values, and R reads
`NA` as one. In Python,
`tx.read_table` handles this (see [Python API](python.md)). In R:

```r
table <- read.delim(path, quote = "", na.strings = character(),
                    colClasses = "character")
table$surprisal <- as.numeric(table$surprisal)
```

`NAME.columns.tsv` lists each feature column with its module, feature,
labels such as `lm_scores.model=gpt2`, and variant identity. `manifest.json`
records the table's identity, the request, the full resolved config and
resources of every variant, notes from planning, and the txtures version.
Use these files, not column names, to find the parameters behind a column.

## What is current

txtures decides what to compute from identities, never from modification
times. A run that finds its table already written with the same identity
does nothing. Otherwise it computes only the missing results and writes the
table again.

Tables are views over stored results, so deleting one loses no computation.

```bash
txtures status [NAME ...]    # each table: current, stale, or broken
txtures gc [--dry-run]       # remove stale and broken tables
```

`status` replans the request each table records. A table is current when
replanning gives the same identity, stale when the input, the definitions, or
a resolved revision has changed, and broken when its files are incomplete,
for example after an interrupted write. `gc` removes stale and broken tables
and the records of finished Slurm jobs. It never removes stored results.

## Stores

Results are kept in two content-addressed stores, which a lab can share:

| Setting | Holds |
| --- | --- |
| `cache` | per-document results of persisted modules, such as `lm_scores` |
| `resources` | the resource store (`store/`), with model properties, and the Hugging Face cache (`huggingface/`), with model files |

Two requests share a stored result only when every part of its identity
matches: the module, its contract version, the resolved config, the
resolved resources, and the document's text and boundaries. Sharing among
inputs and users is therefore safe. Entries are written to a temporary path
and renamed into place, so readers never see partial entries, and
concurrent runs that need the same result compute it once.

Each stored entry has its own `manifest.json`, which records what produced
it. Nothing removes entries from these stores yet.
