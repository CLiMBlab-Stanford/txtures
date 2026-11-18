# Definitions

Scientific parameters live in a definitions store, a directory of YAML files.
A run's results depend only on its input and the definitions it resolves, so
the store, which a lab can track in Git, records how every table was made.

```text
DEFINITIONS/
├── site/site.yml          storage and execution settings (see Settings)
├── configs/MODULE/ID.yml  configs, one directory per module
└── workflows/ID.yml       workflows
```

`txtures def init [PATH]` creates a store with an example config and
workflow. `txtures def validate [PATH]` checks every file in it, including
definitions no workflow uses. Without a store, txtures uses its packaged
definitions alone.

## Configs

A config is a named set of parameters for one module, such as
`configs/lm_scores/pythia410m.yml`. IDs are file stems and must match
`[A-Za-z][A-Za-z0-9_]*`. Unknown keys and values of the wrong type are
errors. [Surprisal](surprisal.md) lists the parameters of each module.

Every module has a packaged `main` config made of its defaults. A store's
`configs/MODULE/main.yml`, if present, overrides some of them for every use
of `main`, and cannot use `extends`.

`extends: ID` starts from another config of the same module and overrides
its top-level keys. A config without `extends` starts from the packaged
defaults, not from the store's `main`:

```yaml
# configs/lm_scores/pythia410m.yml
extends: main
model: EleutherAI/pythia-410m
revision: 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
```

Results depend on resolved values, not on how a config was written. Two
configs that resolve to the same values share every stored result.

## Workflows

A workflow chooses one config, or a list of configs, for each module. The
packaged `main` workflow chooses `main` for every module, and a store's
`workflows/main.yml` overrides part of it. Other workflows use `extends` the
same way configs do:

```yaml
# workflows/pythia.yml
extends: main
lm_scores: pythia410m
```

The workflow alone decides which config each module uses. Configs never name
other configs, so here `surprisal` keeps its `main` config and reads the
`lm_scores` results computed under `pythia410m`.

`txtures run NAME --workflow ID` uses a workflow. `--module` (`-m`), which
can be repeated, limits the table to some feature modules; modules they
depend on still run but add no columns.

## Sweeps

A list given for a sweepable parameter means "compute once for each value."
The results become separate columns of one table.

| Module | Sweepable parameters |
| --- | --- |
| `lm_scores` | `model` (entries may also set `revision`), `window` (entries may also set `stride`), `context` |
| `surprisal` | `correction`, `units` |

```yaml
# configs/lm_scores/small.yml
model:
  gpt2: {model: gpt2, revision: 607a30d783dfa663caf39e06633721c8d4cfcd7e}
  pythia70m: {model: EleutherAI/pythia-70m, revision: a39f36b100fe8a5377810d56c3f4789b9c53ac42}
```

A sweep is a mapping from labels to values, as here, or a list whose labels
are derived from the values. A model entry may carry its own `revision`;
when `model` is swept, `revision` cannot also be set outside the entries.
Several sweeps in one config combine into every combination, and variants
propagate downstream: with two `lm_scores` variants, `surprisal` runs once
for each.

A workflow may instead list config IDs, which suits variants that differ in
several parameters at once. Each config ID is a label:

```yaml
# workflows/compare.yml
extends: main
lm_scores: [gpt2, pythia410m]
```

`txtures plan` reports how many variants each module has. A request that
expands past `execution.max_variants` (default 64) is refused until the
setting is raised.

## Column names

A feature with one variant keeps its plain name, such as `surprisal`. With
several, each column name adds the labels of the swept values in its lineage,
upstream first, joined with `_`:

```text
doc_id  region_index  region  char_start  char_end  surprisal_gpt2  surprisal_pythia70m
```

A sweep with one value adds no label. Labels must match
`[A-Za-z][A-Za-z0-9_]*`. A label derived from a string keeps the part after
the last `/` with characters other than letters and digits removed, so
`EleutherAI/pythia-410m` becomes `pythia410m` and `word_boundary` becomes
`wordboundary`. A number, Boolean, or null becomes the parameter name followed
by the value with `.` and `-` removed, as in `window512`. Two variants with the
same label, or two columns with the same name, are an error; give explicit
labels with a mapping.

Analyses should not parse column names. Each table's `NAME.columns.tsv`
lists every feature column with its module, labels, and variant identity,
and `manifest.json` maps each variant to its full resolved config (see
[Output tables](outputs.md)).
