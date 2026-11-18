# Inputs

An input is one file in the inputs directory, named by its basename:
`stimuli.tsv` is the input `stimuli`. Names must match
`[A-Za-z0-9_][A-Za-z0-9_.-]*`, and two files may not share one. textures
never writes to the inputs directory.

Each input is divided into documents, and each document into regions.
A document is one language-model context: scoring never crosses documents.
A region is one output row. Regions may be grouped into sentences.

## Layouts

**Plain text** (`NAME.txt`): one document per line, numbered from 1.

**Document per row** (`NAME.tsv` with a `text` column): one document per row.
An optional `doc_id` column names documents and must be unique; without it,
documents are numbered from 1. Other columns are document metadata and are
repeated on every output row of that document.

**Region per row** (`NAME.tsv` with a `region` column): one region per row.
Rows with the same `doc_id` form one document and must be contiguous; without
a `doc_id` column, the file is one document. An optional `sentence_id` column
groups regions into sentences, whose rows must also be contiguous. Other
columns are region metadata and pass through row by row.

```text
doc_id	sentence_id	region	condition
1	1	The horse	ambiguous
1	1	raced	ambiguous
1	1	past the barn	ambiguous
1	1	fell.	ambiguous
```

TSV files are read strictly: UTF-8, tab separators, a header on the first
line, and no quote processing. Quote characters are part of the text, so
fields cannot contain tabs or line breaks. A field wrapped in double quotes
draws a warning, because it usually means the file was written with quoting,
as R's `write.table` does by default; write with `quote = FALSE`.

## Sidecar settings

An optional `NAME.json` beside the input sets how text is divided and joined:

```json
{
  "region_delimiter": "|",
  "sentence_delimiter": "<s>",
  "region_joiner": " ",
  "sentence_joiner": " ",
  "language": "en"
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `region_delimiter` | unset | separates regions; unset means each whitespace-delimited word is a region |
| `sentence_delimiter` | unset | separates sentences; unset means the input has no sentences |
| `region_joiner` | `" "` | whitespace placed between regions in the text the model reads |
| `sentence_joiner` | `" "` | whitespace placed between sentences |
| `language` | unset | an IETF language tag such as `en`; no module uses it yet |

The delimiters apply to plain-text and document-per-row inputs. With a
region delimiter, each region is a delimited piece with surrounding
whitespace removed, so `The horse | raced | past the barn | fell.` has four
regions, and the model reads `The horse raced past the barn fell.` Without
one, the text is kept as written, including repeated spaces, and each word is
a region.

Joiners must be empty or whitespace. Every character outside a region is
therefore whitespace, and every region ends where a new word could begin.
Region-per-row inputs use only the joiners. A newline joiner is allowed but
may be refused by `surprisal` for some tokenizers (see
[Surprisal](surprisal.md#limitations)).

## Checks

```bash
textures inputs list           # every input, its size, and whether it is valid
textures inputs check NAME     # every problem in one input
```

Every run reads its input the same way and refuses an invalid one. Problems
include unreadable or non-UTF-8 files; a missing, repeated, or empty column
name; both or neither of `text` and `region`; rows with the wrong number of
fields; metadata columns named like output columns (`doc_id`, `sentence_id`,
`region_index`, `region`, `char_start`, `char_end`, `text`); empty
documents, sentences, regions, or IDs, including blank lines in a `.txt`
file; repeated or non-contiguous IDs; and an invalid sidecar.

Warnings do not stop a run. They flag quoted fields, whitespace other than
spaces and tabs (such as non-breaking spaces), and text that is not in
Unicode NFC form, since each can change tokenization without being visible.

## Identity

An input's identity covers its file bytes, its sidecar, and the version of
the reading rules. Each document's identity covers its text and its region
and sentence boundaries, but not its `doc_id` or metadata. Editing one
document reprocesses only that document. Renaming documents or editing
metadata reprocesses nothing, and identical documents in any input share
stored results.
