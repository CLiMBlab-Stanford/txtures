# Inputs

An input is one file in the inputs directory, named by its basename:
`stimuli.tsv` is the input `stimuli`. Names must match
`[A-Za-z0-9_][A-Za-z0-9_.-]*`, and two files may not share one. A file
elsewhere can be used without copying it; see
[Files outside the inputs directory](#files-outside-the-inputs-directory).
txtures writes to the inputs directory only to add those links.

Each input is divided into documents, and each document into regions.
A document is one language-model context: scoring never crosses documents.
A region is one output row. Between the two, regions may be grouped into
sections, paragraphs, and sentences. Each level nests in the one above it:
document, section, paragraph, sentence, region.

## Layouts

**Plain text** (`NAME.txt`): one document per line, numbered from 1.

**TSV** (`NAME.tsv`) and **CSV** (`NAME.csv`): a `text` column, plus optional ID columns
`document_id`, `section_id`, `paragraph_id`, `sentence_id`, and `region_id`.
Every other column is metadata. The ID columns decide what each row holds:

- With no ID column, each row is a document, numbered from 1.
- With a `region_id` column, or when two rows have the same value in every
  ID column, each row is a region.
- Otherwise each row is a unit of the innermost level that has an ID
  column. With `document_id` and `sentence_id`, for example, each row is a
  sentence.

A row above region level is divided into regions, and a row above sentence
level into sentences, by the sidecar's delimiters as described below.
`txtures inputs check NAME` reports the layout it found, such as
`sentence-per-row`.

Rows with the same `document_id` form one document, in the order the
documents first appear. Without a `document_id` column, a file with other ID
columns is one document, `1`. Any document ID is allowed. Section,
paragraph, sentence, and region IDs must be integers, and within a document
the rows are sorted by them, outer levels first. Without a `region_id`, the
regions of a group keep their file order, so the rows of each innermost
group must be contiguous.

IDs are nested: sentence `1` of paragraph `1` and sentence `1` of
paragraph `2` are different sentences. Files whose IDs are unique across the
whole file read the same way.

A file with one word per row, for example:

```text
document_id	sentence_id	text	condition
1	1	The	ambiguous
1	1	horse	ambiguous
1	1	raced	ambiguous
1	2	It	ambiguous
1	2	fell.	ambiguous
```

The same document with one sentence per row:

```text
document_id	sentence_id	text	condition
1	1	The horse raced	ambiguous
1	2	It fell.	ambiguous
```

Metadata passes through to each region read from its row. In the first file
`condition` belongs to each word, and in the second to each word of the
sentence.

A level that neither an ID column nor a delimiter defines is left undefined.
Modules that work on sentences find them with the
[`sentences`](sentences.md) module when the input has none. Sections and
paragraphs are never found automatically, and `lm_scores` with `context:
sentence` uses only the input's sentences, so it refuses an input without
them.

TSV files are read strictly: UTF-8, tab separators, a header on the first
line, and no quote processing. Quote characters are part of the text, so
fields cannot contain tabs or line breaks. A field wrapped in double quotes
draws a warning, because it usually means the file was written with quoting,
as R's `write.table` does by default; write with `quote = FALSE`.

CSV files have the same columns and rules, separated by commas, with
standard CSV quoting: a field in double quotes may contain commas and line
breaks, and `""` inside it stands for one `"`. They are UTF-8 with a header
on the first line. Problems are reported at the line where the row starts.

## Files outside the inputs directory

`txtures run` and `txtures plan` also accept the path of an input file, or
`NAME=PATH`:

```bash
txtures run ~/data/stories.csv          # the input stories
txtures run pilot=~/data/"Pilot (v2).txt"   # the input pilot
```

txtures links the file into the inputs directory as `NAME.EXT`, keeping the
file's extension (`stories.csv`, `pilot.txt`), and then runs that input as
usual. With a bare path, the name is the file's basename without its
extension, and it must be a valid input name; otherwise give one as
`NAME=PATH`. An argument is read as a path when it contains `/` or ends in
`.txt`, `.tsv`, or `.csv`, and as `NAME=PATH` when the part before the first
`=` is a valid name.

A name that another input already uses is refused. Running the same path
again reuses its link, so later runs can name the input alone
(`txtures run stories`). The link points at the file, so edits to the file
change the input, and moving or deleting it breaks the input. On Slurm the
file must be on a filesystem that the compute nodes can read. A sidecar is
not linked; put `NAME.json` in the inputs directory if the input needs one.
The Python function `txtures.run` accepts the same forms.

## Sidecar settings

An optional `NAME.json` beside the input sets how text is divided and joined:

```json
{
  "region_delimiter": "|",
  "sentence_delimiter": "<s>",
  "region_joiner": " ",
  "sentence_joiner": " ",
  "paragraph_joiner": "\n",
  "section_joiner": "\n\n",
  "language": "en"
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `region_delimiter` | unset | separates regions; unset means each whitespace-delimited word is a region |
| `sentence_delimiter` | unset | separates sentences; unset means the input has no sentences unless it has a `sentence_id` column |
| `region_joiner` | `" "` | whitespace placed between regions in the text the model reads |
| `sentence_joiner` | `" "` | whitespace placed between sentences |
| `paragraph_joiner` | `"\n"` | whitespace placed between paragraphs |
| `section_joiner` | `"\n\n"` | whitespace placed between sections |
| `language` | unset | an IETF language tag such as `en`; no module uses it yet |

The region delimiter applies to rows above region level, and the sentence
delimiter to rows above sentence level. With a region delimiter, each region is a delimited piece with surrounding
whitespace removed, so `The horse | raced | past the barn | fell.` has four
regions, and the model reads `The horse raced past the barn fell.` Without
one, the text is kept as written, including repeated spaces, and each word is
a region.

The language model reads each document as one string, so txtures builds that
string from the document's regions and records each region's span in it as
`char_start` and `char_end`. Joiners set what goes between regions. Between
two regions, txtures places the joiner of the outermost boundary that
separates them: `section_joiner` at a section boundary, `paragraph_joiner` at
a paragraph boundary, `sentence_joiner` at a sentence boundary, and
`region_joiner` otherwise. Joiners change tokenization and therefore computed
values: with `region_joiner: ""`, the regions `walk` and `ed` are read as
`walked`.

Joiners must be empty or whitespace. Every character outside a region is
therefore whitespace, and every region ends where a new word could begin. The
default paragraph and section joiners are newlines, which `surprisal` may
refuse for some tokenizers (see [Surprisal](surprisal.md#limitations)); set
them to `" "` to avoid that.

## Checks

```bash
txtures inputs list           # every input, its size, and whether it is valid
txtures inputs check NAME     # every problem in one input
```

Every run reads its input the same way and refuses an invalid one. Problems
include unreadable or non-UTF-8 files; malformed CSV quoting; a missing `text` column; a repeated
or empty column name; rows with the wrong number of fields; metadata columns
named like output columns (`region_index`, `char_start`, `char_end`);
empty text or IDs, including blank lines in a `.txt` file; section,
paragraph, sentence, or region IDs that are not integers; regions whose
order is undefined (repeated region IDs, or the rows of a group split up
without a `region_id`); a delimiter that does not apply to the file's rows;
and an invalid sidecar.

Files written for earlier versions of txtures, which used `doc_id` and a
`region` column, are refused with a message naming the replacement columns.
Rename `doc_id` to `document_id` and `region` to `text`.

Warnings do not stop a run. They flag quoted fields, whitespace other than
spaces and tabs (such as non-breaking spaces) in the input's text, and text
that is not in Unicode NFC form, since each can change tokenization without
being visible. Whitespace that joiners add draws no warning.

## Identity

An input's identity covers its file bytes, its sidecar, and the version of
the reading rules. Each document's identity covers its text and its region
and segment boundaries, but not its IDs or metadata. Editing one
document reprocesses only that document. Renaming documents or editing
metadata reprocesses nothing, and identical documents in any input share
stored results.
