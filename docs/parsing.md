# Parsing, part of speech, and named entities

`parse` analyzes each document with [Stanza](https://stanfordnlp.github.io/stanza/)
into Universal Dependencies: tokens, part-of-speech tags, morphological
features, lemmas, dependency trees, and named entities. It adds no columns.
`pos`, `named_entity`, and [`dependency_locality`](dependency-locality.md)
read its results, so changing their parameters reuses the stored parses.

## `parse`

| Key | Default | Meaning |
| --- | --- | --- |
| `language` | `en` | Stanza language code |
| `package` | `default` | Stanza package, which selects the training treebank |

Stanza's models come from its Hugging Face repository for the language, at
the resources version of the installed Stanza. The tag is resolved to a
commit hash when a run is planned, and the hash is part of the results'
identity, as for a language model's revision. The models are stored in the
`resources` directory.

Parsing needs sentences. When the input defines them, each is parsed on its
own. Otherwise Stanza splits each document, and a split that falls inside a
region moves to the end of that region. Every Stanza token must lie inside
one region; a token that spans two regions is an error. Stanza splits some
tokens into several syntactic words, such as *don't* into *do* and *n't*;
the words share the token's region.

Like `lm_scores`, `parse` has a Slurm workload: with Slurm set up, a run
whose parses are missing is submitted as a job (see [Slurm](slurm.md)).

## `pos`

| Key | Default | Meaning |
| --- | --- | --- |
| `tagset` | `upos` | `upos` for Universal POS tags, or `xpos` for the treebank's tags (Penn Treebank tags for English) |

`tagset` is sweepable. The output column is `pos`. A region takes the tag of
its head word in the SUD conversion of the parse: the word whose head lies
outside the region, or, when several do, the first that is not punctuation.
Because SUD makes function words heads, *don't* is tagged `AUX`.

## `named_entity`

| Key | Default | Meaning |
| --- | --- | --- |
| `types` | the name types below | entity types that count |

The output column is `named_entity`: 1 if any word of the region lies in an
entity of one of `types`, and 0 otherwise. Stanza's English models tag the
18 OntoNotes types. The default counts the names: `PERSON`, `NORP`, `FAC`,
`ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, and
`LANGUAGE`. It leaves out `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`,
`ORDINAL`, and `CARDINAL`.
