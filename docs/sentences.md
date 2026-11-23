# Sentences

`sentences` finds the sentences of each document once, and every module
that works on sentences reads its result: [`parse`](parsing.md), and through
it `pos`, `named_entity`, and `dependency_locality`, and
[`ngram_surprisal`](ngram.md). It adds no columns.

When the input defines sentences, with a `sentence_id` column or a
`sentence_delimiter` (see [Inputs](inputs.md)), they are used as given.
Otherwise [Stanza](https://stanfordnlp.github.io/stanza/)'s tokenizer splits
each document, and a split that falls inside a region moves to the end of
that region, so every sentence is a run of whole regions.

| Key | Default | Meaning |
| --- | --- | --- |
| `language` | `en` | Stanza language code |
| `package` | `default` | Stanza package, which selects the training treebank |

The models are those of `parse` (see [Parsing](parsing.md#parse)): the same
resource, pinned to a commit of Stanza's Hugging Face repository. With
Slurm set up, a run whose sentences are missing is submitted as a job
without GPUs.
