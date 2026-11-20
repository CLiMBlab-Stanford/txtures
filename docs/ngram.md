# N-gram surprisal

`unigram_surprisal` and `ngram_surprisal` give each region's surprisal
under [KenLM](https://kheafield.com/code/kenlm/) models trained on a corpus,
by default [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext).
Each also counts the region's tokens that the model does not know.

| Module | Columns | Model |
| --- | --- | --- |
| `unigram_surprisal` | `unigram_surprisal`, `unigram_oov` | order 1, unpruned |
| `ngram_surprisal` | `ngram_surprisal`, `ngram_oov` | order 5 by default, pruned |

Both columns are labeled by the corpus, like surprisal by its model (see
[Column names](definitions.md#column-names)): `ngram_surprisal_openwebtext`.

## Configs

The two modules share these keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `dataset` | `Skylion007/openwebtext` | a Hugging Face dataset of Parquet files, or the absolute path of a local directory |
| `revision` | unset | a commit, branch, or tag of the dataset; resolved to a commit and recorded |
| `files` | `plain_text/train-*.parquet` | a glob of the dataset's files to use |
| `text_field` | `text` | the Parquet column that holds each document |
| `max_documents` | unset | use only the first so many documents, in file order |
| `lowercase` | `true` | lowercase the corpus and the text |
| `discount_fallback` | `false` | pass `--discount_fallback` to `lmplz`; small corpora need it |
| `units` | `bits` | `bits` or `nats` |

`ngram_surprisal` adds `order` (default 5, at least 2) and `prune` (default
`[0, 0, 1]`): n-grams seen this many times or fewer are dropped, the first
value for unigrams, the next for bigrams, and the last for every higher
order, as `lmplz --prune` reads them. Unigrams cannot be pruned. `dataset`,
`order`, and `units` are sweepable.

A local corpus is a directory of Parquet files, one document per row, or
text files, one document per line:

```yaml
# configs/ngram_surprisal/mycorpus.yml
dataset: /data/corpora/mycorpus
files: "*.txt"
order: 3
```

It is identified by a digest of its files, so editing them makes the
models be built again.

## The corpus and the models

The corpus is tokenized once, in parallel over its files, and kept in the
resource store with one sentence per line. Text is normalized to NFC, and
tokens follow the Penn Treebank's conventions for English: punctuation is
split from words, numbers such as `3,500.75` stay whole, and the clitics
`n't`, `'s`, `'re`, `'ve`, `'ll`, `'d`, and `'m` are split off (*don't*
becomes *do* *n't*). Documents are split into sentences at `.`, `!`, and
`?` before a capital letter or digit, except after common abbreviations and
initials, and at line breaks.

KenLM's `lmplz`, run from the [tools image](tools.md), estimates modified
Kneser-Ney models on it. The n-gram model is stored in KenLM's binary trie
format. The unigram model's probabilities come from raw counts, with
modified Kneser-Ney discounts and the discounted probability spread over
the vocabulary and `<unk>`; KenLM cannot query an order-1 model, so its
probabilities are stored as a table that txtures reads directly. The
unigram part of a higher-order Kneser-Ney model is not used, because it
measures in how many contexts a word occurs, not how often.

Building these resources takes hours on OpenWebText (about 9 billion
tokens). Where Slurm is set up, each is built by a job without GPUs (see
[Resource builds](slurm.md#resource-builds)): the corpus with 16 cores and
32 GB, the n-gram model with 100 GB, and the unigram table with 32 GB. They
are built once and shared by everyone who uses the `resources` directory;
`txtures resources build NAME -m ngram_surprisal -m unigram_surprisal` builds
them ahead of time.

## Scoring

The text is tokenized the same way as the corpus. Each sentence of the
input, or each sentence the splitter finds when the input defines none
(moved to the end of the region it falls in), is scored as its own
context, starting after `<s>`. The probability of the end of the sentence
is not counted. A region's surprisal is the sum of its tokens' surprisals,
so punctuation attached to a word counts, as for language-model surprisal.

A token the model has never seen gets the probability of `<unk>`, the same
for every such token. The `oov` columns count these tokens in each region,
so that analyses can exclude or control for them.

`ngram_surprisal` stores its results and runs KenLM's `query` on CPUs; with
Slurm set up, missing results are computed by a job without GPUs.
`unigram_surprisal` reads its table when the table is written.
