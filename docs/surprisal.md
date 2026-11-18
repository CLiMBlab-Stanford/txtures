# Surprisal

Surprisal comes from two modules. `lm_scores` runs a causal language model
over each document and stores token-level log-probabilities. `surprisal` sums
them into regions. Changing only a `surprisal` parameter reuses the stored
`lm_scores` results.

## `lm_scores`

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `gpt2` | a Hugging Face repository, or the absolute path of a local `save_pretrained` directory |
| `revision` | unset | a commit hash, branch, or tag; unset means the default branch |
| `dtype` | `float32` | precision of the model weights: `float32`, `float16`, or `bfloat16` |
| `window` | the model's maximum | tokens per forward pass |
| `stride` | half of `window` | tokens advanced between windows; must be smaller than `window` |
| `context` | `document` | `document` or `sentence` |
| `document_start` | `auto` | how the start of a context is marked: `auto`, `bos`, `eos`, or `none` |

`model`, `window`, and `context` are sweepable (see
[Definitions](definitions.md#sweeps)).

A revision that is unset or names a branch or tag is resolved to a commit
hash when a run is planned, and the hash is part of the results' identity. A
new commit on the hub therefore makes earlier tables stale, and every run of
an unpinned model prints a note. Pin `revision` to a commit hash for
reproducible results. If the hub cannot be reached, txtures uses the commit
the local cache records. A local model directory is identified by a digest of
its files, and `revision` may not be set for it.

Log-probabilities are always computed in float32, whatever `dtype` the
weights use.

With `context: sentence`, each sentence is scored as its own context. This
needs an input with sentences. With `context: document`, the default, the
whole document is one context.

A document longer than `window` tokens is scored with a sliding window. Each
token is scored with at least `window - stride` tokens of context, or all
earlier tokens when there are fewer.

## `surprisal`

| Key | Default | Meaning |
| --- | --- | --- |
| `correction` | `word_boundary` | `word_boundary` or `none` |
| `units` | `bits` | `bits` or `nats` |

Both are sweepable. The output column is `surprisal`.

With `correction: none`, a region's surprisal is the sum of the surprisals of
its tokens. That sum misses some of the probability of the word. Most
tokenizers mark the start of a word on its first token, as in GPT-2's
` horse`, so the end of a word is visible only when the next token begins a
new word. With `correction: word_boundary`, the surprisal of a region made of
tokens *t₁…tₖ* after prefix *c* is

    −log P(t₁…tₖ | c) − log P(B | c, t₁…tₖ) + log P(B | c)

where *B* is the set of tokens that begin a word, following Pimentel and
Meister (2024) and Oh and Schuler (2024). The second term accounts for the
word ending after *tₖ*; the third removes the same term from the previous
word. For the first region of a context, *P(B | c)* is 1. The term after the
last region comes from the model's prediction after the final token, so the
last word needs no approximation.

txtures finds *B* by decoding each token after the text `a` and checking
whether the result begins with whitespace. The end-of-sequence token is in
*B*. If a tokenizer does not mark word starts in a way this test detects,
`word_boundary` correction refuses to run, and `correction: none` is the only
choice.

## Tokenization

txtures gives each model the document text as written and tokenizes it with
the tokenizer's defaults. It does not add or move spaces to make words look
alike to the model. With GPT-2-style tokenizers, the first word of a document
has no leading space and later words do. SentencePiece tokenizers such as
Llama's mark the first word themselves. The correction above is exact in
both cases.

A token belongs to the first region that ends after the token's last
character. Whitespace-only tokens between regions therefore belong to the
next region.

The start of each context is marked as the model saw document starts in
training. With `document_start: auto`, txtures reads the convention from the
tokenizer:

1. A tokenizer that adds its beginning-of-sequence token itself (Llama,
   Mistral, Gemma) gets that token (`bos`).
2. One whose beginning- and end-of-sequence tokens are the same (GPT-2,
   Pythia) gets the end-of-sequence token (`eos`), since such models were
   trained on documents separated by it.
3. One that declares no beginning-of-sequence token but has an
   end-of-sequence token (OLMo, Qwen, Falcon) also gets `eos`.
4. One that declares a distinct beginning-of-sequence token without adding it
   gets `bos`.

A tokenizer with neither token is refused until the config sets
`document_start: none`. The resolved choice is recorded in each table's
manifest.

## Limitations

Some tokenizers, including Qwen's, join punctuation and a following newline
into one token. Such a token spans two regions, and `surprisal` refuses the
document. Inputs contain newlines only when a sidecar sets a newline joiner,
so the default joiners avoid the problem.

`lm_scores` needs a fast (Rust-based) tokenizer, which provides the
character spans that map tokens to regions.

## References

Oh, B.-D., & Schuler, W. (2024). Leading whitespaces of language models'
subword vocabulary pose a confound for calculating word probabilities.
*Proceedings of EMNLP 2024*.

Pimentel, T., & Meister, C. (2024). How to compute the probability of a word.
*Proceedings of EMNLP 2024*.
