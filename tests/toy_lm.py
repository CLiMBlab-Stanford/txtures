"""A tiny causal language model and tokenizer, saved as a local model directory.

The vocabulary holds `a`, `b`, ` a`, ` b`, and `<eos>` (plus `<s>` in the
BOS variant). Every string over `a`, `b`, and single spaces has exactly one
tokenization, so string probabilities computed by enumeration equal the
products of token probabilities along that tokenization.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

STRINGS = {0: None, 1: "a", 2: "b", 3: " a", 4: " b"}
"""Decoded text of each token; `None` marks EOS, which ends the text."""


def build(directory: Path, style: str = "eos", positions: int = 16, seed: int = 0) -> Path:
    """Save a toy model to `directory`.

    `style` sets the tokenizer's special tokens: `eos` (BOS and EOS are the
    same token, as in GPT-2), `bos` (a separate BOS token added by default, as
    in Llama), `no_bos` (EOS only, as in OLMo), `quiet_bos` (a separate BOS
    token that is not added by default), or `bare` (neither token).
    """
    vocab = {"<eos>": 0, "a": 1, "b": 2, " a": 3, " b": 4}
    if style in ("bos", "quiet_bos"):
        vocab["<s>"] = 5
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"\s?\S"), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    special = {} if style == "bare" else {"eos_token": "<eos>"}
    if style == "eos":
        special["bos_token"] = "<eos>"
    elif style == "quiet_bos":
        special["bos_token"] = "<s>"
    elif style == "bos":
        special["bos_token"] = "<s>"
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 5)]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special)
    if style == "bos":
        fast.add_bos_token = True
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=positions,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=vocab.get("<s>", 0),
        eos_token_id=0,
        initializer_range=0.5,
    )
    model = GPT2LMHeadModel(config)
    directory.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(directory)
    model.save_pretrained(directory)
    return directory
