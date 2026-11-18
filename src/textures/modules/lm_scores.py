"""Token-level scores from a causal language model."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from textures.modules.base import Module, ModuleConfig


class LMScoresConfig(ModuleConfig):
    """Parameters for scoring tokens with a Hugging Face causal language model."""

    model: str = Field(default="gpt2", min_length=1)
    """Hugging Face repository name."""

    revision: str | None = Field(default=None, min_length=1)
    """Commit hash to load. When unset, the current revision is resolved and recorded."""

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Model weight precision. Log-probabilities are always computed in float32."""

    window: int | None = Field(default=None, gt=1)
    """Tokens per forward pass. When unset, the model's maximum context length."""

    stride: int | None = Field(default=None, gt=0)
    """Tokens advanced between windows. When unset, half the window."""

    context: Literal["document", "sentence"] = "document"
    """Whether each document or each sentence is scored as one context."""

    document_start: Literal["auto", "bos", "eos", "none"] = "auto"
    """How the start of a context is marked for the model."""

    @model_validator(mode="after")
    def _stride_fits_window(self) -> Self:
        if self.window is not None and self.stride is not None and self.stride >= self.window:
            raise ValueError("stride must be smaller than window")
        return self


class LMScores(Module):
    """Score every token of each context with a causal language model."""

    name = "lm_scores"
    contract_version = 1
    Config = LMScoresConfig
    sweeps = {
        "model": ("model", "revision"),
        "window": ("window", "stride"),
        "context": ("context",),
    }
