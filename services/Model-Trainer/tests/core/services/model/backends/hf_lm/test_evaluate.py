"""Tests for HuggingFace LM evaluate module."""

from __future__ import annotations

from contextlib import nullcontext

from model_trainer.core.services.model.backends.hf_lm.evaluate import (
    EvalResult,
    _get_autocast_context,
)


class TestGetAutocastContext:
    """Tests for _get_autocast_context function."""

    def test_returns_nullcontext_for_fp32(self) -> None:
        """Test that fp32 precision returns nullcontext."""
        ctx = _get_autocast_context("fp32", "cuda")
        assert type(ctx) is type(nullcontext())

    def test_returns_nullcontext_for_cpu(self) -> None:
        """Test that CPU device returns nullcontext regardless of precision."""
        ctx_fp16 = _get_autocast_context("fp16", "cpu")
        ctx_bf16 = _get_autocast_context("bf16", "cpu")
        assert type(ctx_fp16) is type(nullcontext())
        assert type(ctx_bf16) is type(nullcontext())


class TestEvalResult:
    """Tests for EvalResult class."""

    def test_init_stores_loss_and_perplexity(self) -> None:
        """Test that EvalResult stores loss and perplexity."""
        result = EvalResult(loss=1.5, perplexity=4.48)
        assert result.loss == 1.5
        assert result.perplexity == 4.48

    def test_init_with_zero_loss(self) -> None:
        """Test EvalResult with zero loss."""
        result = EvalResult(loss=0.0, perplexity=1.0)
        assert result.loss == 0.0
        assert result.perplexity == 1.0

    def test_init_with_inf_perplexity(self) -> None:
        """Test EvalResult with infinite perplexity."""
        result = EvalResult(loss=100.0, perplexity=float("inf"))
        assert result.loss == 100.0
        assert result.perplexity == float("inf")
