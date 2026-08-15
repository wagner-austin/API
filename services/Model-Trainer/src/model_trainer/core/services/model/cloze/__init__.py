"""Cloze (multiple-choice-by-scoring) evaluation of causal language models."""

from __future__ import annotations

from .score import score_cloze_items, sequence_nll

__all__ = [
    "score_cloze_items",
    "sequence_nll",
]
