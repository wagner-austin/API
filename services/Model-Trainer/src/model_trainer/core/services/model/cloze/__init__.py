"""Cloze (multiple-choice-by-scoring) evaluation of causal language models."""

from __future__ import annotations

from .identity import QUESTION_SET_DIGEST_PREFIX, question_set_digest
from .score import score_cloze_items, sequence_nll

__all__ = [
    "QUESTION_SET_DIGEST_PREFIX",
    "question_set_digest",
    "score_cloze_items",
    "sequence_nll",
]
