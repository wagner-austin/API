"""Test backend name() methods for complete coverage."""

from __future__ import annotations

from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend


def test_bpe_backend_name() -> None:
    """Test BPEBackend.name() returns 'bpe' - covers bpe_backend.py line 146."""
    backend = BPEBackend()
    assert backend.name() == "bpe"


def test_sentencepiece_backend_name() -> None:
    """Test SentencePieceBackend.name() returns 'sentencepiece' - covers spm_backend.py line 187."""
    backend = SentencePieceBackend()
    assert backend.name() == "sentencepiece"
