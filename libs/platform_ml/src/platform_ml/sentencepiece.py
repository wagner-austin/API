"""SentencePiece integration with strict typing and warning suppression.

This module provides typed access to the sentencepiece Python API with
centralized handling of the SWIG deprecation warnings. All services that
need sentencepiece functionality should import from here.

Importing this module eagerly loads sentencepiece with warnings suppressed,
so that any later imports (e.g., from transformers) find it already cached
in sys.modules and don't trigger the SWIG deprecation warnings.
"""

from __future__ import annotations

import warnings
from typing import Protocol

# Eagerly import sentencepiece at module load time with warnings suppressed.
# This ensures it's cached in sys.modules before transformers or other libs
# can import it and trigger the SWIG deprecation warnings.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib")
    __import__("sentencepiece")


class SpmTrainerProto(Protocol):
    """Protocol for sentencepiece.SentencePieceTrainer."""

    @staticmethod
    def train(
        input: str,
        model_prefix: str,
        vocab_size: int,
        character_coverage: float,
        model_type: str,
        unk_piece: str,
        pad_piece: str,
        bos_piece: str,
        eos_piece: str,
    ) -> None: ...


class SpmProcessorProto(Protocol):
    """Protocol for sentencepiece.SentencePieceProcessor."""

    def load(self, model_file: str) -> bool: ...
    def encode_as_ids(self, text: str) -> list[int]: ...
    def decode_ids(self, ids: list[int]) -> str: ...


class SpmProcessorCtorProto(Protocol):
    """Protocol for SentencePieceProcessor constructor."""

    def __call__(self) -> SpmProcessorProto: ...


def get_trainer() -> SpmTrainerProto:
    """Get sentencepiece.SentencePieceTrainer with typed interface."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib")
        spm = __import__("sentencepiece", fromlist=["SentencePieceTrainer"])
    trainer: SpmTrainerProto = spm.SentencePieceTrainer
    return trainer


def get_processor_ctor() -> SpmProcessorCtorProto:
    """Get sentencepiece.SentencePieceProcessor constructor with typed interface."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib")
        spm = __import__("sentencepiece", fromlist=["SentencePieceProcessor"])
    ctor: SpmProcessorCtorProto = spm.SentencePieceProcessor
    return ctor


def require_module() -> None:
    """Verify that the sentencepiece Python module is available.

    Since this module eagerly imports sentencepiece at load time,
    this function is a no-op - if we got here, sentencepiece is available.
    """
    pass


def train(
    files: list[str],
    *,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
    unk_piece: str = "[UNK]",
    pad_piece: str = "[PAD]",
    bos_piece: str = "[BOS]",
    eos_piece: str = "[EOS]",
) -> None:
    """Train a SentencePiece model.

    Args:
        files: List of input text file paths.
        model_prefix: Output path prefix for .model and .vocab files.
        vocab_size: Target vocabulary size.
        character_coverage: Character coverage for training (default 1.0).
        model_type: Model type - "bpe", "unigram", "char", or "word".
        unk_piece: Unknown token piece.
        pad_piece: Padding token piece.
        bos_piece: Beginning of sentence token piece.
        eos_piece: End of sentence token piece.
    """
    trainer = get_trainer()
    input_files = ",".join(files)
    trainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        unk_piece=unk_piece,
        pad_piece=pad_piece,
        bos_piece=bos_piece,
        eos_piece=eos_piece,
    )


def encode_ids(model_path: str, text: str) -> list[int]:
    """Encode text to token IDs using a trained model.

    Args:
        model_path: Path to the .model file.
        text: Text to encode.

    Returns:
        List of token IDs.
    """
    ctor = get_processor_ctor()
    sp = ctor()
    sp.load(model_path)
    ids: list[int] = sp.encode_as_ids(text)
    return ids


def decode_ids(model_path: str, ids: list[int]) -> str:
    """Decode token IDs to text using a trained model.

    Args:
        model_path: Path to the .model file.
        ids: List of token IDs to decode.

    Returns:
        Decoded text string.
    """
    ctor = get_processor_ctor()
    sp = ctor()
    sp.load(model_path)
    text: str = sp.decode_ids(ids)
    return text
