"""Shared tokenizer loading with automatic backend detection.

This module provides a single entry point for loading tokenizers from artifact
directories, automatically detecting the correct backend (BPE, char, or
SentencePiece) based on the artifact files present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import load_json_str

from model_trainer.core.contracts.tokenizer import TokenizerHandle

TokenizerKind = Literal["bpe", "char", "sentencepiece"]


def detect_tokenizer_kind(artifact_dir: str) -> TokenizerKind:
    """Detect the tokenizer kind from artifact directory contents.

    Examines the artifact directory to determine which tokenizer backend
    created the artifacts.

    Args:
        artifact_dir: Path to the tokenizer artifact directory.

    Returns:
        The detected tokenizer kind.

    Raises:
        AppError: If no recognized tokenizer artifacts are found.
    """
    base = Path(artifact_dir)
    tok_json = base / "tokenizer.json"
    tok_spm = base / "tokenizer.model"

    if tok_json.exists():
        text = tok_json.read_text(encoding="utf-8")
        obj = load_json_str(text)
        if isinstance(obj, dict) and obj.get("kind") == "char":
            return "char"
        return "bpe"

    if tok_spm.exists():
        return "sentencepiece"

    raise AppError(
        ModelTrainerErrorCode.TOKENIZER_NOT_FOUND,
        f"No tokenizer artifacts found in {artifact_dir}",
        model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_NOT_FOUND),
    )


def load_tokenizer_from_dir(artifact_dir: str) -> TokenizerHandle:
    """Load a tokenizer from an artifact directory with automatic backend detection.

    Detects the tokenizer kind from the artifact files and loads using the
    appropriate backend.

    Args:
        artifact_dir: Path to the tokenizer artifact directory.

    Returns:
        Loaded tokenizer handle ready for encoding/decoding.

    Raises:
        AppError: If no recognized tokenizer artifacts are found or loading fails.
    """
    kind = detect_tokenizer_kind(artifact_dir)
    return _load_by_kind(artifact_dir, kind)


def load_tokenizer_from_path(artifact_path: str) -> TokenizerHandle:
    """Load a tokenizer from a specific artifact file path.

    Determines the tokenizer kind from the file path and parent directory,
    then loads using the appropriate backend.

    Args:
        artifact_path: Path to the tokenizer artifact file (tokenizer.json or
            tokenizer.model).

    Returns:
        Loaded tokenizer handle ready for encoding/decoding.

    Raises:
        AppError: If the tokenizer kind cannot be determined or loading fails.
    """
    path = Path(artifact_path)

    # If path is a directory, delegate to load_tokenizer_from_dir
    if path.is_dir():
        return load_tokenizer_from_dir(artifact_path)

    # Determine kind from file name and contents
    if path.name == "tokenizer.model" or path.suffix == ".model":
        kind: TokenizerKind = "sentencepiece"
    elif path.name == "tokenizer.json" or path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
        obj = load_json_str(text)
        kind = "char" if isinstance(obj, dict) and obj.get("kind") == "char" else "bpe"
    else:
        raise AppError(
            ModelTrainerErrorCode.TOKENIZER_NOT_FOUND,
            f"Unrecognized tokenizer artifact: {artifact_path}",
            model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_NOT_FOUND),
        )

    return _load_by_kind_from_path(artifact_path, kind)


def _load_by_kind(artifact_dir: str, kind: TokenizerKind) -> TokenizerHandle:
    """Load tokenizer from directory using the specified backend.

    Args:
        artifact_dir: Path to the tokenizer artifact directory.
        kind: The tokenizer backend kind to use.

    Returns:
        Loaded tokenizer handle.
    """
    base = Path(artifact_dir)

    if kind == "char":
        from model_trainer.core.services.tokenizer.char_backend import CharBackend

        return CharBackend().load(str(base / "tokenizer.json"))

    if kind == "sentencepiece":
        from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend

        return SentencePieceBackend().load(str(base / "tokenizer.model"))

    # Default to BPE
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    return BPEBackend().load(str(base / "tokenizer.json"))


def _load_by_kind_from_path(artifact_path: str, kind: TokenizerKind) -> TokenizerHandle:
    """Load tokenizer from file path using the specified backend.

    Args:
        artifact_path: Path to the tokenizer artifact file.
        kind: The tokenizer backend kind to use.

    Returns:
        Loaded tokenizer handle.
    """
    if kind == "char":
        from model_trainer.core.services.tokenizer.char_backend import CharBackend

        return CharBackend().load(artifact_path)

    if kind == "sentencepiece":
        from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend

        return SentencePieceBackend().load(artifact_path)

    # Default to BPE
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    return BPEBackend().load(artifact_path)
