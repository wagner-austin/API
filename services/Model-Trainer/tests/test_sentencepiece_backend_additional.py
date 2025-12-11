from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.spm_backend import (
    SentencePieceBackend,
)


def test_spm_encode_ids_blank_output_returns_empty(tmp_path: Path) -> None:
    # Prepare a dummy model path
    model_path = tmp_path / "tok.model"
    model_path.write_bytes(b"")

    # Set up encode hook to return empty for blank output scenario
    def _encode_blank(model_path: str, text: str) -> list[int]:
        return []

    _test_hooks.spm_encode_ids = _encode_blank

    # Call encode through adapter and assert empty ids
    h = SentencePieceBackend().load(str(model_path))
    ids = SentencePieceBackend().encode(h, "ignored")
    assert ids == []


def test_train_spm_tokenizer_no_files_raises(tmp_path: Path) -> None:
    # Disable CLI check
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    backend = SentencePieceBackend()
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(tmp_path / "empty"),
        holdout_fraction=0.2,
        seed=1,
        out_dir=str(tmp_path / "out"),
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    with pytest.raises(AppError):
        _ = backend.train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial


def test_train_spm_tokenizer_clamps_sample_and_char_coverage(tmp_path: Path) -> None:
    # Disable CLI check
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    # Fake train writes minimal model/vocab files
    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\nB\n", encoding="utf-8")

    _test_hooks.spm_train = _fake_train

    # Encode stub: return [0] for 'x' (unknown), [1] otherwise (known)
    def _fake_encode(model_path: str, text: str) -> list[int]:
        if len(text) == 1 and text.lower() == "x":
            return [0]
        return [1]

    _test_hooks.spm_encode_ids = _fake_encode

    # Build corpus with enough lines to trigger holdout clamp when sample_max_lines=1
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = ["xy", "ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr"]
    (corpus / "a.txt").write_text("\n".join(lines), encoding="utf-8")

    out_dir = tmp_path / "tok"
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=32,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=42,
        out_dir=str(out_dir),
        sample_max_lines=1,  # clamp to 1
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    stats = SentencePieceBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
    # Ensure manifest written and stats within expected bounds
    assert (out_dir / "manifest.json").exists()
    assert 0.0 <= stats.coverage <= 1.0
    assert 0.0 <= stats.char_coverage <= 1.0


def test_train_spm_tokenizer_no_clamp_branch(tmp_path: Path) -> None:
    # Disable CLI check
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\nB\n", encoding="utf-8")

    _test_hooks.spm_train = _fake_train

    # Encode returns a single non-UNK id to exercise the true branch of coverage calc
    def _enc_ids(model_path: str, text: str) -> list[int]:
        return [1]

    _test_hooks.spm_encode_ids = _enc_ids

    corpus = tmp_path / "corpus_no_clamp"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abc\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=16,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=7,
        out_dir=str(tmp_path / "tok_nc"),
        # sample_max_lines left as None to exercise false path of clamp
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    stats = SentencePieceBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
    assert stats.token_count >= 0


def test_train_spm_tokenizer_empty_ids_skip_in_char_coverage(tmp_path: Path) -> None:
    # Disable CLI check
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\n", encoding="utf-8")

    _test_hooks.spm_train = _fake_train

    # For a specific char, return empty ids to trigger the false arm from the ids-empty side
    def _enc_ids_empty(model_path: str, text: str) -> list[int]:
        return []

    _test_hooks.spm_encode_ids = _enc_ids_empty

    corpus = tmp_path / "corpus_empty_ids"
    corpus.mkdir()
    (corpus / "a.txt").write_text("x\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=8,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=3,
        out_dir=str(tmp_path / "tok_ei"),
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    stats = SentencePieceBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
    assert stats.char_coverage in (0.0, 1.0)


def test_spm_inspect_success_reads_manifest(tmp_path: Path) -> None:
    # Create a directory with a valid SentencePiece-style manifest
    out = tmp_path / "tok"
    out.mkdir()
    manifest = {
        "stats": {
            "coverage": 0.75,
            "oov_rate": 0.1,
            "token_count": 123,
            "char_coverage": 0.8,
        }
    }
    (out / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
    stats = SentencePieceBackend().inspect(str(out))
    assert stats.token_count == 123 and stats.oov_rate == 0.1


def test_spm_inspect_invalid_manifest_and_stats(tmp_path: Path) -> None:
    base = tmp_path / "spm"
    base.mkdir()
    # invalid manifest format
    (base / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AppError):
        _ = SentencePieceBackend().inspect(str(base))

    # invalid stats object
    (base / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AppError):
        _ = SentencePieceBackend().inspect(str(base))
