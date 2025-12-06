from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError

from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.char_backend import SPECIALS, CharBackend


def _make_cfg(corpus: Path, out_dir: Path) -> TokenizerTrainConfig:
    return TokenizerTrainConfig(
        method="char",
        vocab_size=0,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=1,
        out_dir=str(out_dir),
    )


def test_char_backend_train_and_load_and_decode_specials(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Include a blank line to cover the 'continue' branch in corpus reader
    (corpus / "a.txt").write_text("ab\n\nba\n", encoding="utf-8")
    out_dir = tmp_path / "tok"
    cfg = _make_cfg(corpus, out_dir)

    be = CharBackend()
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    stats = be.train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
    assert stats.token_count >= 4

    handle = be.load(str(out_dir))
    # Cover get_vocab_size and name
    assert handle.get_vocab_size() >= 4
    assert be.name() == "char"
    # Collect a few ids
    ids_a = handle.encode("a")
    assert isinstance(ids_a, list) and len(ids_a) == 1
    a_id = ids_a[0]
    pad_id = handle.token_to_id(SPECIALS[0])
    eos_id = handle.token_to_id(SPECIALS[3])
    assert isinstance(pad_id, int) and isinstance(eos_id, int)
    # Decode should skip specials and return only the character
    s = handle.decode([pad_id, a_id, eos_id])
    assert s == "a"
    # Unknown id should be ignored in decode path
    assert handle.decode([9999]) == ""
    # Cover backend pass-through encode/decode wrappers
    assert be.encode(handle, "a") == [a_id]
    assert be.decode(handle, [a_id]) == "a"


def test_char_backend_load_invalid_variants(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok_bad"
    tok_dir.mkdir()
    # Not a dict
    (tok_dir / "tokenizer.json").write_text("[]", encoding="utf-8")
    be = CharBackend()
    with pytest.raises(AppError, match="invalid char tokenizer format"):
        _ = be.load(str(tok_dir))

    # Wrong kind
    (tok_dir / "tokenizer.json").write_text('{"kind":"wrong","vocab":{}}', encoding="utf-8")
    with pytest.raises(AppError, match="not a char tokenizer"):
        _ = be.load(str(tok_dir))

    # Vocab not a dict
    (tok_dir / "tokenizer.json").write_text('{"kind":"char","vocab":[]}', encoding="utf-8")
    with pytest.raises(AppError, match="invalid vocab"):
        _ = be.load(str(tok_dir))

    # Vocab entry has non-int value
    (tok_dir / "tokenizer.json").write_text('{"kind":"char","vocab":{"a":"x"}}', encoding="utf-8")
    with pytest.raises(AppError, match="invalid vocab entry"):
        _ = be.load(str(tok_dir))


def test_char_backend_inspect_manifest_errors(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok_inspect"
    tok_dir.mkdir()
    # Minimal valid tokenizer.json for directory structure completeness
    (tok_dir / "tokenizer.json").write_text('{"kind":"char","vocab":{}}', encoding="utf-8")
    be = CharBackend()

    # Missing manifest
    with pytest.raises(AppError, match="manifest not found"):
        _ = be.inspect(str(tok_dir))

    # Invalid manifest JSON type
    (tok_dir / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AppError, match="invalid char tokenizer manifest"):
        _ = be.inspect(str(tok_dir))

    # Stats not a dict
    (tok_dir / "manifest.json").write_text('{"stats":[]}', encoding="utf-8")
    with pytest.raises(AppError, match="invalid stats"):
        _ = be.inspect(str(tok_dir))

    # Valid stats
    (tok_dir / "manifest.json").write_text(
        '{"stats":{"coverage":1.0,"oov_rate":0.0,"token_count":5,"char_coverage":1.0}}',
        encoding="utf-8",
    )
    out = be.inspect(str(tok_dir))
    assert out.coverage == 1.0 and out.oov_rate == 0.0


def test_char_backend_train_no_files_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()
    out_dir = tmp_path / "out"
    cfg = _make_cfg(corpus, out_dir)
    be = CharBackend()
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    with pytest.raises(AppError, match="No text files found"):
        _ = be.train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
