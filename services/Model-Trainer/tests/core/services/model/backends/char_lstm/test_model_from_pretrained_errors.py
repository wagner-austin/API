from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from model_trainer.core.services.model.backends.char_lstm.model import CharLSTM


def test_from_pretrained_invalid_config_not_dict(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    mdir.mkdir()
    (mdir / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AppError):
        _ = CharLSTM.from_pretrained(str(mdir))


def test_from_pretrained_invalid_int_and_float_fields(tmp_path: Path) -> None:
    mdir = tmp_path / "m2"
    mdir.mkdir()
    # v1: wrong embed_dim type (str)
    cfg_bad_int = {
        "vocab_size": 8,
        "embed_dim": "x",
        "hidden_dim": 16,
        "num_layers": 2,
        "dropout": 0.1,
        "max_seq_len": 32,
    }
    (mdir / "config.json").write_text(dump_json_str(cfg_bad_int), encoding="utf-8")
    with pytest.raises(AppError):
        _ = CharLSTM.from_pretrained(str(mdir))

    # v2: wrong dropout type (str)
    cfg_bad_float = {
        "vocab_size": 8,
        "embed_dim": 8,
        "hidden_dim": 16,
        "num_layers": 2,
        "dropout": "oops",
        "max_seq_len": 32,
    }
    (mdir / "config.json").write_text(dump_json_str(cfg_bad_float), encoding="utf-8")
    with pytest.raises(AppError):
        _ = CharLSTM.from_pretrained(str(mdir))
