from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def test_train_request_missing_corpus_file_id_raises_validation_error() -> None:
    with pytest.raises(AppError) as exc_info:
        payload: JSONValue = {
            "model_family": "gpt2",
            "model_size": "s",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "tokenizer_id": "tok",
            "user_id": 0,
        }
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc_info.value
    assert err.http_status == 400
    assert "corpus_file_id" in err.message


def test_train_request_extra_corpus_path_forbidden() -> None:
    with pytest.raises(AppError) as exc_info:
        payload2: JSONValue = {
            "model_family": "gpt2",
            "model_size": "s",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "corpus_file_id": "deadbeef",
            "corpus_path": "/path.txt",
            "tokenizer_id": "tok",
            "user_id": 0,
        }
        _ = _decode_train_request(payload2)
    err: AppError[ErrorCode] = exc_info.value
    assert err.http_status == 422
    assert "corpus_path" in err.message
