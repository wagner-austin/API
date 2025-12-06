from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_evaluate_request, _decode_train_request
from model_trainer.api.validators.tokenizers import _decode_tokenizer_train_request


def test_decode_train_request_rejects_non_dict() -> None:
    bad: JSONValue = ["not-a-dict"]
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(bad)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_train_request_invalid_model_family() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "bert",
        "model_size": "s",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "f-1",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_train_request_invalid_numeric_and_required_fields() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "s",
        "max_seq_len": 1,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": -1.0,
        "corpus_file_id": "",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status in (400, 422)


def test_decode_train_request_extra_fields_rejected() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "s",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "extra": "nope",
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 422


def test_decode_train_request_accepts_qwen_family() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "qwen",
        "model_size": "s",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    out = _decode_train_request(payload)
    assert out["model_family"] == "qwen"


def test_decode_train_request_accepts_char_lstm_family() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "char_lstm",
        "model_size": "s",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    out = _decode_train_request(payload)
    assert out["model_family"] == "char_lstm"


def test_decode_train_request_model_size_must_be_string() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": 123,
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_train_request_learning_rate_must_be_number() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "s",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": "slow",
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_train_request_model_family_type_error() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": 123,
        "model_size": "s",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_evaluate_request_invalid_split_type() -> None:
    with pytest.raises(AppError) as exc:
        _ = _decode_evaluate_request({"split": 123})
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_evaluate_request_invalid_split_value() -> None:
    with pytest.raises(AppError) as exc:
        _ = _decode_evaluate_request({"split": "train"})
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_evaluate_request_invalid_path_override_type() -> None:
    payload: dict[str, JSONValue] = {"split": "test", "path_override": 5}
    with pytest.raises(AppError) as exc:
        _ = _decode_evaluate_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_evaluate_request_with_path_override() -> None:
    payload: dict[str, JSONValue] = {"split": "validation", "path_override": "/tmp/p"}
    out = _decode_evaluate_request(payload)
    assert out["split"] == "validation"
    assert out["path_override"] == "/tmp/p"


def test_decode_evaluate_request_with_path_override_for_test_split() -> None:
    payload: dict[str, JSONValue] = {"split": "test", "path_override": "/custom"}
    out = _decode_evaluate_request(payload)
    assert out["split"] == "test"
    assert out["path_override"] == "/custom"


def test_decode_evaluate_request_without_path_override() -> None:
    payload: dict[str, JSONValue] = {"split": "validation"}
    out = _decode_evaluate_request(payload)
    assert out["split"] == "validation"
    assert "path_override" not in out


def test_decode_tokenizer_train_request_rejects_non_dict() -> None:
    bad: JSONValue = ["not-dict"]
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(bad)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_invalid_method() -> None:
    payload: dict[str, JSONValue] = {
        "method": "wordpiece",
        "vocab_size": 128,
        "min_frequency": 2,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_invalid_holdout_fraction() -> None:
    payload: dict[str, JSONValue] = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 2,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.9,
        "seed": 1,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_invalid_seed_and_corpus_id() -> None:
    payload: dict[str, JSONValue] = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 2,
        "corpus_file_id": "",
        "holdout_fraction": 0.1,
        "seed": "abc",
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status in (400, 422)


def test_decode_tokenizer_train_request_method_type_error() -> None:
    payload: dict[str, JSONValue] = {
        "method": 123,
        "vocab_size": 128,
        "min_frequency": 2,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_vocab_size_too_small() -> None:
    payload: dict[str, JSONValue] = {
        "method": "bpe",
        "vocab_size": 10,
        "min_frequency": 2,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_min_frequency_invalid() -> None:
    payload: dict[str, JSONValue] = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 0,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_seed_type_error() -> None:
    payload: dict[str, JSONValue] = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 2,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": "not-int",
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_tokenizer_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_tokenizer_train_request_accepts_char_method() -> None:
    payload: dict[str, JSONValue] = {
        "method": "char",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "cid",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    out = _decode_tokenizer_train_request(payload)
    assert out["method"] == "char"


def test_decode_train_request_accepts_pretrained_run_id() -> None:
    """Cover runs.py line 81 - pretrained_run_id validation when provided."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "pretrained_run_id": "run-abc-123",
    }
    out = _decode_train_request(payload)
    assert out["pretrained_run_id"] == "run-abc-123"


def test_decode_train_request_pretrained_run_id_type_error() -> None:
    """Reject pretrained_run_id if not a string."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "pretrained_run_id": 123,
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400


def test_decode_train_request_freeze_embed_type_error() -> None:
    """Cover runs.py _validate_bool error case - freeze_embed must be boolean."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "freeze_embed": "yes",  # should be bool, not string
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400
    assert "freeze_embed" in str(err.message)


def test_decode_train_request_optimizer_adam() -> None:
    """Cover runs.py optimizer='adam' branch."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "optimizer": "adam",
    }
    out = _decode_train_request(payload)
    assert out["optimizer"] == "adam"


def test_decode_train_request_optimizer_sgd() -> None:
    """Cover runs.py optimizer='sgd' branch."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "optimizer": "sgd",
    }
    out = _decode_train_request(payload)
    assert out["optimizer"] == "sgd"


def test_decode_train_request_freeze_embed_explicit_true() -> None:
    """Cover runs.py _validate_bool valid boolean case - freeze_embed=True."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "freeze_embed": True,
    }
    out = _decode_train_request(payload)
    assert out["freeze_embed"] is True


def test_decode_train_request_accepts_data_loader_knobs() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "data_num_workers": 0,
        "data_pin_memory": True,
    }
    out = _decode_train_request(payload)
    assert out["data_num_workers"] == 0
    assert out["data_pin_memory"] is True


def test_decode_train_request_pin_memory_type_error() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "data_pin_memory": "yes",
    }
    with pytest.raises(AppError) as exc:
        _ = _decode_train_request(payload)
    err: AppError[ErrorCode] = exc.value
    assert err.http_status == 400
