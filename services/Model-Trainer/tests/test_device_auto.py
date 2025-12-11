from __future__ import annotations

import os

import pytest
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request
from model_trainer.core import _test_hooks
from model_trainer.core.compute.device_selector import (
    recommended_batch_size,
    recommended_batch_size_for,
    resolve_device,
    resolve_precision,
)
from model_trainer.core.contracts.queue import TrainRequestPayload
from model_trainer.worker.job_utils import build_cfg


def test_decode_train_request_device_defaults_to_auto() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    out = _decode_train_request(payload)
    assert out["device"] == "auto"


def test_decode_train_request_device_explicit_cuda() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "device": "cuda",
    }
    out = _decode_train_request(payload)
    assert out["device"] == "cuda"


def test_decode_train_request_device_explicit_cpu() -> None:
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "device": "cpu",
    }
    out = _decode_train_request(payload)
    assert out["device"] == "cpu"


def test_resolve_device_auto_cuda() -> None:
    _test_hooks.cuda_is_available = lambda: True
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_cpu() -> None:
    _test_hooks.cuda_is_available = lambda: False
    assert resolve_device("auto") == "cpu"


def test_recommended_batch_size_bumps_on_cuda() -> None:
    assert recommended_batch_size(4, "cuda") == 8
    assert recommended_batch_size(8, "cuda") == 8
    assert recommended_batch_size(4, "cpu") == 4


def test_resolve_device_passthrough_cuda() -> None:
    # No monkeypatch needed; passthrough should not query CUDA
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_passthrough_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_build_cfg_resolves_auto_and_adjusts_batch_size() -> None:
    # CUDA available -> auto resolves to cuda and batch size increases to gpt2 default (32)
    _test_hooks.cuda_is_available = lambda: True
    req: TrainRequestPayload = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "holdout_fraction": 0.1,
        "seed": 1,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "auto",
        "precision": "auto",
        "data_num_workers": None,
        "data_pin_memory": None,
        "early_stopping_patience": 2,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
    }
    cfg = build_cfg(req, corpus_path="/tmp/corpus")
    assert cfg["device"] == "cuda"
    assert cfg["batch_size"] == 32
    expected_workers = min(4, int(os.cpu_count() or 1))
    assert cfg["data_num_workers"] == expected_workers
    assert cfg["data_pin_memory"] is True


def test_build_cfg_auto_cpu_keeps_batch_size() -> None:
    _test_hooks.cuda_is_available = lambda: False
    req: TrainRequestPayload = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "holdout_fraction": 0.1,
        "seed": 1,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "auto",
        "precision": "auto",
        "data_num_workers": None,
        "data_pin_memory": None,
        "early_stopping_patience": 2,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
    }
    cfg = build_cfg(req, corpus_path="/tmp/corpus")
    assert cfg["device"] == "cpu"
    assert cfg["batch_size"] == 4
    # CPU retains default of 0 workers by design (keep lightweight in simple setups)
    assert cfg["data_num_workers"] == 0
    assert cfg["data_pin_memory"] is False


def test_recommended_batch_size_for_families() -> None:
    # char_lstm on CUDA bumps to 64 from small inputs
    assert recommended_batch_size_for("char_lstm", 4, "cuda") == 64
    # gpt2 on CUDA bumps to 32
    assert recommended_batch_size_for("gpt2", 4, "cuda") == 32
    # qwen fallback on CUDA bumps to 16
    assert recommended_batch_size_for("qwen", 4, "cuda") == 16
    # CPU leaves batch unchanged
    assert recommended_batch_size_for("gpt2", 4, "cpu") == 4
    # Larger user-provided batch remains unchanged on CUDA
    assert recommended_batch_size_for("gpt2", 8, "cuda") == 8


# ===== Precision tests =====


def test_resolve_precision_fp32_on_cuda() -> None:
    """fp32 is allowed on any device."""
    assert resolve_precision("fp32", "cuda") == "fp32"


def test_resolve_precision_fp32_on_cpu() -> None:
    """fp32 is allowed on any device."""
    assert resolve_precision("fp32", "cpu") == "fp32"


def test_resolve_precision_fp16_on_cuda() -> None:
    """fp16 is allowed on cuda."""
    assert resolve_precision("fp16", "cuda") == "fp16"


def test_resolve_precision_fp16_on_cpu_raises() -> None:
    """fp16 is NOT allowed on cpu - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"fp16.*not supported on CPU"):
        resolve_precision("fp16", "cpu")


def test_resolve_precision_bf16_on_cuda() -> None:
    """bf16 is allowed on cuda."""
    assert resolve_precision("bf16", "cuda") == "bf16"


def test_resolve_precision_bf16_on_cpu_raises() -> None:
    """bf16 is NOT allowed on cpu - should raise RuntimeError."""
    with pytest.raises(RuntimeError, match=r"bf16.*not supported on CPU"):
        resolve_precision("bf16", "cpu")


def test_resolve_precision_auto_on_cuda() -> None:
    """auto resolves to fp16 on cuda."""
    assert resolve_precision("auto", "cuda") == "fp16"


def test_resolve_precision_auto_on_cpu() -> None:
    """auto resolves to fp32 on cpu."""
    assert resolve_precision("auto", "cpu") == "fp32"


def test_decode_train_request_precision_defaults_to_auto() -> None:
    """Default precision is auto."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    out = _decode_train_request(payload)
    assert out["precision"] == "auto"


def test_decode_train_request_precision_explicit_fp32() -> None:
    """Explicit fp32 is passed through."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "precision": "fp32",
    }
    out = _decode_train_request(payload)
    assert out["precision"] == "fp32"


def test_decode_train_request_precision_explicit_fp16() -> None:
    """Explicit fp16 is passed through."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "precision": "fp16",
    }
    out = _decode_train_request(payload)
    assert out["precision"] == "fp16"


def test_decode_train_request_precision_explicit_bf16() -> None:
    """Explicit bf16 is passed through."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
        "precision": "bf16",
    }
    out = _decode_train_request(payload)
    assert out["precision"] == "bf16"
