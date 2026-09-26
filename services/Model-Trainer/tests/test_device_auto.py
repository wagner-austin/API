from __future__ import annotations

import os

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue
from platform_ml import OptimizerName, RequestedDevice, RequestedPrecision, ResolvedDevice
from platform_ml import torch_types as platform_ml_torch_types
from platform_ml.testing import FakeTorchModule
from platform_ml.torch_types import _TorchModuleProtocol

from model_trainer.api.validators.runs import _decode_train_request
from model_trainer.core.contracts.queue import TrainRequestPayload
from model_trainer.worker.job_utils import build_cfg


def _request(**fields: JSONValue) -> dict[str, JSONValue]:
    """A minimal valid train request body, with ``fields`` added."""
    payload: dict[str, JSONValue] = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    payload.update(fields)
    return payload


def _auto_payload() -> TrainRequestPayload:
    """A queued train request that leaves device and precision to the worker."""
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "tokenizer_id": "tok",
        "holdout_fraction": 0.1,
        "seed": 1,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": OptimizerName.ADAMW,
        "device": RequestedDevice.AUTO,
        "precision": RequestedPrecision.AUTO,
        "data_num_workers": None,
        "data_pin_memory": None,
        "early_stopping_patience": 2,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "loss_mask_prefix_separator": None,
        "hub_model_id": None,
        "finetuning_strategy": "full",
        "lora": None,
        "cartridge": None,
        "quantization": None,
        "gguf_export": None,
    }


def _torch_with_cuda(available: bool) -> None:
    """Point platform_ml's torch hook at a fake reporting ``available``."""
    fake_torch = FakeTorchModule(cuda_available=available)

    def _fake_import() -> _TorchModuleProtocol:
        return fake_torch

    platform_ml_torch_types._import_torch = _fake_import


def test_decode_train_request_device_defaults_to_auto() -> None:
    assert _decode_train_request(_request())["device"] is RequestedDevice.AUTO


def test_decode_train_request_narrows_every_device_word() -> None:
    for device in RequestedDevice:
        out = _decode_train_request(_request(device=device.value))
        assert out["device"] is device


def test_decode_train_request_refuses_an_unknown_device() -> None:
    with pytest.raises(AppError, match=r"^device must be one of: auto, cpu, cuda$"):
        _decode_train_request(_request(device="gpu"))


def test_decode_train_request_precision_defaults_to_auto() -> None:
    assert _decode_train_request(_request())["precision"] is RequestedPrecision.AUTO


def test_decode_train_request_narrows_every_precision_word() -> None:
    for precision in RequestedPrecision:
        out = _decode_train_request(_request(precision=precision.value))
        assert out["precision"] is precision


def test_decode_train_request_refuses_an_unknown_precision() -> None:
    with pytest.raises(AppError, match=r"^precision must be one of: auto, bf16, fp16, fp32$"):
        _decode_train_request(_request(precision="int8"))


def test_build_cfg_resolves_auto_and_keeps_the_declared_batch_size() -> None:
    # CUDA available -> auto resolves to cuda, and the declared batch size is
    # left alone. This comment said 'batch size increases to gpt2 default (32)'
    # until the increase was removed on 2026-09-04.
    _torch_with_cuda(True)
    cfg = build_cfg(_auto_payload(), corpus_path="/tmp/corpus")
    assert cfg["device"] is ResolvedDevice.CUDA
    # THE DECLARED BATCH SIZE SURVIVES ONTO CUDA. This asserted 32 until
    # 2026-09-04, because `recommended_batch_size_for` rewrote any value of 4
    # or less to a family default. A payload declaring 4 then trained at 32
    # and recorded 4, which makes one document two experiments. What a
    # payload declares is not a suggestion.
    assert cfg["batch_size"] == 4
    expected_workers = min(4, int(os.cpu_count() or 1))
    assert cfg["data_num_workers"] == expected_workers
    assert cfg["data_pin_memory"] is True


def test_build_cfg_auto_cpu_keeps_batch_size() -> None:
    _torch_with_cuda(False)
    cfg = build_cfg(_auto_payload(), corpus_path="/tmp/corpus")
    assert cfg["device"] is ResolvedDevice.CPU
    assert cfg["batch_size"] == 4
    # CPU retains default of 0 workers by design (keep lightweight in simple setups)
    assert cfg["data_num_workers"] == 0
    assert cfg["data_pin_memory"] is False
