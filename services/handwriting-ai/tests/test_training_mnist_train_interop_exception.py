from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from PIL import Image
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import Dataset

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import BatchLoaderProtocol
from handwriting_ai.training.mnist_train import train_with_config
from handwriting_ai.training.train_config import TrainConfig, default_train_config


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


@pytest.fixture(autouse=True)
def _mock_monitoring() -> None:
    """Mock monitoring functions that fail on non-container systems."""
    _test_hooks.log_system_info = lambda: None


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, idx % 10


def _cfg(tmp: Path) -> TrainConfig:
    return default_train_config(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-2,
        seed=0,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_train_with_interop_threads_exception(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(2)
    test_base = _TinyBase(1)

    # Speed up training loop to avoid heavy compute
    def _ok_train_epoch(
        model: TorchModule,
        train_loader: BatchLoaderProtocol,
        device: torch.device,
        precision: Literal["fp32", "fp16", "bf16"],
        optimizer: TorchOptimizer,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        _ = precision  # unused in fake
        return 0.0

    _test_hooks.train_epoch = _ok_train_epoch

    # Force branch: interop_threads is not None and set_num_interop_threads raises
    # The exception is caught and logged in train_with_config, training continues
    def _raise_runtime_error(nthreads: int) -> None:
        raise RuntimeError("boom")

    _test_hooks.torch_has_set_num_interop_threads = lambda: True
    _test_hooks.torch_set_interop_threads = _raise_runtime_error

    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    assert result["state_dict"]  # has state dict
    assert result["metadata"]["run_id"]  # has run metadata
