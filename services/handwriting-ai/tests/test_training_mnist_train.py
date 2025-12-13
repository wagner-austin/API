from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import DataLoader, Dataset

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import BatchLoaderProtocol
from handwriting_ai.training import mnist_train as mt
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.mnist_train import (
    _apply_affine,
    _build_model,
    _build_optimizer_and_scheduler,
    _ensure_image,
    _evaluate,
    _set_seed,
    _train_epoch,
    make_loaders,
    train_with_config,
)
from handwriting_ai.training.train_config import TrainConfig, default_train_config


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


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
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        return img, idx % 10


def _cfg(tmp: Path) -> TrainConfig:
    return default_train_config(
        data_root=tmp / "data",  # not used when injecting bases
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-2,
        seed=123,
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


def test_apply_affine_identity() -> None:
    img = Image.new("L", (32, 32), 255)
    img.putpixel((16, 16), 0)
    out = _apply_affine(img, deg_max=0.0, tx_frac=0.0)
    assert type(out) is Image.Image
    assert out.size == img.size
    # center pixel remains dark
    assert out.getpixel((16, 16)) == 0


def test_set_seed_reproducible() -> None:
    _set_seed(123)
    a = torch.rand(3)
    _set_seed(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_build_model_and_evaluate_smoke() -> None:
    model = _build_model()
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    y_out: Tensor = model(x)
    shape_list: list[int] = list(y_out.shape)
    assert shape_list == [1, 10]
    # Verify outputs are valid logits (finite, not all zeros)
    assert torch.isfinite(y_out).all(), "outputs must be finite"
    # Verify softmax produces valid probability distribution
    probs = torch.softmax(y_out, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1)), "softmax must sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "probabilities must be in [0, 1]"
    # Evaluate on tiny loader
    base = _TinyBase(2)
    cfg0 = default_train_config(
        data_root=Path("."),
        out_dir=Path("."),
        model_id="m",
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
    ds = PreprocessDataset(base, cfg0)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(ds, batch_size=2, shuffle=False)
    acc = _evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_make_loaders_and_train_epoch_augment(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg["augment"] = True
    cfg["aug_rotate"] = 5.0
    cfg["aug_translate"] = 0.1
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    ds, train_loader, _ = make_loaders(train_base, test_base, cfg)
    assert type(ds) is PreprocessDataset
    # training DataLoader yields Tensor labels
    model = _build_model()
    opt, _ = _build_optimizer_and_scheduler(model, cfg)
    loss = _train_epoch(
        model,
        train_loader,
        torch.device("cpu"),
        "fp32",
        opt,
        ep=1,
        ep_total=1,
        total_batches=len(train_loader),
    )
    assert type(loss) is float


def test_train_with_config_writes_artifacts(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    cfg = _cfg(tmp_path)
    cfg["out_dir"] = tmp_path / "out"
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    assert result["state_dict"]  # has state dict
    assert result["val_acc"] >= 0  # valid accuracy


def test_train_with_scheduler_and_early_stop(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    cfg = _cfg(tmp_path)
    cfg["out_dir"] = tmp_path / "out_step"
    cfg["epochs"] = 2
    cfg["scheduler"] = "step"
    cfg["step_size"] = 1
    cfg["patience"] = 1
    cfg["min_delta"] = 0.01
    train_base = _TinyBase(6)
    test_base = _TinyBase(2)
    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    assert result["state_dict"]  # has state dict


def test_train_calls_evaluate_in_epoch(tmp_path: Path, write_mnist_raw: MnistRawWriter) -> None:
    cfg = _cfg(tmp_path)
    cfg["out_dir"] = tmp_path / "out_eval"
    cfg["epochs"] = 1
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)

    called = {"n": 0}
    _orig = mt._evaluate

    def _spy(
        model: Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> float:
        called["n"] += 1
        return _orig(model, loader, device)

    mt._evaluate = _spy  # patch
    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    assert result["state_dict"]  # has state dict
    assert called["n"] >= 1


def test_train_interrupt_saves_artifact(tmp_path: Path, write_mnist_raw: MnistRawWriter) -> None:
    # Patch _train_epoch to raise KeyboardInterrupt and ensure artifacts still write
    cfg = _cfg(tmp_path)
    cfg["out_dir"] = tmp_path / "out_interrupt"
    cfg["epochs"] = 2
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)

    def _boom(
        model: Module,
        train_loader: BatchLoaderProtocol,
        device: torch.device,
        precision: Literal["fp32", "fp16", "bf16"],
        optimizer: TorchOptimizer,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        _ = precision  # unused in fake
        raise KeyboardInterrupt

    # Use hook to override train_epoch behavior
    _test_hooks.train_epoch = _boom
    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    # Even when interrupted, we should still get current/best weights and metadata
    assert result["state_dict"]
    assert result["metadata"]["run_id"]


def test_train_threads_log_branch_no_interop(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    # Exercise branch where torch has no interop threads functions (line 228)
    cfg = _cfg(tmp_path)
    cfg["out_dir"] = tmp_path / "out_threads"
    cfg["epochs"] = 1
    train_base = _TinyBase(2)
    test_base = _TinyBase(2)

    # Use hooks to simulate torch not having interop thread functions
    # This covers both the set function (for apply_threads) and get function (for logging)
    _test_hooks.torch_has_set_num_interop_threads = lambda: False
    _test_hooks.torch_has_get_num_interop_threads = lambda: False
    write_mnist_raw(cfg["data_root"], n=8)
    result = train_with_config(cfg, (train_base, test_base))
    assert result["state_dict"]  # has state dict


def test_ensure_image_guard_and_ok() -> None:
    ok_img = Image.new("L", (8, 8), 0)
    img = _ensure_image(ok_img)
    assert type(img) is Image.Image
    with pytest.raises(RuntimeError):
        _ensure_image(42)
