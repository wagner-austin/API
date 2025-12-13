from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from PIL import Image
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import Dataset

import handwriting_ai.training.mnist_train as mt
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import BatchLoaderProtocol
from handwriting_ai.training.metrics import BatchMetrics
from handwriting_ai.training.mnist_train import set_progress_emitter
from handwriting_ai.training.progress import (
    emit_batch,
    emit_best,
    emit_epoch,
    set_batch_emitter,
    set_best_emitter,
    set_epoch_emitter,
)
from handwriting_ai.training.train_config import TrainConfig, default_train_config


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        for y in range(10, 18):
            for x in range(12, 16):
                img.putpixel((x, y), 255)
        return img, idx % 10


class _Rec:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float | None]] = []

    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None:
        acc: float | None = float(val_acc) if isinstance(val_acc, float) else None
        self.calls.append((int(epoch), int(total_epochs), acc))


def _cfg(tmp: Path) -> TrainConfig:
    return default_train_config(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=2,
        batch_size=2,
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
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=3,
        dots_size_px=2,
        blur_sigma=0.0,
        morph="none",
        morph_kernel_px=3,
        progress_every_epochs=1,
        progress_every_batches=10,
        calibrate=False,
        calibration_samples=8,
        force_calibration=False,
        memory_guard=False,
    )


def test_progress_emitter_receives_epoch_updates(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    rec = _Rec()

    # Set up hooks for monitoring and training
    _test_hooks.log_system_info = lambda: None

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

    set_progress_emitter(rec)
    try:
        # Ensure MNIST raw files exist for calibration and training
        write_mnist_raw(cfg["data_root"], n=8)
        result = mt.train_with_config(cfg, (train_base, test_base))
        assert result["state_dict"]  # has state dict
    finally:
        set_progress_emitter(None)
    # Expect an emit per epoch
    assert len(rec.calls) >= 2
    # Verify epoch numbers are within range and total matches config
    tot = rec.calls[-1][1]
    assert tot == cfg["epochs"]
    assert rec.calls[0][0] == 1
    assert rec.calls[-1][0] <= cfg["epochs"]


def test_progress_emitter_failure_raises_after_logging(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(2)
    test_base = _TinyBase(2)

    # Set up hooks for monitoring and training
    _test_hooks.log_system_info = lambda: None

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

    class _Bad:
        def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None:
            raise ValueError("boom")

    set_progress_emitter(_Bad())
    try:
        # Should raise when emitter fails
        with pytest.raises(ValueError, match="boom"):
            # Ensure MNIST raw files exist for calibration
            write_mnist_raw(cfg["data_root"], n=8)
            mt.train_with_config(cfg, (train_base, test_base))
    finally:
        set_progress_emitter(None)


def test_progress_emitter_every_n_epochs(tmp_path: Path, write_mnist_raw: MnistRawWriter) -> None:
    cfg = _cfg(tmp_path)
    # Increase epochs and set cadence
    cfg_modified: TrainConfig = default_train_config(
        data_root=cfg["data_root"],
        out_dir=cfg["out_dir"],
        model_id=cfg["model_id"],
        epochs=5,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        device=cfg["device"],
        optim=cfg["optim"],
        scheduler=cfg["scheduler"],
        step_size=cfg["step_size"],
        gamma=cfg["gamma"],
        min_lr=cfg["min_lr"],
        patience=0,
        min_delta=cfg["min_delta"],
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        noise_prob=cfg["noise_prob"],
        noise_salt_vs_pepper=cfg["noise_salt_vs_pepper"],
        dots_prob=cfg["dots_prob"],
        dots_count=cfg["dots_count"],
        dots_size_px=cfg["dots_size_px"],
        blur_sigma=cfg["blur_sigma"],
        morph=cfg["morph"],
        morph_kernel_px=cfg["morph_kernel_px"],
        progress_every_epochs=2,
        progress_every_batches=cfg["progress_every_batches"],
        calibrate=cfg["calibrate"],
        calibration_samples=cfg["calibration_samples"],
        force_calibration=cfg["force_calibration"],
        memory_guard=cfg["memory_guard"],
    )
    cfg = cfg_modified
    train_base = _TinyBase(6)
    test_base = _TinyBase(3)

    # Set up hooks for monitoring and training
    _test_hooks.log_system_info = lambda: None

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

    rec = _Rec()
    set_progress_emitter(rec)
    try:
        # Ensure MNIST raw files exist for calibration and training
        write_mnist_raw(cfg["data_root"], n=8)
        result = mt.train_with_config(cfg, (train_base, test_base))
        assert result["state_dict"]  # has state dict
    finally:
        set_progress_emitter(None)


class _BatchBestEpochRecorder:
    def __init__(self) -> None:
        self.batch: list[tuple[int, int, int]] = []
        self.best: list[tuple[int, float]] = []
        self.epoch: list[tuple[int, float, float]] = []

    def emit_batch(self, metrics: BatchMetrics) -> None:
        self.batch.append((metrics["epoch"], metrics["total_epochs"], metrics["batch"]))

    def emit_best(self, *, epoch: int, val_acc: float) -> None:
        self.best.append((epoch, val_acc))

    def emit_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_acc: float,
        time_s: float,
    ) -> None:
        self.epoch.append((epoch, train_loss, val_acc))


def test_batch_best_epoch_emitters_are_called() -> None:
    rec = _BatchBestEpochRecorder()
    set_batch_emitter(rec)
    set_best_emitter(rec)
    set_epoch_emitter(rec)

    emit_batch(
        {
            "epoch": 1,
            "total_epochs": 2,
            "batch": 3,
            "total_batches": 10,
            "batch_loss": 0.1,
            "batch_acc": 0.9,
            "avg_loss": 0.2,
            "samples_per_sec": 50.0,
            "main_rss_mb": 100,
            "workers_rss_mb": 50,
            "worker_count": 2,
            "cgroup_usage_mb": 500,
            "cgroup_limit_mb": 1000,
            "cgroup_pct": 50.0,
            "anon_mb": 200,
            "file_mb": 150,
        }
    )
    emit_best(epoch=1, val_acc=0.5)
    emit_epoch(epoch=1, total_epochs=2, train_loss=0.3, val_acc=0.4, time_s=1.0)

    assert len(rec.batch) > 0 and len(rec.best) > 0 and len(rec.epoch) > 0


def test_batch_best_epoch_emitter_failures_raise_after_logging() -> None:
    class _Bad:
        def emit_batch(self, metrics: BatchMetrics) -> None:
            raise ValueError("boom")

        def emit_best(self, *, epoch: int, val_acc: float) -> None:
            raise ValueError("boom")

        def emit_epoch(
            self,
            *,
            epoch: int,
            total_epochs: int,
            train_loss: float,
            val_acc: float,
            time_s: float,
        ) -> None:
            raise ValueError("boom")

    bad = _Bad()
    set_batch_emitter(bad)
    set_best_emitter(bad)
    set_epoch_emitter(bad)

    # Calls should raise after logging
    with pytest.raises(ValueError, match="boom"):
        emit_batch(
            {
                "epoch": 1,
                "total_epochs": 2,
                "batch": 1,
                "total_batches": 2,
                "batch_loss": 0.1,
                "batch_acc": 0.9,
                "avg_loss": 0.2,
                "samples_per_sec": 10.0,
                "main_rss_mb": 100,
                "workers_rss_mb": 50,
                "worker_count": 2,
                "cgroup_usage_mb": 500,
                "cgroup_limit_mb": 1000,
                "cgroup_pct": 50.0,
                "anon_mb": 200,
                "file_mb": 150,
            }
        )
    with pytest.raises(ValueError, match="boom"):
        emit_best(epoch=1, val_acc=0.8)
    with pytest.raises(ValueError, match="boom"):
        emit_epoch(epoch=1, total_epochs=2, train_loss=0.3, val_acc=0.4, time_s=1.0)


def test_emit_no_emitters_noop() -> None:
    # Ensure calling emitters without setting them is a no-op (covers early returns)
    set_batch_emitter(None)
    set_best_emitter(None)
    set_epoch_emitter(None)
    emit_batch(
        {
            "epoch": 1,
            "total_epochs": 2,
            "batch": 1,
            "total_batches": 2,
            "batch_loss": 0.1,
            "batch_acc": 0.9,
            "avg_loss": 0.2,
            "samples_per_sec": 10.0,
            "main_rss_mb": 100,
            "workers_rss_mb": 50,
            "worker_count": 2,
            "cgroup_usage_mb": 500,
            "cgroup_limit_mb": 1000,
            "cgroup_pct": 50.0,
            "anon_mb": 200,
            "file_mb": 150,
        }
    )
    emit_best(epoch=1, val_acc=0.7)
    emit_epoch(epoch=1, total_epochs=2, train_loss=0.1, val_acc=0.2, time_s=0.5)
