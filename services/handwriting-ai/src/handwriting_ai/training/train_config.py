from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from platform_ml import (
    RequestedDevice,
    RequestedPrecision,
    ResolvedDevice,
    ResolvedPrecision,
)
from torch import Tensor


class TrainingResultMetadata(TypedDict):
    """Metadata from a training run for artifact creation."""

    run_id: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    device: ResolvedDevice
    precision: ResolvedPrecision
    optim: str
    scheduler: str
    augment: bool


class TrainingResult(TypedDict):
    """Result from train_with_config() for downstream artifact creation.

    Training returns results, not files. Artifact I/O is handled by the job layer.
    """

    model_id: str
    state_dict: dict[str, Tensor]
    val_acc: float
    metadata: TrainingResultMetadata


class TrainConfig(TypedDict):
    data_root: Path
    out_dir: Path
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: RequestedDevice
    precision: RequestedPrecision
    optim: str
    scheduler: str
    step_size: int
    gamma: float
    min_lr: float
    patience: int
    min_delta: float
    threads: int
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph: str
    morph_kernel_px: int
    progress_every_epochs: int
    progress_every_batches: int
    calibrate: bool
    calibration_samples: int
    force_calibration: bool
    memory_guard: bool


def default_train_config(
    *,
    data_root: Path = Path("./data/mnist"),
    out_dir: Path = Path("./artifacts/digits/models"),
    model_id: str = "mnist_resnet18_v1",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
    device: RequestedDevice = "auto",
    precision: RequestedPrecision = "auto",
    optim: str = "adamw",
    scheduler: str = "none",
    step_size: int = 10,
    gamma: float = 0.5,
    min_lr: float = 1e-5,
    patience: int = 0,
    min_delta: float = 5e-4,
    threads: int = 0,
    augment: bool = False,
    aug_rotate: float = 0.0,
    aug_translate: float = 0.0,
    noise_prob: float = 0.0,
    noise_salt_vs_pepper: float = 0.5,
    dots_prob: float = 0.0,
    dots_count: int = 3,
    dots_size_px: int = 2,
    blur_sigma: float = 0.0,
    morph: str = "none",
    morph_kernel_px: int = 3,
    progress_every_epochs: int = 1,
    progress_every_batches: int = 100,
    calibrate: bool = False,
    calibration_samples: int = 100,
    force_calibration: bool = False,
    memory_guard: bool = False,
) -> TrainConfig:
    """Create a TrainConfig with sensible defaults.

    All parameters are keyword-only to ensure explicit configuration.
    Override specific values as needed for tests or production use.
    """
    return TrainConfig(
        data_root=data_root,
        out_dir=out_dir,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        precision=precision,
        optim=optim,
        scheduler=scheduler,
        step_size=step_size,
        gamma=gamma,
        min_lr=min_lr,
        patience=patience,
        min_delta=min_delta,
        threads=threads,
        augment=augment,
        aug_rotate=aug_rotate,
        aug_translate=aug_translate,
        noise_prob=noise_prob,
        noise_salt_vs_pepper=noise_salt_vs_pepper,
        dots_prob=dots_prob,
        dots_count=dots_count,
        dots_size_px=dots_size_px,
        blur_sigma=blur_sigma,
        morph=morph,
        morph_kernel_px=morph_kernel_px,
        progress_every_epochs=progress_every_epochs,
        progress_every_batches=progress_every_batches,
        calibrate=calibrate,
        calibration_samples=calibration_samples,
        force_calibration=force_calibration,
        memory_guard=memory_guard,
    )
