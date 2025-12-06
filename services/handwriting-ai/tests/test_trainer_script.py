from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.mnist_train import MNISTLike, TrainConfig


class _Base(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


class _FakeMNIST:
    def __init__(self, n: int = 3) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        label = idx % 10
        return img, label


def test_preprocess_dataset_shapes() -> None:
    base: MNISTLike = _FakeMNIST(2)
    cfg = TrainConfig(
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
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=3,
        dots_size_px=2,
        blur_sigma=0.0,
        morph="none",
        morph_kernel_px=3,
        progress_every_epochs=1,
        progress_every_batches=100,
        calibrate=False,
        calibration_samples=100,
        force_calibration=False,
        memory_guard=False,
    )
    ds = PreprocessDataset(base, cfg)
    x0, y0 = ds[0]
    assert list(x0.shape) == [1, 28, 28]
    assert type(int(y0)) is int
