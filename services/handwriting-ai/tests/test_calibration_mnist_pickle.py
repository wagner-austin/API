from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

from PIL import Image

from handwriting_ai.training.calibration.runner import (
    _mnist_read_images_labels,
    _MNISTRawDataset,
    _rebuild_mnist_raw_dataset,
)
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset


class MnistRawWriter(Protocol):
    def __call__(self, root: Path, n: int = 8) -> None: ...


def test_mnist_raw_dataset_pickles_small_and_rebuilds(
    tmp_path: Path, write_mnist_raw: MnistRawWriter
) -> None:
    # Create small MNIST raw files and build dataset
    root = tmp_path / "data"
    write_mnist_raw(root, n=16)
    imgs, labels = _mnist_read_images_labels(root, train=True)
    ds = _MNISTRawDataset(imgs, labels, root=root, train=True)

    # Pickle should be small (spec only), not tens of MB
    blob = pickle.dumps(ds)
    assert len(blob) < 100_000

    # Rebuild via factory to keep strict typing and avoid dynamic typing from pickle.loads
    ds2: _MNISTRawDataset = _rebuild_mnist_raw_dataset(root, True)
    assert len(ds2) == len(ds)


class _BlankBase:
    """Three blank 28x28 images, defined at module level so pickle can name it."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), idx


def test_preprocess_dataset_reduces_to_its_base_and_knobs() -> None:
    """A DataLoader worker receives the dataset rebuilt from its base and knobs alone."""
    cfg: AugmentConfig = {
        "batch_size": 1,
        "augment": True,
        "aug_rotate": 5.0,
        "aug_translate": 0.1,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
        "morph_kernel_px": 1,
    }
    base = _BlankBase()
    ds = PreprocessDataset(base, cfg)
    rebuild, (reduced_base, knobs) = ds.__reduce__()
    assert reduced_base is base
    assert knobs == ds.knobs
    rebuilt = rebuild(reduced_base, knobs)
    assert len(rebuilt) == 3
    assert rebuilt.knobs == ds.knobs
    assert len(pickle.dumps(ds)) < 10_000
