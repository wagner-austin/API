from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

from handwriting_ai.training.calibration.runner import (
    _mnist_read_images_labels,
    _MNISTRawDataset,
    _rebuild_mnist_raw_dataset,
)


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
