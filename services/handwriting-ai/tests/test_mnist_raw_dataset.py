from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai.training.dataset import (
    MNISTRawDataset,
    _mnist_find_raw_dir,
    _mnist_read_images_labels,
    load_mnist_dataset,
)


def _write_mnist_images_gz(path: Path, n: int) -> None:
    """Write a valid MNIST images .gz file with n 28x28 images."""
    with gzip.open(path, "wb") as f:
        # Header: magic (2051), n_images, rows (28), cols (28)
        f.write((2051).to_bytes(4, "big"))
        f.write(n.to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        # Image data: n * 784 bytes
        f.write(bytes([i % 256 for i in range(n * 784)]))


def _write_mnist_labels_gz(path: Path, n: int) -> None:
    """Write a valid MNIST labels .gz file with n labels."""
    with gzip.open(path, "wb") as f:
        # Header: magic (2049), n_labels
        f.write((2049).to_bytes(4, "big"))
        f.write(n.to_bytes(4, "big"))
        # Label data: n bytes, each 0-9
        f.write(bytes([i % 10 for i in range(n)]))


def _create_mnist_files(root: Path, prefix: str, n: int) -> None:
    """Create MNIST image and label files with given prefix and count."""
    _write_mnist_images_gz(root / f"{prefix}-images-idx3-ubyte.gz", n)
    _write_mnist_labels_gz(root / f"{prefix}-labels-idx1-ubyte.gz", n)


# --- _mnist_find_raw_dir tests ---


def test_mnist_find_raw_dir_nested(tmp_path: Path) -> None:
    nested = tmp_path / "MNIST" / "raw"
    nested.mkdir(parents=True)
    result = _mnist_find_raw_dir(tmp_path)
    assert result == nested


def test_mnist_find_raw_dir_flat(tmp_path: Path) -> None:
    # No nested MNIST/raw, should return root
    result = _mnist_find_raw_dir(tmp_path)
    assert result == tmp_path


# --- _mnist_read_images_labels tests ---


def test_mnist_read_images_labels_train(tmp_path: Path) -> None:
    _create_mnist_files(tmp_path, "train", 5)
    images, labels = _mnist_read_images_labels(tmp_path, train=True)
    assert len(images) == 5
    assert len(labels) == 5
    assert all(len(img) == 784 for img in images)
    assert labels == [0, 1, 2, 3, 4]


def test_mnist_read_images_labels_test(tmp_path: Path) -> None:
    _create_mnist_files(tmp_path, "t10k", 3)
    images, labels = _mnist_read_images_labels(tmp_path, train=False)
    assert len(images) == 3
    assert len(labels) == 3


def test_mnist_read_images_missing_images_file(tmp_path: Path) -> None:
    # Only create labels file
    _write_mnist_labels_gz(tmp_path / "train-labels-idx1-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="images file not found"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_labels_missing_labels_file(tmp_path: Path) -> None:
    # Only create images file
    _write_mnist_images_gz(tmp_path / "train-images-idx3-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="labels file not found"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_invalid_images_header_short(tmp_path: Path) -> None:
    img_path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(img_path, "wb") as f:
        f.write(b"short")  # Only 5 bytes, header needs 16
    _write_mnist_labels_gz(tmp_path / "train-labels-idx1-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="Invalid MNIST images header"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_invalid_images_magic(tmp_path: Path) -> None:
    img_path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(img_path, "wb") as f:
        # Wrong magic number
        f.write((9999).to_bytes(4, "big"))
        f.write((5).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
    _write_mnist_labels_gz(tmp_path / "train-labels-idx1-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="Invalid MNIST images format"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_invalid_images_dimensions(tmp_path: Path) -> None:
    img_path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(img_path, "wb") as f:
        # Correct magic, wrong dimensions
        f.write((2051).to_bytes(4, "big"))
        f.write((5).to_bytes(4, "big"))
        f.write((32).to_bytes(4, "big"))  # Wrong: should be 28
        f.write((32).to_bytes(4, "big"))
    _write_mnist_labels_gz(tmp_path / "train-labels-idx1-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="Invalid MNIST images format"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_truncated_images_data(tmp_path: Path) -> None:
    img_path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(img_path, "wb") as f:
        # Valid header claiming 5 images, but provide only partial data
        f.write((2051).to_bytes(4, "big"))
        f.write((5).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write(bytes(100))  # Only 100 bytes, need 5 * 784 = 3920
    _write_mnist_labels_gz(tmp_path / "train-labels-idx1-ubyte.gz", 5)
    with pytest.raises(RuntimeError, match="Truncated MNIST images file"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_invalid_labels_header_short(tmp_path: Path) -> None:
    _write_mnist_images_gz(tmp_path / "train-images-idx3-ubyte.gz", 5)
    lbl_path = tmp_path / "train-labels-idx1-ubyte.gz"
    with gzip.open(lbl_path, "wb") as f:
        f.write(b"short")  # Only 5 bytes, header needs 8
    with pytest.raises(RuntimeError, match="Invalid MNIST labels header"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_invalid_labels_magic(tmp_path: Path) -> None:
    _write_mnist_images_gz(tmp_path / "train-images-idx3-ubyte.gz", 5)
    lbl_path = tmp_path / "train-labels-idx1-ubyte.gz"
    with gzip.open(lbl_path, "wb") as f:
        f.write((9999).to_bytes(4, "big"))  # Wrong magic
        f.write((5).to_bytes(4, "big"))
        f.write(bytes([0, 1, 2, 3, 4]))
    with pytest.raises(RuntimeError, match="Invalid MNIST labels format"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_labels_count_mismatch(tmp_path: Path) -> None:
    _write_mnist_images_gz(tmp_path / "train-images-idx3-ubyte.gz", 5)
    lbl_path = tmp_path / "train-labels-idx1-ubyte.gz"
    with gzip.open(lbl_path, "wb") as f:
        # Claim 10 labels but images file has 5
        f.write((2049).to_bytes(4, "big"))
        f.write((10).to_bytes(4, "big"))
        f.write(bytes(10))
    with pytest.raises(RuntimeError, match="Invalid MNIST labels format"):
        _mnist_read_images_labels(tmp_path, train=True)


def test_mnist_read_truncated_labels_data(tmp_path: Path) -> None:
    _write_mnist_images_gz(tmp_path / "train-images-idx3-ubyte.gz", 5)
    lbl_path = tmp_path / "train-labels-idx1-ubyte.gz"
    with gzip.open(lbl_path, "wb") as f:
        f.write((2049).to_bytes(4, "big"))
        f.write((5).to_bytes(4, "big"))
        f.write(bytes(3))  # Only 3 bytes, need 5
    with pytest.raises(RuntimeError, match="Truncated MNIST labels file"):
        _mnist_read_images_labels(tmp_path, train=True)


# --- MNISTRawDataset tests ---


def test_mnist_raw_dataset_basic() -> None:
    images = [bytes(784) for _ in range(3)]
    labels = [0, 5, 9]
    ds = MNISTRawDataset(images, labels)
    assert len(ds) == 3
    img, lbl = ds[0]
    assert type(img) is Image.Image
    assert img.size == (28, 28)
    assert img.mode == "L"
    assert lbl == 0


def test_mnist_raw_dataset_getitem_all() -> None:
    images = [bytes([i] * 784) for i in range(5)]
    labels = [i % 10 for i in range(5)]
    ds = MNISTRawDataset(images, labels)
    for i in range(5):
        img, lbl = ds[i]
        assert lbl == i % 10
        assert img.size == (28, 28)


def test_mnist_raw_dataset_length_mismatch() -> None:
    images = [bytes(784) for _ in range(5)]
    labels = [0, 1, 2]  # Mismatch
    with pytest.raises(ValueError, match="mismatch"):
        MNISTRawDataset(images, labels)


def test_mnist_raw_dataset_index_out_of_range_positive() -> None:
    images = [bytes(784) for _ in range(3)]
    labels = [0, 1, 2]
    ds = MNISTRawDataset(images, labels)
    with pytest.raises(IndexError, match="out of range"):
        _ = ds[3]


def test_mnist_raw_dataset_index_out_of_range_negative() -> None:
    images = [bytes(784) for _ in range(3)]
    labels = [0, 1, 2]
    ds = MNISTRawDataset(images, labels)
    with pytest.raises(IndexError, match="out of range"):
        _ = ds[-1]


# --- load_mnist_dataset tests ---


def test_load_mnist_dataset_train(tmp_path: Path) -> None:
    _create_mnist_files(tmp_path, "train", 10)
    ds = load_mnist_dataset(tmp_path, train=True)
    assert len(ds) == 10
    img, lbl = ds[0]
    assert type(img) is Image.Image
    assert img.size == (28, 28)
    assert lbl == 0


def test_load_mnist_dataset_test(tmp_path: Path) -> None:
    _create_mnist_files(tmp_path, "t10k", 7)
    ds = load_mnist_dataset(tmp_path, train=False)
    assert len(ds) == 7


def test_load_mnist_dataset_nested_layout(tmp_path: Path) -> None:
    nested = tmp_path / "MNIST" / "raw"
    nested.mkdir(parents=True)
    _create_mnist_files(nested, "train", 4)
    ds = load_mnist_dataset(tmp_path, train=True)
    assert len(ds) == 4


def test_load_mnist_dataset_missing_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        load_mnist_dataset(tmp_path, train=True)
