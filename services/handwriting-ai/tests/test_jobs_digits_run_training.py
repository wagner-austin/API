from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from handwriting_ai.jobs.digits import _run_training
from handwriting_ai.training.train_config import TrainConfig


def _write_mnist_images_gz(path: Path, n: int) -> None:
    """Write a valid MNIST images .gz file with n 28x28 images."""
    with gzip.open(path, "wb") as f:
        f.write((2051).to_bytes(4, "big"))
        f.write(n.to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write((28).to_bytes(4, "big"))
        f.write(bytes([i % 256 for i in range(n * 784)]))


def _write_mnist_labels_gz(path: Path, n: int) -> None:
    """Write a valid MNIST labels .gz file with n labels."""
    with gzip.open(path, "wb") as f:
        f.write((2049).to_bytes(4, "big"))
        f.write(n.to_bytes(4, "big"))
        f.write(bytes([i % 10 for i in range(n)]))


def _create_mnist_files(root: Path, prefix: str, n: int) -> None:
    """Create MNIST image and label files with given prefix and count."""
    _write_mnist_images_gz(root / f"{prefix}-images-idx3-ubyte.gz", n)
    _write_mnist_labels_gz(root / f"{prefix}-labels-idx1-ubyte.gz", n)


def _train_cfg(tmp_path: Path) -> TrainConfig:
    return {
        "data_root": tmp_path,
        "out_dir": tmp_path / "out",
        "model_id": "test-model",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "seed": 42,
        "device": "cpu",
        "optim": "adamw",
        "scheduler": "none",
        "step_size": 1,
        "gamma": 0.5,
        "min_lr": 1e-5,
        "patience": 0,
        "min_delta": 5e-4,
        "threads": 0,
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
        "morph_kernel_px": 1,
        "progress_every_epochs": 1,
        "progress_every_batches": 0,
        "calibrate": False,
        "calibration_samples": 0,
        "force_calibration": False,
        "memory_guard": False,
    }


def test_run_training_with_valid_mnist_data(tmp_path: Path) -> None:
    """Test _run_training loads MNIST data and trains successfully."""
    # Create minimal MNIST files (4 samples for train, 2 for test)
    _create_mnist_files(tmp_path, "train", 4)
    _create_mnist_files(tmp_path, "t10k", 2)

    cfg = _train_cfg(tmp_path)
    result = _run_training(cfg)

    assert result["model_id"] == "test-model"
    state_dict = result["state_dict"]
    # Verify expected ResNet keys exist with tensor values
    assert "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] == 10
    assert "conv1.weight" in state_dict and state_dict["conv1.weight"].ndim == 4
    val_acc = result["val_acc"]
    assert 0.0 <= val_acc <= 1.0, f"val_acc {val_acc} must be in [0,1]"
    metadata = result["metadata"]
    assert metadata["epochs"] == 1


def test_run_training_missing_train_data(tmp_path: Path) -> None:
    """Test _run_training raises when train MNIST data is missing."""
    # Only create test data, not train data
    _create_mnist_files(tmp_path, "t10k", 2)

    cfg = _train_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="not found"):
        _run_training(cfg)


def test_run_training_missing_test_data(tmp_path: Path) -> None:
    """Test _run_training raises when test MNIST data is missing."""
    # Only create train data, not test data
    _create_mnist_files(tmp_path, "train", 4)

    cfg = _train_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="not found"):
        _run_training(cfg)
