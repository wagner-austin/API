from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..inference.types import PreprocessOutput
from ..preprocess import PreprocessOptions, run_preprocess
from .augment import (
    apply_affine,
    ensure_l_mode,
    maybe_add_dots,
    maybe_add_noise,
    maybe_blur,
    maybe_morph,
)
from .train_config import TrainConfig


class AugmentConfig(TypedDict):
    """Minimal config for PreprocessDataset augmentation settings."""

    batch_size: int
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


class MNISTLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


class _AugmentKnobs(TypedDict):
    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str
    morph_kernel_px: int


def _knobs_from_cfg(cfg: TrainConfig | AugmentConfig) -> _AugmentKnobs:
    return {
        "enable": bool(cfg["augment"]),
        "rotate_deg": float(cfg["aug_rotate"]),
        "translate_frac": float(cfg["aug_translate"]),
        "noise_prob": float(cfg["noise_prob"]),
        "noise_salt_vs_pepper": float(cfg["noise_salt_vs_pepper"]),
        "dots_prob": float(cfg["dots_prob"]),
        "dots_count": int(cfg["dots_count"]),
        "dots_size_px": int(cfg["dots_size_px"]),
        "blur_sigma": float(cfg["blur_sigma"]),
        "morph_mode": str(cfg["morph"]),
        "morph_kernel_px": int(cfg["morph_kernel_px"]),
    }


def _normalize_morph(x: str) -> Literal["none", "erode", "dilate"]:
    if x == "erode":
        return "erode"
    if x == "dilate":
        return "dilate"
    return "none"


class PreprocessDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset wrapper that applies optional augmentation then service preprocess."""

    def __init__(self, base: MNISTLike, cfg: TrainConfig | AugmentConfig) -> None:
        self._base = base
        self._opts: PreprocessOptions = {
            "invert": None,
            "center": True,
            "visualize": False,
            "visualize_max_kb": 0,
        }
        self._knobs = _knobs_from_cfg(cfg)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_raw, label = self._base[idx]
        g = ensure_l_mode(img_raw)
        if self._knobs["enable"]:
            g = apply_affine(g, self._knobs["rotate_deg"], self._knobs["translate_frac"])
            g = maybe_add_noise(g, self._knobs["noise_prob"], self._knobs["noise_salt_vs_pepper"])
            g = maybe_add_dots(
                g, self._knobs["dots_prob"], self._knobs["dots_count"], self._knobs["dots_size_px"]
            )
            g = maybe_blur(g, self._knobs["blur_sigma"])
            morph = _normalize_morph(self._knobs["morph_mode"])
            g = maybe_morph(g, morph, self._knobs["morph_kernel_px"])
        out: PreprocessOutput = run_preprocess(g, self._opts)
        t: Tensor = out["tensor"].squeeze(0)
        return t, torch.tensor(int(label), dtype=torch.long)

    @property
    def knobs(self) -> _AugmentKnobs:
        """Expose augmentation knobs with precise typing for downstream code."""
        return self._knobs

    # Ensure spawn-pickle safety for subprocess calibration
    def __reduce__(
        self,
    ) -> tuple[
        Callable[[MNISTLike, _AugmentKnobs], PreprocessDataset], tuple[MNISTLike, _AugmentKnobs]
    ]:
        # Rebuild only from base and knobs; options are constant in __init__
        return _rebuild_preprocess_dataset, (self._base, self._knobs)


def _rebuild_preprocess_dataset(base: MNISTLike, knobs: _AugmentKnobs) -> PreprocessDataset:
    cfg: TrainConfig = {
        "data_root": Path("."),
        "out_dir": Path("."),
        "model_id": "",
        "epochs": 1,
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "seed": 42,
        "device": "cpu",
        "optim": "adamw",
        "scheduler": "none",
        "step_size": 10,
        "gamma": 0.5,
        "min_lr": 1e-5,
        "patience": 0,
        "min_delta": 0.0,
        "threads": 1,
        "augment": bool(knobs["enable"]),
        "aug_rotate": float(knobs["rotate_deg"]),
        "aug_translate": float(knobs["translate_frac"]),
        "noise_prob": float(knobs["noise_prob"]),
        "noise_salt_vs_pepper": float(knobs["noise_salt_vs_pepper"]),
        "dots_prob": float(knobs["dots_prob"]),
        "dots_count": int(knobs["dots_count"]),
        "dots_size_px": int(knobs["dots_size_px"]),
        "blur_sigma": float(knobs["blur_sigma"]),
        "morph": str(knobs["morph_mode"]),
        "morph_kernel_px": int(knobs["morph_kernel_px"]),
        "progress_every_epochs": 1,
        "progress_every_batches": 0,
        "calibrate": False,
        "calibration_samples": 0,
        "force_calibration": False,
        "memory_guard": False,
    }
    return PreprocessDataset(base, cfg)


class DataLoaderConfig:
    """Validated data loader configuration with strict typing."""

    __slots__ = (
        "_batch_size",
        "_num_workers",
        "_persistent_workers",
        "_pin_memory",
        "_prefetch_factor",
    )

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
    ) -> None:
        _validate_loader_values(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers)
        self._prefetch_factor = int(prefetch_factor)

    def __getitem__(
        self,
        key: str,
    ) -> int | bool:
        if key == "batch_size":
            return self._batch_size
        if key == "num_workers":
            return self._num_workers
        if key == "pin_memory":
            return self._pin_memory
        if key == "persistent_workers":
            return self._persistent_workers
        if key == "prefetch_factor":
            return self._prefetch_factor
        raise KeyError(key)

    def as_dict(self) -> dict[str, int | bool]:
        """Return a plain dict view for serialization or logging."""
        return {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "prefetch_factor": self._prefetch_factor,
        }


def _validate_loader_values(
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> None:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be >= 0")
    if int(prefetch_factor) < 0:
        raise ValueError("prefetch_factor must be >= 0")
    if persistent_workers and int(num_workers) == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    # pin_memory is already bool; no additional validation needed


def make_loader_config(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoaderConfig:
    return DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def make_loaders(
    train_base: MNISTLike,
    test_base: MNISTLike,
    cfg: TrainConfig,
    loader_cfg: DataLoaderConfig | None = None,
) -> tuple[
    Dataset[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    train_ds: Dataset[tuple[Tensor, Tensor]] = PreprocessDataset(train_base, cfg)
    test_ds: Dataset[tuple[Tensor, Tensor]] = PreprocessDataset(test_base, cfg)

    bs = int(loader_cfg["batch_size"]) if loader_cfg is not None else int(cfg["batch_size"])
    num_workers = int(loader_cfg["num_workers"]) if loader_cfg is not None else 0
    pin_mem = bool(loader_cfg["pin_memory"]) if loader_cfg is not None else False
    # Only set prefetch_factor / persistent_workers when num_workers > 0
    if num_workers > 0 and loader_cfg is not None:
        train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            prefetch_factor=int(loader_cfg["prefetch_factor"]),
            persistent_workers=bool(loader_cfg["persistent_workers"]),
        )
        test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            prefetch_factor=int(loader_cfg["prefetch_factor"]),
            persistent_workers=bool(loader_cfg["persistent_workers"]),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
        )
        test_loader = DataLoader(
            test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
        )
    return train_ds, train_loader, test_loader


# ---------------------------------------------------------------------------
# MNIST Raw Dataset Loading
# ---------------------------------------------------------------------------


def _mnist_find_raw_dir(root: Path) -> Path:
    """Find the MNIST raw directory, supporting both flat and nested layouts.

    Args:
        root: Base path to search from

    Returns:
        Path to directory containing raw MNIST .gz files
    """
    nested = root / "MNIST" / "raw"
    return nested if nested.exists() else root


def _mnist_read_images_labels(root: Path, train: bool) -> tuple[list[bytes], list[int]]:
    """Read MNIST images and labels from raw .gz files.

    Args:
        root: Path to MNIST data root
        train: If True, load training set; otherwise load test set

    Returns:
        Tuple of (list of raw image bytes, list of integer labels)

    Raises:
        RuntimeError: If files are missing, corrupted, or have invalid format
    """
    raw_dir = _mnist_find_raw_dir(root)
    prefix = "train" if train else "t10k"
    img_path = raw_dir / f"{prefix}-images-idx3-ubyte.gz"
    lbl_path = raw_dir / f"{prefix}-labels-idx1-ubyte.gz"

    if not img_path.exists():
        raise RuntimeError(f"MNIST images file not found: {img_path}")
    if not lbl_path.exists():
        raise RuntimeError(f"MNIST labels file not found: {lbl_path}")

    # Read images
    with gzip.open(img_path, "rb") as f_img:
        header = f_img.read(16)
        if len(header) != 16:
            raise RuntimeError(f"Invalid MNIST images header in {img_path}")
        magic = int.from_bytes(header[0:4], "big")
        n_images = int.from_bytes(header[4:8], "big")
        rows = int.from_bytes(header[8:12], "big")
        cols = int.from_bytes(header[12:16], "big")
        if magic != 2051 or rows != 28 or cols != 28:
            raise RuntimeError(f"Invalid MNIST images format in {img_path}")
        total_bytes = n_images * rows * cols
        img_data = f_img.read(total_bytes)
        if len(img_data) != total_bytes:
            raise RuntimeError(f"Truncated MNIST images file: {img_path}")

    # Read labels
    with gzip.open(lbl_path, "rb") as f_lbl:
        header2 = f_lbl.read(8)
        if len(header2) != 8:
            raise RuntimeError(f"Invalid MNIST labels header in {lbl_path}")
        magic2 = int.from_bytes(header2[0:4], "big")
        n_labels = int.from_bytes(header2[4:8], "big")
        if magic2 != 2049 or n_labels != n_images:
            raise RuntimeError(f"Invalid MNIST labels format in {lbl_path}")
        lbl_data = f_lbl.read(n_labels)
        if len(lbl_data) != n_labels:
            raise RuntimeError(f"Truncated MNIST labels file: {lbl_path}")

    # Parse into lists
    stride = 28 * 28
    images = [img_data[i * stride : (i + 1) * stride] for i in range(n_images)]
    labels = [int(b) for b in lbl_data]

    return images, labels


class MNISTRawDataset:
    """MNIST dataset loaded from raw .gz files.

    Implements MNISTLike protocol, returning (PIL.Image, int) tuples.
    This class provides a strictly-typed, standalone MNIST loader that
    does not depend on torchvision.
    """

    __slots__ = ("_images", "_labels")

    def __init__(self, images: list[bytes], labels: list[int]) -> None:
        """Initialize with pre-loaded image bytes and labels.

        Args:
            images: List of raw 28x28 grayscale image bytes (784 bytes each)
            labels: List of integer labels (0-9)
        """
        if len(images) != len(labels):
            raise ValueError(f"images and labels length mismatch: {len(images)} vs {len(labels)}")
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        i = int(idx)
        if i < 0 or i >= len(self._images):
            raise IndexError(f"index {i} out of range for dataset of size {len(self._images)}")
        img = Image.frombytes("L", (28, 28), self._images[i])
        return img, self._labels[i]


def load_mnist_dataset(root: Path, train: bool) -> MNISTRawDataset:
    """Load MNIST dataset from raw .gz files.

    Args:
        root: Path to MNIST data root (should contain MNIST/raw/ or raw files directly)
        train: If True, load training set (60k samples); otherwise test set (10k samples)

    Returns:
        MNISTRawDataset implementing MNISTLike protocol

    Raises:
        RuntimeError: If MNIST files are missing or corrupted
    """
    images, labels = _mnist_read_images_labels(root, train)
    return MNISTRawDataset(images, labels)
