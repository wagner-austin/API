from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Dataset

from handwriting_ai.training.dataset import (
    DataLoaderConfig,
    PreprocessDataset,
    _rebuild_preprocess_dataset,
    make_loader_config,
    make_loaders,
)
from handwriting_ai.training.train_config import TrainConfig


class _Tiny(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), int(idx)


def _train_cfg(tmp: Path) -> TrainConfig:
    return {
        "data_root": tmp,
        "out_dir": tmp / "out",
        "model_id": "m",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "seed": 0,
        "device": "cpu",
        "precision": "auto",
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


def test_rebuild_preprocess_dataset_roundtrip(tmp_path: Path) -> None:
    base = _Tiny()
    ds = PreprocessDataset(base, _train_cfg(tmp_path))
    rebuilt = _rebuild_preprocess_dataset(base, ds.knobs)
    assert len(rebuilt) == len(ds)
    t, y = rebuilt[0]
    assert t.shape[-2:] == (28, 28)
    assert int(y) == 0


def test_dataloader_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=0,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=-1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=-1,
        )


def test_make_loaders_with_workers_and_prefetch(tmp_path: Path) -> None:
    cfg = _train_cfg(tmp_path)
    loader_cfg = make_loader_config(
        batch_size=1, num_workers=1, pin_memory=False, persistent_workers=True, prefetch_factor=2
    )
    train_base = _Tiny()
    test_base = _Tiny()
    _, train_loader, test_loader = make_loaders(train_base, test_base, cfg, loader_cfg)
    assert train_loader.batch_size == 1 and test_loader.batch_size == 1
    assert train_loader.num_workers == 1 and test_loader.num_workers == 1
    assert train_loader.prefetch_factor == 2 and test_loader.prefetch_factor == 2
    assert train_loader.persistent_workers is True and test_loader.persistent_workers is True


def test_dataloader_config_accessors_cover_invalid_key() -> None:
    cfg = make_loader_config(
        batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=1
    )
    assert cfg["batch_size"] == 2
    assert cfg["prefetch_factor"] == 1
    as_dict = cfg.as_dict()
    assert as_dict["batch_size"] == 2
    with pytest.raises(KeyError):
        _ = cfg["unknown"]
