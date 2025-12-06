from __future__ import annotations

from pathlib import Path

import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.safety import get_memory_guard_config


def _cfg() -> mt.TrainConfig:
    return {
        "data_root": Path("."),
        "out_dir": Path("."),
        "model_id": "m",
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "seed": 0,
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
        "morph_kernel_px": 3,
        "progress_every_epochs": 1,
        "progress_every_batches": 0,
        "calibrate": False,
        "calibration_samples": 0,
        "force_calibration": False,
        "memory_guard": True,
    }


def test_mem_guard_config_sub_1gb() -> None:
    """Memory < 1GB should use 80% threshold."""
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=512 * 1024 * 1024,  # 512MB
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=32,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is True
    assert cur["threshold_percent"] == 80.0
    assert cur["required_consecutive"] == 3


def test_mem_guard_config_1_2gb() -> None:
    """Memory 1-2GB should use 85% threshold."""
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1536 * 1024 * 1024,  # 1.5GB
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is True
    assert cur["threshold_percent"] == 85.0
    assert cur["required_consecutive"] == 3


def test_mem_guard_config_2_4gb() -> None:
    """Memory 2-4GB should use 88% threshold."""
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=3 * 1024 * 1024 * 1024,  # 3GB
        optimal_threads=2,
        optimal_workers=1,
        max_batch_size=128,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is True
    assert cur["threshold_percent"] == 88.0
    assert cur["required_consecutive"] == 3


def test_mem_guard_config_4gb_plus() -> None:
    """Memory >= 4GB should use 92% threshold."""
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=8 * 1024 * 1024 * 1024,  # 8GB
        optimal_threads=4,
        optimal_workers=2,
        max_batch_size=512,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is True
    assert cur["threshold_percent"] == 92.0
    assert cur["required_consecutive"] == 3


def test_mem_guard_config_unlimited() -> None:
    """No memory limit should use conservative 88% default."""
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=None,
        optimal_threads=4,
        optimal_workers=2,
        max_batch_size=512,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is True
    assert cur["threshold_percent"] == 88.0
    assert cur["required_consecutive"] == 3


def test_mem_guard_disabled_by_user() -> None:
    """User can disable memory guard via config."""
    cfg = _cfg()
    cfg_disabled: mt.TrainConfig = {
        "data_root": cfg["data_root"],
        "out_dir": cfg["out_dir"],
        "model_id": cfg["model_id"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "seed": cfg["seed"],
        "device": cfg["device"],
        "optim": cfg["optim"],
        "scheduler": cfg["scheduler"],
        "step_size": cfg["step_size"],
        "gamma": cfg["gamma"],
        "min_lr": cfg["min_lr"],
        "patience": cfg["patience"],
        "min_delta": cfg["min_delta"],
        "threads": cfg["threads"],
        "augment": cfg["augment"],
        "aug_rotate": cfg["aug_rotate"],
        "aug_translate": cfg["aug_translate"],
        "noise_prob": cfg["noise_prob"],
        "noise_salt_vs_pepper": cfg["noise_salt_vs_pepper"],
        "dots_prob": cfg["dots_prob"],
        "dots_count": cfg["dots_count"],
        "dots_size_px": cfg["dots_size_px"],
        "blur_sigma": cfg["blur_sigma"],
        "morph": cfg["morph"],
        "morph_kernel_px": cfg["morph_kernel_px"],
        "progress_every_epochs": cfg["progress_every_epochs"],
        "progress_every_batches": cfg["progress_every_batches"],
        "calibrate": cfg["calibrate"],
        "calibration_samples": cfg["calibration_samples"],
        "force_calibration": cfg["force_calibration"],
        "memory_guard": False,
    }
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    mt._configure_memory_guard_from_limits(cfg_disabled, limits)
    cur = get_memory_guard_config()
    assert cur["enabled"] is False
