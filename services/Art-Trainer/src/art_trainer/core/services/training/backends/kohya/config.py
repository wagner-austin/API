"""TOML config builder for Kohya_ss training.

This module provides functions to build Kohya_ss TOML configuration
from LoraTrainConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import toml

from art_trainer.core.contracts.lora import LoraTrainConfig

# Base model to pretrained model path mapping
BASE_MODEL_PATHS: dict[Literal["sd15", "sdxl", "flux"], str] = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-dev",
}


class KohyaConfig(TypedDict, total=False):
    """Kohya_ss training configuration for TOML serialization.

    Attributes:
        pretrained_model_name_or_path: Path to pretrained model.
        train_data_dir: Directory containing training data.
        output_dir: Directory for output files.
        output_name: Name for output LoRA file.
        max_train_steps: Maximum training steps.
        learning_rate: Learning rate.
        network_dim: Network dimension (rank).
        network_alpha: Network alpha.
        resolution: Training resolution as "WxH" string.
        train_batch_size: Training batch size.
        seed: Random seed.
        caption_extension: Caption file extension.
        shuffle_caption: Whether to shuffle captions.
        keep_tokens: Number of tokens to keep.
        network_module: Network module name.
        save_model_as: Output format.
        mixed_precision: Mixed precision mode.
        save_precision: Save precision mode.
        cache_latents: Whether to cache latents.
        cache_latents_to_disk: Whether to cache latents to disk.
        optimizer_type: Optimizer type.
        lr_scheduler: Learning rate scheduler.
        lr_warmup_steps: Warmup steps.
        gradient_checkpointing: Whether to use gradient checkpointing.
        enable_bucket: Whether to enable bucketing.
        bucket_reso_steps: Bucket resolution steps.
        min_bucket_reso: Minimum bucket resolution.
        max_bucket_reso: Maximum bucket resolution.
        no_half_vae: Disable half-precision VAE (SDXL).
        clip_skip: CLIP skip layers (FLUX).
    """

    pretrained_model_name_or_path: str
    train_data_dir: str
    output_dir: str
    output_name: str
    max_train_steps: int
    learning_rate: float
    network_dim: int
    network_alpha: int
    resolution: str
    train_batch_size: int
    seed: int
    caption_extension: str
    shuffle_caption: bool
    keep_tokens: int
    network_module: str
    save_model_as: str
    mixed_precision: str
    save_precision: str
    cache_latents: bool
    cache_latents_to_disk: bool
    optimizer_type: str
    lr_scheduler: str
    lr_warmup_steps: int
    gradient_checkpointing: bool
    enable_bucket: bool
    bucket_reso_steps: int
    min_bucket_reso: int
    max_bucket_reso: int
    no_half_vae: bool
    clip_skip: int


def build_kohya_config(config: LoraTrainConfig) -> KohyaConfig:
    """Build Kohya_ss TOML configuration from LoraTrainConfig.

    Args:
        config: LoRA training configuration.

    Returns:
        KohyaConfig TypedDict suitable for TOML serialization.
    """
    pretrained_model = BASE_MODEL_PATHS[config["base_model"]]

    kohya_config: KohyaConfig = {
        "pretrained_model_name_or_path": pretrained_model,
        "train_data_dir": config["dataset_dir"],
        "output_dir": config["output_dir"],
        "output_name": f"lora_{config['job_id']}",
        "max_train_steps": config["steps"],
        "learning_rate": config["learning_rate"],
        "network_dim": config["network_rank"],
        "network_alpha": config["network_alpha"],
        "resolution": f"{config['resolution']},{config['resolution']}",
        "train_batch_size": config["batch_size"],
        "seed": config["seed"],
        "caption_extension": config["caption_extension"],
        "shuffle_caption": config["shuffle_caption"],
        "keep_tokens": config["keep_tokens"],
        "network_module": "networks.lora",
        "save_model_as": "safetensors",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": min(100, config["steps"] // 10),
        "gradient_checkpointing": True,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "min_bucket_reso": 256,
        "max_bucket_reso": config["resolution"],
    }

    # Add SDXL-specific settings
    if config["base_model"] == "sdxl":
        kohya_config["no_half_vae"] = True

    # Add FLUX-specific settings
    if config["base_model"] == "flux":
        kohya_config["clip_skip"] = 2

    return kohya_config


def write_kohya_config(config: KohyaConfig, path: Path) -> None:
    """Write Kohya config dictionary to TOML file.

    Uses the config_writer hook if set, otherwise uses toml.dump.

    Args:
        config: KohyaConfig TypedDict.
        path: Path to write the TOML file.
    """
    # Lazy import to avoid circular dependency
    from . import _test_hooks

    if _test_hooks.Hooks.config_writer is not None:
        _test_hooks.Hooks.config_writer(config, path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        toml.dump(config, f)


__all__ = [
    "BASE_MODEL_PATHS",
    "KohyaConfig",
    "build_kohya_config",
    "write_kohya_config",
]
