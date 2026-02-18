"""Tests for Kohya config builder."""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.contracts.lora import LoraTrainConfig
from art_trainer.core.services.training.backends.kohya import _test_hooks
from art_trainer.core.services.training.backends.kohya.config import (
    BASE_MODEL_PATHS,
    KohyaConfig,
    build_kohya_config,
    write_kohya_config,
)

from .testing import FakeConfigWriter


def test_build_kohya_config_sd15() -> None:
    """Test building Kohya config for SD 1.5."""
    config: LoraTrainConfig = {
        "job_id": "test-sd15",
        "base_model": "sd15",
        "training_type": "style",
        "dataset_dir": "/data/dataset",
        "output_dir": "/data/output",
        "steps": 1000,
        "learning_rate": 0.0001,
        "network_rank": 16,
        "network_alpha": 16,
        "resolution": 512,
        "batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }
    kohya_cfg = build_kohya_config(config)
    assert kohya_cfg["pretrained_model_name_or_path"] == BASE_MODEL_PATHS["sd15"]
    assert kohya_cfg["train_data_dir"] == "/data/dataset"
    assert kohya_cfg["output_dir"] == "/data/output"
    assert kohya_cfg["max_train_steps"] == 1000
    assert kohya_cfg["network_dim"] == 16
    assert kohya_cfg["network_alpha"] == 16
    assert kohya_cfg["resolution"] == "512,512"


def test_build_kohya_config_sdxl() -> None:
    """Test building Kohya config for SDXL."""
    config: LoraTrainConfig = {
        "job_id": "test-sdxl",
        "base_model": "sdxl",
        "training_type": "character",
        "dataset_dir": "/data/dataset",
        "output_dir": "/data/output",
        "steps": 2000,
        "learning_rate": 0.0001,
        "network_rank": 32,
        "network_alpha": 16,
        "resolution": 1024,
        "batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }
    kohya_cfg = build_kohya_config(config)
    assert kohya_cfg["pretrained_model_name_or_path"] == BASE_MODEL_PATHS["sdxl"]
    assert kohya_cfg["resolution"] == "1024,1024"
    assert kohya_cfg["no_half_vae"] is True


def test_build_kohya_config_flux() -> None:
    """Test building Kohya config for FLUX."""
    config: LoraTrainConfig = {
        "job_id": "test-flux",
        "base_model": "flux",
        "training_type": "concept",
        "dataset_dir": "/data/dataset",
        "output_dir": "/data/output",
        "steps": 3000,
        "learning_rate": 0.00005,
        "network_rank": 64,
        "network_alpha": 32,
        "resolution": 1024,
        "batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": False,
        "keep_tokens": 0,
    }
    kohya_cfg = build_kohya_config(config)
    assert kohya_cfg["pretrained_model_name_or_path"] == BASE_MODEL_PATHS["flux"]
    assert kohya_cfg["clip_skip"] == 2


def test_write_kohya_config_with_fake_writer() -> None:
    """Test write_kohya_config uses hook when set."""
    fake_writer = FakeConfigWriter()
    _test_hooks.Hooks.config_writer = fake_writer

    config: KohyaConfig = {
        "pretrained_model_name_or_path": "test-model",
        "train_data_dir": "/test/data",
        "output_dir": "/test/output",
        "output_name": "test_lora",
        "max_train_steps": 100,
        "learning_rate": 0.0001,
        "network_dim": 16,
        "network_alpha": 16,
        "resolution": "512x512",
        "train_batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
        "network_module": "networks.lora",
        "save_model_as": "safetensors",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100,
        "gradient_checkpointing": True,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "min_bucket_reso": 256,
        "max_bucket_reso": 1024,
    }
    path = Path("/fake/path/config.toml")
    write_kohya_config(config, path)

    assert len(fake_writer.written_configs) == 1
    written_config, written_path = fake_writer.written_configs[0]
    assert written_config == config
    assert written_path == path


def test_write_kohya_config_real_toml(tmp_path: Path) -> None:
    """Test write_kohya_config writes real TOML file when hook is None."""
    # Make sure hook is None to use real TOML writing
    _test_hooks.Hooks.config_writer = None

    config: KohyaConfig = {
        "pretrained_model_name_or_path": "test-model",
        "train_data_dir": "/test/data",
        "output_dir": "/test/output",
        "output_name": "test_lora",
        "max_train_steps": 100,
        "learning_rate": 0.0001,
        "network_dim": 16,
        "network_alpha": 16,
        "resolution": "512x512",
        "train_batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
        "network_module": "networks.lora",
        "save_model_as": "safetensors",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100,
        "gradient_checkpointing": True,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "min_bucket_reso": 256,
        "max_bucket_reso": 1024,
    }
    path = tmp_path / "output" / "config.toml"
    write_kohya_config(config, path)

    # Verify file was created
    assert path.exists()

    # Verify file contains expected keys by reading as text and checking
    content = path.read_text(encoding="utf-8")
    assert 'pretrained_model_name_or_path = "test-model"' in content
    assert "max_train_steps = 100" in content
