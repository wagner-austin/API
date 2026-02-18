"""Tests for LoRA training contracts."""

from __future__ import annotations

from art_trainer.core.contracts.lora import LoraTrainConfig, LoraTrainOutcome


def test_lora_train_config_structure() -> None:
    """Test that LoraTrainConfig has correct structure."""
    config: LoraTrainConfig = {
        "job_id": "test-job-123",
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
    assert config["job_id"] == "test-job-123"
    assert config["base_model"] == "sd15"
    assert config["training_type"] == "style"
    assert config["steps"] == 1000
    assert config["learning_rate"] == 0.0001


def test_lora_train_outcome_success() -> None:
    """Test LoraTrainOutcome for successful training."""
    outcome: LoraTrainOutcome = {
        "success": True,
        "lora_path": "/output/lora.safetensors",
        "final_loss": 0.05,
        "error_message": None,
    }
    assert outcome["success"] is True
    assert outcome["lora_path"] == "/output/lora.safetensors"
    assert outcome["final_loss"] == 0.05
    assert outcome["error_message"] is None


def test_lora_train_outcome_failure() -> None:
    """Test LoraTrainOutcome for failed training."""
    outcome: LoraTrainOutcome = {
        "success": False,
        "lora_path": None,
        "final_loss": None,
        "error_message": "Out of memory",
    }
    assert outcome["success"] is False
    assert outcome["lora_path"] is None
    assert outcome["final_loss"] is None
    assert outcome["error_message"] == "Out of memory"
