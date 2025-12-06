from __future__ import annotations

from pathlib import Path

from ...core.config.settings import Settings


def artifacts_root(settings: Settings) -> Path:
    return Path(settings["app"]["artifacts_root"])


# Tokenizers
def tokenizers_dir(settings: Settings) -> Path:
    return artifacts_root(settings) / "tokenizers"


def tokenizer_dir(settings: Settings, tokenizer_id: str) -> Path:
    return tokenizers_dir(settings) / tokenizer_id


def tokenizer_logs_path(settings: Settings, tokenizer_id: str) -> Path:
    return tokenizer_dir(settings, tokenizer_id) / "logs.jsonl"


# Models (training runs)
def models_dir(settings: Settings) -> Path:
    return artifacts_root(settings) / "models"


def model_dir(settings: Settings, run_id: str) -> Path:
    return models_dir(settings) / run_id


def model_logs_path(settings: Settings, run_id: str) -> Path:
    return model_dir(settings, run_id) / "logs.jsonl"


def model_eval_dir(settings: Settings, run_id: str) -> Path:
    return model_dir(settings, run_id) / "eval"
