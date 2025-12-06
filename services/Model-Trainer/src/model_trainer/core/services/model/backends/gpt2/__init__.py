from __future__ import annotations

# GPT-2 backend public API re-exports (kept minimal).
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel

from .evaluate import EvalResult, evaluate_gpt2
from .generate import generate_gpt2
from .io import load_prepared_gpt2_from_handle, save_prepared_gpt2
from .prepare import prepare_gpt2_with_handle
from .score import score_gpt2
from .train import train_prepared_gpt2

__all__ = [
    "EvalResult",
    "ModelTrainConfig",
    "PreparedLMModel",
    "evaluate_gpt2",
    "generate_gpt2",
    "load_prepared_gpt2_from_handle",
    "prepare_gpt2_with_handle",
    "save_prepared_gpt2",
    "score_gpt2",
    "train_prepared_gpt2",
]
