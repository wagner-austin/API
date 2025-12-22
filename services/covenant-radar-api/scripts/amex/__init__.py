"""AMEX Competition ensemble pipeline module.

Provides a complete pipeline for the Kaggle AMEX Default Prediction competition:
- Multi-model training with k-fold cross-validation
- OOF prediction collection and AMEX metric computation
- Ensemble weight optimization
- Submission generation

Usage:
    python -m scripts.amex [OPTIONS]

Components:
    types.py - TypedDicts for pipeline configuration and results
    _hooks.py - Dependency injection hooks for testability
    pipeline.py - Core pipeline functions
    __main__.py - CLI entry point

Example:
    config = make_default_config()
    result = run_pipeline(
        train_dir=train_path,
        test_dir=test_path,
        output_path=submission_path,
        config=config,
    )
    final_score = result["ensemble_result"]["optimized_score"]
"""

from scripts.amex.pipeline import (
    build_dataset_config,
    build_test_config,
    generate_ensemble_predictions,
    load_test_data,
    load_training_data,
    optimize_ensemble,
    run_pipeline,
    train_all_models,
    train_single_model,
    write_submission,
)
from scripts.amex.types import (
    AMEXPipelineConfig,
    EnsembleResult,
    ModelOOFResult,
    PipelineResult,
    make_default_config,
)

__all__ = [
    "AMEXPipelineConfig",
    "EnsembleResult",
    "ModelOOFResult",
    "PipelineResult",
    "build_dataset_config",
    "build_test_config",
    "generate_ensemble_predictions",
    "load_test_data",
    "load_training_data",
    "make_default_config",
    "optimize_ensemble",
    "run_pipeline",
    "train_all_models",
    "train_single_model",
    "write_submission",
]
