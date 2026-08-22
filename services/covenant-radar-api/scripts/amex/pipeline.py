"""AMEX Competition ensemble pipeline.

Trains multiple models with k-fold cross-validation, collects OOF predictions,
optimizes ensemble weights, and generates submission predictions.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import ClassifierBackend, PreparedClassifier
from covenant_ml.datasets import LoadedDataset
from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    ModelOOFPredictions,
    make_default_optimization_config,
)
from covenant_ml.metrics import compute_amex_metric
from covenant_ml.types import BackendName
from covenant_ml.validation import run_cross_validation
from covenant_ml.validation.runner import FoldTrainer
from numpy.typing import NDArray
from platform_core.logging import get_logger

from scripts.amex._configs import (
    _BackendTrainer,
    build_dataset_config,
    build_test_config,
)
from scripts.amex._hooks import (
    get_console,
    get_ensemble_optimizer,
    get_registry,
    get_timeseries_loader,
)
from scripts.amex.types import (
    AMEXPipelineConfig,
    EnsembleResult,
    ModelOOFResult,
    PipelineResult,
)

_log = get_logger(__name__)


# =============================================================================
# Dataset Configuration Builder
# =============================================================================


def load_training_data(
    data_dir: Path,
    config: AMEXPipelineConfig,
) -> LoadedDataset:
    """Load and aggregate AMEX training data.

    Args:
        data_dir: Directory containing training data.
        config: Pipeline configuration.

    Returns:
        LoadedDataset with aggregated features and labels.

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data format is invalid.
    """
    console = get_console()
    console.write(f"Loading training data from {data_dir}")

    dataset_config = build_dataset_config(
        data_dir,
        config["aggregation"],
        config["include_rank_features"],
        config["include_diff_features"],
        config["include_window_features"],
        config["window_sizes"],
    )

    loader = get_timeseries_loader()
    dataset = loader(dataset_config, data_dir.parent)

    console.write(
        f"Loaded {dataset['meta']['n_samples']} samples "
        f"with {dataset['meta']['n_features']} features"
    )
    return dataset


def train_single_model(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    feature_names: tuple[str, ...],
    backend_name: BackendName,
    n_folds: int,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
    output_dir: Path,
) -> ModelOOFResult:
    """Train a single model with k-fold CV and collect OOF predictions.

    Args:
        x: Feature matrix.
        y: Labels.
        feature_names: Feature column names.
        backend_name: ML backend to use.
        n_folds: Number of CV folds.
        n_estimators: Number of estimators.
        learning_rate: Learning rate.
        random_state: Random seed.
        output_dir: Directory for model artifacts.

    Returns:
        ModelOOFResult with OOF predictions and CV scores.
    """
    console = get_console()
    console.write(f"Training {backend_name} with {n_folds}-fold CV")

    # Get backend
    registry = get_registry()
    backend: ClassifierBackend = registry.get(backend_name)

    # Create model output directory
    model_dir = output_dir / backend_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer: FoldTrainer = _BackendTrainer(
        backend=backend,
        backend_name=backend_name,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        feature_names=feature_names,
        output_dir=model_dir,
    )

    # Run cross-validation
    cv_result = run_cross_validation(
        x=x,
        y=y,
        n_folds=n_folds,
        random_state=random_state,
        trainer=trainer,
        progress_callback=None,
    )

    # Compute AMEX scores for each fold
    cv_scores: list[float] = []
    for fold_result in cv_result["fold_results"]:
        val_indices = fold_result["val_indices"]
        val_preds = fold_result["val_predictions"]
        val_labels = y[val_indices]
        amex_result = compute_amex_metric(val_labels, val_preds)
        cv_scores.append(amex_result["score"])

    mean_cv_score = sum(cv_scores) / len(cv_scores)

    console.write(f"{backend_name} CV complete: mean AMEX score = {mean_cv_score:.5f}")

    # Build fold indices array
    n_samples = len(y)
    fold_indices: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    for fold_result in cv_result["fold_results"]:
        fold_num = fold_result["fold_number"]
        val_indices = fold_result["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            fold_indices[idx] = fold_num

    return ModelOOFResult(
        model_name=backend_name,
        oof_predictions=cv_result["oof_predictions"],
        fold_indices=fold_indices,
        cv_scores=tuple(cv_scores),
        mean_cv_score=mean_cv_score,
    )


def train_all_models(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    feature_names: tuple[str, ...],
    config: AMEXPipelineConfig,
    output_dir: Path,
) -> tuple[ModelOOFResult, ...]:
    """Train all models specified in config.

    Args:
        x: Feature matrix.
        y: Labels.
        feature_names: Feature column names.
        config: Pipeline configuration.
        output_dir: Directory for model artifacts.

    Returns:
        Tuple of ModelOOFResult for each backend.
    """
    results: list[ModelOOFResult] = []

    for backend_name in config["backends"]:
        result = train_single_model(
            x=x,
            y=y,
            feature_names=feature_names,
            backend_name=backend_name,
            n_folds=config["n_folds"],
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            random_state=config["random_state"],
            output_dir=output_dir,
        )
        results.append(result)

    return tuple(results)


# =============================================================================
# Ensemble Optimization
# =============================================================================


def optimize_ensemble(
    model_results: tuple[ModelOOFResult, ...],
    labels: NDArray[np.int64],
    random_state: int,
) -> EnsembleResult:
    """Optimize ensemble weights using OOF predictions.

    Args:
        model_results: OOF results from each model.
        labels: True labels.
        random_state: Random seed.

    Returns:
        EnsembleResult with optimized weights.
    """
    console = get_console()
    console.write("Optimizing ensemble weights")

    # Configure scipy (in case it hasn't been done)

    # Build EnsembleOOFData
    model_preds: list[ModelOOFPredictions] = []
    for result in model_results:
        model_preds.append(
            ModelOOFPredictions(
                model_name=result["model_name"],
                predictions=result["oof_predictions"],
                fold_indices=result["fold_indices"],
            )
        )

    oof_data = EnsembleOOFData(
        model_predictions=tuple(model_preds),
        labels=labels,
        n_samples=len(labels),
        n_models=len(model_results),
    )

    # Optimize
    opt_config = make_default_optimization_config(random_state)
    optimizer = get_ensemble_optimizer()
    opt_result = optimizer(oof_data, opt_config)

    # Convert weights to tuple
    weight_arr = opt_result["weights"]["weights"]
    weights_tuple: tuple[float, ...] = tuple(float(w) for w in weight_arr.flat)

    improvement = opt_result["best_score"] - opt_result["initial_score"]

    console.write(
        f"Ensemble optimization complete: "
        f"initial = {opt_result['initial_score']:.5f}, "
        f"optimized = {opt_result['best_score']:.5f}, "
        f"improvement = {improvement:+.5f}"
    )

    return EnsembleResult(
        model_names=opt_result["weights"]["model_names"],
        weights=weights_tuple,
        initial_score=opt_result["initial_score"],
        optimized_score=opt_result["best_score"],
        improvement=improvement,
    )


# =============================================================================
# Prediction and Submission
# =============================================================================


def load_test_data(
    data_dir: Path,
    config: AMEXPipelineConfig,
) -> LoadedDataset:
    """Load and aggregate AMEX test data.

    Args:
        data_dir: Directory containing test data.
        config: Pipeline configuration.

    Returns:
        LoadedDataset with aggregated features (y will be zeros).

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data format is invalid.
    """
    console = get_console()
    console.write(f"Loading test data from {data_dir}")

    dataset_config = build_test_config(
        data_dir,
        config["aggregation"],
        config["include_rank_features"],
        config["include_diff_features"],
        config["include_window_features"],
        config["window_sizes"],
    )

    loader = get_timeseries_loader()
    dataset = loader(dataset_config, data_dir.parent)

    console.write(f"Loaded {dataset['meta']['n_samples']} test samples")
    return dataset


def generate_ensemble_predictions(
    x_test: NDArray[np.float64],
    model_results: tuple[ModelOOFResult, ...],
    ensemble_result: EnsembleResult,
    config: AMEXPipelineConfig,
    output_dir: Path,
) -> NDArray[np.float64]:
    """Generate ensemble predictions for test data.

    This function loads the final fold models from each backend and
    generates weighted ensemble predictions.

    Args:
        x_test: Test feature matrix.
        model_results: OOF results (used for model names).
        ensemble_result: Optimized ensemble weights.
        config: Pipeline configuration.
        output_dir: Directory with saved models.

    Returns:
        Ensemble predictions array.
    """
    console = get_console()
    n_samples = x_test.shape[0]

    console.write(f"Generating predictions for {n_samples} test samples")

    # Get registry
    registry = get_registry()

    # Collect predictions from each model
    model_predictions: list[NDArray[np.float64]] = []

    for i, result in enumerate(model_results):
        backend_name_str = result["model_name"]
        # Convert string back to BackendName literal
        backend_name: BackendName = config["backends"][i]

        backend: ClassifierBackend = registry.get(backend_name)

        # Use last fold model for final predictions
        last_fold = config["n_folds"] - 1
        model_path = output_dir / backend_name_str / f"fold_{last_fold}" / "model"

        # Find actual model file
        model_files = list(model_path.parent.glob("*.pkl")) + list(model_path.parent.glob("*.json"))
        if not model_files:
            msg = f"No model file found in {model_path.parent}"
            raise FileNotFoundError(msg)

        actual_path = model_files[0]
        model: PreparedClassifier = backend.load(path=str(actual_path))

        # Get predictions
        raw_preds: NDArray[np.float64] = model.predict_proba(x_test)
        if raw_preds.ndim == 2:
            preds: NDArray[np.float64] = raw_preds[:, 1]
        else:
            preds = raw_preds

        model_predictions.append(preds)
        console.write(f"  {backend_name}: predictions generated")

    # Compute weighted ensemble
    ensemble_preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i, preds in enumerate(model_predictions):
        weight = ensemble_result["weights"][i]
        ensemble_preds += weight * preds

    return ensemble_preds


def write_submission(
    output_path: Path,
    predictions: NDArray[np.float64],
    n_samples: int,
) -> int:
    """Write predictions to submission CSV file.

    Args:
        output_path: Path to write submission CSV.
        predictions: Predicted probabilities.
        n_samples: Number of samples.

    Returns:
        Number of rows written.
    """
    console = get_console()
    console.write(f"Writing submission to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("customer_ID,prediction\n")
        for i in range(n_samples):
            # Use generic customer IDs (actual IDs would come from test data)
            customer_id = f"customer_{i:08d}"
            pred = float(predictions.item(i))
            f.write(f"{customer_id},{pred:.6f}\n")

    console.write(f"Wrote {n_samples} predictions")
    return n_samples


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    config: AMEXPipelineConfig,
    model_output_dir: Path | None = None,
) -> PipelineResult:
    """Run the full AMEX competition pipeline.

    Steps:
    1. Load training data with competition features
    2. Train each model with k-fold CV
    3. Optimize ensemble weights on OOF predictions
    4. Load test data
    5. Generate ensemble predictions
    6. Write submission CSV

    Args:
        train_dir: Directory with training data.
        test_dir: Directory with test data.
        output_path: Path to write submission CSV.
        config: Pipeline configuration.
        model_output_dir: Directory for model artifacts.

    Returns:
        PipelineResult with all results.

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data format is invalid.
    """
    console = get_console()
    console.write("=" * 60)
    console.write("AMEX Competition Pipeline")
    console.write("=" * 60)

    # Determine model output directory
    output_dir = model_output_dir if model_output_dir is not None else output_path.parent

    # Step 1: Load training data
    train_data = load_training_data(train_dir, config)
    n_samples_train = train_data["meta"]["n_samples"]
    n_features = train_data["meta"]["n_features"]

    # Step 2: Train all models
    model_results = train_all_models(
        x=train_data["x"],
        y=train_data["y"],
        feature_names=train_data["meta"]["feature_names"],
        config=config,
        output_dir=output_dir,
    )

    # Step 3: Optimize ensemble
    ensemble_result = optimize_ensemble(
        model_results=model_results,
        labels=train_data["y"],
        random_state=config["random_state"],
    )

    # Step 4: Load test data
    test_data = load_test_data(test_dir, config)
    n_samples_test = test_data["meta"]["n_samples"]

    # Step 5: Generate predictions
    predictions = generate_ensemble_predictions(
        x_test=test_data["x"],
        model_results=model_results,
        ensemble_result=ensemble_result,
        config=config,
        output_dir=output_dir,
    )

    # Step 6: Write submission
    write_submission(output_path, predictions, n_samples_test)

    console.write("=" * 60)
    console.write("Pipeline complete!")
    console.write(f"Submission saved to: {output_path}")
    console.write("=" * 60)

    return PipelineResult(
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test,
        n_features=n_features,
        model_results=model_results,
        ensemble_result=ensemble_result,
        submission_path=str(output_path),
    )


__all__ = [
    "build_dataset_config",
    "build_test_config",
    "generate_ensemble_predictions",
    "load_test_data",
    "load_training_data",
    "optimize_ensemble",
    "run_pipeline",
    "train_all_models",
    "train_single_model",
    "write_submission",
]
